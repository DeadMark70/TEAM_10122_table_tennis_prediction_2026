"""R21B smoke: Extended OpenTTGames auxiliary pretraining -> AICUP fine-tune.

This script is intentionally conservative:
- external data is used only for auxiliary masked-field pretraining;
- no external rows are mapped into AICUP actionId/pointId/server labels;
- evaluation remains the existing AICUP GroupKFold-by-match sampled-prefix CV.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

import baseline_v10a_pretrain_transformer as v10
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, add_role_and_score_features, sample_validation_prefixes, validate_raw_data
from baseline_v2 import blend_probs
from baseline_v3 import apply_segmented_multipliers, tune_segmented_multipliers
from baseline_v5_gru import build_sequence_arrays, build_train_meta, fit_numeric_stats


EXT_FIELDS = [
    "event_type",
    "player_side",
    "stroke_hand",
    "technique",
    "safe_action_family",
    "lean",
    "feet",
    "rally_ending_type",
    "safe_terminal_label",
]

EXT_VOCABS = {
    "event_type": ["unknown", "stroke", "bounce", "net", "rally_ending", "empty_event"],
    "player_side": ["", "left", "right"],
    "stroke_hand": ["", "forehand", "backhand"],
    "technique": ["", "block", "chop", "flick", "lob", "loop", "push", "serve", "smash"],
    "safe_action_family": ["", "serve_family", "attack_family", "control_family", "defensive_or_control_family"],
    "lean": ["", "back_heavy", "front_heavy", "right_leaning", "left_leaning", "neutral", "unknown"],
    "feet": ["", "both_feet_planted", "both_feet_lifted", "right_foot_lifted", "left_foot_lifted", "unknown"],
    "rally_ending_type": ["", "net", "not_hitting_ball", "winner", "double_bounce", "out", "miss_on_own_side"],
    "safe_terminal_label": ["", "1"],
}


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R21B OpenTTGames pretraining smoke.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--external-events", default="external_data/openttgames/processed/openttgames_events.csv")
    parser.add_argument("--cv-report", default="cv_report_r21b.csv")
    parser.add_argument("--feature-report", default="feature_report_r21b.json")
    parser.add_argument("--oof-proba", default="oof_proba_r21b.pkl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=24)
    parser.add_argument("--numeric-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--external-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=4)
    parser.add_argument("--lr-external", type=float, default=7e-4)
    parser.add_argument("--lr-finetune", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mask-prob", type=float, default=0.35)
    parser.add_argument("--multiplier-bins", choices=["global", "two", "five"], default="two")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_segments(events: pd.DataFrame, max_len: int) -> list[pd.DataFrame]:
    segments: list[pd.DataFrame] = []
    for _, group in events.sort_values(["split", "video_id", "frame"]).groupby(["split", "video_id"], sort=False):
        cur = []
        for row in group.itertuples(index=False):
            if getattr(row, "event_type") == "empty_event":
                if cur and any(r.event_type == "stroke" for r in cur):
                    segments.append(pd.DataFrame([r._asdict() for r in cur]).tail(max_len).reset_index(drop=True))
                cur = []
            else:
                cur.append(row)
        if cur and any(r.event_type == "stroke" for r in cur):
            segments.append(pd.DataFrame([r._asdict() for r in cur]).tail(max_len).reset_index(drop=True))
    return segments


class OpenTTGamesDataset(Dataset):
    def __init__(self, segments: list[pd.DataFrame], max_len: int, mask_prob: float, seed: int) -> None:
        self.max_len = int(max_len)
        self.mask_prob = float(mask_prob)
        self.seed = int(seed)
        self.cat = []
        self.targets = []
        self.lengths = []
        self.num = []
        self.mask_ids = np.asarray([len(EXT_VOCABS[f]) + 1 for f in EXT_FIELDS], dtype=np.int64)
        maps = {f: {v: i for i, v in enumerate(EXT_VOCABS[f])} for f in EXT_FIELDS}
        for seg in segments:
            seg = seg.reset_index(drop=True).tail(max_len)
            length = len(seg)
            cat = np.zeros((max_len, len(EXT_FIELDS)), dtype=np.int64)
            target = np.full((max_len, len(EXT_FIELDS)), -100, dtype=np.int64)
            num = np.zeros((max_len, 2), dtype=np.float32)
            frames = seg["frame"].to_numpy(dtype=np.float32)
            if length:
                frame0 = frames[0]
                denom = max(float(frames[-1] - frame0), 1.0)
            for i, row in enumerate(seg.itertuples(index=False)):
                for j, field in enumerate(EXT_FIELDS):
                    val = getattr(row, field)
                    if pd.isna(val):
                        val = ""
                    val = str(val)
                    cls = maps[field].get(val, 0)
                    cat[i, j] = cls + 1
                    target[i, j] = cls
                if length:
                    num[i, 0] = (float(row.frame) - frame0) / denom
                    num[i, 1] = i / max(length - 1, 1)
            self.cat.append(cat)
            self.targets.append(target)
            self.lengths.append(length)
            self.num.append(num)
        self.cat = np.stack(self.cat).astype(np.int64)
        self.targets = np.stack(self.targets).astype(np.int64)
        self.lengths = np.asarray(self.lengths, dtype=np.int64)
        self.num = np.stack(self.num).astype(np.float32)

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + idx)
        cat = self.cat[idx].copy()
        target = self.targets[idx].copy()
        length = int(self.lengths[idx])
        masked_targets = np.full_like(target, -100)
        if length > 0:
            for j in range(len(EXT_FIELDS)):
                mask = rng.random(length) < self.mask_prob
                if not mask.any():
                    mask[int(rng.integers(0, length))] = True
                masked_targets[:length, j][mask] = target[:length, j][mask]
                cat[:length, j][mask] = self.mask_ids[j]
        return {
            "cat": torch.from_numpy(cat).long(),
            "num": torch.from_numpy(self.num[idx]).float(),
            "lengths": torch.tensor(length, dtype=torch.long),
            "targets": torch.from_numpy(masked_targets).long(),
        }


class ExternalTransformer(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        cards = [len(EXT_VOCABS[f]) + 2 for f in EXT_FIELDS]
        self.embeddings = nn.ModuleList([nn.Embedding(card, args.emb_dim, padding_idx=0) for card in cards])
        self.numeric = nn.Sequential(nn.Linear(2, args.numeric_dim), nn.LayerNorm(args.numeric_dim), nn.GELU())
        self.input_proj = nn.Sequential(
            nn.Linear(args.emb_dim * len(EXT_FIELDS) + args.numeric_dim, args.d_model),
            nn.LayerNorm(args.d_model),
            nn.GELU(),
        )
        self.pos_emb = nn.Embedding(args.max_len, args.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.num_heads,
            dim_feedforward=args.d_model * 4,
            dropout=args.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=args.num_layers)
        self.heads = nn.ModuleList([nn.Linear(args.d_model, len(EXT_VOCABS[f])) for f in EXT_FIELDS])

    def forward(self, cat: torch.Tensor, num: torch.Tensor, lengths: torch.Tensor) -> list[torch.Tensor]:
        bsz, seq_len, _ = cat.shape
        x = torch.cat([emb(cat[:, :, i]) for i, emb in enumerate(self.embeddings)] + [self.numeric(num)], dim=-1)
        x = self.input_proj(x)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos)
        pad_mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        return [head(h) for head in self.heads]


def ext_loss(outputs: list[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    losses = []
    for j, logits in enumerate(outputs):
        target = targets[:, :, j]
        if target.ge(0).any():
            losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100))
    return torch.stack(losses).mean() if losses else outputs[0].sum() * 0.0


def train_external_encoder(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    events = pd.read_csv(args.external_events).fillna("")
    segments = make_segments(events, args.max_len)
    ds = OpenTTGamesDataset(segments, args.max_len, args.mask_prob, args.seed + 2100)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    model = ExternalTransformer(args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_external, weight_decay=args.weight_decay)
    rows = []
    for epoch in range(1, args.external_epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss = ext_loss(model(batch["cat"], batch["num"], batch["lengths"]), batch["targets"])
            loss.backward()
            clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        rows.append({"epoch": epoch, "external_loss": float(np.mean(losses)), "segments": len(ds)})
        print(f"external epoch {epoch:02d}: loss={rows[-1]['external_loss']:.5f} segments={len(ds)}")
    return {
        "encoder": {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()},
        "pos_emb": {k: v.detach().cpu().clone() for k, v in model.pos_emb.state_dict().items()},
        "external_rows": rows,
    }


def class_weights(values: pd.Series, classes: list[int], beta: float = 0.25) -> torch.Tensor:
    counts = values.value_counts().to_dict()
    weights = np.array([float(counts.get(cls, 1)) ** (-beta) for cls in classes], dtype=np.float32)
    return torch.from_numpy(weights / weights.mean())


def normalize_rows(prob: np.ndarray) -> np.ndarray:
    return prob / prob.sum(axis=1, keepdims=True)


def compose_v3_subset(path: str, valid_meta: pd.DataFrame):
    with open(path, "rb") as f:
        oof = pickle.load(f)
    tuning = oof["tuning"]
    meta = oof["valid_meta"].reset_index(drop=True)
    key_cols = ["rally_uid", "prefix_len"]
    merged = valid_meta[key_cols].reset_index().merge(meta[key_cols].reset_index(), on=key_cols, how="left", suffixes=("", "_v3"))
    if merged["index_v3"].isna().any():
        raise ValueError("Could not align valid rows to V3 OOF.")
    idx = merged["index_v3"].astype(int).to_numpy()
    action = blend_probs(oof["lgbm_action"], oof["ngram_action"], tuning.action_ngram_weight)[idx]
    point = blend_probs(oof["lgbm_point"], oof["ngram_point"], tuning.point_ngram_weight)[idx]
    sw = tuning.server_weights
    server = (
        sw["direct"] * oof["lgbm_server"]
        + sw["ngram"] * oof["ngram_server"]
        + sw["parity"] * oof["parity_server"]
        + sw["remaining"] * oof["remaining_server"]
    )[idx]
    return action, point, server, tuning


def evaluate(meta, action_prob, point_prob, server_prob, action_mult=None, point_mult=None, bins_mode="global") -> dict[str, float]:
    if action_mult is None:
        action_pred = np.asarray(ACTION_CLASSES)[np.argmax(action_prob, axis=1)]
    else:
        action_pred = apply_segmented_multipliers(meta, action_prob, action_mult, ACTION_CLASSES, bins_mode)
    if point_mult is None:
        point_pred = np.asarray(POINT_CLASSES)[np.argmax(point_prob, axis=1)]
    else:
        point_pred = apply_segmented_multipliers(meta, point_prob, point_mult, POINT_CLASSES, bins_mode)
    action_f1 = f1_score(meta["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
    point_f1 = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
    server_auc = roc_auc_score(meta["serverGetPoint"], server_prob)
    return {
        "action_macro_f1": float(action_f1),
        "point_macro_f1": float(point_f1),
        "server_auc": float(server_auc),
        "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
    }


def train_aicup_model(
    train: pd.DataFrame,
    train_meta: pd.DataFrame,
    valid_arrays,
    valid_meta: pd.DataFrame,
    num_mean: np.ndarray,
    num_std: np.ndarray,
    cat_cards: list[int],
    mask_ids: list[int],
    class_sizes: list[int],
    args: argparse.Namespace,
    device: torch.device,
    external_state: dict | None,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    train_arrays = build_sequence_arrays(train, train_meta, args.max_len, num_mean, num_std)
    action_w = class_weights(train_meta["next_actionId"], ACTION_CLASSES)
    point_w = class_weights(train_meta[train_meta["next_pointId"].gt(0)]["next_pointId"] - 1, list(range(9)))
    mask_field_indices = [v10.CAT_FIELDS.index(f) for f in v10.MASK_FIELDS]
    model = v10.StrokeTransformer(
        cat_cards,
        class_sizes,
        mask_field_indices,
        len(v10.NUM_FIELDS),
        args.max_len,
        args.d_model,
        args.emb_dim,
        args.numeric_dim,
        args.num_layers,
        args.num_heads,
        args.dropout,
    ).to(device)
    if external_state is not None:
        model.encoder.load_state_dict(external_state["encoder"], strict=True)
        model.pos_emb.load_state_dict(external_state["pos_emb"], strict=True)
    ds = v10.StrokeEvalDataset(train_arrays)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(args.seed + 91))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_finetune, weight_decay=args.weight_decay)
    best_state = None
    best = {"overall": -1.0}
    rows = []
    for epoch in range(1, args.finetune_epochs + 1):
        losses = v10.train_epoch(model, loader, opt, action_w, point_w, args, device, pretrain=False)
        a, p, s = v10.predict_model(model, valid_arrays, args.batch_size, device)
        metrics = evaluate(valid_meta, a, p, s)
        rows.append({"model": label, "epoch": epoch, **metrics, "loss": losses["loss"]})
        print(f"{label} finetune {epoch:02d}: overall={metrics['overall']:.6f} action={metrics['action_macro_f1']:.6f} point={metrics['point_macro_f1']:.6f} server={metrics['server_auc']:.6f}")
        if metrics["overall"] > best["overall"]:
            best = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    a, p, s = v10.predict_model(model, valid_arrays, args.batch_size, device)
    return a, p, s, {"best_metrics": best, "epoch_rows": rows}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)
    prefix_meta = build_train_meta(train)
    num_mean, num_std = fit_numeric_stats(train)
    cat_cards, mask_ids, class_sizes = v10.cat_cardinalities_with_mask(train, test)

    external_state = train_external_encoder(args, device)

    rally_meta = prefix_meta[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=args.folds)
    test_lengths = test.groupby("rally_uid").size().to_numpy(dtype=int)
    all_rows = []
    valid_parts = []
    outputs = {"scratch_action": [], "scratch_point": [], "scratch_server": [], "ext_action": [], "ext_point": [], "ext_server": []}

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(rally_meta, groups=rally_meta["match"]), start=1):
        if fold > args.fold_limit:
            break
        tr_ids = set(rally_meta.iloc[tr_idx]["rally_uid"])
        va_ids = set(rally_meta.iloc[va_idx]["rally_uid"])
        tr_meta = prefix_meta[prefix_meta["rally_uid"].isin(tr_ids)].copy().reset_index(drop=True)
        va_pool = prefix_meta[prefix_meta["rally_uid"].isin(va_ids)].copy()
        va_idx_sample = sample_validation_prefixes(va_pool, test_lengths, args.seed + fold)
        va_meta = va_pool.loc[va_idx_sample].copy().reset_index(drop=True)
        valid_arrays = build_sequence_arrays(train, va_meta, args.max_len, num_mean, num_std)
        print(f"fold {fold}: train={len(tr_meta)} valid={len(va_meta)} device={device}")
        scratch_a, scratch_p, scratch_s, scratch_info = train_aicup_model(
            train, tr_meta, valid_arrays, va_meta, num_mean, num_std, cat_cards, mask_ids, class_sizes, args, device, None, "scratch"
        )
        ext_a, ext_p, ext_s, ext_info = train_aicup_model(
            train, tr_meta, valid_arrays, va_meta, num_mean, num_std, cat_cards, mask_ids, class_sizes, args, device, external_state, "opentt_pretrained"
        )
        for row in scratch_info["epoch_rows"] + ext_info["epoch_rows"]:
            row["fold"] = fold
            all_rows.append(row)
        valid_parts.append(va_meta)
        outputs["scratch_action"].append(scratch_a)
        outputs["scratch_point"].append(scratch_p)
        outputs["scratch_server"].append(scratch_s)
        outputs["ext_action"].append(ext_a)
        outputs["ext_point"].append(ext_p)
        outputs["ext_server"].append(ext_s)

    valid_meta = pd.concat(valid_parts, ignore_index=True)
    for key in list(outputs):
        outputs[key] = np.vstack(outputs[key]) if "server" not in key else np.concatenate(outputs[key])
    v3_action, v3_point, v3_server, v3_tuning = compose_v3_subset(args.v3_oof, valid_meta)

    summary_rows = []
    for label, action, point, server in [
        ("scratch_single", outputs["scratch_action"], outputs["scratch_point"], outputs["scratch_server"]),
        ("opentt_single", outputs["ext_action"], outputs["ext_point"], outputs["ext_server"]),
    ]:
        summary_rows.append({"variant": label, "action_weight": 1.0, "server_weight": 1.0, **evaluate(valid_meta, action, point, server)})
    for label, action, server in [
        ("v3_plus_scratch", outputs["scratch_action"], outputs["scratch_server"]),
        ("v3_plus_opentt", outputs["ext_action"], outputs["ext_server"]),
    ]:
        for aw in [0.0, 0.1, 0.2, 0.3, 0.4]:
            action_prob = blend_probs(v3_action, action, aw)
            action_mult = tune_segmented_multipliers(valid_meta, action_prob, ACTION_CLASSES, "action", args.multiplier_bins)
            for sw in [0.0, 0.1, 0.2, 0.3]:
                server_prob = (1.0 - sw) * v3_server + sw * server
                summary_rows.append(
                    {
                        "variant": label,
                        "action_weight": aw,
                        "server_weight": sw,
                        **evaluate(valid_meta, action_prob, v3_point, server_prob, action_mult, v3_tuning.point_multipliers, args.multiplier_bins),
                    }
                )
    report = pd.DataFrame(summary_rows).sort_values("overall", ascending=False)
    report.to_csv(args.cv_report, index=False)
    with open(args.oof_proba, "wb") as f:
        pickle.dump({"valid_meta": valid_meta, **outputs, "summary": report}, f)
    feature_report = {
        "args": vars(args),
        "external_pretrain": {
            "fields": EXT_FIELDS,
            "vocabs": EXT_VOCABS,
            "rows": external_state["external_rows"],
        },
        "epoch_rows": all_rows,
        "selected": report.iloc[0].to_dict(),
        "decision": "R21B smoke only; no submission generated.",
    }
    Path(args.feature_report).write_text(json.dumps(feature_report, indent=2), encoding="utf-8")
    print("R21B selected diagnostic:")
    print(json.dumps(feature_report["selected"], indent=2))
    print(f"wrote {args.cv_report}")
    print(f"wrote {args.oof_proba}")
    print(f"wrote {args.feature_report}")


if __name__ == "__main__":
    main()
