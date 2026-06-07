"""V208 action-first TT-ShuttleNet.

This experiment applies the useful ShuttleNet idea to actionId instead of point:
action/shot type is the main target, while point/depth/terminal are auxiliary
heads that regularize the representation.  Exported submissions change action
only; point stays at V188 r186_w005 cap5 and server stays at R121.

No ShuttleSet/CoachAI/TTMATCH external rows are read.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from analysis_r179_action_physics_hierarchy import normalize_rows_safe
from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions
from analysis_v188_point_intent_gru import row_log_blend, set_seed
from analysis_v195_distribution_matched_point_gru import MATCH_COLS, distribution_match_weights, prepare_data
from baseline_lgbm import ACTION_CLASSES
from train_v203_tt_shuttlenet import AREA_IDXS, TYPE_IDXS, TTBatch, collate


OUTDIR = Path("v208_action_ttshuttlenet")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
SRC_DEST = Path("src/analysis/analysis_v208_action_ttshuttlenet.py")

POINT_ANCHOR = UPLOAD_DIR / "submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv"
SERVER_ANCHOR = UPLOAD_DIR / "submission_r121_traj_min_w0p2.csv"
V173_ACTION = UPLOAD_DIR / "submission_v173__av173_best_action__pr119_public_point__sr121_min_w0p2.csv"

BATCH_SIZE = 512
EPOCHS = 8
PATIENCE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEAK_CLASSES = {3, 4, 7, 8, 9, 11, 12, 14}
BLEND_WEIGHTS = [0.03, 0.05, 0.075, 0.10]
ACTION_CAPS = [0.01, 0.02, 0.03, 0.05]


@dataclass
class ActionBatch:
    type_strokes: torch.Tensor
    area_strokes: torch.Tensor
    lengths: torch.Tensor
    static: torch.Tensor
    action: torch.Tensor
    point: torch.Tensor
    family: torch.Tensor
    terminal: torch.Tensor


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class GrUTuning:
    action_gru_weight: float
    point_gru_weight: float
    server_gru_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class TransformerTuning:
    action_weight: float
    point_weight: float
    server_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


def action_family_targets(action: np.ndarray) -> np.ndarray:
    y = np.asarray(action, dtype=int)
    out = np.zeros(len(y), dtype=np.int64)
    out[(1 <= y) & (y <= 7)] = 1
    out[(8 <= y) & (y <= 11)] = 2
    out[(12 <= y) & (y <= 14)] = 3
    out[(15 <= y) & (y <= 18)] = 4
    return out


def weak_action_mask(labels: torch.Tensor) -> torch.Tensor:
    allowed = torch.tensor(sorted(WEAK_CLASSES), device=labels.device, dtype=labels.dtype)
    return (labels[..., None] == allowed).any(dim=-1)


def blend_action_probs(base_prob: np.ndarray, model_prob: np.ndarray, weight: float) -> np.ndarray:
    return normalize_rows_safe((1.0 - float(weight)) * np.asarray(base_prob, dtype=float) + float(weight) * np.asarray(model_prob, dtype=float))


def class_gated_action_labels(base_labels: np.ndarray, blended_prob: np.ndarray, allowed_targets: set[int]) -> np.ndarray:
    base = np.asarray(base_labels, dtype=int)
    pred = np.asarray(blended_prob).argmax(axis=1).astype(int)
    mask = np.array([int(p) in allowed_targets for p in pred], dtype=bool) & (pred != base)
    out = base.copy()
    out[mask] = pred[mask]
    return out


def apply_action_residual(
    base_labels: np.ndarray,
    prob: np.ndarray,
    max_churn: float,
    allowed_targets: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base_labels, dtype=int)
    pred = np.asarray(prob).argmax(axis=1).astype(int)
    changed = pred != base
    if allowed_targets is not None:
        changed &= np.array([int(p) in allowed_targets for p in pred], dtype=bool)
    max_rows = int(np.floor(len(base) * float(max_churn)))
    if changed.sum() > max_rows:
        base_score = prob[np.arange(len(prob)), base]
        pred_score = prob[np.arange(len(prob)), pred]
        gain = pred_score - base_score
        cand = np.where(changed)[0]
        keep = cand[np.argsort(gain[cand])[::-1][:max_rows]]
        capped = np.zeros(len(base), dtype=bool)
        capped[keep] = True
        changed = capped
    out = base.copy()
    out[changed] = pred[changed]
    return out, changed


class ActionDataset(Dataset):
    def __init__(self, strokes: np.ndarray, lengths: np.ndarray, static: np.ndarray, action: np.ndarray, point: np.ndarray):
        self.type_strokes = torch.as_tensor(strokes[:, :, TYPE_IDXS], dtype=torch.long)
        self.area_strokes = torch.as_tensor(strokes[:, :, AREA_IDXS], dtype=torch.long)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long)
        self.static = torch.as_tensor(static, dtype=torch.float32)
        self.action = torch.as_tensor(np.asarray(action, dtype=np.int64).copy(), dtype=torch.long)
        self.point = torch.as_tensor(np.asarray(point, dtype=np.int64).copy(), dtype=torch.long)
        self.family = torch.as_tensor(action_family_targets(action), dtype=torch.long)
        self.terminal = torch.as_tensor((np.asarray(point, dtype=int) == 0).astype(np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.action)

    def __getitem__(self, idx: int) -> ActionBatch:
        return ActionBatch(
            self.type_strokes[idx],
            self.area_strokes[idx],
            self.lengths[idx],
            self.static[idx],
            self.action[idx],
            self.point[idx],
            self.family[idx],
            self.terminal[idx],
        )


def action_collate(batch: list[ActionBatch]) -> ActionBatch:
    return ActionBatch(
        type_strokes=torch.stack([b.type_strokes for b in batch]),
        area_strokes=torch.stack([b.area_strokes for b in batch]),
        lengths=torch.stack([b.lengths for b in batch]),
        static=torch.stack([b.static for b in batch]),
        action=torch.stack([b.action for b in batch]),
        point=torch.stack([b.point for b in batch]),
        family=torch.stack([b.family for b in batch]),
        terminal=torch.stack([b.terminal for b in batch]),
    )


class ActionTTShuttleNet(nn.Module):
    def __init__(self, vocab_sizes: list[int], static_dim: int, emb_dim: int = 8, hidden: int = 64):
        super().__init__()
        type_vocab = [vocab_sizes[i] for i in TYPE_IDXS]
        area_vocab = [vocab_sizes[i] for i in AREA_IDXS]
        self.type_emb = nn.ModuleList([nn.Embedding(v, emb_dim, padding_idx=0) for v in type_vocab])
        self.area_emb = nn.ModuleList([nn.Embedding(v, emb_dim, padding_idx=0) for v in area_vocab])
        self.type_gru = nn.GRU(emb_dim * len(type_vocab), hidden, batch_first=True)
        self.area_gru = nn.GRU(emb_dim * len(area_vocab), hidden, batch_first=True)
        self.static_net = nn.Sequential(nn.Linear(static_dim, 64), nn.ReLU(), nn.Dropout(0.10), nn.Linear(64, 32), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(hidden * 2 + 32, 3), nn.Softmax(dim=1))
        self.shared = nn.Sequential(nn.Linear(hidden * 2 + 32, 128), nn.ReLU(), nn.Dropout(0.15), nn.Linear(128, 96), nn.ReLU())
        self.action = nn.Linear(96, 19)
        self.family = nn.Linear(96, 5)
        self.point = nn.Linear(96, 10)
        self.terminal = nn.Linear(96, 2)

    @staticmethod
    def encode(gru: nn.GRU, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = gru(packed)
        return h[-1]

    def forward(self, type_strokes: torch.Tensor, area_strokes: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor) -> dict[str, torch.Tensor]:
        type_x = torch.cat([emb(type_strokes[:, :, i]) for i, emb in enumerate(self.type_emb)], dim=2)
        area_x = torch.cat([emb(area_strokes[:, :, i]) for i, emb in enumerate(self.area_emb)], dim=2)
        type_h = self.encode(self.type_gru, type_x, lengths)
        area_h = self.encode(self.area_gru, area_x, lengths)
        static_h = self.static_net(static)
        w = self.gate(torch.cat([type_h, area_h, static_h], dim=1))
        z = torch.cat([type_h * w[:, [0]], area_h * w[:, [1]], static_h * w[:, [2]]], dim=1)
        h = self.shared(z)
        return {"action": self.action(h), "family": self.family(h), "point": self.point(h), "terminal": self.terminal(h)}


def action_loss(outputs: dict[str, torch.Tensor], batch: ActionBatch, aux: bool) -> torch.Tensor:
    action = batch.action.to(outputs["action"].device)
    family = batch.family.to(outputs["action"].device)
    point = batch.point.to(outputs["action"].device)
    terminal = batch.terminal.to(outputs["action"].device)
    loss = F.cross_entropy(outputs["action"], action)
    loss = loss + 0.25 * F.cross_entropy(outputs["family"], family)
    weak = weak_action_mask(action)
    if weak.any():
        loss = loss + 0.15 * F.cross_entropy(outputs["action"][weak], action[weak])
    if aux:
        loss = loss + 0.10 * F.cross_entropy(outputs["point"], point)
        loss = loss + 0.05 * F.cross_entropy(outputs["terminal"], terminal)
    return loss


def train_model(
    train_ds: ActionDataset,
    valid_ds: ActionDataset,
    vocab_sizes: list[int],
    static_dim: int,
    seed: int,
    sample_weights: np.ndarray | None,
    aux: bool,
) -> tuple[ActionTTShuttleNet, float]:
    set_seed(seed)
    model = ActionTTShuttleNet(vocab_sizes, static_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    if sample_weights is None:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=action_collate)
    else:
        sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=action_collate)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=action_collate)
    best_state = None
    best_loss = float("inf")
    bad = 0
    for _ in range(EPOCHS):
        model.train()
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            out = model(batch.type_strokes.to(DEVICE), batch.area_strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))
            loss = action_loss(out, batch, aux)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        val_loss = 0.0
        n = 0
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                out = model(batch.type_strokes.to(DEVICE), batch.area_strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))
                loss = action_loss(out, batch, aux)
                val_loss += float(loss.item()) * len(batch.action)
                n += len(batch.action)
        val_loss /= max(n, 1)
        if val_loss + 1e-5 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss


def predict_action_proba(model: ActionTTShuttleNet, dataset: ActionDataset) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=action_collate)
    probs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.type_strokes.to(DEVICE), batch.area_strokes.to(DEVICE), batch.lengths.to(DEVICE), batch.static.to(DEVICE))["action"]
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return normalize_rows_safe(np.vstack(probs))


def make_dataset(data: dict, source: str, idx: np.ndarray | slice) -> ActionDataset:
    if source == "oof":
        return ActionDataset(data["oof_seq"][idx], data["oof_len"][idx], data["x_oof"][idx], data["rows"]["next_actionId"].to_numpy(dtype=int)[idx], data["y_oof"][idx])
    if source == "full":
        return ActionDataset(data["full_seq"][idx], data["full_len"][idx], data["x_full"][idx], data["full_pool"]["next_actionId"].to_numpy(dtype=int)[idx], data["y_full"][idx])
    if source == "test":
        n = len(data["test_seq"][idx])
        return ActionDataset(data["test_seq"][idx], data["test_len"][idx], data["x_test_fullstats"][idx], np.zeros(n, dtype=int), np.zeros(n, dtype=int))
    raise ValueError(source)


def run_scheme(data: dict, aux: bool, tag: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = data["rows"]
    train_rows = data["full_pool"]
    weights = distribution_match_weights(train_rows, data["test_rows"], MATCH_COLS)
    test_ds = make_dataset(data, "test", slice(None))
    oof = np.zeros((len(rows), 19), dtype=float)
    test_probs = []
    folds = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train_mask = ~train_rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train_idx = np.where(train_mask)[0]
        train_ds = make_dataset(data, "full", train_idx)
        valid_ds = make_dataset(data, "oof", valid)
        model, val_loss = train_model(train_ds, valid_ds, data["vocab_sizes"], data["x_full"].shape[1], 2080 + int(fold) + (100 if aux else 0), weights[train_idx], aux)
        oof[valid] = predict_action_proba(model, valid_ds)
        test_probs.append(predict_action_proba(model, test_ds))
        pred = oof[valid].argmax(axis=1).astype(int)
        y = rows.loc[valid, "next_actionId"].astype(int).to_numpy()
        folds.append({"scheme": tag, "fold": int(fold), "val_loss": float(val_loss), "raw_action_macro_f1": float(f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0))})
    return normalize_rows_safe(oof), normalize_rows_safe(np.mean(test_probs, axis=0)), folds


def eval_action_candidate(name: str, y: np.ndarray, pred: np.ndarray, anchor: np.ndarray, meta: dict) -> dict:
    score = float(f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0))
    anchor_score = float(f1_score(y, anchor, labels=ACTION_CLASSES, average="macro", zero_division=0))
    rec = {
        "candidate": name,
        "action_macro_f1": score,
        "delta_vs_v173_anchor": score - anchor_score,
        "action_churn_vs_v173_anchor": float(np.mean(pred != anchor)),
        "changed_rows": int(np.sum(pred != anchor)),
    }
    rec.update(meta)
    return rec


def load_sub(path: Path, rally_uids: np.ndarray | None = None) -> pd.DataFrame:
    sub = pd.read_csv(path)
    if rally_uids is None:
        return sub
    return pd.DataFrame({"rally_uid": rally_uids.astype(int)}).merge(sub, on="rally_uid", how="left", validate="one_to_one")


def write_action_submission(name: str, action: np.ndarray, point_src: pd.DataFrame, server_src: pd.DataFrame) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "rally_uid": point_src["rally_uid"].astype(int),
            "actionId": np.asarray(action, dtype=int),
            "pointId": point_src["pointId"].astype(int),
            "serverGetPoint": server_src["serverGetPoint"].astype(float),
        }
    )
    path = OUTDIR / name
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    out.to_csv(path, index=False, float_format="%.8f")
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"submission": name, "path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    random.seed(208)
    np.random.seed(208)
    set_seed(208)
    data = prepare_data()
    state = rebuild_v173_best_actions()
    y = data["rows"]["next_actionId"].astype(int).to_numpy()
    v173_oof = state["v173_pred_oof"].astype(int)
    v173_test = state["v173_pred_test"].astype(int)
    v173_oof_prob = np.eye(19)[v173_oof]
    v173_test_prob = np.eye(19)[v173_test]

    records = [eval_action_candidate("v173_rebuilt_anchor", y, v173_oof, v173_oof, {"scheme": "anchor", "weight": 0.0})]
    pred_store = {}
    fold_rows = []
    for tag, aux in [("action_family_only", False), ("action_point_aux", True)]:
        oof_prob, test_prob, folds = run_scheme(data, aux, tag)
        fold_rows.extend(folds)
        raw_pred = oof_prob.argmax(axis=1).astype(int)
        records.append(eval_action_candidate(f"v208_{tag}_raw_diagnostic", y, raw_pred, v173_oof, {"scheme": tag, "weight": 1.0, "gate": "raw"}))
        for w in BLEND_WEIGHTS:
            blend = blend_action_probs(v173_oof_prob, oof_prob, w)
            blend_test = blend_action_probs(v173_test_prob, test_prob, w)
            pred = blend.argmax(axis=1).astype(int)
            test_pred = blend_test.argmax(axis=1).astype(int)
            name = f"v208_{tag}_w{str(w).replace('.', 'p')}"
            records.append(eval_action_candidate(name, y, pred, v173_oof, {"scheme": tag, "weight": w, "gate": "all"}))
            pred_store[name] = test_pred
            gated = class_gated_action_labels(v173_oof, blend, WEAK_CLASSES)
            gated_test = class_gated_action_labels(v173_test, blend_test, WEAK_CLASSES)
            gname = f"v208_{tag}_classgate_w{str(w).replace('.', 'p')}"
            records.append(eval_action_candidate(gname, y, gated, v173_oof, {"scheme": tag, "weight": w, "gate": "weak_class_targets"}))
            pred_store[gname] = gated_test
        for cap in ACTION_CAPS:
            pred, changed = apply_action_residual(v173_oof, oof_prob, cap)
            test_pred, test_changed = apply_action_residual(v173_test, test_prob, cap)
            name = f"v208_{tag}_residual_cap{str(cap).replace('.', 'p')}"
            rec = eval_action_candidate(name, y, pred, v173_oof, {"scheme": tag, "weight": 1.0, "gate": "residual_cap", "cap": cap})
            rec["test_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
            rec["test_changed_rows"] = int(test_changed.sum())
            records.append(rec)
            pred_store[name] = test_pred
            weak_pred, weak_changed = apply_action_residual(v173_oof, oof_prob, cap, allowed_targets=WEAK_CLASSES)
            weak_test, weak_test_changed = apply_action_residual(v173_test, test_prob, cap, allowed_targets=WEAK_CLASSES)
            wname = f"v208_{tag}_weakresid_cap{str(cap).replace('.', 'p')}"
            wrec = eval_action_candidate(wname, y, weak_pred, v173_oof, {"scheme": tag, "weight": 1.0, "gate": "weak_residual_cap", "cap": cap})
            wrec["test_churn_vs_v173"] = float(np.mean(weak_test != v173_test))
            wrec["test_changed_rows"] = int(weak_test_changed.sum())
            records.append(wrec)
            pred_store[wname] = weak_test

    search = pd.DataFrame(records).sort_values(["action_macro_f1", "action_churn_vs_v173_anchor"], ascending=[False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "v208_action_search.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTDIR / "v208_fold_metrics.csv", index=False)

    point = load_sub(POINT_ANCHOR)
    rally_uids = point["rally_uid"].astype(int).to_numpy()
    server = load_sub(SERVER_ANCHOR, rally_uids)
    generated = []
    chosen = []
    for gate in ["residual_cap", "weak_residual_cap"]:
        subset = search[(search["candidate"].str.startswith("v208_")) & (search["gate"].eq(gate)) & (search["cap"].astype(float).le(0.03))]
        if subset.empty:
            continue
        rec = subset.iloc[0].to_dict()
        chosen.append(str(rec["candidate"]))
    for name in list(dict.fromkeys(chosen)):
        sub_name = f"submission_{name}__pv188cap5__sr121.csv"
        info = write_action_submission(sub_name, pred_store[name], point, server)
        info.update(search[search["candidate"].eq(name)].iloc[0].to_dict())
        generated.append(info)

    model_rows = search[search["candidate"].str.startswith("v208_") & (search["changed_rows"].astype(float).gt(0))]
    best_model_delta = float(model_rows["delta_vs_v173_anchor"].max()) if not model_rows.empty else 0.0
    verdict = "GENERATED_LOCAL_POSITIVE" if best_model_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    report = {
        "verdict": verdict,
        "generated": generated,
        "best": search.head(12).to_dict(orient="records"),
        "best_model_delta_vs_v173_anchor": best_model_delta,
        "notes": [
            "V208 trains action as the main ShuttleNet-style shot-type target.",
            "Point/depth/terminal information is auxiliary only in the action_point_aux variant.",
            "Generated submissions change action only; point is V188 cap5 and server is R121.",
            "Raw action replacement is diagnostic only.",
            "If all changed V208 variants are below the V173 action anchor, generated files are diagnostics and should not enter the upload queue.",
            "TTMATCH, ShuttleSet, and CoachAI external rows are not read.",
        ],
    }
    (OUTDIR / "v208_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v208_report.md").write_text(
        "# V208 Action-First TT-ShuttleNet\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Generated submissions: `{len(generated)}`\n\n"
        "## Generated\n\n"
        + "\n".join(f"- `{g['submission']}` action OOF `{g['action_macro_f1']:.6f}`, delta `{g['delta_vs_v173_anchor']:.6f}`, churn `{g['action_churn_vs_v173_anchor']:.6f}`" for g in generated)
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_v208_action_ttshuttlenet.py", SRC_DEST)
    print(json.dumps({"verdict": report["verdict"], "generated": len(generated), "outdir": str(OUTDIR)}, indent=2))


if __name__ == "__main__":
    main()
