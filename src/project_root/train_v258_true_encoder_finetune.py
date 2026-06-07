from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from analysis_v233_public_like_validation_lab import weighted_macro_f1
from analysis_v243_v247_action_experiment_common import context_weights, load_action_context
from analysis_v258_encoder_finetune_helpers import (
    action_family_id,
    blend_probabilities,
    kd_cross_entropy,
    normalize_rows_safe,
    pad_sequence,
)
from baseline_lgbm import ACTION_CLASSES

ROOT = Path(".")
OUTDIR = ROOT / "v258_true_encoder_finetune"
V257_ENCODER_PATH = ROOT / "v257_shuttlenet_repretrain" / "v257_encoder.pt"
RANDOM_STATE = 258
MAX_LEN = 8
N_ACTION = 19
N_FAMILY = 5
WEAK_ACTIONS = [0, 3, 4, 5, 7, 8, 9, 12, 14]


class V258ActionModel(nn.Module):
    def __init__(self, n_shot: int = 33, n_family: int = 6, n_player: int = 40, hidden: int = 96):
        super().__init__()
        self.shot_emb = nn.Embedding(n_shot, 32, padding_idx=0)
        self.family_emb = nn.Embedding(n_family, 16, padding_idx=0)
        self.player_emb = nn.Embedding(n_player, 16, padding_idx=0)
        self.xy_proj = nn.Linear(2, 16)
        self.gru = nn.GRU(80, hidden, batch_first=True)
        self.action_head = nn.Linear(hidden, N_ACTION)
        self.family_head = nn.Linear(hidden, N_FAMILY)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = torch.cat(
            [
                self.shot_emb(batch["shot"]),
                self.family_emb(batch["family"]),
                self.player_emb(batch["player"]),
                torch.relu(self.xy_proj(batch["xy"])),
            ],
            dim=-1,
        )
        h, _ = self.gru(x)
        mask = (batch["shot"] != 0).float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        return {
            "action": self.action_head(pooled),
            "family": self.family_head(pooled),
            "hidden": pooled,
        }


def load_v257_encoder_weights(model: V258ActionModel) -> dict:
    if not V257_ENCODER_PATH.exists():
        return {"loaded": False, "reason": "missing_v257_encoder", "matched_keys": []}
    payload = torch.load(V257_ENCODER_PATH, map_location="cpu")
    state = payload.get("model_state", payload)
    current = model.state_dict()
    matched = {}
    for key, value in state.items():
        if key == "family_head.weight" or key == "family_head.bias":
            continue
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            matched[key] = value
    current.update(matched)
    model.load_state_dict(current)
    return {"loaded": bool(matched), "matched_keys": sorted(matched)}


def lag_index(col: str, suffix: str) -> int:
    return int(col.replace("lag", "").replace(suffix, ""))


def build_prefix_tensors(rows: pd.DataFrame, max_len: int = MAX_LEN) -> dict[str, np.ndarray]:
    action_cols = sorted(
        [col for col in rows.columns if col.startswith("lag") and col.endswith("_actionId")],
        key=lambda col: lag_index(col, "_actionId"),
    )[:max_len]
    point_cols = sorted(
        [col for col in rows.columns if col.startswith("lag") and col.endswith("_pointId")],
        key=lambda col: lag_index(col, "_pointId"),
    )[:max_len]
    if not action_cols:
        raise RuntimeError("No lag action columns found for V258 prefix tensor build.")

    shot = []
    family = []
    player = []
    xy = []
    for _, row in rows.iterrows():
        actions = [int(pd.to_numeric(row.get(col, 0), errors="coerce") or 0) + 1 for col in action_cols]
        families = [action_family_id(action - 1) + 1 for action in actions]
        points = [int(pd.to_numeric(row.get(col, 0), errors="coerce") or 0) for col in point_cols] if point_cols else [0] * len(actions)
        if len(points) < len(actions):
            points = points + [0] * (len(actions) - len(points))
        x_proxy = [((p - 1) % 3) / 2.0 if 1 <= p <= 9 else 0.5 for p in points]
        y_proxy = [((p - 1) // 3) / 2.0 if 1 <= p <= 9 else 0.5 for p in points]
        shot.append(pad_sequence(actions, max_len, pad=0))
        family.append(pad_sequence(families, max_len, pad=0))
        player.append(pad_sequence([(i % 2) + 1 for i in range(len(actions))], max_len, pad=0))
        xy.append(np.stack([pad_sequence(x_proxy, max_len, pad=0), pad_sequence(y_proxy, max_len, pad=0)], axis=1))
    return {
        "shot": np.asarray(shot, dtype=np.int64),
        "family": np.asarray(family, dtype=np.int64),
        "player": np.asarray(player, dtype=np.int64),
        "xy": np.asarray(xy, dtype=np.float32),
    }


def subset_tensors(tensors: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[idx] for key, value in tensors.items()}


def predict_prob(model: V258ActionModel, tensors: dict[str, np.ndarray], device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(next(iter(tensors.values()))), batch_size):
            sl = slice(start, start + batch_size)
            batch = {k: torch.as_tensor(v[sl], device=device) for k, v in tensors.items()}
            chunks.append(torch.softmax(model(batch)["action"], dim=1).cpu().numpy())
    return normalize_rows_safe(np.vstack(chunks))


def train_one_fold(
    mode: str,
    train_tensors: dict[str, np.ndarray],
    y_train: np.ndarray,
    teacher_train: np.ndarray,
    valid_tensors: dict[str, np.ndarray],
    test_tensors: dict[str, np.ndarray],
    fold: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    torch.manual_seed(RANDOM_STATE + int(fold))
    np.random.seed(RANDOM_STATE + int(fold))
    model = V258ActionModel()
    load_info = load_v257_encoder_weights(model)
    model.to(device)

    if mode == "freeze_encoder":
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("action_head") or name.startswith("family_head")
    elif mode == "last_layer":
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("gru") or name.startswith("action_head") or name.startswith("family_head")
    else:
        for param in model.parameters():
            param.requires_grad = True

    lr = 3e-4 if mode != "full_low_lr" else 8e-5
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    y_family = np.array([action_family_id(y) for y in y_train], dtype=np.int64)

    train_loss = []
    epochs = 5 if mode != "full_low_lr" else 4
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(RANDOM_STATE + fold * 17 + epoch).permutation(len(y_train))
        total = 0.0
        count = 0
        for start in range(0, len(order), 256):
            idx = order[start : start + 256]
            batch = {k: torch.as_tensor(v[idx], device=device) for k, v in train_tensors.items()}
            out = model(batch)
            action_target = torch.as_tensor(y_train[idx], dtype=torch.long, device=device)
            family_target = torch.as_tensor(y_family[idx], dtype=torch.long, device=device)
            teacher = torch.as_tensor(teacher_train[idx], dtype=torch.float32, device=device)
            loss = F.cross_entropy(out["action"], action_target)
            loss = loss + 0.2 * F.cross_entropy(out["family"], family_target)
            loss = loss + 0.05 * kd_cross_entropy(out["action"], teacher, temperature=2.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            total += float(loss.item()) * len(idx)
            count += len(idx)
        train_loss.append(total / max(count, 1))

    valid_prob = predict_prob(model, valid_tensors, device)
    test_prob = predict_prob(model, test_tensors, device)
    metrics = {
        "fold": int(fold),
        "mode": mode,
        "train_loss": float(train_loss[-1]),
        "epochs": int(epochs),
        "loaded_v257": bool(load_info["loaded"]),
        "matched_keys": "|".join(load_info["matched_keys"]),
    }
    return valid_prob, test_prob, metrics


def evaluate_candidate(name: str, y: np.ndarray, pred: np.ndarray, anchor: np.ndarray, weights: np.ndarray) -> dict:
    score = f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0)
    base = f1_score(y, anchor, labels=ACTION_CLASSES, average="macro", zero_division=0)
    iw = weighted_macro_f1(y, pred, weights)
    base_iw = weighted_macro_f1(y, anchor, weights)
    weak = f1_score(y, pred, labels=WEAK_ACTIONS, average="macro", zero_division=0)
    weak_base = f1_score(y, anchor, labels=WEAK_ACTIONS, average="macro", zero_division=0)
    return {
        "candidate": name,
        "action_macro_f1": float(score),
        "delta_vs_v173_anchor": float(score - base),
        "iw_delta_vs_v173": float(iw - base_iw),
        "weak_delta_vs_v173": float(weak - weak_base),
        "action_churn_vs_v173_anchor": float(np.mean(pred != anchor)),
        "changed_rows": int(np.sum(pred != anchor)),
    }


def one_hot(labels: np.ndarray, n_classes: int = N_ACTION) -> np.ndarray:
    out = np.zeros((len(labels), n_classes), dtype=float)
    out[np.arange(len(labels)), labels.astype(int)] = 1.0
    return out


def classgate(anchor: np.ndarray, prob: np.ndarray) -> np.ndarray:
    raw = prob.argmax(axis=1)
    conf = prob.max(axis=1)
    out = anchor.copy()
    raw_family = np.array([action_family_id(x) for x in raw])
    anchor_family = np.array([action_family_id(x) for x in anchor])
    weakish = np.isin(raw, WEAK_ACTIONS)
    take = weakish & (conf >= 0.35) & (raw_family != anchor_family)
    out[take] = raw[take]
    return out


def run_mode(mode: str, ctx: dict, train_tensors: dict[str, np.ndarray], test_tensors: dict[str, np.ndarray], device: torch.device) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = ctx["rows"]
    y = ctx["y"]
    oof = np.zeros((len(rows), N_ACTION), dtype=float)
    test_sum = np.zeros((len(ctx["test_rows"]), N_ACTION), dtype=float)
    metrics = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        valid_prob, test_prob, fold_metrics = train_one_fold(
            mode,
            subset_tensors(train_tensors, train),
            y[train],
            ctx["v173_prob_oof"][train],
            subset_tensors(train_tensors, valid),
            test_tensors,
            int(fold),
            device,
        )
        oof[valid] = valid_prob
        test_sum += test_prob
        metrics.append(fold_metrics)
    return normalize_rows_safe(oof), normalize_rows_safe(test_sum / max(len(metrics), 1)), metrics


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    ctx = load_action_context()
    train_tensors = build_prefix_tensors(ctx["rows"])
    test_tensors = build_prefix_tensors(ctx["test_rows"])
    weights = context_weights(ctx["rows"], ctx["test_rows"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modes = ["freeze_encoder", "last_layer", "full_low_lr"]
    mode_results = []
    all_metrics = []
    best = None
    for mode in modes:
        oof_prob, test_prob, metrics = run_mode(mode, ctx, train_tensors, test_tensors, device)
        raw_pred = oof_prob.argmax(axis=1)
        raw_rec = evaluate_candidate(f"v258_{mode}_raw_action", ctx["y"], raw_pred, ctx["v173_oof"], weights)
        mode_results.append((mode, raw_rec, oof_prob, test_prob))
        all_metrics.extend(metrics)
        if best is None or raw_rec["delta_vs_v173_anchor"] > best[1]["delta_vs_v173_anchor"]:
            best = (mode, raw_rec, oof_prob, test_prob)

    best_mode, _, best_oof, best_test = best
    candidates = {
        "v173_anchor": ctx["v173_oof"],
        "v258_raw_action": best_oof.argmax(axis=1),
        "v258_v173blend_w0p05": blend_probabilities(ctx["v173_oof"], best_oof, 0.05).argmax(axis=1),
        "v258_v173blend_w0p10": blend_probabilities(ctx["v173_oof"], best_oof, 0.10).argmax(axis=1),
        "v258_v173blend_w0p20": blend_probabilities(ctx["v173_oof"], best_oof, 0.20).argmax(axis=1),
        "v258_classgate": classgate(ctx["v173_oof"], best_oof),
    }
    records = [evaluate_candidate(name, ctx["y"], pred, ctx["v173_oof"], weights) for name, pred in candidates.items()]
    search = pd.DataFrame(records).sort_values(
        ["delta_vs_v173_anchor", "iw_delta_vs_v173", "weak_delta_vs_v173"],
        ascending=[False, False, False],
    )
    non_anchor = search[search["candidate"].ne("v173_anchor")]
    best_candidate = non_anchor.iloc[0].to_dict() if len(non_anchor) else {}
    best_delta = float(best_candidate.get("delta_vs_v173_anchor", 0.0))
    best_iw = float(best_candidate.get("iw_delta_vs_v173", 0.0))
    if best_delta >= 0.003 and best_iw >= 0.001:
        verdict = "CANDIDATE_FOR_PUBLIC_PROBE"
    elif best_delta > 0 and best_iw >= 0:
        verdict = "LOCAL_WEAK_POSITIVE_NEEDS_REVIEW"
    else:
        verdict = "LOCAL_NEGATIVE_DO_NOT_SUBMIT"

    np.save(OUTDIR / "v258_oof_action_prob.npy", best_oof)
    np.save(OUTDIR / "v258_test_action_prob.npy", best_test)
    np.savez(
        OUTDIR / "v258_candidate_test_actions.npz",
        v258_raw_action=best_test.argmax(axis=1),
        v258_v173blend_w0p05=blend_probabilities(ctx["v173_test"], best_test, 0.05).argmax(axis=1),
        v258_v173blend_w0p10=blend_probabilities(ctx["v173_test"], best_test, 0.10).argmax(axis=1),
        v258_v173blend_w0p20=blend_probabilities(ctx["v173_test"], best_test, 0.20).argmax(axis=1),
        v258_classgate=classgate(ctx["v173_test"], best_test),
    )
    search.to_csv(OUTDIR / "v258_action_search.csv", index=False)
    pd.DataFrame(all_metrics).to_csv(OUTDIR / "v258_training_metrics.csv", index=False)
    report = {
        "best_mode": best_mode,
        "best_candidate": best_candidate,
        "verdict": verdict,
        "device": str(device),
        "modes": [rec for _, rec, _, _ in mode_results],
        "prob_shapes": {"oof": list(best_oof.shape), "test": list(best_test.shape)},
    }
    (OUTDIR / "v258_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v258_report.md").write_text(
        "# V258 True Encoder Fine-Tune\n\n"
        f"- Best mode: `{best_mode}`\n"
        f"- Best candidate: `{best_candidate.get('candidate', 'none')}`\n"
        f"- OOF delta vs V173: `{best_delta:.6f}`\n"
        f"- Public-like/IW delta: `{best_iw:.6f}`\n"
        f"- Verdict: `{verdict}`\n"
        f"- Device: `{device}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"outdir": str(OUTDIR), "best_mode": best_mode, "best_delta": best_delta, "public_like_delta": best_iw, "verdict": verdict}))


if __name__ == "__main__":
    main()
