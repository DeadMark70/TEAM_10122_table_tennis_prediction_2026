"""R23 metric-aware neural loss audit.

This script reuses the compact V7 Transformer sequence pipeline and compares
loss recipes that are closer to the competition metrics:

- weighted_ce: V7-style weighted CE/BCE control.
- focal_softf1: focal CE plus warm-ramped macro soft-F1 for action/point.
- uncertainty: homoscedastic uncertainty weighting over the same task losses.

It is a research script. It does not train a full-test model or generate a
submission; it writes OOF probabilities and an ensemble diagnostic against V3.
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
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, add_role_and_score_features, sample_validation_prefixes, validate_raw_data
from baseline_v2 import blend_probs
from baseline_v3 import apply_segmented_multipliers, tune_segmented_multipliers
from baseline_v5_gru import (
    build_sequence_arrays,
    build_test_meta,
    build_train_meta,
    cat_cardinalities,
    class_weights,
    fit_numeric_stats,
)
from baseline_v7_transformer import (
    TransformerDataset,
    TransformerModel,
    evaluate,
    load_tabular_oof,
    predict_model,
    weighted_classes,
)


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
class R23Tuning:
    variant: str
    action_weight: float
    point_weight: float
    server_weight: float
    action_multipliers: dict[str, list[float]]
    point_multipliers: dict[str, list[float]]
    metrics: dict[str, float]
    bins_mode: str


class UncertaintyWeights(nn.Module):
    def __init__(self, n_tasks: int) -> None:
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        out = losses[0].new_tensor(0.0)
        for idx, loss in enumerate(losses):
            out = out + torch.exp(-self.log_vars[idx]) * loss + self.log_vars[idx]
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R23 metric-aware loss audit.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--tabular-oof", default="oof_proba_v3.pkl")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["weighted_ce", "focal_softf1", "uncertainty"],
        default=["weighted_ce", "focal_softf1", "uncertainty"],
    )
    parser.add_argument("--cv-report", default="cv_report_r23.csv")
    parser.add_argument("--summary", default="r23_loss_summary.csv")
    parser.add_argument("--prefix-len-report", default="prefix_len_report_r23.csv")
    parser.add_argument("--class-report-action", default="class_report_r23_action.csv")
    parser.add_argument("--class-report-point", default="class_report_r23_point.csv")
    parser.add_argument("--feature-report", default="feature_report_r23.json")
    parser.add_argument("--oof-proba", default="oof_proba_r23.pkl")
    parser.add_argument("--recommendation", default="r23_recommendation.md")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=24)
    parser.add_argument("--numeric-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--short-point-weight", type=float, default=1.5)
    parser.add_argument("--action-beta", type=float, default=0.25)
    parser.add_argument("--point-beta", type=float, default=0.35)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--soft-f1-weight", type=float, default=0.25)
    parser.add_argument("--soft-f1-warmup", type=int, default=2)
    parser.add_argument("--multiplier-bins", choices=["global", "two", "five"], default="two")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def focal_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    gamma: float,
    reduction: str = "mean",
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
    pt = torch.exp(-ce.detach()).clamp(1e-6, 1.0)
    loss = ((1.0 - pt) ** gamma) * ce
    if reduction == "none":
        return loss
    return loss.mean()


def macro_soft_f1_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    prob = F.softmax(logits, dim=-1)
    truth = F.one_hot(target, num_classes=num_classes).float()
    tp = (truth * prob).sum(dim=0)
    fp = ((1.0 - truth) * prob).sum(dim=0)
    fn = (truth * (1.0 - prob)).sum(dim=0)
    f1 = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)
    return 1.0 - f1.mean()


def binary_task_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if weight is not None:
        raw = raw * weight
    return raw.mean()


def task_losses(outputs, batch, action_weights, point_weights, args, device, epoch: int, variant: str) -> list[torch.Tensor]:
    action_weights = action_weights.to(device)
    point_weights = point_weights.to(device)
    use_focal = variant in {"focal_softf1"}
    if use_focal:
        action_loss = focal_ce_loss(outputs["action"], batch["action"], action_weights, args.focal_gamma)
    else:
        action_loss = F.cross_entropy(outputs["action"], batch["action"], weight=action_weights)

    terminal_loss = binary_task_loss(outputs["terminal"], batch["terminal"])
    point_mask = batch["point_mask"] > 0.5
    if point_mask.any():
        if use_focal:
            point_raw = focal_ce_loss(
                outputs["point"][point_mask],
                batch["point_nonterminal"][point_mask],
                point_weights,
                args.focal_gamma,
                reduction="none",
            )
        else:
            point_raw = F.cross_entropy(
                outputs["point"][point_mask],
                batch["point_nonterminal"][point_mask],
                weight=point_weights,
                reduction="none",
            )
        short_weight = torch.where(batch["prefix_len"][point_mask] <= 2, args.short_point_weight, 1.0)
        point_loss = (point_raw * short_weight).mean()
    else:
        point_loss = outputs["point"].sum() * 0.0

    if variant == "focal_softf1" and epoch > args.soft_f1_warmup:
        ramp = min(1.0, (epoch - args.soft_f1_warmup) / max(1, args.epochs - args.soft_f1_warmup))
        action_loss = (1.0 - args.soft_f1_weight * ramp) * action_loss + args.soft_f1_weight * ramp * macro_soft_f1_loss(
            outputs["action"], batch["action"], 19
        )
        if point_mask.any():
            point_loss = (1.0 - args.soft_f1_weight * ramp) * point_loss + args.soft_f1_weight * ramp * macro_soft_f1_loss(
                outputs["point"][point_mask], batch["point_nonterminal"][point_mask], 9
            )

    server_loss = binary_task_loss(outputs["server"], batch["server"], batch["server_weight"])
    parity_loss = binary_task_loss(outputs["parity"], batch["parity"])
    remaining_loss = F.cross_entropy(outputs["remaining"], batch["remaining"])
    return [action_loss, terminal_loss, point_loss, server_loss, parity_loss, remaining_loss]


def combine_losses(losses: list[torch.Tensor], variant: str, uncertainty: UncertaintyWeights | None) -> torch.Tensor:
    if variant == "uncertainty":
        assert uncertainty is not None
        return uncertainty(losses)
    action_loss, terminal_loss, point_loss, server_loss, parity_loss, remaining_loss = losses
    return (
        0.35 * action_loss
        + 0.15 * terminal_loss
        + 0.25 * point_loss
        + 0.15 * server_loss
        + 0.05 * parity_loss
        + 0.05 * remaining_loss
    )


def train_fold(train_arrays, valid_arrays, cat_cards, action_w, point_w, args, seed: int, variant: str):
    device = torch.device(args.device)
    model = TransformerModel(
        cat_cards,
        len(train_arrays.num[0, 0]),
        args.max_len,
        args.d_model,
        args.emb_dim,
        args.numeric_dim,
        args.num_layers,
        args.num_heads,
        args.dropout,
    ).to(device)
    uncertainty = UncertaintyWeights(6).to(device) if variant == "uncertainty" else None
    params = list(model.parameters()) + ([] if uncertainty is None else list(uncertainty.parameters()))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TransformerDataset(train_arrays),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state, best_unc, best_metrics = None, None, {"overall": -1.0}
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        if uncertainty is not None:
            uncertainty.train()
        losses_epoch: list[float] = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            out = model(batch["cat"], batch["num"], batch["lengths"])
            losses = task_losses(out, batch, action_w, point_w, args, device, epoch, variant)
            loss = combine_losses(losses, variant, uncertainty)
            loss.backward()
            clip_grad_norm_(params, args.grad_clip)
            opt.step()
            losses_epoch.append(float(loss.detach().cpu()))
        a, p, s = predict_model(model, valid_arrays, args.batch_size, device)
        metrics = evaluate(valid_arrays.meta, a, p, s)
        print(
            f"  {variant} epoch {epoch:02d}: loss={np.mean(losses_epoch):.5f} "
            f"overall={metrics['overall']:.6f} action={metrics['action_macro_f1']:.6f} "
            f"point={metrics['point_macro_f1']:.6f} server={metrics['server_auc']:.6f}"
        )
        if metrics["overall"] > best_metrics["overall"] + 1e-6:
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if uncertainty is not None:
                best_unc = {k: v.detach().cpu().clone() for k, v in uncertainty.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if uncertainty is not None and best_unc is not None:
        uncertainty.load_state_dict(best_unc)
    return model, best_metrics


def sample_valid(prefix_meta, test_lengths, seed: int):
    idx = sample_validation_prefixes(prefix_meta, test_lengths, seed)
    return prefix_meta.loc[idx].copy().reset_index(drop=True)


def run_cv_variant(train, prefix_meta, test_lengths, cat_cards, num_mean, num_std, args, variant: str):
    rally_meta = prefix_meta[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=args.folds)
    valid_parts, action_parts, point_parts, server_parts, fold_rows = [], [], [], [], []
    for fold, (tr_r, va_r) in enumerate(splitter.split(rally_meta, groups=rally_meta["match"]), start=1):
        if fold > args.fold_limit:
            break
        tr_ids = set(rally_meta.iloc[tr_r]["rally_uid"])
        va_ids = set(rally_meta.iloc[va_r]["rally_uid"])
        tr_meta = prefix_meta[prefix_meta["rally_uid"].isin(tr_ids)].copy().reset_index(drop=True)
        va_pool = prefix_meta[prefix_meta["rally_uid"].isin(va_ids)].copy()
        va_meta = sample_valid(va_pool, test_lengths, args.seed + fold)
        train_arrays = build_sequence_arrays(train, tr_meta, args.max_len, num_mean, num_std)
        valid_arrays = build_sequence_arrays(train, va_meta, args.max_len, num_mean, num_std)
        action_w = weighted_classes(tr_meta["next_actionId"], ACTION_CLASSES, args.action_beta)
        point_w = weighted_classes(tr_meta[tr_meta["next_pointId"].gt(0)]["next_pointId"] - 1, list(range(9)), args.point_beta)
        print(f"fold {fold} variant={variant}: train={len(tr_meta)} valid={len(va_meta)}")
        model, _ = train_fold(train_arrays, valid_arrays, cat_cards, action_w, point_w, args, args.seed + fold * 100, variant)
        a, p, s = predict_model(model, valid_arrays, args.batch_size, torch.device(args.device))
        m = evaluate(va_meta, a, p, s)
        m.update({"fold": fold, "variant": variant, "train_rows": len(tr_meta), "valid_rows": len(va_meta)})
        fold_rows.append(m)
        valid_parts.append(va_meta)
        action_parts.append(a)
        point_parts.append(p)
        server_parts.append(s)
    valid_meta = pd.concat(valid_parts, ignore_index=True)
    report = pd.DataFrame(fold_rows)
    mean = {"fold": 0, "variant": variant, "train_rows": 0, "valid_rows": 0}
    for col in ["action_macro_f1", "point_macro_f1", "server_auc", "overall"]:
        mean[col] = float(report[col].mean())
    report = pd.concat([report, pd.DataFrame([mean])], ignore_index=True)
    return {
        "valid_meta": valid_meta,
        "action": np.vstack(action_parts),
        "point": np.vstack(point_parts),
        "server": np.concatenate(server_parts),
        "fold_report": report,
    }


def compose_v3(oof: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    meta = oof["valid_meta"].reset_index(drop=True)
    tuning = oof["tuning"]
    action = blend_probs(oof["lgbm_action"], oof["ngram_action"], tuning.action_ngram_weight)
    point = blend_probs(oof["lgbm_point"], oof["ngram_point"], tuning.point_ngram_weight)
    sw = tuning.server_weights
    server = (
        sw["direct"] * oof["lgbm_server"]
        + sw["ngram"] * oof["ngram_server"]
        + sw["parity"] * oof["parity_server"]
        + sw["remaining"] * oof["remaining_server"]
    )
    return meta, action, point, np.clip(server, 1e-6, 1.0 - 1e-6)


def tune_against_v3(meta, action, point, server, v3_oof, args, variant: str) -> tuple[R23Tuning, dict[str, np.ndarray]]:
    v3_meta, v3_action, v3_point, v3_server = compose_v3(v3_oof)
    check_cols = ["rally_uid", "prefix_len", "next_actionId", "next_pointId", "serverGetPoint"]
    if not meta[check_cols].reset_index(drop=True).equals(v3_meta[check_cols].reset_index(drop=True)):
        raise ValueError("R23 and V3 OOF rows are not aligned.")
    best = None
    best_probs = None
    for aw in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for pw in [0.0, 0.05, 0.1, 0.2, 0.3]:
            for sw in [0.0, 0.1, 0.2, 0.3]:
                act = blend_probs(v3_action, action, aw)
                pnt = blend_probs(v3_point, point, pw)
                srv = np.clip((1.0 - sw) * v3_server + sw * server, 1e-6, 1.0 - 1e-6)
                action_mult = tune_segmented_multipliers(meta, act, ACTION_CLASSES, "action", args.multiplier_bins)
                point_mult = tune_segmented_multipliers(meta, pnt, POINT_CLASSES, "point", args.multiplier_bins)
                action_pred = apply_segmented_multipliers(meta, act, action_mult, ACTION_CLASSES, args.multiplier_bins)
                point_pred = apply_segmented_multipliers(meta, pnt, point_mult, POINT_CLASSES, args.multiplier_bins)
                action_f1 = f1_score(meta["next_actionId"], action_pred, average="macro", labels=ACTION_CLASSES, zero_division=0)
                point_f1 = f1_score(meta["next_pointId"], point_pred, average="macro", labels=POINT_CLASSES, zero_division=0)
                server_auc = roc_auc_score(meta["serverGetPoint"], srv)
                metrics = {
                    "action_macro_f1": float(action_f1),
                    "point_macro_f1": float(point_f1),
                    "server_auc": float(server_auc),
                    "overall": float(0.4 * action_f1 + 0.4 * point_f1 + 0.2 * server_auc),
                }
                if best is None or metrics["overall"] > best.metrics["overall"] + 1e-12:
                    best = R23Tuning(variant, float(aw), float(pw), float(sw), action_mult, point_mult, metrics, args.multiplier_bins)
                    best_probs = {"action": act, "point": pnt, "server": srv}
    assert best is not None and best_probs is not None
    return best, best_probs


def prefix_rows(meta, action_prob, point_prob, server_prob, tuning: R23Tuning) -> pd.DataFrame:
    action_pred = apply_segmented_multipliers(meta, action_prob, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode)
    point_pred = apply_segmented_multipliers(meta, point_prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode)
    rows = []
    for label, mask in [
        ("1", meta["prefix_len"].eq(1).to_numpy()),
        ("2", meta["prefix_len"].eq(2).to_numpy()),
        ("3", meta["prefix_len"].eq(3).to_numpy()),
        ("4-6", meta["prefix_len"].between(4, 6).to_numpy()),
        ("7+", meta["prefix_len"].ge(7).to_numpy()),
    ]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        try:
            server_auc = roc_auc_score(meta.iloc[idx]["serverGetPoint"], server_prob[idx])
        except ValueError:
            server_auc = np.nan
        rows.append(
            {
                "prefix_len_bin": label,
                "count": int(len(idx)),
                "action_macro_f1": f1_score(
                    meta.iloc[idx]["next_actionId"], action_pred[idx], average="macro", labels=ACTION_CLASSES, zero_division=0
                ),
                "point_macro_f1": f1_score(
                    meta.iloc[idx]["next_pointId"], point_pred[idx], average="macro", labels=POINT_CLASSES, zero_division=0
                ),
                "server_auc": server_auc,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)
    prefix_meta = build_train_meta(train)
    test_meta = build_test_meta(test)
    num_mean, num_std = fit_numeric_stats(train)
    cat_cards = cat_cardinalities(train, test)
    print(f"device: {args.device}")
    print(f"variants: {args.variants}")
    print(f"train prefix rows: {len(prefix_meta):,}")

    with open(args.tabular_oof, "rb") as f:
        v3_oof = pickle.load(f)

    variant_outputs = {}
    cv_reports = []
    summary_rows = []
    best_tuning = None
    best_probs = None
    best_oof = None

    for variant in args.variants:
        oof = run_cv_variant(train, prefix_meta, test_meta["prefix_len"].to_numpy(dtype=int), cat_cards, num_mean, num_std, args, variant)
        tuning, probs = tune_against_v3(oof["valid_meta"], oof["action"], oof["point"], oof["server"], v3_oof, args, variant)
        single = evaluate(oof["valid_meta"], oof["action"], oof["point"], oof["server"])
        report = oof["fold_report"].copy()
        for k, v in single.items():
            report[f"single_{k}"] = v
        for k, v in tuning.metrics.items():
            report[f"selected_{k}"] = v
        report["selected_action_weight"] = tuning.action_weight
        report["selected_point_weight"] = tuning.point_weight
        report["selected_server_weight"] = tuning.server_weight
        cv_reports.append(report)
        summary_rows.append(
            {
                "variant": variant,
                "single_action": single["action_macro_f1"],
                "single_point": single["point_macro_f1"],
                "single_server": single["server_auc"],
                "single_overall": single["overall"],
                "selected_action_weight": tuning.action_weight,
                "selected_point_weight": tuning.point_weight,
                "selected_server_weight": tuning.server_weight,
                **{f"selected_{k}": v for k, v in tuning.metrics.items()},
            }
        )
        variant_outputs[variant] = {**oof, "tuning": tuning, "selected_probs": probs, "single": single}
        if best_tuning is None or tuning.metrics["overall"] > best_tuning.metrics["overall"] + 1e-12:
            best_tuning = tuning
            best_probs = probs
            best_oof = oof

    assert best_tuning is not None and best_probs is not None and best_oof is not None
    pd.concat(cv_reports, ignore_index=True).to_csv(args.cv_report, index=False)
    pd.DataFrame(summary_rows).sort_values("selected_overall", ascending=False).to_csv(args.summary, index=False)
    prefix_rows(best_oof["valid_meta"], best_probs["action"], best_probs["point"], best_probs["server"], best_tuning).to_csv(
        args.prefix_len_report, index=False
    )
    action_pred = apply_segmented_multipliers(
        best_oof["valid_meta"], best_probs["action"], best_tuning.action_multipliers, ACTION_CLASSES, best_tuning.bins_mode
    )
    point_pred = apply_segmented_multipliers(
        best_oof["valid_meta"], best_probs["point"], best_tuning.point_multipliers, POINT_CLASSES, best_tuning.bins_mode
    )
    pd.DataFrame(
        classification_report(best_oof["valid_meta"]["next_actionId"], action_pred, labels=ACTION_CLASSES, zero_division=0, output_dict=True)
    ).T.to_csv(args.class_report_action)
    pd.DataFrame(
        classification_report(best_oof["valid_meta"]["next_pointId"], point_pred, labels=POINT_CLASSES, zero_division=0, output_dict=True)
    ).T.to_csv(args.class_report_point)

    with open(args.oof_proba, "wb") as f:
        pickle.dump({"variants": variant_outputs, "best_variant": best_tuning.variant, "best_tuning": best_tuning}, f)

    metadata = {
        "args": vars(args),
        "best_variant": best_tuning.variant,
        "best_selected": {
            "action_weight": best_tuning.action_weight,
            "point_weight": best_tuning.point_weight,
            "server_weight": best_tuning.server_weight,
            "metrics": best_tuning.metrics,
            "bins_mode": best_tuning.bins_mode,
        },
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rec = [
        "# R23 Metric-Aware Loss Audit",
        "",
        "Status: completed. No submission generated.",
        "",
        "## Best Variant",
        "",
        "```text",
        f"variant = {best_tuning.variant}",
        f"action_weight = {best_tuning.action_weight}",
        f"point_weight = {best_tuning.point_weight}",
        f"server_weight = {best_tuning.server_weight}",
        f"action = {best_tuning.metrics['action_macro_f1']:.6f}",
        f"point  = {best_tuning.metrics['point_macro_f1']:.6f}",
        f"server = {best_tuning.metrics['server_auc']:.6f}",
        f"overall = {best_tuning.metrics['overall']:.6f}",
        "```",
        "",
        "## Decision",
        "",
        "- Submit only if the selected overall clearly exceeds the current safe reference by at least +0.0015.",
        "- If selected point weight is zero, the loss recipe is only useful as an action/server branch.",
    ]
    Path(args.recommendation).write_text("\n".join(rec) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary_rows).sort_values("selected_overall", ascending=False).to_string(index=False))
    print("selected", json.dumps(metadata["best_selected"], indent=2))
    print(f"wrote {args.cv_report}")
    print(f"wrote {args.summary}")
    print(f"wrote {args.prefix_len_report}")
    print(f"wrote {args.class_report_action}")
    print(f"wrote {args.class_report_point}")
    print(f"wrote {args.oof_proba}")
    print(f"wrote {args.feature_report}")
    print(f"wrote {args.recommendation}")


if __name__ == "__main__":
    main()
