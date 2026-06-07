"""V24 SSL + metric-aware fine-tuning audit.

V24 combines two previous findings:
- V14 has a safe SSL pipeline, including optional public-test observed-prefix
  transitions.
- R23 showed focal CE + warm-ramped macro soft-F1 improves the neural point
  branch, although the representation was still too weak.

This script tests whether SSL representation learning plus metric-aligned
fine-tuning creates a point/action branch that is useful when blended with V3.
It is intentionally research-only: no submission is generated.
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
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, add_role_and_score_features, sample_validation_prefixes, validate_raw_data
from baseline_v2 import blend_probs
from baseline_v3 import apply_segmented_multipliers, tune_segmented_multipliers
from baseline_v5_gru import build_sequence_arrays, build_test_meta, build_train_meta, fit_numeric_stats
from baseline_v14_transductive_ssl import (
    V12EvalDataset,
    V12PretrainDataset,
    V12Transformer,
    build_test_internal_meta,
    cat_cardinalities_with_mask,
    class_weights,
    concat_sequence_arrays,
    load_v3_subset,
    mark_train_ssl_meta,
    mask_loss,
    supervised_loss as ce_supervised_loss,
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
class V24Tuning:
    ssl_mode: str
    action_weight: float
    point_weight: float
    server_weight: float
    action_multipliers: dict[str, list[float]]
    point_multipliers: dict[str, list[float]]
    metrics: dict[str, float]
    bins_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V24 SSL + focal soft-F1 fine-tuning audit.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--ssl-modes", nargs="+", choices=["train_only", "train_test"], default=["train_only", "train_test"])
    parser.add_argument("--cv-report", default="cv_report_v24.csv")
    parser.add_argument("--summary", default="v24_ssl_focal_summary.csv")
    parser.add_argument("--prefix-report", default="prefix_len_report_v24.csv")
    parser.add_argument("--class-report-action", default="class_report_v24_action.csv")
    parser.add_argument("--class-report-point", default="class_report_v24_point.csv")
    parser.add_argument("--feature-report", default="feature_report_v24.json")
    parser.add_argument("--oof-proba", default="oof_proba_v24.pkl")
    parser.add_argument("--recommendation", default="v24_recommendation.md")
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
    parser.add_argument("--retrieval-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr-pretrain", type=float, default=7e-4)
    parser.add_argument("--lr-finetune", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--view-loss-weight", type=float, default=0.05)
    parser.add_argument("--action-beta", type=float, default=0.25)
    parser.add_argument("--point-beta", type=float, default=0.35)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--soft-f1-weight", type=float, default=0.25)
    parser.add_argument("--soft-f1-warmup", type=int, default=2)
    parser.add_argument("--short-point-weight", type=float, default=1.5)
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


def weighted_mean(loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(loss.device).float()
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


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


def metric_supervised_loss(outputs, batch, action_w, point_w, args, device, epoch: int) -> torch.Tensor:
    next_mask = batch.get("has_next_label", torch.ones_like(batch["terminal"])).to(device)
    outcome_mask = batch.get("has_outcome_label", torch.ones_like(batch["terminal"])).to(device)
    action_w = action_w.to(device)
    point_w = point_w.to(device)

    action_raw = focal_ce_loss(outputs["action"], batch["action"], action_w, args.focal_gamma, reduction="none")
    action_loss = weighted_mean(action_raw, next_mask)

    terminal_raw = F.binary_cross_entropy_with_logits(outputs["terminal"], batch["terminal"], reduction="none")
    terminal_loss = weighted_mean(terminal_raw, outcome_mask)

    point10_raw = focal_ce_loss(outputs["point10"], batch["point10"], None, args.focal_gamma, reduction="none")
    point10_loss = weighted_mean(point10_raw, next_mask)

    point_mask = batch["point_mask"] > 0.5
    if point_mask.any():
        point_raw = focal_ce_loss(
            outputs["point"][point_mask],
            batch["point_nonterminal"][point_mask],
            point_w,
            args.focal_gamma,
            reduction="none",
        )
        short_weight = torch.where(batch["lengths"][point_mask] <= 2, args.short_point_weight, 1.0)
        point_loss = weighted_mean(point_raw, next_mask[point_mask] * short_weight)
    else:
        point_loss = outputs["point"].sum() * 0.0

    if epoch > args.soft_f1_warmup:
        ramp = min(1.0, (epoch - args.soft_f1_warmup) / max(1, args.finetune_epochs - args.soft_f1_warmup))
        sf1 = args.soft_f1_weight * ramp
        action_loss = (1.0 - sf1) * action_loss + sf1 * macro_soft_f1_loss(outputs["action"], batch["action"], 19)
        if point_mask.any():
            point_loss = (1.0 - sf1) * point_loss + sf1 * macro_soft_f1_loss(
                outputs["point"][point_mask], batch["point_nonterminal"][point_mask], 9
            )

    server_raw = F.binary_cross_entropy_with_logits(outputs["server"], batch["server"], reduction="none")
    server_loss = weighted_mean(server_raw, batch["server_weight"].to(device) * outcome_mask)
    parity_raw = F.binary_cross_entropy_with_logits(outputs["parity"], batch["parity"], reduction="none")
    parity_loss = weighted_mean(parity_raw, outcome_mask)
    remaining_raw = F.cross_entropy(outputs["remaining"], batch["remaining"], reduction="none")
    remaining_loss = weighted_mean(remaining_raw, outcome_mask)

    return (
        0.25 * action_loss
        + 0.12 * terminal_loss
        + 0.24 * point_loss
        + 0.16 * point10_loss
        + 0.13 * server_loss
        + 0.05 * parity_loss
        + 0.05 * remaining_loss
    )


def pretrain_epoch(model, loader, opt, action_w, point_w, args, device) -> dict[str, float]:
    model.train()
    losses, mask_losses, sup_losses, view_losses = [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        out = model(batch["cat"], batch["num"], batch["lengths"])
        sup = ce_supervised_loss(out, batch, action_w, point_w, device)
        m = mask_loss(out, batch, model.mask_field_indices)
        out_view = model(batch["cat_view"], batch["num_view"], batch["lengths_view"])
        view = (1.0 - (out["embedding"] * out_view["embedding"]).sum(dim=-1)).mean()
        loss = 0.45 * m + 0.50 * sup + args.view_loss_weight * view
        loss.backward()
        clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        mask_losses.append(float(m.detach().cpu()))
        sup_losses.append(float(sup.detach().cpu()))
        view_losses.append(float(view.detach().cpu()))
    return {
        "loss": float(np.mean(losses)),
        "mask_loss": float(np.mean(mask_losses)),
        "supervised_loss": float(np.mean(sup_losses)),
        "view_loss": float(np.mean(view_losses)),
    }


def finetune_epoch(model, loader, opt, action_w, point_w, args, device, epoch: int) -> dict[str, float]:
    model.train()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        out = model(batch["cat"], batch["num"], batch["lengths"])
        loss = metric_supervised_loss(out, batch, action_w, point_w, args, device, epoch)
        loss.backward()
        clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return {"loss": float(np.mean(losses))}


def predict_model(model, arrays, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(V12EvalDataset(arrays), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    action_parts, point_parts, server_parts = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["cat"].to(device), batch["num"].to(device), batch["lengths"].to(device))
            action = F.softmax(out["action"], dim=-1).cpu().numpy()
            terminal = torch.sigmoid(out["terminal"]).cpu().numpy()
            point_nonterm = F.softmax(out["point"], dim=-1).cpu().numpy()
            point = np.zeros((len(action), 10), dtype=np.float32)
            point[:, 0] = terminal
            point[:, 1:] = (1.0 - terminal[:, None]) * point_nonterm
            point = point / point.sum(axis=1, keepdims=True)
            server = torch.sigmoid(out["server"]).cpu().numpy()
            action_parts.append(action)
            point_parts.append(point)
            server_parts.append(server)
    return np.vstack(action_parts), np.vstack(point_parts), np.concatenate(server_parts)


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


def train_model(train_arrays, valid_arrays, pretrain_arrays, cat_cards, class_sizes, action_w, point_w, args, seed: int):
    set_seed(seed)
    device = torch.device(args.device)
    model = V12Transformer(
        cat_cards,
        class_sizes,
        train_arrays.num.shape[-1],
        args.max_len,
        args.d_model,
        args.emb_dim,
        args.numeric_dim,
        args.num_layers,
        args.num_heads,
        args.dropout,
        args.retrieval_dim,
    ).to(device)

    pre_loader = DataLoader(
        V12PretrainDataset(pretrain_arrays, [card - 1 for card in cat_cards], seed + 9000),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay)
    for epoch in range(1, args.pretrain_epochs + 1):
        losses = pretrain_epoch(model, pre_loader, opt, action_w, point_w, args, device)
        print(
            f"  pretrain {epoch:02d}: loss={losses['loss']:.5f} mask={losses['mask_loss']:.5f} "
            f"sup={losses['supervised_loss']:.5f} view={losses['view_loss']:.5f}"
        )

    ft_loader = DataLoader(
        V12EvalDataset(train_arrays),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed + 77),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_finetune, weight_decay=args.weight_decay)
    best_state, best_score = None, {"overall": -1.0}
    bad = 0
    for epoch in range(1, args.finetune_epochs + 1):
        losses = finetune_epoch(model, ft_loader, opt, action_w, point_w, args, device, epoch)
        a, p, s = predict_model(model, valid_arrays, args.batch_size, device)
        metrics = evaluate(valid_arrays.meta, a, p, s)
        print(
            f"  finetune {epoch:02d}: loss={losses['loss']:.5f} overall={metrics['overall']:.6f} "
            f"action={metrics['action_macro_f1']:.6f} point={metrics['point_macro_f1']:.6f}"
        )
        if metrics["overall"] > best_score["overall"] + 1e-6:
            best_score = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score


def run_cv_mode(train, test, prefix_meta, test_ssl_meta, test_lengths, num_mean, num_std, cat_cards, class_sizes, args, ssl_mode: str):
    rally_meta = prefix_meta[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=args.folds)
    test_ssl_arrays = (
        build_sequence_arrays(test, test_ssl_meta, args.max_len, num_mean, num_std)
        if ssl_mode == "train_test" and len(test_ssl_meta)
        else None
    )
    valid_parts, action_parts, point_parts, server_parts, fold_rows = [], [], [], [], []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(rally_meta, groups=rally_meta["match"]), start=1):
        if fold > args.fold_limit:
            break
        tr_ids = set(rally_meta.iloc[tr_idx]["rally_uid"])
        va_ids = set(rally_meta.iloc[va_idx]["rally_uid"])
        tr_meta = prefix_meta[prefix_meta["rally_uid"].isin(tr_ids)].copy().reset_index(drop=True)
        va_pool = prefix_meta[prefix_meta["rally_uid"].isin(va_ids)].copy()
        va_idx_sample = sample_validation_prefixes(va_pool, test_lengths, args.seed + fold)
        va_meta = va_pool.loc[va_idx_sample].copy().reset_index(drop=True)

        train_arrays = build_sequence_arrays(train, tr_meta, args.max_len, num_mean, num_std)
        valid_arrays = build_sequence_arrays(train, va_meta, args.max_len, num_mean, num_std)
        train_ssl_arrays = build_sequence_arrays(train, mark_train_ssl_meta(tr_meta), args.max_len, num_mean, num_std)
        pretrain_arrays = concat_sequence_arrays([train_ssl_arrays, test_ssl_arrays]) if test_ssl_arrays is not None else train_ssl_arrays

        action_w = class_weights(tr_meta["next_actionId"], ACTION_CLASSES, args.action_beta)
        point_w = class_weights(tr_meta[tr_meta["next_pointId"].gt(0)]["next_pointId"] - 1, list(range(9)), args.point_beta)
        print(
            f"fold {fold} ssl_mode={ssl_mode}: train={len(tr_meta):,} ssl={len(pretrain_arrays.meta):,} "
            f"valid={len(va_meta):,}"
        )
        model, _ = train_model(
            train_arrays,
            valid_arrays,
            pretrain_arrays,
            cat_cards,
            class_sizes,
            action_w,
            point_w,
            args,
            args.seed + 1000 * (1 + args.ssl_modes.index(ssl_mode)) + fold,
        )
        a, p, s = predict_model(model, valid_arrays, args.batch_size, torch.device(args.device))
        metrics = evaluate(va_meta, a, p, s)
        metrics.update({"fold": fold, "ssl_mode": ssl_mode, "train_rows": len(tr_meta), "valid_rows": len(va_meta)})
        fold_rows.append(metrics)
        valid_parts.append(va_meta)
        action_parts.append(a)
        point_parts.append(p)
        server_parts.append(s)

    valid_meta = pd.concat(valid_parts, ignore_index=True)
    report = pd.DataFrame(fold_rows)
    mean = {"fold": 0, "ssl_mode": ssl_mode, "train_rows": 0, "valid_rows": 0}
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


def tune_against_v3(meta, action, point, server, args, ssl_mode: str) -> tuple[V24Tuning, dict[str, np.ndarray]]:
    v3_action, v3_point, v3_server, _ = load_v3_subset(args.v3_oof, meta)
    best = None
    best_probs = None
    for aw in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        for pw in [0.0, 0.05, 0.1, 0.2, 0.3]:
            for sw in [0.0, 0.1, 0.2, 0.3]:
                act = blend_probs(v3_action, action, aw)
                pnt = blend_probs(v3_point, point, pw)
                srv = np.clip((1.0 - sw) * v3_server + sw * server, 1e-6, 1.0 - 1e-6)
                action_mult = tune_segmented_multipliers(meta, act, ACTION_CLASSES, "action", args.multiplier_bins)
                point_mult = tune_segmented_multipliers(meta, pnt, POINT_CLASSES, "point", args.multiplier_bins)
                metrics = evaluate(meta, act, pnt, srv, action_mult, point_mult, args.multiplier_bins)
                if best is None or metrics["overall"] > best.metrics["overall"] + 1e-12:
                    best = V24Tuning(ssl_mode, aw, pw, sw, action_mult, point_mult, metrics, args.multiplier_bins)
                    best_probs = {"action": act, "point": pnt, "server": srv}
    assert best is not None and best_probs is not None
    return best, best_probs


def prefix_rows(meta, action_prob, point_prob, server_prob, tuning: V24Tuning) -> pd.DataFrame:
    action_pred = apply_segmented_multipliers(meta, action_prob, tuning.action_multipliers, ACTION_CLASSES, tuning.bins_mode)
    point_pred = apply_segmented_multipliers(meta, point_prob, tuning.point_multipliers, POINT_CLASSES, tuning.bins_mode)
    rows = []
    for label, mask in [
        ("1", meta["prefix_len"].eq(1).to_numpy()),
        ("2", meta["prefix_len"].eq(2).to_numpy()),
        ("3", meta["prefix_len"].eq(3).to_numpy()),
        ("4-6", meta["prefix_len"].between(4, 6).to_numpy()),
        ("7+", meta["prefix_len"].ge(7).to_numpy()),
        ("le2", meta["prefix_len"].le(2).to_numpy()),
        ("ge3", meta["prefix_len"].ge(3).to_numpy()),
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
    test_ssl_meta = build_test_internal_meta(test)
    num_mean, num_std = fit_numeric_stats(train)
    cat_cards, _, class_sizes = cat_cardinalities_with_mask(train, test)
    print(f"device: {args.device}")
    print(f"ssl_modes: {args.ssl_modes}")
    print(f"train prefix rows: {len(prefix_meta):,}")
    print(f"public test SSL rows: {len(test_ssl_meta):,}")

    mode_outputs = {}
    cv_reports = []
    summary_rows = []
    best_tuning = None
    best_probs = None
    best_oof = None
    test_lengths = test_meta["prefix_len"].to_numpy(dtype=int)

    for ssl_mode in args.ssl_modes:
        oof = run_cv_mode(train, test, prefix_meta, test_ssl_meta, test_lengths, num_mean, num_std, cat_cards, class_sizes, args, ssl_mode)
        tuning, probs = tune_against_v3(oof["valid_meta"], oof["action"], oof["point"], oof["server"], args, ssl_mode)
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
                "ssl_mode": ssl_mode,
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
        mode_outputs[ssl_mode] = {**oof, "tuning": tuning, "selected_probs": probs, "single": single}
        if best_tuning is None or tuning.metrics["overall"] > best_tuning.metrics["overall"] + 1e-12:
            best_tuning = tuning
            best_probs = probs
            best_oof = oof

    assert best_tuning is not None and best_probs is not None and best_oof is not None
    pd.concat(cv_reports, ignore_index=True).to_csv(args.cv_report, index=False)
    pd.DataFrame(summary_rows).sort_values("selected_overall", ascending=False).to_csv(args.summary, index=False)
    prefix_rows(best_oof["valid_meta"], best_probs["action"], best_probs["point"], best_probs["server"], best_tuning).to_csv(
        args.prefix_report, index=False
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
        pickle.dump({"modes": mode_outputs, "best_mode": best_tuning.ssl_mode, "best_tuning": best_tuning}, f)

    metadata = {
        "args": vars(args),
        "public_test_ssl_rows": int(len(test_ssl_meta)),
        "best_mode": best_tuning.ssl_mode,
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
        "# V24 SSL + Focal Soft-F1 Audit",
        "",
        "Status: completed. No submission generated.",
        "",
        "## Best Selected Ensemble",
        "",
        "```text",
        f"ssl_mode = {best_tuning.ssl_mode}",
        f"action_weight = {best_tuning.action_weight}",
        f"point_weight = {best_tuning.point_weight}",
        f"server_weight = {best_tuning.server_weight}",
        f"action = {best_tuning.metrics['action_macro_f1']:.6f}",
        f"point  = {best_tuning.metrics['point_macro_f1']:.6f}",
        f"server = {best_tuning.metrics['server_auc']:.6f}",
        f"overall = {best_tuning.metrics['overall']:.6f}",
        "```",
        "",
        "## Decision Rule",
        "",
        "- Continue this line only if the selected point weight is non-zero and point improves over V3 by at least +0.003.",
        "- Treat action-only gains as useful ensemble branches, not as point breakthroughs.",
    ]
    Path(args.recommendation).write_text("\n".join(rec) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary_rows).sort_values("selected_overall", ascending=False).to_string(index=False))
    print("selected", json.dumps(metadata["best_selected"], indent=2))
    print(f"wrote {args.cv_report}")
    print(f"wrote {args.summary}")
    print(f"wrote {args.prefix_report}")
    print(f"wrote {args.class_report_action}")
    print(f"wrote {args.class_report_point}")
    print(f"wrote {args.oof_proba}")
    print(f"wrote {args.feature_report}")
    print(f"wrote {args.recommendation}")


if __name__ == "__main__":
    main()
