"""Generate full-test submission for R1 + V10B OOF-selected ensemble."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import baseline_v10a_pretrain_transformer as v10
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, add_role_and_score_features, validate_raw_data
from baseline_v2 import blend_probs
from baseline_v3 import apply_segmented_multipliers
from baseline_v5_gru import build_sequence_arrays, build_test_meta, build_train_meta, fit_numeric_stats
from generate_r1_submission import compose_v3_full


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
    parser = argparse.ArgumentParser(description="Generate V10B+R1 submission.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--v3-oof", default="oof_proba_v3.pkl")
    parser.add_argument("--r1-sequence-proba", default="r1_full_sequence_proba.pkl")
    parser.add_argument("--v10b-feature-report", default="feature_report_v10b.json")
    parser.add_argument("--v10b-r1-selected", default="v10b_r1_selected.json")
    parser.add_argument("--v10b-full-proba", default="v10b_full_sequence_proba.pkl")
    parser.add_argument("--submission", default="submission_v10b_r1.csv")
    parser.add_argument("--feature-report", default="feature_report_v10b_r1.json")
    parser.add_argument("--reuse-v10b-proba", action="store_true")
    parser.add_argument("--force-v3-point", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def namespace_from_feature_report(path: str, device: str):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = report["args"].copy()
    raw["device"] = device
    return type("Args", (), raw)()


def train_full_v10b(train: pd.DataFrame, test: pd.DataFrame, args: argparse.Namespace) -> dict[str, np.ndarray | pd.DataFrame]:
    v10_args = namespace_from_feature_report(args.v10b_feature_report, args.device)
    v10.set_seed(int(v10_args.seed))
    prefix_meta = build_train_meta(train)
    test_meta = build_test_meta(test)
    num_mean, num_std = fit_numeric_stats(train)
    cat_cards, mask_ids, class_sizes = v10.cat_cardinalities_with_mask(train, test)
    mask_field_indices = [v10.CAT_FIELDS.index(f) for f in v10.MASK_FIELDS]
    train_arrays = build_sequence_arrays(train, prefix_meta, int(v10_args.max_len), num_mean, num_std)
    test_arrays = build_sequence_arrays(test, test_meta, int(v10_args.max_len), num_mean, num_std)
    action_w = v10.class_weights(prefix_meta["next_actionId"], ACTION_CLASSES)
    point_w = v10.class_weights(prefix_meta[prefix_meta["next_pointId"].gt(0)]["next_pointId"] - 1, list(range(9)))
    device = torch.device(v10_args.device)

    model = v10.StrokeTransformer(
        cat_cards,
        class_sizes,
        mask_field_indices,
        len(v10.NUM_FIELDS),
        int(v10_args.max_len),
        int(v10_args.d_model),
        int(v10_args.emb_dim),
        int(v10_args.numeric_dim),
        int(v10_args.num_layers),
        int(v10_args.num_heads),
        float(v10_args.dropout),
    ).to(device)

    pre_ds = v10.StrokePretrainDataset(train_arrays, mask_ids, float(v10_args.mask_prob), int(v10_args.seed) + 999)
    pre_loader = DataLoader(
        pre_ds,
        batch_size=int(v10_args.batch_size),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(int(v10_args.seed)),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(v10_args.lr_pretrain), weight_decay=float(v10_args.weight_decay))
    for epoch in range(1, int(v10_args.pretrain_epochs) + 1):
        losses = v10.train_epoch(model, pre_loader, opt, action_w, point_w, v10_args, device, pretrain=True)
        print(f"pretrain epoch {epoch:02d}: loss={losses['loss']:.5f} mask={losses['mask_loss']:.5f} causal={losses['supervised_loss']:.5f}")

    ft_ds = v10.StrokeEvalDataset(train_arrays)
    ft_loader = DataLoader(
        ft_ds,
        batch_size=int(v10_args.batch_size),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(int(v10_args.seed) + 77),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(v10_args.lr_finetune), weight_decay=float(v10_args.weight_decay))
    for epoch in range(1, int(v10_args.finetune_epochs) + 1):
        losses = v10.train_epoch(model, ft_loader, opt, action_w, point_w, v10_args, device, pretrain=False)
        print(f"finetune epoch {epoch:02d}: loss={losses['loss']:.5f}")

    action, point, server = v10.predict_model(model, test_arrays, int(v10_args.batch_size), device)
    out = {"test_meta": test_meta, "v10_action": action, "v10_point": point, "v10_server": server}
    with open(args.v10b_full_proba, "wb") as f:
        pickle.dump(out, f)
    return out


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    train = add_role_and_score_features(train)
    test = add_role_and_score_features(test)
    selected = json.loads(Path(args.v10b_r1_selected).read_text(encoding="utf-8"))

    if args.reuse_v10b_proba and Path(args.v10b_full_proba).exists():
        with open(args.v10b_full_proba, "rb") as f:
            v10_full = pickle.load(f)
    else:
        v10_full = train_full_v10b(train, test, args)

    with open(args.r1_sequence_proba, "rb") as f:
        r1_seq = pickle.load(f)
    with open(args.v3_oof, "rb") as f:
        v3_oof = pickle.load(f)
    test_prefix, _, v3_point, v3_server = compose_v3_full(train, test, v3_oof["tuning"])
    test_meta = v10_full["test_meta"].reset_index(drop=True)
    if not test_meta["rally_uid"].reset_index(drop=True).equals(test_prefix["rally_uid"].reset_index(drop=True)):
        raise ValueError("V10B and V3 test rows are not aligned.")

    r1_action = 0.4 * r1_seq["gru_action"] + 0.6 * r1_seq["tr_action"]
    r1_action = r1_action / r1_action.sum(axis=1, keepdims=True)
    r1_point = v3_point
    r1_server = 0.8 * v3_server + 0.1 * r1_seq["gru_server"] + 0.1 * r1_seq["tr_server"]

    action_prob = blend_probs(r1_action, v10_full["v10_action"], float(selected["action_v10_weight"]))
    point_weight = 0.0 if args.force_v3_point else float(selected["point_v10_weight"])
    point_prob = blend_probs(r1_point, v10_full["v10_point"], point_weight)
    server_prob = (1.0 - float(selected["server_v10_weight"])) * r1_server + float(selected["server_v10_weight"]) * v10_full["v10_server"]

    action_pred = apply_segmented_multipliers(test_meta, action_prob, selected["action_multipliers"], ACTION_CLASSES, "two")
    point_mult = v3_oof["tuning"].point_multipliers if args.force_v3_point else selected["point_multipliers"]
    point_pred = apply_segmented_multipliers(test_meta, point_prob, point_mult, POINT_CLASSES, "two")
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int).to_numpy(),
            "actionId": action_pred.astype(int),
            "pointId": point_pred.astype(int),
            "serverGetPoint": np.round(np.clip(server_prob, 1e-6, 1.0 - 1e-6), 8),
        }
    )
    if len(sub) != test["rally_uid"].nunique():
        raise ValueError("Submission row count mismatch.")
    if sub.isna().any().any():
        raise ValueError("Submission contains NaN.")
    sub.to_csv(args.submission, index=False, float_format="%.8f")
    metadata = {
        "source_oof_overall": float(selected["overall"]),
        "weights": {
            "action_v10": float(selected["action_v10_weight"]),
            "point_v10": point_weight,
            "server_v10": float(selected["server_v10_weight"]),
        },
        "force_v3_point": bool(args.force_v3_point),
        "rows": int(len(sub)),
    }
    Path(args.feature_report).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.submission} ({len(sub):,} rows)")
    print(f"wrote {args.feature_report}")


if __name__ == "__main__":
    main()
