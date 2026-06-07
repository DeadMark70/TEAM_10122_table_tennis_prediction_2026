"""R97 style-injected GRU wrapper.

This keeps baseline_v5_gru.py intact. It creates temporary train/test CSV files
with per-stroke hitter/receiver style vectors, monkey-patches V5 NUM_FIELDS to
include those dense style features, then delegates training to V5.

The style vectors are intentionally simple and transparent:
  - observed hitter action rates, point rates, spin rates
  - observed receiver action/point rates
  - smoothed with global train priors and test-observed prefixes

Use this as a diagnostic deep sequence branch. It is slower than the tabular
post-process experiments, so the default command should use fewer epochs first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_v5_gru as v5
from baseline_lgbm import ACTION_CLASSES, POINT_CLASSES, validate_raw_data


OUTDIR = Path("r97_style_gru")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R97 style-injected GRU via V5 wrapper.")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test_new.csv")
    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-full-train", action="store_true")
    return parser.parse_args()


def smooth_rates(counts: np.ndarray, global_prior: np.ndarray, alpha: float) -> np.ndarray:
    return (counts + alpha * global_prior) / (counts.sum() + alpha)


def build_player_style(observed: pd.DataFrame, train: pd.DataFrame, alpha: float = 30.0) -> tuple[dict[int, np.ndarray], list[str]]:
    action_prior = train["actionId"].value_counts().reindex(ACTION_CLASSES, fill_value=0).to_numpy(dtype=float)
    action_prior = (action_prior + 1.0) / (action_prior.sum() + len(ACTION_CLASSES))
    point_prior = train["pointId"].value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=float)
    point_prior = (point_prior + 1.0) / (point_prior.sum() + len(POINT_CLASSES))
    spin_values = list(range(int(max(train["spinId"].max(), observed["spinId"].max())) + 1))
    spin_prior = train["spinId"].value_counts().reindex(spin_values, fill_value=0).to_numpy(dtype=float)
    spin_prior = (spin_prior + 1.0) / (spin_prior.sum() + len(spin_values))

    names = [f"style_action_{c:02d}" for c in ACTION_CLASSES]
    names += [f"style_point_{c:02d}" for c in POINT_CLASSES]
    names += [f"style_spin_{c:02d}" for c in spin_values]
    names += ["style_n_obs", "style_entropy_action", "style_entropy_point"]

    global_vec = np.concatenate(
        [
            action_prior,
            point_prior,
            spin_prior,
            np.array([
                0.0,
                float(-np.sum(action_prior * np.log(np.clip(action_prior, 1e-12, 1.0)))),
                float(-np.sum(point_prior * np.log(np.clip(point_prior, 1e-12, 1.0)))),
            ]),
        ]
    )

    styles: dict[int, np.ndarray] = {}
    for pid, g in observed.groupby("gamePlayerId", sort=False):
        act = g["actionId"].value_counts().reindex(ACTION_CLASSES, fill_value=0).to_numpy(dtype=float)
        pt = g["pointId"].value_counts().reindex(POINT_CLASSES, fill_value=0).to_numpy(dtype=float)
        sp = g["spinId"].value_counts().reindex(spin_values, fill_value=0).to_numpy(dtype=float)
        act_rate = smooth_rates(act, action_prior, alpha)
        pt_rate = smooth_rates(pt, point_prior, alpha)
        sp_rate = smooth_rates(sp, spin_prior, alpha)
        extras = np.array(
            [
                np.log1p(len(g)),
                float(-np.sum(act_rate * np.log(np.clip(act_rate, 1e-12, 1.0)))),
                float(-np.sum(pt_rate * np.log(np.clip(pt_rate, 1e-12, 1.0)))),
            ]
        )
        styles[int(pid)] = np.concatenate([act_rate, pt_rate, sp_rate, extras])
    styles[-1] = global_vec
    return styles, names


def add_style_columns(df: pd.DataFrame, styles: dict[int, np.ndarray], names: list[str]) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    hitter = np.vstack([styles.get(int(pid), styles[-1]) for pid in out["gamePlayerId"].to_numpy()])
    receiver = np.vstack([styles.get(int(pid), styles[-1]) for pid in out["gamePlayerOtherId"].to_numpy()])
    diff = hitter - receiver
    for j, name in enumerate(names):
        out[f"h_{name}"] = hitter[:, j].astype(np.float32)
        out[f"r_{name}"] = receiver[:, j].astype(np.float32)
        out[f"d_{name}"] = diff[:, j].astype(np.float32)
    return out


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_raw_data(train, test)
    observed = pd.concat([train, test], ignore_index=True)
    styles, style_names = build_player_style(observed, train)
    train_aug = add_style_columns(train, styles, style_names)
    test_aug = add_style_columns(test, styles, style_names)
    train_aug_path = outdir / "train_r97_style_aug.csv"
    test_aug_path = outdir / "test_r97_style_aug.csv"
    train_aug.to_csv(train_aug_path, index=False)
    test_aug.to_csv(test_aug_path, index=False)

    style_num_fields = [f"{prefix}_{name}" for name in style_names for prefix in ("h", "r", "d")]
    v5.NUM_FIELDS = v5.NUM_FIELDS + style_num_fields

    sys.argv = [
        "baseline_v5_gru.py",
        "--train",
        str(train_aug_path),
        "--test",
        str(test_aug_path),
        "--submission",
        str(outdir / "submission_r97_style_gru.csv"),
        "--cv-report",
        str(outdir / "cv_report_r97_style_gru.csv"),
        "--prefix-len-report",
        str(outdir / "prefix_len_report_r97_style_gru.csv"),
        "--class-report-action",
        str(outdir / "class_report_r97_action.csv"),
        "--class-report-point",
        str(outdir / "class_report_r97_point.csv"),
        "--feature-report",
        str(outdir / "feature_report_r97_style_gru.json"),
        "--oof-proba",
        str(outdir / "oof_proba_r97_style_gru.pkl"),
        "--tabular-oof",
        "",
        "--epochs",
        str(args.epochs),
        "--folds",
        str(args.folds),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--lr",
        str(args.lr),
        "--device",
        args.device,
        "--multiplier-bins",
        "two",
    ]
    if args.skip_full_train:
        sys.argv.append("--skip-full-train")
    v5.main()


if __name__ == "__main__":
    main()
