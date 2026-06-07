"""R62 pressure-aware action ensemble.

R58-R61 showed that player-style clusters help action OOF, especially when
used in a class-aware way. This experiment asks whether score pressure should
change the blend between the stable R42 action base, the R57 style expert, and
the V49 robust/unseen-player expert.

Point/server are intentionally kept fixed to the current R34 branch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_r58_r61_style_gated_ensembles import (
    PRIMARY_STYLE,
    build_test_style_probs,
    load_artifact,
    prepare_prefix_tables,
    reconstruct_oof_meta_and_style,
    write_submission as write_r58_submission,
)
from analysis_r48_action_meta_stacker import build_current_oof_action
from baseline_lgbm import ACTION_CLASSES
from baseline_v3 import apply_segmented_multipliers
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows


OUTDIR = Path("r62_pressure_gated_ensemble")
PRESSURE_COLS = [
    "is_deuce_like",
    "server_at_game_point_like",
    "receiver_at_game_point_like",
    "scoreTotal",
    "serverScoreDiff",
    "serverScore",
    "receiverScore",
]


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


def apply_action(prob: np.ndarray, meta: pd.DataFrame, mult: dict) -> np.ndarray:
    return apply_segmented_multipliers(meta, prob, mult, ACTION_CLASSES, "two")


def add_pressure_columns(meta: pd.DataFrame, prefix_base: pd.DataFrame) -> pd.DataFrame:
    keys = ["rally_uid", "prefix_len", "next_actionId"]
    cols = keys + [c for c in PRESSURE_COLS if c in prefix_base.columns]
    pressure = prefix_base[cols].copy()
    out = meta.merge(pressure, on=keys, how="left", validate="one_to_one")
    if out[PRESSURE_COLS].isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError(f"Missing pressure columns after merge: {bad}")
    return out


def pressure_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    score_total = df["scoreTotal"].to_numpy(dtype=int)
    score_diff_abs = np.abs(df["serverScoreDiff"].to_numpy(dtype=int))
    deuce = df["is_deuce_like"].to_numpy(dtype=int).astype(bool)
    gp = (
        df["server_at_game_point_like"].to_numpy(dtype=int).astype(bool)
        | df["receiver_at_game_point_like"].to_numpy(dtype=int).astype(bool)
    )
    late = score_total >= 16
    close = score_diff_abs <= 2
    return {
        "deuce_gp": deuce | gp,
        "late_close": late & close,
        "late_or_gp": late | gp,
        "pressure_any": deuce | gp | (late & close),
    }


def row_three_blend(base: np.ndarray, style: np.ndarray, robust: np.ndarray, ws: np.ndarray, wr: np.ndarray) -> np.ndarray:
    ws = np.asarray(ws, dtype=float).reshape(-1, 1)
    wr = np.asarray(wr, dtype=float).reshape(-1, 1)
    keep = np.clip(1.0 - ws - wr, 0.0, 1.0)
    return normalize_rows(keep * base + ws * style + wr * robust)


def row_class_three_blend(
    base: np.ndarray,
    style: np.ndarray,
    robust: np.ndarray,
    ws: np.ndarray,
    wr: np.ndarray,
    classes: list[int],
) -> np.ndarray:
    ws = np.asarray(ws, dtype=float)
    wr = np.asarray(wr, dtype=float)
    out = base.copy()
    for cls in classes:
        keep = np.clip(1.0 - ws - wr, 0.0, 1.0)
        out[:, cls] = keep * base[:, cls] + ws * style[:, cls] + wr * robust[:, cls]
    return normalize_rows(out)


def metrics_row(name: str, prob: np.ndarray, meta: pd.DataFrame, y: np.ndarray, base_pred: np.ndarray, mult: dict, extra: dict) -> dict:
    pred = apply_action(prob, meta, mult)
    row = {
        "candidate": name,
        "action_macro_f1": float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
        "churn_vs_r42": float(np.mean(pred != base_pred)),
        "pred8_count": int((pred == 8).sum()),
        "pred9_count": int((pred == 9).sum()),
        "pred11_count": int((pred == 11).sum()),
        "pred12_count": int((pred == 12).sum()),
        "pred13_count": int((pred == 13).sum()),
        "pred14_count": int((pred == 14).sum()),
    }
    for mask_name, mask in extra.pop("_masks", {}).items():
        if mask.sum() > 0:
            row[f"{mask_name}_action_f1"] = float(
                f1_score(y[mask], pred[mask], average="macro", labels=ACTION_CLASSES, zero_division=0)
            )
            row[f"{mask_name}_count"] = int(mask.sum())
    row.update(extra)
    return row


def make_pressure_weights(n: int, mask: np.ndarray, normal_w: float, pressure_w: float) -> np.ndarray:
    out = np.full(n, normal_w, dtype=float)
    out[mask] = pressure_w
    return out


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)

    art = load_artifact()
    train_raw, test_raw, prefix_base, test_prefix_base, base_features = prepare_prefix_tables()
    meta, _, r57_oof_probs, _, _ = reconstruct_oof_meta_and_style(train_raw, prefix_base, test_prefix_base, base_features)
    test_probs, _, _ = build_test_style_probs(train_raw, test_raw, prefix_base, test_prefix_base, base_features)
    meta = add_pressure_columns(meta, prefix_base)

    y = meta["next_actionId"].to_numpy(dtype=int)
    mult = art["selected"]["action_multipliers"]

    current_oof = build_current_oof_action()
    golden_oof = art["experts_oof"]["v47_v64_oof_soft"]
    r42_oof_full = normalize_rows(0.80 * current_oof + 0.20 * golden_oof)
    src = art["valid_meta"].copy().reset_index(drop=True)
    src["_row"] = np.arange(len(src))
    align = meta[["rally_uid", "prefix_len", "next_actionId"]].merge(
        src[["rally_uid", "prefix_len", "next_actionId", "_row"]],
        on=["rally_uid", "prefix_len", "next_actionId"],
        how="left",
        validate="one_to_one",
    )
    if align["_row"].isna().any():
        raise ValueError("Could not align artifact valid_meta to R62 meta.")
    idx = align["_row"].to_numpy(dtype=int)

    r42_oof = r42_oof_full[idx]
    style_oof = r57_oof_probs[PRIMARY_STYLE]
    robust_oof = art["experts_oof"]["v49_robust_unseen"][idx]
    base_pred = apply_action(r42_oof, meta, mult)
    base_f1 = float(f1_score(y, base_pred, average="macro", labels=ACTION_CLASSES, zero_division=0))

    r42_test = normalize_rows(0.80 * art["current_test_action"] + 0.20 * art["experts_test"]["v47_golden_test_soft"])
    style_test = test_probs[PRIMARY_STYLE]
    robust_test = art["experts_test"]["v49_robust_unseen"]
    test_masks = pressure_masks(test_prefix_base)
    oof_masks = pressure_masks(meta)

    rows: list[dict] = [
        {
            "candidate": "r42_base",
            "experiment": "base",
            "action_macro_f1": base_f1,
            "churn_vs_r42": 0.0,
        }
    ]
    oof_by_name = {"r42_base": r42_oof}
    test_by_name = {"r42_base": r42_test}

    class_sets = {
        "rare_control": [8, 9, 11, 12],
        "control_defense": [8, 9, 11, 12, 13, 14],
        "low_action": [0, 3, 4, 7, 8, 9, 11, 12, 14],
    }

    # A: row-wise pressure gate. Small search by design.
    for mask_name, mask in oof_masks.items():
        tmask = test_masks[mask_name]
        for ns in [0.0, 0.05, 0.10]:
            for nr in [0.0, 0.05, 0.10]:
                for ps in [0.0, 0.10, 0.20, 0.30, 0.40]:
                    for pr in [0.0, 0.10, 0.20, 0.30]:
                        if ns + nr > 0.25 or ps + pr > 0.60:
                            continue
                        ws = make_pressure_weights(len(meta), mask, ns, ps)
                        wr = make_pressure_weights(len(meta), mask, nr, pr)
                        prob = row_three_blend(r42_oof, style_oof, robust_oof, ws, wr)
                        name = f"r62_row_{mask_name}_ns{ns}_nr{nr}_ps{ps}_pr{pr}"
                        row = metrics_row(
                            name,
                            prob,
                            meta,
                            y,
                            base_pred,
                            mult,
                            {
                                "experiment": "R62_row",
                                "mask": mask_name,
                                "normal_style_w": ns,
                                "normal_robust_w": nr,
                                "pressure_style_w": ps,
                                "pressure_robust_w": pr,
                                "pressure_coverage": float(mask.mean()),
                                "_masks": {mask_name: mask, "non_pressure": ~mask},
                            },
                        )
                        rows.append(row)
                        oof_by_name[name] = prob
                        tws = make_pressure_weights(len(test_prefix_base), tmask, ns, ps)
                        twr = make_pressure_weights(len(test_prefix_base), tmask, nr, pr)
                        test_by_name[name] = row_three_blend(r42_test, style_test, robust_test, tws, twr)

    # B: class-aware pressure gate, starting from the proven R59 idea but making
    # pressure rows more/less style-driven and optionally robust-assisted.
    for mask_name, mask in oof_masks.items():
        tmask = test_masks[mask_name]
        for set_name, classes in class_sets.items():
            for ns in [0.20, 0.30, 0.40]:
                for nr in [0.0, 0.05]:
                    for ps in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]:
                        for pr in [0.0, 0.10, 0.20, 0.30]:
                            if ns + nr > 0.50 or ps + pr > 0.70:
                                continue
                            ws = make_pressure_weights(len(meta), mask, ns, ps)
                            wr = make_pressure_weights(len(meta), mask, nr, pr)
                            prob = row_class_three_blend(r42_oof, style_oof, robust_oof, ws, wr, classes)
                            name = f"r62_cls_{set_name}_{mask_name}_ns{ns}_nr{nr}_ps{ps}_pr{pr}"
                            row = metrics_row(
                                name,
                                prob,
                                meta,
                                y,
                                base_pred,
                                mult,
                                {
                                    "experiment": "R62_class",
                                    "class_set": set_name,
                                    "classes": str(classes),
                                    "mask": mask_name,
                                    "normal_style_w": ns,
                                    "normal_robust_w": nr,
                                    "pressure_style_w": ps,
                                    "pressure_robust_w": pr,
                                    "pressure_coverage": float(mask.mean()),
                                    "_masks": {mask_name: mask, "non_pressure": ~mask},
                                },
                            )
                            rows.append(row)
                            oof_by_name[name] = prob
                            tws = make_pressure_weights(len(test_prefix_base), tmask, ns, ps)
                            twr = make_pressure_weights(len(test_prefix_base), tmask, nr, pr)
                            test_by_name[name] = row_class_three_blend(r42_test, style_test, robust_test, tws, twr, classes)

    search = pd.DataFrame(rows).sort_values(["action_macro_f1", "churn_vs_r42"], ascending=[False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "r62_pressure_gate_search.csv", index=False)

    # Generate plausible candidates: prefer low churn first, then one higher-risk option.
    current_sub = test_prefix_base[["rally_uid", "prefix_len"]].merge(pd.read_csv(CURRENT_SUB_PATH), on="rally_uid", how="left")
    if current_sub[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError("Current R34 submission did not align.")

    generated = []
    used: set[str] = set()
    candidate_pool = pd.concat(
        [
            search[(search["candidate"] != "r42_base") & (search["churn_vs_r42"] <= 0.06)].head(4),
            search[(search["candidate"] != "r42_base") & (search["churn_vs_r42"] <= 0.08)].head(4),
            search[(search["candidate"] != "r42_base") & (search["churn_vs_r42"] <= 0.10)].head(2),
        ],
        ignore_index=True,
    )
    for _, row in candidate_pool.iterrows():
        label = str(row["candidate"])
        if label in used or label not in test_by_name:
            continue
        pred = apply_action(test_by_name[label], test_prefix_base, mult)
        safe = label.replace(".", "p").replace(" ", "_")
        name = f"submission_{safe}_current_point_server.csv"
        info = write_r58_submission(
            test_prefix_base,
            pred,
            current_sub,
            name,
            {
                "source_oof_action_f1": float(row["action_macro_f1"]),
                "source_oof_churn": float(row["churn_vs_r42"]),
                "experiment": str(row.get("experiment", "")),
            },
        )
        # Move/copy metadata path names into R62 output as well; write_r58 writes
        # into the R58/R61 output dir, so mirror the file for discoverability.
        src_path = Path(info["path"])
        dst_path = OUTDIR / name
        dst_path.write_bytes(src_path.read_bytes())
        info["path"] = str(dst_path)
        generated.append(info)
        used.add(label)
        if len(generated) >= 8:
            break

    pd.DataFrame(generated).to_csv(OUTDIR / "r62_generated_candidates.csv", index=False)
    report = {
        "base_action_f1": base_f1,
        "best_overall_rows": search.head(20).to_dict(orient="records"),
        "best_churn_le_6": search[(search["candidate"] != "r42_base") & (search["churn_vs_r42"] <= 0.06)].head(10).to_dict(orient="records"),
        "best_churn_le_8": search[(search["candidate"] != "r42_base") & (search["churn_vs_r42"] <= 0.08)].head(10).to_dict(orient="records"),
        "generated": generated,
        "note": "R62 changes action only. Point/server are fixed to current R34 branch.",
    }
    (OUTDIR / "r62_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(search.head(25).to_string(index=False))
    print(pd.DataFrame(generated).to_string(index=False))


if __name__ == "__main__":
    main()
