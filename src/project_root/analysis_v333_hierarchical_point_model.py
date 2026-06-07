"""V333 hierarchical terminal/depth/side point model.

This experiment keeps the package action/server fixed to the clean V306
submission and only exports local point candidates when the reconstructed V306
OOF point anchor and local evidence gate both pass.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from baseline_lgbm import (
    POINT_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    validate_raw_data,
)
from analysis_v261_action_conditioned_point_residual import (
    action_family,
    normalize_rows_safe,
    numeric_feature_columns,
    train_oof_prob,
)
from analysis_v305_rebuild_v261_from_literal_v188 import align_train_to_literal_meta, point_column
from analysis_v306_point0_addition_probe import apply_point0_additions, load_artifacts


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTDIR = ROOT / "v333_hierarchical_point_model"
ANCHOR_SUBMISSION = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
BANNED_PATH_TOKENS = ("TTMATCH", "OLD_SERVER", "OLDSERVER", "upload_candidates", "selected", "submissions")
MIN_POINT_DELTA = 0.0008
MIN_CHANGED_ROWS = 5
MAX_CHANGED_ROWS = 36
V306_CAP = 0.01


@dataclass(frozen=True)
class VariantSpec:
    name: str
    filename: str
    budget: int
    selector: str


VARIANT_SPECS: tuple[VariantSpec, ...] = (
    VariantSpec("v333_hier_soft_cap12", "submission_v333_hier_soft_cap12__v173action_v300server.csv", 12, "soft"),
    VariantSpec("v333_hier_soft_cap18", "submission_v333_hier_soft_cap18__v173action_v300server.csv", 18, "soft"),
    VariantSpec(
        "v333_nonterminal_only_cap12",
        "submission_v333_nonterminal_only_cap12__v173action_v300server.csv",
        12,
        "nonterminal_only",
    ),
    VariantSpec(
        "v333_depth_confident_cap18",
        "submission_v333_depth_confident_cap18__v173action_v300server.csv",
        18,
        "depth_confident",
    ),
    VariantSpec("v333_no_p0_add_cap24", "submission_v333_no_p0_add_cap24__v173action_v300server.csv", 24, "no_p0_add"),
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def path_has_banned_token(path: Path | str) -> bool:
    upper = str(path).upper()
    return any(token.upper() in upper for token in BANNED_PATH_TOKENS)


def protected_output_path(outdir: Path, filename: str) -> Path:
    path = Path(outdir) / filename
    parts = {part.lower() for part in path.parts}
    if any("upload_candidates" in part for part in parts) or "selected" in parts or "submissions" in parts:
        raise ValueError(f"refusing non-local V333 export path: {path}")
    if path.parent != Path(outdir):
        raise ValueError(f"refusing non-local V333 export path: {path}")
    if path_has_banned_token(path):
        raise ValueError(f"refusing banned V333 export path: {path}")
    return path


def point_to_depth_side(point_id: int) -> tuple[int, int]:
    point = int(point_id)
    if point == 0:
        return -1, -1
    if not 1 <= point <= 9:
        raise ValueError(f"pointId outside 0..9: {point_id}")
    point0 = point - 1
    return point0 // 3, point0 % 3


def depth_side_to_point(depth: int, side: int) -> int:
    d = int(depth)
    s = int(side)
    if d not in (0, 1, 2) or s not in (0, 1, 2):
        raise ValueError(f"depth/side outside 0..2: {depth}, {side}")
    return d * 3 + s + 1


def point_depth3(point_id: int) -> int:
    return point_to_depth_side(point_id)[0]


def point_side3(point_id: int) -> int:
    return point_to_depth_side(point_id)[1]


def point_distribution(values: np.ndarray) -> str:
    counts = np.bincount(np.asarray(values, dtype=int), minlength=10)
    return json.dumps({str(i): int(v) for i, v in enumerate(counts) if v > 0}, sort_keys=True)


def macro_f1(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y_true, pred, labels=POINT_CLASSES, average="macro", zero_division=0))


def point0_stats(anchor: np.ndarray, pred: np.ndarray) -> tuple[int, int]:
    base = np.asarray(anchor, dtype=int)
    out = np.asarray(pred, dtype=int)
    return int(np.sum((base != 0) & (out == 0))), int(np.sum((base == 0) & (out != 0)))


def build_base_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_raw = pd.read_csv(ROOT / "train.csv")
    test_raw = pd.read_csv(ROOT / "test_new.csv")
    validate_raw_data(train_raw, test_raw)
    train_raw = add_role_and_score_features(train_raw)
    test_raw = add_role_and_score_features(test_raw)
    train_df = build_train_prefix_table(train_raw, 6)
    test_df = build_test_prefix_table(test_raw, 6)

    train_df = add_point_geometry(train_df)
    test_df = add_point_geometry(test_df)

    train_df["fold"] = -1
    rally_meta = train_df[["rally_uid", "match"]].drop_duplicates("rally_uid").reset_index(drop=True)
    splitter = GroupKFold(n_splits=5)
    for fold, (_, valid_idx) in enumerate(splitter.split(rally_meta, groups=rally_meta["match"])):
        valid_rallies = set(rally_meta.iloc[valid_idx]["rally_uid"].astype(int))
        train_df.loc[train_df["rally_uid"].isin(valid_rallies), "fold"] = fold
    if train_df["fold"].lt(0).any():
        raise RuntimeError("fold assignment failed")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def add_point_geometry(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lag0_point_depth"] = out["lag0_pointId"].astype(int).map(point_depth3)
    out["lag0_point_side"] = out["lag0_pointId"].astype(int).map(point_side3)
    out["lag0_action_family"] = out["lag0_actionId"].astype(int).map(action_family)
    out["v333_phase"] = pd.cut(
        out["prefix_len"].astype(int),
        bins=[0, 1, 3, 6, 99],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(int)
    return out


def load_package_anchor(path: Path = ANCHOR_SUBMISSION) -> pd.DataFrame:
    if path_has_banned_token(path):
        raise ValueError(f"banned anchor path for V333: {path}")
    if not path.exists():
        raise FileNotFoundError(f"missing package anchor: {path}")
    anchor = pd.read_csv(path)
    if list(anchor.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"anchor columns {list(anchor.columns)} != {SUBMISSION_COLUMNS}")
    if len(anchor) != 1845:
        raise ValueError(f"anchor row count {len(anchor)} != 1845")
    return anchor


def add_foldsafe_proxy_columns_no_upload(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    package_anchor: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    base_features = numeric_feature_columns(train_df, include_proxy=False)
    for col in base_features:
        if col not in test_df:
            test_df[col] = 0

    action_oof, action_test, action_folds = train_oof_prob(
        train_df,
        test_df,
        train_df["next_actionId"].astype(int).to_numpy(),
        list(range(19)),
        base_features,
        seed=2610,
        n_estimators=120,
        min_samples_leaf=5,
    )
    terminal_oof, terminal_test, terminal_folds = train_oof_prob(
        train_df,
        test_df,
        train_df["next_pointId"].eq(0).astype(int).to_numpy(),
        [0, 1],
        base_features,
        seed=2710,
        n_estimators=120,
        min_samples_leaf=8,
    )

    out_train = train_df.copy()
    out_test = test_df.merge(
        package_anchor[["rally_uid", "actionId", "pointId"]],
        on="rally_uid",
        how="left",
        validate="one_to_one",
    )
    if out_test[["actionId", "pointId"]].isna().any().any():
        raise ValueError("package anchor did not align to test prefix rows")
    out_train["v261_action_proxy"] = action_oof.argmax(axis=1).astype(int)
    out_train["v261_action_family"] = out_train["v261_action_proxy"].map(action_family)
    out_train["v261_terminal_proxy"] = terminal_oof[:, 1]
    out_train["v261_anchor_point"] = -1
    out_train["v261_anchor_depth"] = -1
    out_train["v261_anchor_side"] = -1

    out_test["v261_action_proxy"] = out_test["actionId"].astype(int)
    out_test["v261_action_family"] = out_test["v261_action_proxy"].map(action_family)
    out_test["v261_terminal_proxy"] = terminal_test[:, 1]
    out_test["v261_anchor_point"] = out_test["pointId"].astype(int)
    out_test["v261_anchor_depth"] = out_test["v261_anchor_point"].map(point_depth3)
    out_test["v261_anchor_side"] = out_test["v261_anchor_point"].map(point_side3)
    out_test = out_test.drop(columns=["actionId", "pointId"])

    folds = [{"stage": "action_proxy", **r} for r in action_folds]
    folds += [{"stage": "terminal_proxy", **r} for r in terminal_folds]
    return out_train, out_test, folds


def _align_by_key(base: pd.DataFrame, values: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    key = ["rally_uid", "prefix_len", "next_actionId", "next_pointId"]
    missing = [c for c in key if c not in base or c not in meta]
    if missing:
        raise KeyError(f"cannot align V173 action OOF; missing columns: {missing}")
    lookup = meta[key].copy()
    lookup["_v333_value"] = np.asarray(values, dtype=int)
    merged = base[key].merge(lookup, on=key, how="left", validate="one_to_one")
    if merged["_v333_value"].isna().any():
        raise ValueError("V173 OOF rows did not align to V333 train frame")
    return merged["_v333_value"].astype(int).to_numpy()


def try_rebuild_v173_action_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    package_anchor: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        from analysis_r184_receiver_affordance_refiner import rebuild_v173_best_actions

        state = rebuild_v173_best_actions()
        train_action = _align_by_key(train_df, state["v173_pred_oof"], state["rows"])
        test_lookup = pd.DataFrame(
            {
                "rally_uid": state["rally_uids"].astype(int),
                "_v333_v173_action": np.asarray(state["v173_pred_test"], dtype=int),
            }
        )
        merged = test_df[["rally_uid"]].merge(test_lookup, on="rally_uid", how="left", validate="one_to_one")
        if merged["_v333_v173_action"].isna().any():
            raise ValueError("V173 test rows did not align to V333 test frame")
        test_action = merged["_v333_v173_action"].astype(int).to_numpy()
        package_action = package_anchor["actionId"].astype(int).to_numpy()
        exact = bool(np.array_equal(test_action, package_action))
        if not exact:
            return (
                train_df["v261_action_proxy"].astype(int).to_numpy(),
                package_action,
                {
                    "action_anchor_source": "fallback_v261_action_proxy",
                    "v173_rebuild_status": "test_mismatch",
                    "v173_test_mismatch_rows": int(np.sum(test_action != package_action)),
                },
            )
        return (
            train_action,
            test_action,
            {
                "action_anchor_source": "rebuilt_v173_pred_oof",
                "v173_rebuild_status": "exact_test_match",
                "v173_best_candidate": str(state.get("best_candidate", "")),
            },
        )
    except Exception as exc:  # pragma: no cover - exercised only when optional artifacts are absent.
        return (
            train_df["v261_action_proxy"].astype(int).to_numpy(),
            package_anchor["actionId"].astype(int).to_numpy(),
            {
                "action_anchor_source": "fallback_v261_action_proxy",
                "v173_rebuild_status": "unavailable",
                "v173_rebuild_error": f"{type(exc).__name__}: {exc}",
            },
        )


def reconstruct_v306_point_anchor() -> dict[str, Any]:
    package_anchor = load_package_anchor()
    artifacts = load_artifacts()
    train_df, test_df = build_base_frames()
    train_df, test_df, proxy_folds = add_foldsafe_proxy_columns_no_upload(train_df, test_df, package_anchor)
    train_df = align_train_to_literal_meta(train_df, artifacts["meta"])

    for col in train_df.columns:
        if col not in test_df and pd.api.types.is_numeric_dtype(train_df[col]):
            test_df[col] = 0

    point_features = numeric_feature_columns(train_df, include_proxy=True)
    point_features = [c for c in point_features if c in test_df]
    y = train_df["next_pointId"].astype(int).to_numpy()
    model_oof_prob, _, point_folds = train_oof_prob(
        train_df,
        test_df,
        y,
        POINT_CLASSES,
        point_features,
        seed=30510,
        n_estimators=260,
        min_samples_leaf=4,
    )

    cap5_oof = artifacts["cap5_oof"]
    oof_base = cap5_oof[point_column(cap5_oof)].astype(int).to_numpy()
    if len(oof_base) != len(y):
        raise ValueError(f"V188 cap5 OOF length {len(oof_base)} != y length {len(y)}")
    oof_budget = int(np.floor(len(oof_base) * V306_CAP))
    v306_oof, v306_oof_added, _ = apply_point0_additions(oof_base, model_oof_prob, oof_budget)

    package_point = package_anchor["pointId"].astype(int).to_numpy()
    train_action, test_action, action_meta = try_rebuild_v173_action_features(train_df, test_df, package_anchor)
    train_df = add_anchor_feature_columns(train_df, v306_oof, train_action)
    test_df = add_anchor_feature_columns(test_df, package_point, test_action)

    return {
        "status": "exact_oof_reconstructed",
        "anchor_source": "v306_reconstructed_from_v305_v188_and_v261_oof",
        "train_df": train_df,
        "test_df": test_df,
        "package_anchor": package_anchor,
        "y": y,
        "v306_oof_point": v306_oof,
        "v306_test_point": package_point,
        "v188_cap5_oof_point": oof_base,
        "v306_oof_point0_additions": int(v306_oof_added.sum()),
        "v306_oof_budget": oof_budget,
        "folds": proxy_folds + [{"stage": "v306_anchor_reconstruct_point", **r} for r in point_folds],
        **action_meta,
    }


def add_anchor_feature_columns(df: pd.DataFrame, point_anchor: np.ndarray, action_anchor: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["v333_v173_action_anchor"] = np.asarray(action_anchor, dtype=int)
    out["v333_v173_action_family"] = out["v333_v173_action_anchor"].map(action_family)
    out["v333_v306_point_anchor"] = np.asarray(point_anchor, dtype=int)
    out["v333_v306_anchor_terminal"] = out["v333_v306_point_anchor"].eq(0).astype(int)
    out["v333_v306_anchor_depth"] = out["v333_v306_point_anchor"].map(point_depth3)
    out["v333_v306_anchor_side"] = out["v333_v306_point_anchor"].map(point_side3)
    return out


def predict_aligned(model: ExtraTreesClassifier, frame: pd.DataFrame, classes: Iterable[int]) -> np.ndarray:
    classes = list(classes)
    raw = model.predict_proba(frame)
    out = np.zeros((len(frame), len(classes)), dtype=float)
    model_classes = [int(c) for c in model.classes_]
    for src_idx, cls in enumerate(model_classes):
        if cls in classes:
            out[:, classes.index(cls)] = raw[:, src_idx]
    return normalize_rows_safe(out)


def constant_probability(n_rows: int, classes: Iterable[int], observed: np.ndarray | None = None) -> np.ndarray:
    classes = list(classes)
    out = np.zeros((n_rows, len(classes)), dtype=float)
    if observed is None or len(observed) == 0:
        out[:, :] = 1.0 / len(classes)
        return out
    counts = np.bincount([classes.index(int(v)) for v in observed if int(v) in classes], minlength=len(classes))
    if counts.sum() <= 0:
        out[:, :] = 1.0 / len(classes)
    else:
        out[:, :] = counts / counts.sum()
    return out


def fit_predict_head(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    pred_x: pd.DataFrame,
    classes: Iterable[int],
    seed: int,
) -> np.ndarray:
    y = np.asarray(train_y, dtype=int)
    classes = list(classes)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return constant_probability(len(pred_x), classes, y)
    model = ExtraTreesClassifier(
        n_estimators=220,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(train_x.fillna(0), y)
    return predict_aligned(model, pred_x.fillna(0), classes)


def compose_point_probabilities(
    terminal_prob: np.ndarray,
    depth_prob: np.ndarray,
    side_prob_by_depth: dict[int, np.ndarray],
) -> np.ndarray:
    term = normalize_rows_safe(terminal_prob)[:, 1]
    depth = normalize_rows_safe(depth_prob)
    if depth.shape[1] != 3:
        raise ValueError("depth_prob must have three columns")
    out = np.zeros((len(term), 10), dtype=float)
    out[:, 0] = term
    nonterminal = 1.0 - term
    for depth_id in (0, 1, 2):
        side = normalize_rows_safe(side_prob_by_depth[depth_id])
        if side.shape[1] != 3:
            raise ValueError("side probabilities must have three columns")
        for side_id in (0, 1, 2):
            point = depth_side_to_point(depth_id, side_id)
            out[:, point] = nonterminal * depth[:, depth_id] * side[:, side_id]
    return normalize_rows_safe(out)


def train_hierarchical_point_probabilities(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    y = train_df["next_pointId"].astype(int).to_numpy()
    y_terminal = (y == 0).astype(int)
    y_depth = np.array([point_depth3(v) for v in y], dtype=int)
    y_side = np.array([point_side3(v) for v in y], dtype=int)

    oof_terminal = np.zeros((len(train_df), 2), dtype=float)
    oof_depth = np.zeros((len(train_df), 3), dtype=float)
    oof_side: dict[int, np.ndarray] = {depth_id: np.zeros((len(train_df), 3), dtype=float) for depth_id in (0, 1, 2)}
    test_terminal_sum = np.zeros((len(test_df), 2), dtype=float)
    test_depth_sum = np.zeros((len(test_df), 3), dtype=float)
    test_side_sum: dict[int, np.ndarray] = {depth_id: np.zeros((len(test_df), 3), dtype=float) for depth_id in (0, 1, 2)}
    fold_rows: list[dict[str, Any]] = []

    for fold in sorted(train_df["fold"].astype(int).unique()):
        valid = train_df["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        x_train = train_df.loc[train, features]
        x_valid = train_df.loc[valid, features]
        x_test = test_df.loc[:, features]

        term_valid = fit_predict_head(x_train, y_terminal[train], x_valid, [0, 1], 33300 + int(fold))
        term_test = fit_predict_head(x_train, y_terminal[train], x_test, [0, 1], 33300 + int(fold))
        oof_terminal[valid] = term_valid
        test_terminal_sum += term_test

        nonterminal_train = train & (y != 0)
        depth_valid = fit_predict_head(
            train_df.loc[nonterminal_train, features],
            y_depth[nonterminal_train],
            x_valid,
            [0, 1, 2],
            33400 + int(fold),
        )
        depth_test = fit_predict_head(
            train_df.loc[nonterminal_train, features],
            y_depth[nonterminal_train],
            x_test,
            [0, 1, 2],
            33400 + int(fold),
        )
        oof_depth[valid] = depth_valid
        test_depth_sum += depth_test

        for depth_id in (0, 1, 2):
            side_train = train & (y_depth == depth_id)
            side_valid = fit_predict_head(
                train_df.loc[side_train, features],
                y_side[side_train],
                x_valid,
                [0, 1, 2],
                33500 + depth_id * 10 + int(fold),
            )
            side_test = fit_predict_head(
                train_df.loc[side_train, features],
                y_side[side_train],
                x_test,
                [0, 1, 2],
                33500 + depth_id * 10 + int(fold),
            )
            oof_side[depth_id][valid] = side_valid
            test_side_sum[depth_id] += side_test

        fold_rows.append({"stage": "v333_hierarchical_heads", "fold": int(fold), "train_rows": int(train.sum()), "valid_rows": int(valid.sum())})

    n_folds = max(1, len(fold_rows))
    oof_prob = compose_point_probabilities(oof_terminal, oof_depth, oof_side)
    test_prob = compose_point_probabilities(
        test_terminal_sum / n_folds,
        test_depth_sum / n_folds,
        {depth_id: test_side_sum[depth_id] / n_folds for depth_id in (0, 1, 2)},
    )
    return oof_prob, test_prob, fold_rows


def select_variant_predictions(
    anchor: np.ndarray,
    prob: np.ndarray,
    budget: int,
    selector: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(anchor, dtype=int)
    p = normalize_rows_safe(prob)
    top = p.argmax(axis=1).astype(int)
    base_prob = p[np.arange(len(p)), np.clip(base, 0, p.shape[1] - 1)]
    margin = p[np.arange(len(p)), top] - base_prob
    eligible = (top != base) & np.isfinite(margin) & (margin > 0.0)
    if selector == "nonterminal_only":
        eligible &= (base != 0) & (top != 0)
    elif selector == "depth_confident":
        base_depth = np.array([point_depth3(v) for v in base], dtype=int)
        top_depth = np.array([point_depth3(v) for v in top], dtype=int)
        eligible &= (base != 0) & (top != 0) & (base_depth != top_depth)
    elif selector == "no_p0_add":
        eligible &= ~((base != 0) & (top == 0))
    elif selector != "soft":
        raise ValueError(f"unknown selector: {selector}")

    changed = np.zeros(len(base), dtype=bool)
    if budget > 0 and eligible.any():
        idx = np.where(eligible)[0]
        order = idx[np.argsort(-margin[idx], kind="mergesort")]
        changed[order[: min(int(budget), len(order))]] = True
    out = base.copy()
    out[changed] = top[changed]
    return out, changed, margin


def changed_row_precision(y_true: np.ndarray, anchor: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    base = np.asarray(anchor, dtype=int)
    out = np.asarray(pred, dtype=int)
    changed = out != base
    rows = int(changed.sum())
    correct = int(np.sum(changed & (out == y)))
    return {
        "changed_oof_rows": rows,
        "changed_oof_correct": correct,
        "changed_oof_precision": float(correct / rows) if rows else 0.0,
    }


def evidence_passes(row: dict[str, Any] | pd.Series, *, anchor_is_fallback: bool) -> bool:
    data = row.to_dict() if isinstance(row, pd.Series) else row
    if anchor_is_fallback:
        return False
    return bool(
        float(data.get("point_oof_delta_vs_v306", 0.0)) >= MIN_POINT_DELTA
        and MIN_CHANGED_ROWS <= int(data.get("test_changed_rows", 0)) <= MAX_CHANGED_ROWS
        and int(data.get("test_point0_total", 0)) <= int(data.get("anchor_point0_total", 0)) + MAX_CHANGED_ROWS
    )


def build_export_frame(anchor: pd.DataFrame, point: np.ndarray) -> pd.DataFrame:
    out = anchor.copy()
    out["pointId"] = np.asarray(point, dtype=int)
    out = out[SUBMISSION_COLUMNS]
    if not out["actionId"].equals(anchor["actionId"]):
        raise AssertionError("V333 export changed actionId")
    if not out["serverGetPoint"].equals(anchor["serverGetPoint"]):
        raise AssertionError("V333 export changed serverGetPoint")
    return out


def write_report(report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    OUTDIR.mkdir(exist_ok=True)
    search = pd.DataFrame(rows)
    if not search.empty:
        search = search.sort_values(
            ["evidence_pass", "point_oof_delta_vs_v306", "test_changed_rows"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    search.to_csv(OUTDIR / "v333_point_search.csv", index=False)
    (OUTDIR / "v333_report.json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    best = report.get("best_candidate") or {}
    (OUTDIR / "v333_report.md").write_text(
        "# V333 Hierarchical Point Model\n\n"
        f"- Verdict: `{report.get('verdict', 'UNKNOWN')}`\n"
        f"- Anchor status: `{report.get('anchor_status', 'UNKNOWN')}`\n"
        f"- Best candidate: `{best.get('candidate', 'none')}`\n"
        f"- Best delta vs V306: `{float(best.get('point_oof_delta_vs_v306', 0.0)):.6f}`\n"
        f"- Generated CSVs: `{len(report.get('generated_submissions', []))}`\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    generated: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    anchor_is_fallback = False
    try:
        state = reconstruct_v306_point_anchor()
    except Exception as exc:
        anchor_is_fallback = True
        report = {
            "verdict": "NO_EXPORT_ANCHOR_FALLBACK",
            "anchor_status": "fallback_unavailable",
            "anchor_error": f"{type(exc).__name__}: {exc}",
            "generated_submissions": [],
            "notes": [
                "V333 exports no CSV unless the V306 point OOF anchor is reconstructed.",
                "No upload directory writes, TTMATCH, old-server sources, or manual row edits are used.",
            ],
        }
        write_report(report, rows)
        return

    y = state["y"]
    train_df = state["train_df"]
    test_df = state["test_df"]
    v306_oof = state["v306_oof_point"]
    v306_test = state["v306_test_point"]
    package_anchor = state["package_anchor"]
    for col in train_df.columns:
        if col not in test_df and pd.api.types.is_numeric_dtype(train_df[col]):
            test_df[col] = 0
    features = numeric_feature_columns(train_df, include_proxy=True)
    features = [c for c in features if c in test_df]
    blocked = {"v333_v306_point_anchor"}
    features = [c for c in features if c not in blocked]

    hier_oof_prob, hier_test_prob, hier_folds = train_hierarchical_point_probabilities(train_df, test_df, features)
    raw_hier_oof_score = macro_f1(y, hier_oof_prob.argmax(axis=1).astype(int))
    v306_oof_score = macro_f1(y, v306_oof)
    anchor_point0_total = int(np.sum(v306_test == 0))

    for spec in VARIANT_SPECS:
        cap = spec.budget / len(v306_test)
        oof_budget = int(np.floor(len(v306_oof) * cap))
        oof_pred, oof_changed, oof_margin = select_variant_predictions(v306_oof, hier_oof_prob, oof_budget, spec.selector)
        test_pred, test_changed, test_margin = select_variant_predictions(v306_test, hier_test_prob, spec.budget, spec.selector)
        score = macro_f1(y, oof_pred)
        p0_add, p0_remove = point0_stats(v306_test, test_pred)
        precision = changed_row_precision(y, v306_oof, oof_pred)
        row = {
            "candidate": spec.name,
            "selector": spec.selector,
            "budget": spec.budget,
            "oof_budget": oof_budget,
            "v306_point_macro_f1": v306_oof_score,
            "raw_hier_point_macro_f1": raw_hier_oof_score,
            "point_macro_f1": score,
            "point_oof_delta_vs_v306": score - v306_oof_score,
            "test_changed_rows": int(test_changed.sum()),
            "oof_changed_rows": int(oof_changed.sum()),
            "test_point0_additions": p0_add,
            "test_point0_removals": p0_remove,
            "anchor_point0_total": anchor_point0_total,
            "test_point0_total": int(np.sum(test_pred == 0)),
            "test_distribution": point_distribution(test_pred),
            "anchor_distribution": point_distribution(v306_test),
            "test_margin_min_changed": float(test_margin[test_changed].min()) if test_changed.any() else 0.0,
            "test_margin_mean_changed": float(test_margin[test_changed].mean()) if test_changed.any() else 0.0,
            "oof_margin_mean_changed": float(oof_margin[oof_changed].mean()) if oof_changed.any() else 0.0,
            **precision,
        }
        row["evidence_pass"] = evidence_passes(row, anchor_is_fallback=anchor_is_fallback)
        row["decision"] = "EXPORT_LOCAL" if row["evidence_pass"] else "DO_NOT_EXPORT"
        if row["evidence_pass"]:
            path = protected_output_path(OUTDIR, spec.filename)
            out = build_export_frame(package_anchor, test_pred)
            out.to_csv(path, index=False, float_format="%.8f")
            row["submission"] = spec.filename
            row["path"] = relative_path(path)
            generated.append({"candidate": spec.name, "submission": spec.filename, "path": relative_path(path)})
        rows.append(row)

    search = pd.DataFrame(rows)
    passed = search[search["evidence_pass"].astype(bool)] if not search.empty else pd.DataFrame()
    best_source = passed if not passed.empty else search
    best = best_source.sort_values(["point_oof_delta_vs_v306", "test_changed_rows"], ascending=[False, True]).head(1)
    best_dict = best.iloc[0].to_dict() if not best.empty else {}
    report = {
        "verdict": "HAS_EVIDENCE_CANDIDATE" if generated else "NO_EXPORT_NO_EVIDENCE",
        "anchor_status": state["status"],
        "anchor_source": state["anchor_source"],
        "action_anchor_source": state.get("action_anchor_source"),
        "v173_rebuild_status": state.get("v173_rebuild_status"),
        "v306_point_macro_f1": v306_oof_score,
        "raw_hier_point_macro_f1": raw_hier_oof_score,
        "best_candidate": best_dict,
        "generated_submissions": generated,
        "folds": state["folds"] + hier_folds,
        "feature_count": len(features),
        "v306_oof_point0_additions": state["v306_oof_point0_additions"],
        "policy": {
            "fixed_action_server": "v306_point0_addition_probe/submission_v306_p0_cap0p01__v173action_v300server.csv",
            "no_ttm": True,
            "no_old_server": True,
            "no_upload_dir_writes": True,
            "manual_row_edits": False,
        },
        "notes": [
            "Point probabilities use P(0)=P(terminal) and P(1..9)=P(nonterminal)*P(depth)*P(side|depth).",
            "Exports preserve package actionId and serverGetPoint exactly.",
            "No CSV is exported unless V306 OOF anchor reconstruction and the point evidence gate pass.",
        ],
    }
    write_report(report, rows)


if __name__ == "__main__":
    main()
