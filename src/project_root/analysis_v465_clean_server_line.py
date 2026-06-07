"""V465 clean server-only rank/MAD pipeline.

This script preserves V362 actionId and pointId, trains serverGetPoint signals
only from clean train labels plus train/test_new-observed fields, and writes
small server-only candidates for review.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import validate_submission_schema

try:  # pragma: no cover - availability is environment dependent
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

try:  # pragma: no cover - availability depends on sklearn version
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "v465_clean_server_line"
ANCHOR_RELATIVE = Path("v362_point_hierarchical_specialists") / (
    "submission_v362_depth_agree_only__v173action_v300server.csv"
)
TRAIN_RELATIVE = Path("train.csv")
TEST_NEW_RELATIVE = Path("test_new.csv")
EXPECTED_ROWS = 1845

SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
BANNED_PATH_TOKENS = (
    "TTMATCH",
    "OLD_SERVER",
    "OLDSERVER",
    "OLD-SERVER",
    "UPLOAD_CANDIDATES_20260519",
)
PROB_MIN = 1e-6
PROB_MAX = 1.0 - 1e-6
TARGET_MADS = (0.003, 0.005, 0.010, 0.015)


@dataclass(frozen=True)
class ServerSignal:
    name: str
    oof: np.ndarray
    test: np.ndarray
    auc: float


@dataclass(frozen=True)
class ServerSource:
    name: str
    server: np.ndarray
    path: str


@dataclass(frozen=True)
class ServerCandidate:
    candidate: str
    server: np.ndarray
    source_names: tuple[str, ...]
    train_oof_auc: float
    train_oof_auc_delta_vs_anchor_proxy: float
    risk: str
    decision: str


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def no_banned_input_guard(paths: Iterable[Path | str]) -> None:
    bad = [str(p) for p in paths if any(token in str(p).upper() for token in BANNED_PATH_TOKENS)]
    if bad:
        raise ValueError(f"banned clean-server input path: {bad}")


def _validate_output_path(path: Path, outdir: Path) -> Path:
    outdir_resolved = Path(outdir).resolve()
    path_resolved = Path(path).resolve()
    if outdir_resolved not in path_resolved.parents and path_resolved != outdir_resolved:
        raise ValueError(f"output path escapes V465 outdir: {path_resolved}")
    no_banned_input_guard([path_resolved])
    return path_resolved


def clip_prob(values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.5, posinf=PROB_MAX, neginf=PROB_MIN)
    return np.clip(arr, PROB_MIN, PROB_MAX)


def rank_normalize_to_anchor(source: np.ndarray | pd.Series, anchor: np.ndarray | pd.Series) -> np.ndarray:
    source_arr = np.asarray(source, dtype=float)
    anchor_sorted = np.sort(clip_prob(anchor))
    if len(source_arr) != len(anchor_sorted):
        raise ValueError("source and anchor lengths differ")
    if len(source_arr) == 0:
        return np.asarray([], dtype=float)
    if len(source_arr) == 1:
        return np.array([anchor_sorted[0]], dtype=float)
    finite = source_arr[np.isfinite(source_arr)]
    fill = float(np.median(finite)) if len(finite) else 0.5
    source_arr = np.nan_to_num(source_arr, nan=fill, posinf=fill, neginf=fill)
    ranks = pd.Series(source_arr).rank(method="average").to_numpy(dtype=float) - 1.0
    return clip_prob(np.interp(ranks, np.arange(len(anchor_sorted), dtype=float), anchor_sorted))


def blend_to_target_mad(
    anchor: np.ndarray | pd.Series,
    target: np.ndarray | pd.Series,
    *,
    target_mad: float,
) -> np.ndarray:
    if not np.isfinite(target_mad) or target_mad < 0.0:
        raise ValueError(f"target_mad must be finite and non-negative, got {target_mad}")
    anchor_arr = clip_prob(anchor)
    target_arr = clip_prob(target)
    if len(anchor_arr) != len(target_arr):
        raise ValueError("anchor and target lengths differ")
    delta = target_arr - anchor_arr
    full_mad = float(np.mean(np.abs(delta)))
    if full_mad == 0.0 or target_mad == 0.0:
        return anchor_arr.copy()
    return clip_prob(anchor_arr + min(1.0, target_mad / full_mad) * delta)


def package_server_only(anchor: pd.DataFrame, server: np.ndarray, *, expected_rows: int) -> pd.DataFrame:
    out = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    out["serverGetPoint"] = clip_prob(server)
    validate_submission_schema(out, expected_rows=expected_rows)
    if not out["actionId"].astype(int).equals(anchor["actionId"].astype(int)):
        raise AssertionError("V465 changed actionId")
    if not out["pointId"].astype(int).equals(anchor["pointId"].astype(int)):
        raise AssertionError("V465 changed pointId")
    return out


def build_scoreboard_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    out = pd.DataFrame(index=df.index)
    for col in ["match", "numberGame", "rally_id", "scoreSelf", "scoreOther", "strikeNumber"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    score_self = pd.to_numeric(df.get("scoreSelf", 0), errors="coerce").fillna(0)
    score_other = pd.to_numeric(df.get("scoreOther", 0), errors="coerce").fillna(0)
    strike = pd.to_numeric(df.get("strikeNumber", 1), errors="coerce").fillna(1)
    out["score_total"] = score_self + score_other
    out["score_margin"] = score_self - score_other
    out["abs_score_margin"] = out["score_margin"].abs()
    out["is_close_score"] = (out["abs_score_margin"] <= 2).astype(int)
    out["is_deuce_like"] = ((score_self >= 10) & (score_other >= 10)).astype(int)
    out["game_point_like"] = ((score_self >= 10) | (score_other >= 10)).astype(int)
    out["phase_code"] = np.select([strike <= 1, strike <= 3, strike <= 6], [0, 1, 2], default=3)
    return out.replace([np.inf, -np.inf], np.nan).fillna(-1.0)


def action_family(action: int) -> int:
    action = int(action)
    if action == 0:
        return 0
    if 1 <= action <= 7:
        return 1
    if 8 <= action <= 11:
        return 2
    if 12 <= action <= 14:
        return 3
    return 4


def point_depth(point: int) -> int:
    point = int(point)
    if point == 0:
        return 0
    return 1 + ((point - 1) // 3)


def build_action_point_conditioned_features(anchor: pd.DataFrame) -> pd.DataFrame:
    action = anchor["actionId"].astype(int)
    point = anchor["pointId"].astype(int)
    out = pd.DataFrame(index=anchor.index)
    out["anchor_actionId"] = action
    out["anchor_pointId"] = point
    out["anchor_action_family"] = action.map(action_family)
    out["anchor_point_depth"] = point.map(point_depth)
    out["anchor_point0"] = (point == 0).astype(int)
    out["anchor_long_point"] = point.isin([7, 8, 9]).astype(int)
    out["terminal_action0"] = (action == 0).astype(int)
    return out.replace([np.inf, -np.inf], np.nan).fillna(-1.0)


def _build_train_action_point_features(train: pd.DataFrame) -> pd.DataFrame:
    if {"actionId", "pointId"}.issubset(train.columns):
        return build_action_point_conditioned_features(train)
    fallback = pd.DataFrame({"actionId": np.zeros(len(train)), "pointId": np.zeros(len(train))}, index=train.index)
    return build_action_point_conditioned_features(fallback)


def _build_numeric_observed_features(frame: pd.DataFrame) -> pd.DataFrame:
    banned = set(SUBMISSION_COLUMNS) | {"rally_uid", "serverGetPoint"}
    out = pd.DataFrame(index=frame.index)
    for col in frame.columns:
        if col in banned:
            continue
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if numeric.notna().any():
            out[f"raw_{col}"] = numeric
    return out.replace([np.inf, -np.inf], np.nan).fillna(-1.0)


def _align_feature_columns(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = sorted(set(train_x.columns) | set(test_x.columns))
    train_aligned = train_x.reindex(columns=columns, fill_value=-1.0)
    test_aligned = test_x.reindex(columns=columns, fill_value=-1.0)
    return train_aligned.astype(float), test_aligned.astype(float)


def _anchor_predictions_for_test_rows(test_new: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    """Repeat anchor target action/point predictions onto each observed test prefix row."""
    if "rally_uid" not in test_new.columns:
        raise ValueError("test_new must contain rally_uid")
    anchor_map = anchor.set_index("rally_uid")[["actionId", "pointId"]]
    mapped = anchor_map.reindex(test_new["rally_uid"])
    fallback = test_new.reindex(columns=["actionId", "pointId"]).reset_index(drop=True)
    mapped = mapped.reset_index(drop=True)
    for col in ["actionId", "pointId"]:
        if col not in fallback.columns:
            fallback[col] = 0
        mapped[col] = mapped[col].fillna(pd.to_numeric(fallback[col], errors="coerce"))
    mapped = mapped.fillna(0)
    return mapped


def build_feature_matrices(train: pd.DataFrame, test_new: pd.DataFrame, anchor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = [
        build_scoreboard_features(train),
        _build_train_action_point_features(train),
        _build_numeric_observed_features(train),
    ]
    test_anchor_ap = _anchor_predictions_for_test_rows(test_new, anchor)
    test_parts = [
        build_scoreboard_features(test_new),
        build_action_point_conditioned_features(test_anchor_ap),
        _build_numeric_observed_features(test_new),
    ]
    train_x = pd.concat(train_parts, axis=1)
    test_x = pd.concat(test_parts, axis=1)
    train_x = train_x.loc[:, ~train_x.columns.duplicated()]
    test_x = test_x.loc[:, ~test_x.columns.duplicated()]
    return _align_feature_columns(train_x, test_x)


def aggregate_signals_to_anchor(
    signals: list[ServerSignal],
    test_new: pd.DataFrame,
    anchor: pd.DataFrame,
) -> list[ServerSignal]:
    """Aggregate row-level test predictions to one server probability per submission rally_uid."""
    if "rally_uid" not in test_new.columns:
        raise ValueError("test_new must contain rally_uid")
    anchor_uids = anchor["rally_uid"]
    out: list[ServerSignal] = []
    for signal in signals:
        if len(signal.test) != len(test_new):
            raise ValueError(f"{signal.name} test prediction length does not match test_new rows")
        row_pred = pd.DataFrame({"rally_uid": test_new["rally_uid"].to_numpy(), "server": signal.test})
        by_uid = row_pred.groupby("rally_uid", sort=False)["server"].mean()
        if not anchor_uids.isin(by_uid.index).all():
            missing = anchor_uids[~anchor_uids.isin(by_uid.index)].head(5).tolist()
            raise ValueError(f"test predictions missing anchor rally_uid values: {missing}")
        out.append(
            ServerSignal(
                name=f"{signal.name}_uidmean",
                oof=signal.oof,
                test=clip_prob(by_uid.reindex(anchor_uids).to_numpy(dtype=float)),
                auc=signal.auc,
            )
        )
    return out


def _support_safe_folds(y: np.ndarray) -> int:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    return min(5, int(counts.min()))


def _support_safe_group_folds(y: np.ndarray, groups: np.ndarray) -> int:
    group_frame = pd.DataFrame({"group": groups, "y": y})
    group_labels = group_frame.groupby("group", sort=False)["y"].max()
    counts = np.bincount(group_labels.to_numpy(dtype=int), minlength=2)
    return min(5, int(counts.min()), int(len(group_labels)))


def fit_server_signals(
    x: pd.DataFrame,
    y: np.ndarray | pd.Series,
    x_test: pd.DataFrame,
    *,
    random_state: int,
    groups: np.ndarray | pd.Series | None = None,
) -> list[ServerSignal]:
    y_arr = np.asarray(y, dtype=int)
    if len(y_arr) != len(x):
        raise ValueError("x and y length mismatch")
    if len(np.unique(y_arr)) != 2:
        raise ValueError("server target is not binary")
    group_arr = None if groups is None else np.asarray(groups)
    if group_arr is not None and len(group_arr) != len(y_arr):
        raise ValueError("groups and y length mismatch")
    folds = _support_safe_group_folds(y_arr, group_arr) if group_arr is not None else _support_safe_folds(y_arr)
    if folds < 2:
        raise ValueError("not enough class support for stratified OOF")

    models: list[tuple[str, Any]] = [
        (
            "logistic_balanced",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=160,
                min_samples_leaf=1 if len(y_arr) < 40 else 8,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
        ),
    ]
    if LGBMClassifier is not None:
        models.append(
            (
                "lgbm_balanced",
                LGBMClassifier(
                    n_estimators=180,
                    learning_rate=0.035,
                    num_leaves=15,
                    class_weight="balanced",
                    random_state=random_state,
                    verbose=-1,
                    n_jobs=1,
                ),
            )
        )
    else:
        models.append(
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=160,
                    min_samples_leaf=1 if len(y_arr) < 40 else 8,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=1,
                ),
            )
        )

    signals: list[ServerSignal] = []
    def make_split_iter():
        if group_arr is not None and StratifiedGroupKFold is not None:
            return StratifiedGroupKFold(
                n_splits=folds,
                shuffle=True,
                random_state=random_state,
            ).split(x, y_arr, group_arr)
        if group_arr is not None:
            return GroupKFold(n_splits=folds).split(x, y_arr, group_arr)
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state).split(x, y_arr)

    for name, prototype in models:
        oof = np.zeros(len(y_arr), dtype=float)
        test_fold_predictions = []
        for train_idx, valid_idx in make_split_iter():
            model = _clone_model(prototype)
            model.fit(x.iloc[train_idx], y_arr[train_idx])
            oof[valid_idx] = _predict_positive(model, x.iloc[valid_idx])
            test_fold_predictions.append(_predict_positive(model, x_test))
        test_pred = np.mean(test_fold_predictions, axis=0)
        auc = float(roc_auc_score(y_arr, clip_prob(oof)))
        signals.append(ServerSignal(name=name, oof=clip_prob(oof), test=clip_prob(test_pred), auc=auc))
    return signals


def _clone_model(model: Any) -> Any:
    from sklearn.base import clone

    return clone(model)


def _predict_positive(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        return clip_prob(proba[:, 1])
    pred = model.predict(x)
    return clip_prob(pred)


def _corr(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _mean_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise ValueError("cannot average an empty target list")
    return clip_prob(np.mean(np.vstack([clip_prob(arr) for arr in arrays]), axis=0))


def build_candidate_servers(
    anchor: pd.DataFrame,
    signals: list[ServerSignal],
    existing_clean_sources: list[ServerSource],
) -> list[ServerCandidate]:
    if not signals:
        raise ValueError("at least one server signal is required")
    anchor_server = clip_prob(anchor["serverGetPoint"])
    best_signal = max(signals, key=lambda signal: signal.auc)
    train_oof_auc = float(np.mean([signal.auc for signal in signals]))
    anchor_proxy_auc = _anchor_proxy_auc(signals, anchor_server)
    targets: dict[str, tuple[np.ndarray, tuple[str, ...]]] = {
        "strict_model_mean": (_mean_arrays([signal.test for signal in signals]), tuple(signal.name for signal in signals)),
        "strict_model_rankmean": (
            _mean_arrays([rank_normalize_to_anchor(signal.test, anchor_server) for signal in signals]),
            tuple(f"{signal.name}:rank" for signal in signals),
        ),
        "scoreboard_value": (best_signal.test, (best_signal.name,)),
        "actionpoint_value": (
            _mean_arrays([signals[0].test, best_signal.test]),
            (signals[0].name, best_signal.name),
        ),
    }
    if existing_clean_sources:
        targets["clean_source_rankmean"] = (
            _mean_arrays([rank_normalize_to_anchor(source.server, anchor_server) for source in existing_clean_sources]),
            tuple(source.name for source in existing_clean_sources),
        )

    ensemble_arrays = [rank_normalize_to_anchor(signal.test, anchor_server) for signal in signals]
    ensemble_names = [f"{signal.name}:rank" for signal in signals]
    for source in existing_clean_sources:
        ensemble_arrays.append(rank_normalize_to_anchor(source.server, anchor_server))
        ensemble_names.append(source.name)
    targets["ensemble_rankmean"] = (_mean_arrays(ensemble_arrays), tuple(ensemble_names))

    candidates: list[ServerCandidate] = []
    seen_servers: set[bytes] = set()
    for target_name, (target, source_names) in targets.items():
        normalized_target = rank_normalize_to_anchor(target, anchor_server)
        for mad in TARGET_MADS:
            server = blend_to_target_mad(anchor_server, normalized_target, target_mad=mad)
            fingerprint = np.round(server, 10).tobytes()
            if fingerprint in seen_servers:
                continue
            seen_servers.add(fingerprint)
            actual_mad = float(np.mean(np.abs(server - anchor_server)))
            mad_key = str(mad).replace(".", "p")
            risk = "safe" if actual_mad <= 0.005 else "exploratory"
            decision = "review" if actual_mad <= 0.010 else "hold"
            candidates.append(
                ServerCandidate(
                    candidate=f"{target_name}_mad{mad_key}",
                    server=server,
                    source_names=source_names,
                    train_oof_auc=train_oof_auc,
                    train_oof_auc_delta_vs_anchor_proxy=train_oof_auc - anchor_proxy_auc,
                    risk=risk,
                    decision=decision,
                )
            )
    return candidates


def _anchor_proxy_auc(signals: list[ServerSignal], anchor_server: np.ndarray) -> float:
    # The anchor has no train-row predictions. A neutral baseline keeps the
    # decision board explicit without inventing old-test label alignment.
    del signals, anchor_server
    return 0.5


def load_submission(path: Path, *, expected_rows: int) -> pd.DataFrame:
    no_banned_input_guard([path])
    frame = pd.read_csv(path)
    validate_submission_schema(frame, expected_rows=expected_rows)
    return frame


def _resolve_source_path(raw_path: object, *, root: Path, search_path: Path) -> Path:
    path = Path(str(raw_path))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    candidates.extend([root / path, search_path.parent / path.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_existing_clean_sources(root: Path, anchor: pd.DataFrame, *, expected_rows: int) -> list[ServerSource]:
    specs = [
        ("v300", root / "v300_clean_server_blend_recycler" / "v300_server_search.csv"),
        ("v319", root / "v319_clean_server_value_state" / "v319_server_value_state_search.csv"),
        ("v321", root / "v321_server_robust_rankblend" / "v321_server_rankblend_search.csv"),
        ("v408", root / "v408_clean_server_microblend_recheck" / "ranked_candidates.csv"),
    ]
    sources: list[ServerSource] = []
    anchor_uids = anchor["rally_uid"].reset_index(drop=True)
    seen: set[bytes] = set()
    for family, search_path in specs:
        if not search_path.exists():
            continue
        no_banned_input_guard([search_path])
        search = pd.read_csv(search_path)
        if "path" not in search.columns:
            continue
        for idx, raw_path in enumerate(search["path"].head(4)):
            path = _resolve_source_path(raw_path, root=root, search_path=search_path)
            no_banned_input_guard([path])
            if not path.exists():
                continue
            try:
                sub = load_submission(path, expected_rows=expected_rows)
            except Exception:
                continue
            if set(sub["rally_uid"]) != set(anchor_uids):
                continue
            aligned = sub.set_index("rally_uid").loc[anchor_uids].reset_index()
            server = clip_prob(aligned["serverGetPoint"])
            fingerprint = np.round(server, 10).tobytes()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            sources.append(
                ServerSource(
                    name=f"{family}_{idx}",
                    server=server,
                    path=str(path.resolve()),
                )
            )
    return sources


def _candidate_filename(candidate: str) -> str:
    safe = candidate.replace(".", "p").replace("/", "_").replace("\\", "_")
    return f"submission_v465_{safe}__v362action_v362point.csv"


def _selected_rows(anchor: pd.DataFrame, server: np.ndarray, *, candidate: str) -> pd.DataFrame:
    anchor_server = clip_prob(anchor["serverGetPoint"])
    changed = np.abs(clip_prob(server) - anchor_server)
    out = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    out["candidate"] = candidate
    out["server_anchor"] = anchor_server
    out["server_candidate"] = clip_prob(server)
    out["server_abs_delta"] = changed
    return out.loc[changed > 1e-12].sort_values("server_abs_delta", ascending=False)


def _write_report_md(path: Path, report: dict[str, Any], board: pd.DataFrame) -> None:
    top = board.head(8)
    lines = [
        "# V465 clean server line",
        "",
        f"Generated candidates: {report['candidate_count']}",
        f"Recommended: {report['recommended_candidate']}",
        "",
        "Policy: no old-server direct labels, no TTMATCH, no upload_candidates_20260519, no rally_uid chronological inference.",
        "",
        "## Top candidates",
        "",
        _simple_markdown_table(top),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _simple_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(none)"
    columns = list(frame.columns)
    rows = [["" if pd.isna(value) else str(value) for value in row] for row in frame.to_numpy()]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _candidate_row(
    *,
    candidate: ServerCandidate,
    anchor: pd.DataFrame,
    submission_path: Path,
    selected_path: Path,
) -> dict[str, Any]:
    anchor_server = clip_prob(anchor["serverGetPoint"])
    server = clip_prob(candidate.server)
    return {
        "candidate": candidate.candidate,
        "path": str(submission_path.resolve()),
        "selected_path": str(selected_path.resolve()),
        "selected_rows": int(np.sum(np.abs(server - anchor_server) > 1e-12)),
        "action_churn": 0,
        "point_churn": 0,
        "server_changed": int(np.sum(np.abs(server - anchor_server) > 1e-12)),
        "server_mad": float(np.mean(np.abs(server - anchor_server))),
        "server_corr": _corr(server, anchor_server),
        "server_min": float(np.min(server)),
        "server_max": float(np.max(server)),
        "train_oof_auc": candidate.train_oof_auc,
        "train_oof_auc_delta_vs_anchor_proxy": candidate.train_oof_auc_delta_vs_anchor_proxy,
        "source_names": "|".join(candidate.source_names),
        "risk": candidate.risk,
        "decision": candidate.decision,
    }


def run_pipeline(
    *,
    root: Path = ROOT,
    outdir: Path | None = None,
    expected_rows: int = EXPECTED_ROWS,
) -> dict[str, Any]:
    root = Path(root)
    outdir = Path(outdir) if outdir is not None else root / OUT_DIR.name
    no_banned_input_guard([root / ANCHOR_RELATIVE, root / TRAIN_RELATIVE, root / TEST_NEW_RELATIVE, outdir])

    outdir.mkdir(parents=True, exist_ok=True)
    anchor = load_submission(root / ANCHOR_RELATIVE, expected_rows=expected_rows)
    train = pd.read_csv(root / TRAIN_RELATIVE)
    test_new = pd.read_csv(root / TEST_NEW_RELATIVE)
    if "serverGetPoint" not in train.columns:
        raise ValueError("train.csv must contain serverGetPoint")

    y = pd.to_numeric(train["serverGetPoint"], errors="coerce")
    if y.isna().any():
        raise ValueError("train serverGetPoint contains non-numeric values")
    y_arr = (y.to_numpy(dtype=float) >= 0.5).astype(int)
    train_x, test_x = build_feature_matrices(train, test_new, anchor)
    row_signals = fit_server_signals(
        train_x,
        y_arr,
        test_x,
        random_state=465,
        groups=train["rally_uid"] if "rally_uid" in train.columns else None,
    )
    signals = aggregate_signals_to_anchor(row_signals, test_new, anchor)
    clean_sources = load_existing_clean_sources(root, anchor, expected_rows=expected_rows)
    candidates = build_candidate_servers(anchor, signals, clean_sources)

    rows = []
    for candidate in candidates:
        filename = _candidate_filename(candidate.candidate)
        submission_path = _validate_output_path(outdir / filename, outdir)
        selected_path = _validate_output_path(outdir / f"selected_rows_{candidate.candidate}.csv", outdir)
        submission = package_server_only(anchor, candidate.server, expected_rows=expected_rows)
        selected = _selected_rows(anchor, candidate.server, candidate=candidate.candidate)
        submission.to_csv(submission_path, index=False)
        selected.to_csv(selected_path, index=False)
        rows.append(
            _candidate_row(
                candidate=candidate,
                anchor=anchor,
                submission_path=submission_path,
                selected_path=selected_path,
            )
        )

    board = pd.DataFrame(rows)
    if not board.empty:
        board["_risk_order"] = board["risk"].map({"safe": 0, "exploratory": 1}).fillna(9)
        board["_decision_order"] = board["decision"].map({"review": 0, "hold": 1}).fillna(9)
        board = board.sort_values(
            ["_decision_order", "_risk_order", "server_mad", "train_oof_auc"],
            ascending=[True, True, True, False],
        ).drop(columns=["_risk_order", "_decision_order"])
    search_path = _validate_output_path(outdir / "v465_server_search.csv", outdir)
    board.to_csv(search_path, index=False)
    recommended = str(board.iloc[0]["candidate"]) if not board.empty else None
    report = {
        "pipeline": "v465_clean_server_line",
        "anchor": str((root / ANCHOR_RELATIVE).resolve()),
        "candidate_count": int(len(board)),
        "recommended_candidate": recommended,
        "signals": [{"name": signal.name, "auc": signal.auc} for signal in signals],
        "clean_source_count": int(len(clean_sources)),
        "policy": {
            "no_old_server": True,
            "no_ttmatch": True,
            "no_upload_candidates_20260519": True,
            "no_rally_uid_chronological_inference": True,
            "preserve_action_point_from_v362": True,
        },
        "search_path": str(search_path.resolve()),
    }
    report_json_path = _validate_output_path(outdir / "v465_report.json", outdir)
    report_json_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")
    _write_report_md(_validate_output_path(outdir / "v465_report.md", outdir), report, board)
    print(json.dumps(_json_safe({"candidate_count": len(board), "recommended_candidate": recommended}), sort_keys=True))
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
