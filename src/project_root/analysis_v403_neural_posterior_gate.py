"""V403 neural/posterior gate for point candidate alternatives.

This script scores existing clean candidate point alternatives. It never emits
raw model argmax predictions; submissions are anchor copies with only selected
candidate point rows changed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from analysis_v335_moe_anchor_contract import (
    SUBMISSION_COLUMNS,
    point_distribution_report,
    safe_output_path,
    validate_submission_schema,
    write_json,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v403_neural_posterior_gate"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
TRAIN_PATHS = (ROOT / "train.csv", ROOT / "data" / "raw" / "train.csv")
TEST_PATHS = (ROOT / "test_new.csv", ROOT / "data" / "raw" / "test_new.csv")
DEFAULT_CANDIDATE_PATHS = (
    ROOT / "v400_public_component_recombination" / "candidate_posterior_scores.csv",
    ROOT / "v400_public_component_recombination" / "scored_candidates.csv",
    ROOT / "v401_action_point_compatibility" / "candidate_posterior_scores.csv",
    ROOT / "v401_action_point_compatibility" / "ranked_row_candidates.csv",
    ROOT / "v402_rare_point_specialist_lab" / "candidate_posterior_scores.csv",
    ROOT / "v402_rare_point_specialist_lab" / "ranked_row_candidates.csv",
    ROOT / "v362_point_hierarchical_specialists" / "scored_candidates.csv",
)
SUBMISSION_TOP9 = "submission_v403_posterior_top9__v173action_v300server.csv"
SUBMISSION_TOP15 = "submission_v403_posterior_top15__v173action_v300server.csv"


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return relative_path(value)
    return value


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return None


def action_family(action_id: Any) -> str:
    value = int(action_id)
    if value in {15, 16, 17, 18}:
        return "serve"
    if value in {1, 2, 3, 4, 5, 6}:
        return "short_control"
    if value in {7, 8, 9, 10, 11, 12}:
        return "drive_rally"
    return "special"


def phase_from_strike(strike_number: Any) -> str:
    try:
        value = int(strike_number)
    except (TypeError, ValueError):
        return "unknown"
    if value <= 1:
        return "receive"
    if value == 2:
        return "third_ball"
    if value == 3:
        return "fourth_ball"
    return "rally"


def point_depth(point: Any) -> str:
    value = int(point)
    if value == 0:
        return "terminal"
    if value <= 3:
        return "short"
    if value <= 6:
        return "half"
    return "long"


def read_submission(path: Path, expected_rows: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_submission_schema(frame, expected_rows=expected_rows)
    return frame.loc[:, SUBMISSION_COLUMNS].copy()


def read_optional_context(test_path: Path | None) -> pd.DataFrame:
    if test_path is None or not Path(test_path).exists():
        return pd.DataFrame(columns=["rally_uid", "strikeNumber", "strengthId", "spinId", "phase"])
    frame = pd.read_csv(test_path)
    if "rally_uid" not in frame.columns:
        return pd.DataFrame(columns=["rally_uid", "strikeNumber", "strengthId", "spinId", "phase"])
    keep = [col for col in ["rally_uid", "strikeNumber", "strengthId", "spinId"] if col in frame.columns]
    out = frame.loc[:, keep].drop_duplicates("rally_uid").copy()
    if "strikeNumber" in out.columns:
        out["phase"] = out["strikeNumber"].map(phase_from_strike)
    else:
        out["phase"] = "unknown"
    return out


def _canonical_candidate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    rename = {}
    if "anchor_point" in out.columns and "base_point" not in out.columns:
        rename["anchor_point"] = "base_point"
    if "new_point" in out.columns and "candidate_point" not in out.columns:
        rename["new_point"] = "candidate_point"
    if "posterior_score" in out.columns and "score" not in out.columns:
        rename["posterior_score"] = "score"
    out = out.rename(columns=rename)
    return out


def load_candidate_pool(
    anchor: pd.DataFrame,
    candidate_paths: tuple[Path, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = candidate_paths if candidate_paths is not None else DEFAULT_CANDIDATE_PATHS
    scanned: list[dict[str, Any]] = []
    rows: list[pd.DataFrame] = []
    anchor_points = anchor.set_index("rally_uid")["pointId"].astype(int)

    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            scanned.append({"path": relative_path(path), "status": "missing"})
            continue
        if any(part.lower() in {"ttmatch", "old_labels", "old-server"} for part in path.parts):
            scanned.append({"path": relative_path(path), "status": "blocked_leaky_source"})
            continue
        try:
            frame = _canonical_candidate_columns(pd.read_csv(path))
        except Exception as exc:  # pragma: no cover - defensive artifact guard
            scanned.append({"path": relative_path(path), "status": f"read_error:{exc}"})
            continue
        if not {"rally_uid", "candidate_point"}.issubset(frame.columns):
            scanned.append({"path": relative_path(path), "status": "missing_candidate_columns"})
            continue
        frame = frame.copy()
        frame["source_path"] = relative_path(path)
        frame["rally_uid"] = pd.to_numeric(frame["rally_uid"], errors="coerce")
        frame["candidate_point"] = pd.to_numeric(frame["candidate_point"], errors="coerce")
        frame = frame.dropna(subset=["rally_uid", "candidate_point"])
        frame["rally_uid"] = frame["rally_uid"].astype(int)
        frame["candidate_point"] = frame["candidate_point"].astype(int)
        frame = frame[frame["rally_uid"].isin(anchor_points.index)].copy()
        if frame.empty:
            scanned.append({"path": relative_path(path), "status": "no_anchor_overlap"})
            continue
        if "base_point" not in frame.columns:
            frame["base_point"] = frame["rally_uid"].map(anchor_points)
        frame["base_point"] = pd.to_numeric(frame["base_point"], errors="coerce")
        frame = frame.dropna(subset=["base_point"])
        frame["base_point"] = frame["base_point"].astype(int)
        for col in ["agreement_count", "source_dir_count", "score", "depth_support"]:
            if col not in frame.columns:
                frame[col] = 0.0
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
        for col in ["is_point0_addition", "depth_agree", "side_agree", "bank_agree"]:
            if col not in frame.columns:
                frame[col] = False
            frame[col] = frame[col].fillna(False).astype(bool)
        frame = frame[
            frame["candidate_point"].between(0, 9)
            & frame["base_point"].between(0, 9)
            & frame["candidate_point"].ne(frame["base_point"])
        ].copy()
        frame = frame[~((frame["base_point"] != 0) & (frame["candidate_point"] == 0))].copy()
        frame = frame[~frame["is_point0_addition"]].copy()
        scanned.append({"path": relative_path(path), "status": "loaded", "rows": int(len(frame))})
        if not frame.empty:
            rows.append(frame)

    if not rows:
        return pd.DataFrame(), {"scanned": scanned, "rows": 0}

    pool = pd.concat(rows, ignore_index=True, sort=False)
    pool = pool.sort_values(
        ["agreement_count", "source_dir_count", "score"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    pool = pool.drop_duplicates(["rally_uid", "candidate_point"], keep="first").reset_index(drop=True)
    return pool, {"scanned": scanned, "rows": int(len(pool))}


def _build_train_examples(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    required = {"actionId", "pointId"}
    if not required.issubset(train.columns):
        return pd.DataFrame(), pd.Series(dtype=int), "missing actionId/pointId exact labels"
    clean = train.dropna(subset=["actionId", "pointId"]).copy()
    clean["pointId"] = pd.to_numeric(clean["pointId"], errors="coerce")
    clean["actionId"] = pd.to_numeric(clean["actionId"], errors="coerce")
    clean = clean.dropna(subset=["actionId", "pointId"])
    clean = clean[clean["pointId"].between(0, 9) & clean["actionId"].between(0, 18)].copy()
    if len(clean) < 20:
        return pd.DataFrame(), pd.Series(dtype=int), "too few aligned train labels"

    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    for row in clean.itertuples(index=False):
        action = int(getattr(row, "actionId"))
        truth = int(getattr(row, "pointId"))
        strike = getattr(row, "strikeNumber", None)
        strength = getattr(row, "strengthId", 0)
        spin = getattr(row, "spinId", 0)
        candidates = {truth}
        candidates.update(range(0, 10, 2))
        candidates.add((truth + 1) % 10)
        for candidate in sorted(candidates):
            rows.append(
                {
                    "actionId": action,
                    "action_family": action_family(action),
                    "phase": phase_from_strike(strike),
                    "candidate_point": int(candidate),
                    "candidate_depth": point_depth(candidate),
                    "strengthId": int(strength) if pd.notna(strength) else 0,
                    "spinId": int(spin) if pd.notna(spin) else 0,
                    "agreement_count": 0.0,
                    "source_dir_count": 0.0,
                    "score": 0.0,
                }
            )
            labels.append(1 if int(candidate) == truth else 0)
    y = pd.Series(labels, dtype=int)
    if y.nunique() < 2:
        return pd.DataFrame(), pd.Series(dtype=int), "single-class train target"
    return pd.DataFrame(rows), y, "ok"


def train_posterior_model(train_path: Path | None) -> tuple[Any | None, list[str], str, dict[str, Any]]:
    if train_path is None or not Path(train_path).exists():
        return None, [], "fallback_evidence_scorer", {"reason": "missing train.csv"}
    try:
        train = pd.read_csv(train_path)
    except Exception as exc:  # pragma: no cover - defensive artifact guard
        return None, [], "fallback_evidence_scorer", {"reason": f"train read error: {exc}"}
    X, y, reason = _build_train_examples(train)
    if X.empty:
        return None, [], "fallback_evidence_scorer", {"reason": reason, "train_rows": int(len(train))}

    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover - depends on environment
        return None, [], "fallback_evidence_scorer", {"reason": f"sklearn unavailable: {exc}"}

    X_enc = pd.get_dummies(X, columns=["action_family", "phase", "candidate_depth"], dummy_na=True)
    model = LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear", random_state=403)
    model.fit(X_enc, y)
    return model, list(X_enc.columns), "logistic_regression_posterior", {
        "reason": "trained on AICUP train exact labels",
        "train_rows": int(len(train)),
        "examples": int(len(X_enc)),
        "positive_rate": float(y.mean()),
    }


def add_context_features(pool: pd.DataFrame, anchor: pd.DataFrame, test_context: pd.DataFrame) -> pd.DataFrame:
    out = pool.merge(anchor[["rally_uid", "actionId", "pointId"]], on="rally_uid", how="left", suffixes=("", "_anchor"))
    if "pointId" in out.columns:
        out["anchor_point"] = out["pointId"].astype(int)
        out = out.drop(columns=["pointId"])
    if not test_context.empty:
        out = out.merge(test_context, on="rally_uid", how="left")
    for col in ["strengthId", "spinId"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    if "phase" not in out.columns:
        out["phase"] = "unknown"
    out["phase"] = out["phase"].fillna("unknown")
    out["actionId"] = pd.to_numeric(out["actionId"], errors="coerce").fillna(0).astype(int)
    out["action_family"] = out["actionId"].map(action_family)
    out["candidate_depth"] = out["candidate_point"].map(point_depth)
    return out


def fallback_scores(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)

    def scaled(col: str) -> pd.Series:
        values = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).astype(float)
        max_value = float(values.max())
        if max_value <= 0:
            return pd.Series(np.zeros(len(values)), index=frame.index)
        return values / max_value

    score = (
        0.40 * scaled("agreement_count")
        + 0.20 * scaled("source_dir_count")
        + 0.20 * scaled("score")
        + 0.08 * frame["depth_agree"].astype(float)
        + 0.06 * frame["side_agree"].astype(float)
        + 0.06 * frame["bank_agree"].astype(float)
    )
    return score.clip(lower=0.0, upper=1.0).astype(float)


def score_candidates(
    pool: pd.DataFrame,
    anchor: pd.DataFrame,
    train_path: Path | None,
    test_path: Path | None,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    if pool.empty:
        return pool.copy(), "fallback_evidence_scorer", {"reason": "empty candidate pool"}
    test_context = read_optional_context(test_path)
    candidates = add_context_features(pool, anchor, test_context)
    model, feature_columns, model_used, model_report = train_posterior_model(train_path)
    candidates["fallback_score"] = fallback_scores(candidates)
    if model is None:
        candidates["posterior_score"] = candidates["fallback_score"]
    else:
        encoded = pd.get_dummies(
            candidates[
                [
                    "actionId",
                    "action_family",
                    "phase",
                    "candidate_point",
                    "candidate_depth",
                    "strengthId",
                    "spinId",
                    "agreement_count",
                    "source_dir_count",
                    "score",
                ]
            ],
            columns=["action_family", "phase", "candidate_depth"],
            dummy_na=True,
        )
        encoded = encoded.reindex(columns=feature_columns, fill_value=0)
        model_prob = model.predict_proba(encoded)[:, 1]
        candidates["posterior_score"] = (0.70 * model_prob) + (0.30 * candidates["fallback_score"].to_numpy())
    candidates["posterior_score"] = pd.to_numeric(candidates["posterior_score"], errors="coerce").fillna(0.0)
    candidates["posterior_rank"] = candidates["posterior_score"].rank(method="first", ascending=False).astype(int)
    candidates = candidates.sort_values(
        ["posterior_score", "agreement_count", "source_dir_count", "score"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    candidates["posterior_rank"] = np.arange(1, len(candidates) + 1)
    return candidates, model_used, model_report


def build_submission(anchor: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    out = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    point = out["pointId"].astype(int).copy()
    by_uid = pd.Series(point.to_numpy(), index=out["rally_uid"].astype(int))
    for row in selected.itertuples(index=False):
        uid = int(getattr(row, "rally_uid"))
        candidate_point = int(getattr(row, "candidate_point"))
        base_point = int(by_uid.loc[uid])
        if base_point != 0 and candidate_point == 0:
            continue
        by_uid.loc[uid] = candidate_point
    out["pointId"] = out["rally_uid"].astype(int).map(by_uid).astype(int)
    if not out["actionId"].equals(anchor["actionId"]):
        raise AssertionError("actionId changed")
    if not out["serverGetPoint"].equals(anchor["serverGetPoint"]):
        raise AssertionError("serverGetPoint changed")
    validate_submission_schema(out, expected_rows=len(anchor))
    return out


def audit_candidate(anchor: pd.DataFrame, submission: pd.DataFrame, selected: pd.DataFrame) -> dict[str, Any]:
    point_report = point_distribution_report(anchor["pointId"], submission["pointId"])
    action_churn = int((anchor["actionId"].astype(int) != submission["actionId"].astype(int)).sum())
    server_changed = int(
        (
            pd.to_numeric(anchor["serverGetPoint"], errors="coerce")
            != pd.to_numeric(submission["serverGetPoint"], errors="coerce")
        ).sum()
    )
    return {
        "selected_rows": "|".join(str(int(v)) for v in selected["rally_uid"].tolist()),
        "selected_row_count": int(len(selected)),
        "action_churn": action_churn,
        "point_churn": int(point_report["changed_rows"]),
        "point0_additions": int(point_report["point0_additions"]),
        "server_changed": server_changed,
        "risk": "safe" if len(selected) <= 9 else "normal",
        "evidence": "posterior_over_clean_candidate_sources",
        "posterior_score_sum": float(selected["posterior_score"].sum()) if not selected.empty else 0.0,
    }


def write_candidate_pack(
    *,
    outdir: Path,
    anchor: pd.DataFrame,
    scores: pd.DataFrame,
    candidate: str,
    filename: str,
    limit: int,
) -> dict[str, Any]:
    selected = scores.head(limit).copy()
    submission = build_submission(anchor, selected)
    path = safe_output_path(outdir, filename)
    submission.to_csv(path, index=False)
    selected_path = safe_output_path(outdir, f"selected_rows_{candidate}.csv")
    selected.to_csv(selected_path, index=False)
    return {
        "candidate": candidate,
        "path": relative_path(path),
        "selected_rows_path": relative_path(selected_path),
        **audit_candidate(anchor, submission, selected),
    }


def run_pipeline(
    *,
    outdir: Path = OUTDIR,
    anchor_path: Path = ANCHOR_PATH,
    train_path: Path | None = None,
    test_path: Path | None = None,
    candidate_paths: tuple[Path, ...] | None = None,
    expected_rows: int | None = 1845,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if train_path is None:
        train_path = first_existing(TRAIN_PATHS)
    if test_path is None:
        test_path = first_existing(TEST_PATHS)

    if not Path(anchor_path).exists():
        report = {
            "version": "V403",
            "decision": "NO_EXPORT",
            "model_used": "fallback_evidence_scorer",
            "reason": "missing anchor submission",
            "policy": policy_report(outdir),
        }
        write_json(safe_output_path(outdir, "search_report.json"), report)
        return report

    anchor = read_submission(Path(anchor_path), expected_rows=expected_rows)
    pool, source_report = load_candidate_pool(anchor, candidate_paths)
    scores, model_used, model_report = score_candidates(pool, anchor, train_path, test_path)
    score_path = safe_output_path(outdir, "candidate_posterior_scores.csv")
    scores.to_csv(score_path, index=False)

    ranked_rows: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    if not scores.empty:
        for candidate, filename, limit in [
            ("v403_posterior_top9", SUBMISSION_TOP9, 9),
            ("v403_posterior_top15", SUBMISSION_TOP15, 15),
        ]:
            row = write_candidate_pack(
                outdir=outdir,
                anchor=anchor,
                scores=scores,
                candidate=candidate,
                filename=filename,
                limit=min(limit, len(scores)),
            )
            ranked_rows.append(row)
            generated.append(
                {
                    "candidate": candidate,
                    "path": row["path"],
                    "selected_row_count": row["selected_row_count"],
                    "point_churn": row["point_churn"],
                    "point0_additions": row["point0_additions"],
                }
            )
    ranked = pd.DataFrame(ranked_rows)
    ranked_path = safe_output_path(outdir, "ranked_candidates.csv")
    ranked.to_csv(ranked_path, index=False)

    report = {
        "version": "V403",
        "decision": "HAS_EXPORT" if generated else "NO_EXPORT",
        "anchor": relative_path(Path(anchor_path)),
        "anchor_rows": int(len(anchor)),
        "model_used": model_used,
        "model_report": model_report,
        "candidate_source_report": source_report,
        "candidate_posterior_scores": relative_path(score_path),
        "ranked_candidates": relative_path(ranked_path),
        "generated_submission_count": int(len(generated)),
        "generated_submissions": generated,
        "policy": policy_report(outdir),
    }
    write_json(safe_output_path(outdir, "search_report.json"), report)
    return report


def policy_report(outdir: Path) -> dict[str, Any]:
    return {
        "no_ttmatch_inputs": True,
        "no_old_labels": True,
        "no_raw_argmax_submission": True,
        "no_point0_additions": True,
        "preserve_anchor_action": True,
        "preserve_anchor_server": True,
        "output_dir": relative_path(Path(outdir)),
    }


def main() -> None:
    report = run_pipeline()
    print(
        json.dumps(
            json_safe(
                {
                    "decision": report["decision"],
                    "model_used": report["model_used"],
                    "generated_submission_count": report["generated_submission_count"],
                    "generated_submissions": report["generated_submissions"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
