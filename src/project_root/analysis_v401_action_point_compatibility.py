"""V401 train-derived action/point compatibility scorer.

This experiment ranks public-positive point alternatives by compatibility with
the V173 action anchor and observed incoming context. It packages point-only
submissions against the V362 public-proven anchor.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
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
OUTDIR = ROOT / "v401_action_point_compatibility"
ANCHOR_PATH = (
    ROOT
    / "v362_point_hierarchical_specialists"
    / "submission_v362_depth_agree_only__v173action_v300server.csv"
)
TRAIN_PATHS = (ROOT / "train.csv", ROOT / "data" / "raw" / "train.csv")
TEST_PATHS = (ROOT / "test_new.csv", ROOT / "data" / "raw" / "test_new.csv")
SOURCE_DIRS = (
    "v400_public_component_recombination",
    "v338_joint_moe_pack",
    "v341_no_p0_point_pack",
    "v306_point0_addition_probe",
    "v300_clean_server_blend_recycler",
    "v362_point_hierarchical_specialists",
)
POINT_VALUES = tuple(range(10))
DEPTH_VALUES = ("terminal", "short", "half", "long")


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


def point_to_depth(point: int) -> str:
    value = int(point)
    if value == 0:
        return "terminal"
    if 1 <= value <= 3:
        return "short"
    if 4 <= value <= 6:
        return "half"
    if 7 <= value <= 9:
        return "long"
    raise ValueError(f"pointId outside 0..9: {point}")


def action_family(action: int) -> str:
    value = int(action)
    if value in {15, 16, 17, 18}:
        return "serve"
    if value in {0, 1, 2, 3, 4}:
        return "control"
    if value in {5, 6, 7, 8, 9}:
        return "drive"
    if value in {10, 11, 12, 13, 14}:
        return "attack"
    return "unknown"


def phase_from_strike(strike_number: int) -> str:
    value = int(strike_number)
    if value <= 1:
        return "receive"
    if value == 2:
        return "third_ball"
    if value == 3:
        return "fourth_ball"
    return "rally"


def prefix_len_bin(prefix_len: int) -> str:
    value = int(prefix_len)
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value == 3:
        return "3"
    if value <= 6:
        return "4_6"
    return "7p"


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_submission(path: Path, expected_rows: int | None = 1845) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_submission_schema(frame, expected_rows=expected_rows)
    return frame.loc[:, SUBMISSION_COLUMNS].copy()


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"rally_uid", "strikeNumber", "actionId", "pointId"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing required train/test columns: {sorted(required - set(frame.columns))}")
    out = frame.sort_values(["rally_uid", "strikeNumber"]).copy()
    same_prev = out["rally_uid"].eq(out["rally_uid"].shift(1))
    prev_action = out["actionId"].shift(1).where(same_prev, out["actionId"])
    prev_point = out["pointId"].shift(1).where(same_prev, out["pointId"])
    out["phase"] = out["strikeNumber"].astype(int).map(phase_from_strike)
    out["lag0_action"] = prev_action.astype(int)
    out["lag0_action_family"] = out["lag0_action"].map(action_family)
    out["lag0_point"] = prev_point.astype(int)
    out["lag0_depth"] = out["lag0_point"].map(point_to_depth)
    out["lag0_spin"] = out["spinId"] if "spinId" in out.columns else -1
    out["lag0_strength"] = out["strengthId"] if "strengthId" in out.columns else -1
    out["prefix_len_bin"] = out["strikeNumber"].astype(int).map(prefix_len_bin)
    out["action_family"] = out["actionId"].astype(int).map(action_family)
    out["point_depth"] = out["pointId"].astype(int).map(point_to_depth)
    return out


@dataclass
class SmoothedCompatibilityScorer:
    alpha: float
    action_phase_point: dict[tuple[int, str], Counter]
    family_phase_depth_point: dict[tuple[str, str, str], Counter]
    family_phase_lagfamily_depth: dict[tuple[str, str, str], Counter]
    action_phase_p0: dict[tuple[int, str], Counter]
    global_point: Counter
    global_depth: Counter
    phases: set[str]
    families: set[str]

    def _smoothed_prob(self, counts: Counter, value: Any, values: tuple[Any, ...]) -> float:
        total = sum(counts.values())
        return (float(counts.get(value, 0)) + self.alpha) / (float(total) + self.alpha * len(values))

    def point_prob_action_phase(self, action: int, phase: str, point: int) -> float:
        counts = self.action_phase_point.get((int(action), str(phase)), self.global_point)
        return self._smoothed_prob(counts, int(point), POINT_VALUES)

    def point_prob_family_depth(self, family: str, phase: str, lag0_depth: str, point: int) -> float:
        key = (str(family), str(phase), str(lag0_depth))
        counts = self.family_phase_depth_point.get(key, self.global_point)
        return self._smoothed_prob(counts, int(point), POINT_VALUES)

    def depth_prob_family_lagfamily(self, family: str, phase: str, lag0_family: str, depth: str) -> float:
        key = (str(family), str(phase), str(lag0_family))
        counts = self.family_phase_lagfamily_depth.get(key, self.global_depth)
        return self._smoothed_prob(counts, str(depth), DEPTH_VALUES)

    def point0_prob_action_phase(self, action: int, phase: str, is_point0: bool) -> float:
        counts = self.action_phase_p0.get((int(action), str(phase)), Counter({0: 1, 1: 1}))
        return self._smoothed_prob(counts, int(bool(is_point0)), (0, 1))

    def log_point_probability(self, action: int, point: int, context: dict[str, Any]) -> float:
        phase = str(context.get("phase", "rally"))
        family = action_family(int(action))
        lag0_depth = str(context.get("lag0_depth", "terminal"))
        lag0_family = str(context.get("lag0_action_family", "unknown"))
        point = int(point)
        depth = point_to_depth(point)
        p_action = self.point_prob_action_phase(action, phase, point)
        p_family = self.point_prob_family_depth(family, phase, lag0_depth, point)
        p_depth = self.depth_prob_family_lagfamily(family, phase, lag0_family, depth)
        p_p0 = self.point0_prob_action_phase(action, phase, point == 0)
        prob = (0.45 * p_action) + (0.30 * p_family) + (0.15 * p_depth) + (0.10 * p_p0)
        return math.log(max(prob, 1e-12))


def build_compatibility_scorer(train: pd.DataFrame, alpha: float = 0.5) -> SmoothedCompatibilityScorer:
    features = _feature_frame(train)
    action_phase_point: dict[tuple[int, str], Counter] = defaultdict(Counter)
    family_phase_depth_point: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    family_phase_lagfamily_depth: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    action_phase_p0: dict[tuple[int, str], Counter] = defaultdict(Counter)
    global_point: Counter = Counter()
    global_depth: Counter = Counter()

    for row in features.itertuples(index=False):
        action = int(row.actionId)
        point = int(row.pointId)
        phase = str(row.phase)
        family = str(row.action_family)
        lag0_depth = str(row.lag0_depth)
        lag0_family = str(row.lag0_action_family)
        depth = str(row.point_depth)
        action_phase_point[(action, phase)][point] += 1
        family_phase_depth_point[(family, phase, lag0_depth)][point] += 1
        family_phase_lagfamily_depth[(family, phase, lag0_family)][depth] += 1
        action_phase_p0[(action, phase)][int(point == 0)] += 1
        global_point[point] += 1
        global_depth[depth] += 1

    if not global_point:
        global_point.update({value: 1 for value in POINT_VALUES})
    if not global_depth:
        global_depth.update({value: 1 for value in DEPTH_VALUES})

    return SmoothedCompatibilityScorer(
        alpha=float(alpha),
        action_phase_point=dict(action_phase_point),
        family_phase_depth_point=dict(family_phase_depth_point),
        family_phase_lagfamily_depth=dict(family_phase_lagfamily_depth),
        action_phase_p0=dict(action_phase_p0),
        global_point=global_point,
        global_depth=global_depth,
        phases=set(features["phase"].astype(str).unique()),
        families=set(features["action_family"].astype(str).unique()),
    )


def build_test_contexts(test: pd.DataFrame, anchor: pd.DataFrame) -> dict[int, dict[str, Any]]:
    if test.empty:
        return {
            int(row.rally_uid): {
                "phase": "rally",
                "lag0_depth": point_to_depth(int(row.pointId)),
                "lag0_action_family": action_family(int(row.actionId)),
                "prefix_len_bin": "1",
            }
            for row in anchor.itertuples(index=False)
        }
    features = _feature_frame(test)
    last = features.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False).tail(1)
    contexts: dict[int, dict[str, Any]] = {}
    for row in last.itertuples(index=False):
        next_strike = int(row.strikeNumber) + 1
        contexts[int(row.rally_uid)] = {
            "phase": phase_from_strike(next_strike),
            "lag0_action": int(row.actionId),
            "lag0_action_family": action_family(int(row.actionId)),
            "lag0_point": int(row.pointId),
            "lag0_depth": point_to_depth(int(row.pointId)),
            "lag0_spin": int(getattr(row, "spinId", -1)),
            "lag0_strength": int(getattr(row, "strengthId", -1)),
            "prefix_len_bin": prefix_len_bin(next_strike),
        }
    for row in anchor.itertuples(index=False):
        contexts.setdefault(
            int(row.rally_uid),
            {
                "phase": "rally",
                "lag0_depth": point_to_depth(int(row.pointId)),
                "lag0_action_family": action_family(int(row.actionId)),
                "prefix_len_bin": "1",
            },
        )
    return contexts


def discover_source_submissions(root: Path = ROOT) -> list[Path]:
    paths: list[Path] = []
    for dirname in SOURCE_DIRS:
        directory = root / dirname
        if not directory.exists() or not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.csv")):
            if path.name.startswith(("selected_", "ranked_", "scored_", "candidate_", "joint_", "v300_", "v306_")):
                continue
            if "submission" not in path.name:
                continue
            if path.resolve() == ANCHOR_PATH.resolve():
                continue
            paths.append(path)
    return paths


def load_source_submissions(anchor: pd.DataFrame, paths: Iterable[Path]) -> tuple[list[tuple[Path, pd.DataFrame]], list[str]]:
    loaded: list[tuple[Path, pd.DataFrame]] = []
    skipped: list[str] = []
    anchor_ids = set(anchor["rally_uid"].astype(int).tolist())
    for path in paths:
        try:
            frame = load_submission(path, expected_rows=len(anchor))
        except Exception as exc:  # noqa: BLE001 - guarded discovery must skip malformed files.
            skipped.append(f"{relative_path(path)}: {exc}")
            continue
        if set(frame["rally_uid"].astype(int).tolist()) != anchor_ids:
            skipped.append(f"{relative_path(path)}: rally_uid mismatch")
            continue
        loaded.append((path, frame))
    return loaded, skipped


def build_candidate_pool(anchor: pd.DataFrame, sources: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    anchor_by_uid = anchor.reset_index(drop=False).rename(columns={"index": "row_id"}).set_index("rally_uid")
    alternatives: dict[tuple[int, int], dict[str, Any]] = {}
    for path, source in sources:
        source_by_uid = source.set_index("rally_uid")
        common = anchor_by_uid.index.intersection(source_by_uid.index)
        for rally_uid in common:
            base = anchor_by_uid.loc[rally_uid]
            src = source_by_uid.loc[rally_uid]
            if int(src["actionId"]) != int(base["actionId"]):
                continue
            anchor_point = int(base["pointId"])
            candidate_point = int(src["pointId"])
            if candidate_point == anchor_point:
                continue
            if candidate_point == 0 and anchor_point != 0:
                continue
            key = (int(rally_uid), candidate_point)
            record = alternatives.setdefault(
                key,
                {
                    "rally_uid": int(rally_uid),
                    "row_id": int(base["row_id"]),
                    "actionId": int(base["actionId"]),
                    "anchor_point": anchor_point,
                    "candidate_point": candidate_point,
                    "source_agreement": 0,
                    "source_count": 0,
                    "sources_list": [],
                },
            )
            record["source_agreement"] += 1
            record["source_count"] += 1
            record["sources_list"].append(relative_path(path))

    rows = []
    for record in alternatives.values():
        row = dict(record)
        row["sources"] = "|".join(row.pop("sources_list"))
        rows.append(row)
    columns = [
        "rally_uid",
        "row_id",
        "actionId",
        "anchor_point",
        "candidate_point",
        "source_agreement",
        "source_count",
        "sources",
    ]
    return pd.DataFrame(rows, columns=columns)


def score_candidate_pool(
    pool: pd.DataFrame,
    scorer: SmoothedCompatibilityScorer,
    contexts: dict[int, dict[str, Any]],
    threshold: float = 0.0,
) -> pd.DataFrame:
    if pool.empty:
        return pool.assign(
            old_score=pd.Series(dtype=float),
            new_score=pd.Series(dtype=float),
            compat_delta=pd.Series(dtype=float),
            phase=pd.Series(dtype=str),
            lag0_depth=pd.Series(dtype=str),
            lag0_action_family=pd.Series(dtype=str),
        )
    rows = []
    for row in pool.itertuples(index=False):
        context = contexts.get(int(row.rally_uid), {})
        old_score = scorer.log_point_probability(int(row.actionId), int(row.anchor_point), context)
        new_score = scorer.log_point_probability(int(row.actionId), int(row.candidate_point), context)
        out = row._asdict()
        out["old_score"] = old_score
        out["new_score"] = new_score
        out["compat_delta"] = new_score - old_score
        out["phase"] = str(context.get("phase", "rally"))
        out["lag0_depth"] = str(context.get("lag0_depth", "terminal"))
        out["lag0_action_family"] = str(context.get("lag0_action_family", "unknown"))
        rows.append(out)
    scored = pd.DataFrame(rows)
    scored = scored[
        (scored["compat_delta"] > float(threshold))
        & (scored["source_agreement"] >= 1)
        & ~((scored["candidate_point"].astype(int) == 0) & (scored["anchor_point"].astype(int) != 0))
    ].copy()
    if scored.empty:
        return scored
    scored = scored.sort_values(
        ["compat_delta", "source_agreement", "source_count", "rally_uid", "candidate_point"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)
    return scored


def _dedupe_by_row(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored
    return scored.sort_values(
        ["compat_delta", "source_agreement", "source_count", "rally_uid"],
        ascending=[False, False, False, True],
    ).drop_duplicates("row_id", keep="first").reset_index(drop=True)


def package_candidate(
    anchor: pd.DataFrame,
    selected: pd.DataFrame,
    name: str,
    outdir: Path,
) -> dict[str, Any]:
    out = anchor.loc[:, SUBMISSION_COLUMNS].copy()
    if not selected.empty:
        for row in selected.itertuples(index=False):
            out.at[int(row.row_id), "pointId"] = int(row.candidate_point)
    validate_submission_schema(out, expected_rows=len(anchor))
    if not out["actionId"].equals(anchor["actionId"]):
        raise AssertionError("actionId changed while packaging V401 candidate")
    if not out["serverGetPoint"].equals(anchor["serverGetPoint"]):
        raise AssertionError("serverGetPoint changed while packaging V401 candidate")

    submission_path = safe_output_path(outdir, f"submission_{name}__v173action_v300server.csv")
    selected_path = safe_output_path(outdir, f"selected_rows_{name}.csv")
    out.to_csv(submission_path, index=False)
    selected.to_csv(selected_path, index=False)

    point_report = point_distribution_report(anchor["pointId"], out["pointId"])
    return {
        "candidate": name,
        "path": relative_path(submission_path),
        "selected_rows": relative_path(selected_path),
        "selected_row_count": int(len(selected)),
        "action_churn": 0,
        "point_churn": int(point_report["changed_rows"]),
        "point0_additions": int(point_report["point0_additions"]),
        "server_changed": 0,
        "risk": "medium" if len(selected) > 15 else "low",
        "evidence": "train_smoothed_action_point_compatibility",
    }


def _empty_report(reason: str, outdir: Path) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "v401",
        "status": "empty",
        "reason": reason,
        "generated_candidates": [],
        "selected_counts": {},
    }
    write_json(safe_output_path(outdir, "search_report.json"), report)
    return report


def run_pipeline(*, outdir: Path = OUTDIR) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_path = first_existing(TRAIN_PATHS)
    test_path = first_existing(TEST_PATHS)
    if not ANCHOR_PATH.exists():
        return _empty_report("missing_anchor", outdir)
    if train_path is None:
        return _empty_report("missing_train", outdir)

    anchor = load_submission(ANCHOR_PATH, expected_rows=1845)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if test_path is not None else pd.DataFrame()
    scorer = build_compatibility_scorer(train, alpha=0.5)
    contexts = build_test_contexts(test, anchor)
    source_paths = discover_source_submissions(ROOT)
    sources, skipped_sources = load_source_submissions(anchor, source_paths)
    pool = build_candidate_pool(anchor, sources)
    scored = score_candidate_pool(pool, scorer, contexts, threshold=0.0)
    unique_scored = _dedupe_by_row(scored)

    safe_output_path(outdir, "candidate_row_scores.csv")
    unique_scored.to_csv(safe_output_path(outdir, "candidate_row_scores.csv"), index=False)

    packages = [
        ("v401_compat_top9", unique_scored.head(9)),
        ("v401_compat_top15", unique_scored.head(15)),
        (
            "v401_compat_nonterminal_top24",
            unique_scored[
                (unique_scored["anchor_point"].astype(int) != 0)
                & (unique_scored["candidate_point"].astype(int) != 0)
            ].head(24),
        ),
    ]
    summaries = [package_candidate(anchor, selected.copy(), name, outdir) for name, selected in packages]
    ranked = pd.DataFrame(summaries)
    ranked.to_csv(safe_output_path(outdir, "ranked_candidates.csv"), index=False)

    report = {
        "version": "v401",
        "status": "ok",
        "anchor": relative_path(ANCHOR_PATH),
        "train_path": relative_path(train_path),
        "test_path": relative_path(test_path) if test_path is not None else None,
        "source_paths": [relative_path(path) for path in source_paths],
        "loaded_source_count": len(sources),
        "skipped_sources": skipped_sources,
        "candidate_row_count": int(len(pool)),
        "passing_candidate_row_count": int(len(unique_scored)),
        "generated_candidates": [Path(item["path"]).name for item in summaries],
        "selected_counts": {item["candidate"]: int(item["selected_row_count"]) for item in summaries},
        "ranked_candidates": summaries,
        "tables": {
            "action_phase_point_keys": len(scorer.action_phase_point),
            "family_phase_depth_point_keys": len(scorer.family_phase_depth_point),
            "family_phase_lagfamily_depth_keys": len(scorer.family_phase_lagfamily_depth),
            "action_phase_p0_keys": len(scorer.action_phase_p0),
        },
    }
    write_json(safe_output_path(outdir, "search_report.json"), json_safe(report))
    return report


def main() -> None:
    report = run_pipeline()
    print(json.dumps(json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
