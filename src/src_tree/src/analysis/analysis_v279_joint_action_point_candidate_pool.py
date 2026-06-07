from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
ANCHOR_PATH = Path("v261_action_conditioned_point_residual/submission_v261_cap0p01__v173action_r121server.csv")
OUTDIR = Path("v279_joint_action_point_candidate_pool")
SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    role: str


def action_family(action_id: int) -> str:
    if action_id == 0:
        return "Zero"
    if 1 <= action_id <= 7:
        return "Attack"
    if 8 <= action_id <= 11:
        return "Control"
    if 12 <= action_id <= 14:
        return "Defensive"
    if 15 <= action_id <= 18:
        return "Serve"
    raise ValueError(action_id)


def point_depth(point_id: int) -> int:
    if point_id == 0:
        return 0
    if 1 <= point_id <= 9:
        return 1 + (point_id - 1) // 3
    raise ValueError(point_id)


def is_pair_compatible(action_id: int, point_id: int) -> bool:
    family = action_family(int(action_id))
    point_id = int(point_id)
    depth = point_depth(point_id)

    if family == "Serve":
        return False
    if action_id == 0:
        return point_id == 0
    if point_id == 0:
        return action_id in {1, 2, 3, 10, 11, 12, 13}
    if family == "Zero":
        return False

    if family == "Attack":
        return depth in {2, 3} or action_id in {4, 7}
    if family == "Control":
        return depth in {1, 2, 3}
    if family == "Defensive":
        return point_id in {5, 7, 8, 9}
    return False


def compatibility_score(action_id: int, point_id: int) -> float:
    if not is_pair_compatible(action_id, point_id):
        return 0.0

    action_id = int(action_id)
    point_id = int(point_id)
    family = action_family(action_id)
    depth = point_depth(point_id)

    if action_id == 0 and point_id == 0:
        return 1.0
    if point_id == 0:
        if action_id in {1, 2, 3, 13}:
            return 0.9
        return 0.55
    if family == "Attack":
        if point_id in {7, 8, 9}:
            return 0.95
        if point_id in {4, 5, 6}:
            return 0.78
        return 0.6
    if family == "Control":
        if depth in {1, 2}:
            return 0.9
        return 0.7
    if family == "Defensive":
        if point_id in {7, 8, 9}:
            return 0.9
        return 0.75
    return 0.0


def expected_pair_columns() -> list[str]:
    return [
        "rally_uid",
        "row_index",
        "anchor_action",
        "anchor_point",
        "candidate_action",
        "candidate_point",
        "action_source",
        "point_source",
        "pair_key",
        "compatibility_score",
        "action_changed",
        "point_changed",
        "pair_changed",
        "action_agreement_count",
        "point_agreement_count",
        "pair_agreement_count",
        "action_source_count",
        "point_source_count",
        "is_anchor_pair",
        "is_terminal_pair",
        "is_nonterminal_pair",
    ]


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _anchor_rally_uids() -> pd.Series:
    anchor = pd.read_csv(_absolute(ANCHOR_PATH), usecols=["rally_uid"])
    return anchor["rally_uid"]


def load_submission(path: Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(_absolute(path))
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"{path} columns must be exactly {SUBMISSION_COLUMNS}, got {list(df.columns)}")
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"{path} row count must be {EXPECTED_ROWS}, got {len(df)}")
    if not df["rally_uid"].equals(_anchor_rally_uids()):
        raise ValueError(f"{path} rally_uid does not match anchor")
    return df.copy()


def _glob_specs(pattern: str, role: str) -> list[SourceSpec]:
    matches = sorted(Path().glob(pattern))
    if matches:
        return [SourceSpec(path.stem, path, role) for path in matches]
    return [SourceSpec(pattern, Path(pattern), role)]


def discover_sources() -> tuple[list[SourceSpec], list[SourceSpec]]:
    action_sources = [
        SourceSpec("anchor_v261_v173_action", ANCHOR_PATH, "action"),
        SourceSpec(
            "v191_v166_action",
            Path("upload_candidates_20260519/submission_v191_v166_best_action__pv188_r186_w005_cap5__sr121.csv"),
            "action",
        ),
    ]
    for pattern in [
        "upload_candidates_20260519/submission_v220*.csv",
        "upload_candidates_20260519/submission_v216*.csv",
        "upload_candidates_20260519/submission_v217*.csv",
        "upload_candidates_20260519/submission_v230*.csv",
        "upload_candidates_20260519/submission_v231*.csv",
        "upload_candidates_20260519/submission_v232*.csv",
    ]:
        action_sources.extend(_glob_specs(pattern, "action"))

    point_sources = [
        SourceSpec("anchor_v261_cap1_point", ANCHOR_PATH, "point"),
        SourceSpec(
            "v188_cap5_point",
            Path("upload_candidates_20260519/submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv"),
            "point",
        ),
        SourceSpec(
            "v272_cap0p005_point",
            Path("v272_action_conditioned_point_residual/submission_v272_point_actioncond_cap0p005__v173action_r121server.csv"),
            "point",
        ),
        SourceSpec(
            "v272_cap0p010_point",
            Path("v272_action_conditioned_point_residual/submission_v272_point_actioncond_cap0p010__v173action_r121server.csv"),
            "point",
        ),
        SourceSpec(
            "v272_cap0p015_point",
            Path("v272_action_conditioned_point_residual/submission_v272_point_actioncond_cap0p015__v173action_r121server.csv"),
            "point",
        ),
        SourceSpec(
            "v277_cap0p010_nonterminal_point",
            Path("v277_v272b_point_refinement/submission_v277_v272_cap0p010_nonterminal_only__v173action_r121server.csv"),
            "point",
        ),
        SourceSpec(
            "v277_cap0p010_no_point0_add_point",
            Path("v277_v272b_point_refinement/submission_v277_v272_cap0p010_no_point0_add__v173action_r121server.csv"),
            "point",
        ),
        SourceSpec(
            "v277_cap0p015_nonterminal_point",
            Path("v277_v272b_point_refinement/submission_v277_v272_cap0p015_nonterminal_only__v173action_r121server.csv"),
            "point",
        ),
    ]
    return action_sources, point_sources


def _load_sources(specs: list[SourceSpec]) -> tuple[list[tuple[SourceSpec, pd.DataFrame]], list[dict[str, object]]]:
    loaded: list[tuple[SourceSpec, pd.DataFrame]] = []
    summary: list[dict[str, object]] = []
    for spec in specs:
        exists = _absolute(spec.path).exists()
        row = {"role": spec.role, "source": spec.name, "path": str(spec.path), "status": "missing", "rows": 0, "message": ""}
        if exists:
            try:
                df = load_submission(spec.path)
            except Exception as exc:
                row["status"] = "skipped_invalid"
                row["message"] = str(exc)
            else:
                loaded.append((spec, df))
                row["status"] = "loaded"
                row["rows"] = len(df)
        summary.append(row)
    return loaded, summary


def build_candidate_pool() -> tuple[pd.DataFrame, pd.DataFrame]:
    action_specs, point_specs = discover_sources()
    loaded_actions, action_summary = _load_sources(action_specs)
    loaded_points, point_summary = _load_sources(point_specs)
    if not loaded_actions or not loaded_points:
        raise RuntimeError("At least one action source and one point source are required")

    anchor = load_submission(ANCHOR_PATH)
    all_loaded_by_name: dict[str, pd.DataFrame] = {}
    for spec, df in loaded_actions + loaded_points:
        all_loaded_by_name.setdefault(spec.name, df)

    rows: list[dict[str, object]] = []
    for idx, anchor_row in anchor.iterrows():
        rally_uid = int(anchor_row["rally_uid"])
        anchor_action = int(anchor_row["actionId"])
        anchor_point = int(anchor_row["pointId"])

        action_votes = [(spec.name, int(df.at[idx, "actionId"])) for spec, df in loaded_actions]
        point_votes = [(spec.name, int(df.at[idx, "pointId"])) for spec, df in loaded_points]
        pair_votes = [
            (int(df.at[idx, "actionId"]), int(df.at[idx, "pointId"]))
            for df in all_loaded_by_name.values()
        ]
        action_counts = Counter(value for _, value in action_votes)
        point_counts = Counter(value for _, value in point_votes)
        pair_counts = Counter(pair_votes)

        action_sources = {
            action: "|".join(name for name, value in action_votes if value == action)
            for action in sorted(action_counts)
        }
        point_sources = {
            point: "|".join(name for name, value in point_votes if value == point)
            for point in sorted(point_counts)
        }

        candidate_actions = sorted(set(action_counts) | {anchor_action})
        candidate_points = sorted(set(point_counts) | {anchor_point})
        for action in candidate_actions:
            for point in candidate_points:
                is_anchor_pair = action == anchor_action and point == anchor_point
                if not is_anchor_pair and not is_pair_compatible(action, point):
                    continue
                rows.append(
                    {
                        "rally_uid": rally_uid,
                        "row_index": idx,
                        "anchor_action": anchor_action,
                        "anchor_point": anchor_point,
                        "candidate_action": action,
                        "candidate_point": point,
                        "action_source": action_sources.get(action, ""),
                        "point_source": point_sources.get(point, ""),
                        "pair_key": f"{action}_{point}",
                        "compatibility_score": compatibility_score(action, point),
                        "action_changed": action != anchor_action,
                        "point_changed": point != anchor_point,
                        "pair_changed": (action, point) != (anchor_action, anchor_point),
                        "action_agreement_count": action_counts.get(action, 0),
                        "point_agreement_count": point_counts.get(point, 0),
                        "pair_agreement_count": pair_counts.get((action, point), 0),
                        "action_source_count": len(loaded_actions),
                        "point_source_count": len(loaded_points),
                        "is_anchor_pair": is_anchor_pair,
                        "is_terminal_pair": point == 0,
                        "is_nonterminal_pair": point != 0,
                    }
                )

    candidates = pd.DataFrame(rows, columns=expected_pair_columns())
    summary = pd.DataFrame(action_summary + point_summary)
    return candidates, summary


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    candidates, summary = build_candidate_pool()
    candidates.to_csv(OUTDIR / "v279_pair_candidates.csv", index=False)
    summary.to_csv(OUTDIR / "v279_source_summary.csv", index=False)
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR),
                "rows": int(len(candidates)),
                "unique_rallies": int(candidates["rally_uid"].nunique()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
