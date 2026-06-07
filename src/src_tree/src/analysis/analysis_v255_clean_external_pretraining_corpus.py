"""V255 clean external canonical pretraining corpus builder.

Builds a coarse external corpus from V254-approved GREEN/YELLOW datasets.
It does not map external labels to AICUP exact actionId, train a model, or
generate submissions.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis_v255_external_pretraining_helpers import canonical_phase_from_event, parse_vector_string


ROOT = Path(__file__).resolve().parent
EXTERNAL = ROOT / "external_data"
V254_POLICY = ROOT / "v254_external_acquisition_audit" / "v254_training_use_policy.csv"
OUTDIR = ROOT / "v255_clean_external_pretraining_corpus"
SRC_DEST = ROOT / "src" / "analysis" / "analysis_v255_clean_external_pretraining_corpus.py"


CANONICAL_COLUMNS = [
    "source_dataset",
    "source_path",
    "sequence_id",
    "event_index",
    "event_type",
    "coarse_family",
    "phase",
    "terminal_like",
    "landing_x",
    "landing_y",
    "landing_z",
    "speed",
    "spin",
    "player_context",
    "raw_label",
    "risk_tier",
    "allowed_target",
]


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def row(
    source_dataset: str,
    source_path: Path | str,
    sequence_id: str,
    event_index: int,
    event_type: str,
    coarse_family: str,
    phase: str,
    terminal_like: bool,
    risk_tier: str,
    allowed_target: str,
    landing_x: float | None = np.nan,
    landing_y: float | None = np.nan,
    landing_z: float | None = np.nan,
    speed: float | None = np.nan,
    spin: float | None = np.nan,
    player_context: str = "",
    raw_label: str = "",
) -> dict[str, Any]:
    return {
        "source_dataset": source_dataset,
        "source_path": str(source_path).replace("\\", "/"),
        "sequence_id": sequence_id,
        "event_index": int(event_index),
        "event_type": event_type,
        "coarse_family": coarse_family,
        "phase": phase,
        "terminal_like": bool(terminal_like),
        "landing_x": landing_x,
        "landing_y": landing_y,
        "landing_z": landing_z,
        "speed": speed,
        "spin": spin,
        "player_context": player_context,
        "raw_label": raw_label,
        "risk_tier": risk_tier,
        "allowed_target": allowed_target,
    }


def read_policy() -> pd.DataFrame:
    if not V254_POLICY.exists():
        raise FileNotFoundError(f"Missing V254 policy: {V254_POLICY}")
    policy = pd.read_csv(V254_POLICY)
    return policy[policy["tier"].isin(["GREEN", "YELLOW"])].copy()


def policy_map(policy: pd.DataFrame) -> dict[str, tuple[str, str]]:
    return {r["dataset"]: (r["tier"], r["allowed_use"]) for _, r in policy.iterrows()}


def load_openttgames_events(policy: dict[str, tuple[str, str]]) -> pd.DataFrame:
    dataset = "openttgames"
    if dataset not in policy:
        return empty_frame()
    tier, allowed = policy[dataset]
    path = EXTERNAL / dataset / "processed" / "openttgames_events.csv"
    if not path.exists():
        return empty_frame()
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        event = str(r.get("event_type", r.get("event_raw_label", ""))).lower()
        is_stroke = bool(r.get("is_stroke", False))
        is_bounce = bool(r.get("is_bounce", False))
        is_net = bool(r.get("is_net", False))
        terminal = bool(r.get("is_rally_ending", False)) or is_net or "net" in event
        if is_stroke:
            family = str(r.get("safe_action_family", "Attack") or "Attack")
            if family not in {"Zero", "Attack", "Control", "Defensive", "Serve"}:
                family = "Attack"
        elif is_bounce:
            family = "Control"
        elif terminal:
            family = "Zero"
        else:
            family = "Zero"
        seq = str(r.get("video_id", "opentt")) + ":" + str(r.get("split", ""))
        rows.append(
            row(
                dataset,
                path.relative_to(ROOT),
                seq,
                int(i),
                event,
                family,
                canonical_phase_from_event(event, int(i % 8) + 1),
                terminal,
                tier,
                allowed,
                raw_label=str(r.get("event_raw_label", "")),
                player_context=str(r.get("player_side", "")),
            )
        )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def load_sony_events(policy: dict[str, tuple[str, str]]) -> pd.DataFrame:
    dataset = "sonytabletennis"
    if dataset not in policy:
        return empty_frame()
    tier, allowed = policy[dataset]
    path = EXTERNAL / dataset / "data" / "match_data.csv"
    if not path.exists():
        return empty_frame()
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        event = str(r.get("type", "")).lower()
        pos = parse_vector_string(r.get("ball_pos", ""))
        vel = parse_vector_string(r.get("ball_vel_out", ""))
        spin_vec = parse_vector_string(r.get("ball_spin_out", ""))
        speed = float(np.linalg.norm(vel)) if vel else np.nan
        spin = float(np.linalg.norm(spin_vec)) if spin_vec else np.nan
        if "shot" in event:
            family = "Attack"
        elif "net" in event:
            family = "Zero"
        else:
            family = "Control"
        terminal = "net" in event
        rows.append(
            row(
                dataset,
                path.relative_to(ROOT),
                str(r.get("rally_id", "")),
                i,
                event,
                family,
                canonical_phase_from_event(event, i + 1),
                terminal,
                tier,
                allowed,
                landing_x=pos[0] if len(pos) > 0 else np.nan,
                landing_y=pos[1] if len(pos) > 1 else np.nan,
                landing_z=pos[2] if len(pos) > 2 else np.nan,
                speed=speed,
                spin=spin,
                player_context=str(r.get("player_id", "")),
                raw_label=event,
            )
        )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def load_deepmind_states(policy: dict[str, tuple[str, str]]) -> pd.DataFrame:
    dataset = "DeepMindrobottabletennis"
    if dataset not in policy:
        return empty_frame()
    tier, allowed = policy[dataset]
    rows = []
    for split in ["rallies", "serves"]:
        path = EXTERNAL / dataset / f"{split}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for i, r in enumerate(data):
            vel = [r.get("vel_x", 0.0), r.get("vel_y", 0.0), r.get("vel_z", 0.0)]
            spin_vec = [r.get("w_vel_x", 0.0), r.get("w_vel_y", 0.0), r.get("w_vel_z", 0.0)]
            rows.append(
                row(
                    dataset,
                    path.relative_to(ROOT),
                    split,
                    i,
                    split,
                    "Serve" if split == "serves" else "Attack",
                    "serve_like" if split == "serves" else "rally_like",
                    False,
                    tier,
                    allowed,
                    landing_x=float(r.get("pos_x", np.nan)),
                    landing_y=float(r.get("pos_y", np.nan)),
                    landing_z=float(r.get("pos_z", np.nan)),
                    speed=float(np.linalg.norm(vel)),
                    spin=float(np.linalg.norm(spin_vec)),
                    raw_label=split,
                )
            )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def load_tt3d_trajectories(policy: dict[str, tuple[str, str]], max_files: int = 180) -> pd.DataFrame:
    dataset = "TT3D"
    if dataset not in policy:
        return empty_frame()
    tier, allowed = policy[dataset]
    root = EXTERNAL / dataset / "evaluation" / "3D_gt"
    if not root.exists():
        return empty_frame()
    rows = []
    for path in sorted(root.glob("*.csv"))[:max_files]:
        df = pd.read_csv(path)
        for i, r in df.iterrows():
            rows.append(
                row(
                    dataset,
                    path.relative_to(ROOT),
                    path.stem,
                    i,
                    "trajectory_point",
                    "Control",
                    "rally_like",
                    False,
                    tier,
                    allowed,
                    landing_x=float(r.get("X", np.nan)),
                    landing_y=float(r.get("Y", np.nan)),
                    landing_z=float(r.get("Z", np.nan)),
                    raw_label="trajectory_point",
                )
            )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def load_matchdynamics_rows(policy: dict[str, tuple[str, str]]) -> pd.DataFrame:
    dataset = "TT-MatchDynamics"
    if dataset not in policy:
        return empty_frame()
    tier, allowed = policy[dataset]
    path = EXTERNAL / dataset / "table_tennis_data.csv"
    if not path.exists():
        return empty_frame()
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        top = int(r.get("Topspin/Backspin Indicator", 0))
        fh = int(r.get("Forehand/Backhand Indicator", 0))
        family = "Attack" if top == 1 else "Control"
        rows.append(
            row(
                dataset,
                path.relative_to(ROOT),
                str(r.get("date", "matchdynamics")),
                i,
                "spin_landing",
                family,
                "rally_like",
                False,
                tier,
                allowed,
                landing_x=float(r.get("X", np.nan)),
                landing_y=float(r.get("Y", np.nan)),
                player_context=f"fhbh={fh}",
                raw_label=f"topspin={top};fhbh={fh}",
            )
        )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def load_coachai_rows(policy: dict[str, tuple[str, str]], limit_files: int = 120) -> pd.DataFrame:
    dataset = "CoachAI-Projects-main"
    if dataset not in policy:
        return empty_frame()
    tier, allowed = policy[dataset]
    root = EXTERNAL / dataset
    paths = sorted(root.glob("**/set*.csv"))[:limit_files]
    rows = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        for i, r in df.head(400).iterrows():
            text = " ".join(str(r.get(c, "")) for c in df.columns[:12]).lower()
            if any(k in text for k in ["smash", "drive", "clear", "lob"]):
                family = "Attack"
            elif any(k in text for k in ["drop", "net", "short"]):
                family = "Control"
            else:
                family = "Control"
            terminal = any(k in text for k in ["lose", "win", "error", "out"])
            rows.append(
                row(
                    dataset,
                    path.relative_to(ROOT),
                    path.parent.name,
                    int(i),
                    "badminton_stroke",
                    family,
                    canonical_phase_from_event("badminton_stroke", int(i) + 1),
                    terminal,
                    tier,
                    allowed,
                    raw_label=text[:120],
                )
            )
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def build_corpus() -> pd.DataFrame:
    policy = policy_map(read_policy())
    parts = [
        load_openttgames_events(policy),
        load_sony_events(policy),
        load_deepmind_states(policy),
        load_tt3d_trajectories(policy),
        load_matchdynamics_rows(policy),
        load_coachai_rows(policy),
    ]
    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if out.empty:
        return empty_frame()
    return out[CANONICAL_COLUMNS].copy()


def write_outputs(corpus: pd.DataFrame) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(OUTDIR / "v255_canonical_external_events.csv", index=False)
    summary = (
        corpus.groupby(["source_dataset", "risk_tier"], dropna=False)
        .agg(
            rows=("source_dataset", "size"),
            sequences=("sequence_id", "nunique"),
            terminal_rate=("terminal_like", "mean"),
            speed_mean=("speed", "mean"),
            spin_mean=("spin", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUTDIR / "v255_source_summary.csv", index=False)
    target_summary = (
        corpus.groupby(["source_dataset", "coarse_family", "phase"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    target_summary.to_csv(OUTDIR / "v255_training_targets_summary.csv", index=False)
    report = {
        "rows": int(len(corpus)),
        "sources": sorted(corpus["source_dataset"].unique().tolist()),
        "ttmatch_rows": int((corpus["source_dataset"] == "TTMATCH").sum()),
        "source_summary": summary.to_dict(orient="records"),
    }
    (OUTDIR / "v255_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# V255 Clean External Pretraining Corpus",
        "",
        f"Rows: `{len(corpus)}`",
        f"Sources: `{', '.join(report['sources'])}`",
        "TTMATCH rows: `0`",
        "",
        "Use policy: external rows supervise only coarse family, phase, terminal, trajectory, spin/velocity, and landing-intent SSL targets.",
        "",
    ]
    for _, r in summary.iterrows():
        lines.append(f"- `{r['source_dataset']}` ({r['risk_tier']}): {int(r['rows'])} rows")
    (OUTDIR / "v255_report.md").write_text("\n".join(lines), encoding="utf-8")
    SRC_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), SRC_DEST)


def main() -> None:
    corpus = build_corpus()
    write_outputs(corpus)
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR.relative_to(ROOT)),
                "rows": int(len(corpus)),
                "sources": int(corpus["source_dataset"].nunique()) if not corpus.empty else 0,
                "ttmatch_rows": int((corpus["source_dataset"] == "TTMATCH").sum()) if not corpus.empty else 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
