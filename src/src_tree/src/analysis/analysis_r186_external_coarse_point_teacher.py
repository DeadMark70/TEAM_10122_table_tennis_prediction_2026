"""R186 external coarse point-intent teacher.

External data is converted only into coarse landing-intent supervision:
terminal, depth, width, safety, spin, and early-pressure.  It is never mapped
to AI CUP receiver-relative pointId 1..9 labels.

TTMATCH is intentionally excluded.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_r179_action_physics_hierarchy import action_family, phase_name, point_depth
from analysis_r67_r70_meta_priors import prepare_prefix_features


OUTDIR = Path("r186_external_coarse_point_teacher")
SRC_DEST = Path("src/analysis/analysis_r186_external_coarse_point_teacher.py")

EXTERNAL = Path("external_data")
TTMD = EXTERNAL / "TT-MatchDynamics" / "table_tennis_data.csv"
DM_RALLIES = EXTERNAL / "DeepMindrobottabletennis" / "rallies.json"
DM_SERVES = EXTERNAL / "DeepMindrobottabletennis" / "serves.json"
OPENTT_EVENTS = EXTERNAL / "openttgames" / "processed" / "openttgames_events.csv"
COACHAI_TRACK2 = EXTERNAL / "CoachAI-Projects-main" / "CoachAI-Challenge-IJCAI2023" / "Track 2_ Stroke Forecasting" / "data" / "train.csv"

DEPTH_LABELS = ["short", "half", "long"]
WIDTH_LABELS = ["center", "wide"]
SAFETY_LABELS = ["safe_middle", "pressure_wide", "risky_edge"]
TERMINAL_LABELS = ["nonterminal", "terminalish"]
SPIN_LABELS = ["unknown", "topspin", "backspin", "sidespin_or_fast"]
PHASE_LABELS = ["receive", "third_ball", "fourth_ball", "rally"]


def qcut3(values: pd.Series) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    if v.nunique(dropna=True) < 3:
        return pd.Series(["half"] * len(v), index=v.index)
    labels = pd.qcut(v.rank(method="first"), 3, labels=DEPTH_LABELS)
    return labels.astype(str)


def width_from_x(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    centered = x - x.median()
    absx = centered.abs()
    cut = absx.quantile(0.60) if absx.notna().any() else 0.0
    return pd.Series(np.where(absx <= cut, "center", "wide"), index=x.index)


def safety_from_xy(x: pd.Series, y: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    xc = (x - x.median()).abs()
    yc = (y - y.median()).abs()
    xq = xc.rank(pct=True)
    yq = yc.rank(pct=True)
    risky = (xq >= 0.90) | (yq >= 0.90)
    wide = xq >= 0.60
    return pd.Series(np.where(risky, "risky_edge", np.where(wide, "pressure_wide", "safe_middle")), index=x.index)


def phase_from_round(values: pd.Series) -> pd.Series:
    n = pd.to_numeric(values, errors="coerce").fillna(4).astype(int)
    return pd.Series(np.select([n <= 1, n == 2, n == 3], ["receive", "third_ball", "fourth_ball"], default="rally"), index=values.index)


def source_rows_ttmd() -> pd.DataFrame:
    if not TTMD.exists():
        return pd.DataFrame()
    raw = pd.read_csv(TTMD)
    out = pd.DataFrame(index=raw.index)
    out["source"] = "tt_matchdynamics"
    out["source_row"] = np.arange(len(raw))
    out["phase"] = np.where(raw["Winning in First Three Strokes"].astype(int).eq(1), "third_ball", "rally")
    out["depth"] = qcut3(raw["Y"])
    out["width"] = width_from_x(raw["X"])
    out["safety"] = safety_from_xy(raw["X"], raw["Y"])
    out["terminal"] = np.where(raw["Winning in First Three Strokes"].astype(int).eq(1), "terminalish", "nonterminal")
    out["spin"] = np.where(raw["Topspin/Backspin Indicator"].astype(int).eq(1), "topspin", "backspin")
    out["action_family"] = np.where(raw["Forehand/Backhand Indicator"].astype(int).eq(1), "attack", "control")
    out["early_pressure"] = raw["Winning in First Three Strokes"].astype(float)
    out["reliability"] = 0.90
    out["notes"] = "landing x/y plus spin and first-three outcome; no receiver-relative side mapping"
    return out


def source_rows_deepmind() -> pd.DataFrame:
    rows = []
    for path, phase in [(DM_RALLIES, "rally"), (DM_SERVES, "receive")]:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = pd.DataFrame(data)
        speed = np.sqrt(raw["vel_x"] ** 2 + raw["vel_y"] ** 2 + raw["vel_z"] ** 2)
        spin = np.sqrt(raw["w_vel_x"] ** 2 + raw["w_vel_y"] ** 2 + raw["w_vel_z"] ** 2)
        out = pd.DataFrame(index=raw.index)
        out["source"] = "deepmind_robot_table_tennis"
        out["source_row"] = raw["id"].astype(int)
        out["phase"] = phase
        # Initial ball state has no landing target; depth is a weak physics proxy.
        out["depth"] = qcut3(np.abs(raw["vel_y"]))
        out["width"] = width_from_x(raw["pos_x"])
        out["safety"] = np.where((speed.rank(pct=True) > 0.90) | (spin.rank(pct=True) > 0.90), "risky_edge", "safe_middle")
        out["terminal"] = "nonterminal"
        out["spin"] = np.where(raw["w_vel_x"].abs() >= raw["w_vel_y"].abs(), "topspin", "sidespin_or_fast")
        out["action_family"] = np.where(phase == "receive", "serve_receive_physics", "rally_physics")
        out["early_pressure"] = (speed.rank(pct=True) + spin.rank(pct=True)) / 2.0
        out["reliability"] = 0.55
        out["notes"] = "robot initial ball-state physics; no human tactical landing label"
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def source_rows_coachai() -> pd.DataFrame:
    if not COACHAI_TRACK2.exists():
        return pd.DataFrame()
    raw = pd.read_csv(COACHAI_TRACK2)
    out = pd.DataFrame(index=raw.index)
    out["source"] = "coachai_shuttleset_track2"
    out["source_row"] = np.arange(len(raw))
    out["phase"] = phase_from_round(raw["ball_round"])
    out["depth"] = qcut3(raw["landing_y"])
    out["width"] = width_from_x(raw["landing_x"])
    out["safety"] = safety_from_xy(raw["landing_x"], raw["landing_y"])
    terminal = raw["lose_reason"].notna() | raw["getpoint_player"].notna() | raw["ball_round"].eq(raw["rally_length"])
    out["terminal"] = np.where(terminal, "terminalish", "nonterminal")
    out["spin"] = "unknown"
    control_types = {"net shot", "drop", "short service"}
    attack_types = {"smash", "drive", "push/rush"}
    defensive_types = {"clear", "lob", "long service"}
    typ = raw["type"].astype(str).str.lower()
    out["action_family"] = np.select(
        [typ.isin(control_types), typ.isin(attack_types), typ.isin(defensive_types)],
        ["control", "attack", "defensive"],
        default="unknown",
    )
    out["early_pressure"] = np.where(terminal, 1.0, 0.0)
    out["reliability"] = 0.45
    out["notes"] = "badminton landing intent only; no direct table-tennis pointId mapping"
    return out


def source_rows_opentt() -> pd.DataFrame:
    if not OPENTT_EVENTS.exists():
        return pd.DataFrame()
    raw = pd.read_csv(OPENTT_EVENTS)
    strokes = raw[raw["is_stroke"].fillna(0).astype(int).eq(1)].copy()
    if strokes.empty:
        strokes = raw[raw["is_rally_ending"].fillna(0).astype(int).eq(1)].copy()
    out = pd.DataFrame(index=strokes.index)
    out["source"] = "openttgames"
    out["source_row"] = strokes.index.to_numpy()
    out["phase"] = "rally"
    out["depth"] = "half"
    out["width"] = "center"
    out["safety"] = np.where(strokes["is_rally_ending"].fillna(0).astype(int).eq(1), "risky_edge", "safe_middle")
    out["terminal"] = np.where(strokes["is_rally_ending"].fillna(0).astype(int).eq(1), "terminalish", "nonterminal")
    out["spin"] = "unknown"
    out["action_family"] = strokes["safe_action_family"].fillna("unknown").astype(str)
    out["early_pressure"] = strokes["is_rally_ending"].fillna(0).astype(float)
    out["reliability"] = 0.35
    out["notes"] = "terminal/action-family event supervision; weak/no landing geometry"
    return out.reset_index(drop=True)


def distribution(rows: pd.DataFrame, col: str, labels: list[str], weight_col: str = "reliability") -> dict[str, float]:
    if rows.empty:
        return {k: 0.0 for k in labels}
    weights = rows[weight_col].astype(float)
    vals = rows[col].astype(str)
    out = {}
    denom = float(weights.sum())
    for label in labels:
        out[label] = float(weights[vals.eq(label)].sum() / denom) if denom > 0 else 0.0
    return out


def make_prior_table(rows: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("terminal", TERMINAL_LABELS),
        ("depth", DEPTH_LABELS),
        ("width", WIDTH_LABELS),
        ("safety", SAFETY_LABELS),
        ("spin", SPIN_LABELS),
    ]
    records = []
    groupings = [
        ("global", []),
        ("source", ["source"]),
        ("phase", ["phase"]),
        ("source_phase", ["source", "phase"]),
        ("phase_action_family", ["phase", "action_family"]),
    ]
    for scope, keys in groupings:
        grouped = [((), rows)] if not keys else rows.groupby(keys, dropna=False)
        for key, part in grouped:
            key_tuple = key if isinstance(key, tuple) else (key,)
            rec = {"scope": scope, "rows": int(len(part)), "weight_sum": float(part["reliability"].sum())}
            for i, k in enumerate(keys):
                rec[k] = key_tuple[i]
            for col, labels in specs:
                dist = distribution(part, col, labels)
                for label, value in dist.items():
                    rec[f"{col}_{label}"] = value
            records.append(rec)
    return pd.DataFrame(records)


def aicup_action_family(action_id: int) -> str:
    fam = action_family(int(action_id))
    return {"Zero": "zero", "Attack": "attack", "Control": "control", "Defensive": "defensive", "Serve": "serve"}.get(fam, "unknown")


def priors_for_aicup(table: pd.DataFrame, df: pd.DataFrame, split: str) -> pd.DataFrame:
    global_row = table[table["scope"].eq("global")].iloc[0].to_dict()
    phase_rows = table[table["scope"].eq("phase")].set_index("phase")
    pf_rows = table[table["scope"].eq("phase_action_family")].set_index(["phase", "action_family"])
    records = []
    for row in df.itertuples(index=False):
        phase = phase_name(getattr(row, "phase_id"), getattr(row, "prefix_len"))
        fam = aicup_action_family(getattr(row, "lag0_actionId"))
        rec = {"split": split, "rally_uid": int(getattr(row, "rally_uid")), "prefix_len": int(getattr(row, "prefix_len")), "phase": phase, "lag0_action_family": fam}
        src = global_row
        if (phase, fam) in pf_rows.index:
            src = pf_rows.loc[(phase, fam)].to_dict()
        elif phase in phase_rows.index:
            src = phase_rows.loc[phase].to_dict()
        for col, labels in [("terminal", TERMINAL_LABELS), ("depth", DEPTH_LABELS), ("width", WIDTH_LABELS), ("safety", SAFETY_LABELS)]:
            for label in labels:
                rec[f"T_{col}_{label}"] = float(src.get(f"{col}_{label}", global_row.get(f"{col}_{label}", 0.0)))
        records.append(rec)
    return pd.DataFrame(records)


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    parts = [
        source_rows_ttmd(),
        source_rows_deepmind(),
        source_rows_coachai(),
        source_rows_opentt(),
    ]
    rows = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    rows.to_csv(OUTDIR / "r186_external_coarse_teacher_rows.csv", index=False)

    summary = rows.groupby("source", dropna=False).agg(
        rows=("source", "size"),
        reliability_sum=("reliability", "sum"),
        terminalish_rate=("terminal", lambda s: float((s == "terminalish").mean())),
        wide_rate=("width", lambda s: float((s == "wide").mean())),
        risky_rate=("safety", lambda s: float((s == "risky_edge").mean())),
    ).reset_index()
    summary.to_csv(OUTDIR / "r186_source_summary.csv", index=False)

    table = make_prior_table(rows)
    table.to_csv(OUTDIR / "r186_coarse_prior_table.csv", index=False)

    _, _, prefix, test_prefix, _ = prepare_prefix_features()
    train_priors = priors_for_aicup(table, prefix, "train_prefix")
    test_priors = priors_for_aicup(table, test_prefix, "test_prefix")
    train_priors.to_csv(OUTDIR / "r186_aicup_train_prefix_coarse_priors.csv", index=False)
    test_priors.to_csv(OUTDIR / "r186_aicup_test_prefix_coarse_priors.csv", index=False)

    report = {
        "verdict": "COARSE_TEACHER_READY",
        "external_sources_used": sorted(rows["source"].unique().tolist()),
        "excluded_sources": ["external_data/TTMATCH"],
        "rows": int(len(rows)),
        "source_summary": summary.to_dict(orient="records"),
        "artifacts": [
            "r186_external_coarse_teacher_rows.csv",
            "r186_source_summary.csv",
            "r186_coarse_prior_table.csv",
            "r186_aicup_train_prefix_coarse_priors.csv",
            "r186_aicup_test_prefix_coarse_priors.csv",
        ],
        "notes": [
            "External data is converted only to terminal/depth/width/safety/spin coarse labels.",
            "No external row is converted to AI CUP pointId 1..9.",
            "TT-MatchDynamics x/y is used for coarse depth/width/safety only; receiver-relative FH/BH is not inferred.",
            "DeepMind robot data is used as weak physics prior, not human tactical label.",
            "CoachAI is used as coarse landing intent prior only.",
            "OpenTTGames contributes terminal/action-family supervision and weak/no landing geometry.",
            "TTMATCH is excluded from this pipeline.",
        ],
    }
    (OUTDIR / "r186_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "r186_report.md").write_text(
        "# R186 External Coarse Point Teacher\n\n"
        f"- Verdict: `{report['verdict']}`\n"
        f"- Rows: `{report['rows']}`\n"
        f"- Sources: `{', '.join(report['external_sources_used'])}`\n"
        "- Excluded: `external_data/TTMATCH`\n\n"
        "## Artifacts\n\n"
        + "\n".join(f"- `{a}`" for a in report["artifacts"])
        + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in report["notes"])
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2("analysis_r186_external_coarse_point_teacher.py", SRC_DEST)
    print(json.dumps({"rows": len(rows), "artifacts": report["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
