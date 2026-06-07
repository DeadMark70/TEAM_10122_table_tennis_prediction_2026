"""V266 clean autoresearch loop.

This is the first deliberately conservative autonomous clean-line loop.
It never reads TTMATCH or old-server artifacts.  The first iteration only
derives low-risk server microblend candidates from the existing V263C clean
server teacher on top of the public-positive V261 cap1 anchor.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


OUTDIR = Path("v266_clean_autoresearch_loop")
UPLOAD_DIR = Path("upload_candidates_20260519")
ANCHOR_PATH = Path("v261_action_conditioned_point_residual/submission_v261_cap0p01__v173action_r121server.csv")
V263C_W02_PATH = Path("v263_questionnaire_baseline/submission_v263c_server_w0p02__v173_v261cap1.csv")
V263C_SEARCH_PATH = Path("v263_questionnaire_baseline/v263c_server_search.csv")
EXPECTED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
CURRENT_PUBLIC_PL = 0.3576720
V263C_TEACHER_WEIGHT = 0.02
SERVER_WEIGHTS = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]


def load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{path} columns are {list(df.columns)}, expected {EXPECTED_COLUMNS}")
    if len(df) != 1845:
        raise ValueError(f"{path} has {len(df)} rows, expected 1845")
    return df


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def cap_prob(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)


def derive_v263_teacher(anchor: pd.DataFrame, v263_w02: pd.DataFrame) -> tuple[np.ndarray, int]:
    merged = anchor[["rally_uid", "serverGetPoint"]].merge(
        v263_w02[["rally_uid", "serverGetPoint"]],
        on="rally_uid",
        how="left",
        validate="one_to_one",
        suffixes=("_anchor", "_w02"),
    )
    if merged["serverGetPoint_w02"].isna().any():
        raise ValueError("V263C w0.02 rows are not aligned with anchor.")
    anchor_prob = cap_prob(merged["serverGetPoint_anchor"].to_numpy())
    w02_prob = cap_prob(merged["serverGetPoint_w02"].to_numpy())
    raw_teacher = (w02_prob - (1.0 - V263C_TEACHER_WEIGHT) * anchor_prob) / V263C_TEACHER_WEIGHT
    clipped = int(((raw_teacher < 0.0) | (raw_teacher > 1.0)).sum())
    return cap_prob(raw_teacher), clipped


def copy_to_upload(path: Path) -> None:
    if UPLOAD_DIR.exists():
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, UPLOAD_DIR / path.name)


def known_proxy_delta_per_weight() -> float:
    if not V263C_SEARCH_PATH.exists():
        return float("nan")
    search = pd.read_csv(V263C_SEARCH_PATH)
    hit = search[search["candidate"].astype(str).eq("v263c_server_w0p02__v173_v261cap1")]
    if hit.empty or "delta_vs_proxy_base" not in hit:
        return float("nan")
    return float(hit.iloc[0]["delta_vs_proxy_base"]) / V263C_TEACHER_WEIGHT


def write_submission(path: Path, anchor: pd.DataFrame, server_prob: np.ndarray) -> None:
    out = anchor.copy()
    out["serverGetPoint"] = cap_prob(server_prob)
    out[EXPECTED_COLUMNS].to_csv(path, index=False, float_format="%.8f")
    copy_to_upload(path)


def make_weight_name(weight: float) -> str:
    return str(weight).replace(".", "p")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    anchor = load_submission(ANCHOR_PATH)
    v263_w02 = load_submission(V263C_W02_PATH)
    teacher, teacher_clipped = derive_v263_teacher(anchor, v263_w02)
    base_server = cap_prob(anchor["serverGetPoint"].to_numpy())
    proxy_slope = known_proxy_delta_per_weight()

    rows = []
    for weight in SERVER_WEIGHTS:
        server = cap_prob((1.0 - weight) * base_server + weight * teacher)
        name = f"submission_v266_server_teacher_w{make_weight_name(weight)}__v173_v261cap1.csv"
        path = OUTDIR / name
        write_submission(path, anchor, server)
        server_mad = float(np.mean(np.abs(server - base_server)))
        row = {
            "candidate": name,
            "branch": "v266_clean_autoresearch_loop_server",
            "tier": "clean_no_old_autoresearch",
            "path": str(path),
            "rows": int(len(anchor)),
            "action_changed_vs_anchor": 0,
            "point_changed_vs_anchor": 0,
            "server_mad_vs_anchor": server_mad,
            "server_corr_vs_anchor": corr(server, base_server),
            "server_weight": float(weight),
            "teacher_source": str(V263C_W02_PATH),
            "teacher_derived_from_weight": V263C_TEACHER_WEIGHT,
            "teacher_clipped_rows": teacher_clipped,
            "proxy_delta_linear_from_v263c": float(proxy_slope * weight) if np.isfinite(proxy_slope) else np.nan,
            "risk_tier": "safe_probe" if weight <= 0.020 else "medium_probe",
            "verdict": "CANDIDATE_FOR_REVIEW" if weight <= 0.020 else "AGGRESSIVE_REVIEW",
        }
        rows.append(row)

    search = pd.DataFrame(rows)
    search.to_csv(OUTDIR / "v266_candidate_search.csv", index=False)

    best_safe = search[search["risk_tier"].eq("safe_probe")].sort_values(
        ["proxy_delta_linear_from_v263c", "server_mad_vs_anchor"], ascending=[False, True]
    ).iloc[0]
    best_explore = search.sort_values(["proxy_delta_linear_from_v263c", "server_mad_vs_anchor"], ascending=[False, True]).iloc[0]
    state = {
        "branch": "v266_clean_autoresearch_loop",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "clean_only": True,
            "no_ttmatch": True,
            "no_old_server": True,
            "no_auto_upload": True,
        },
        "current_clean_anchor": {
            "path": str(ANCHOR_PATH),
            "public_pl": CURRENT_PUBLIC_PL,
            "action": "V173",
            "point": "V261 cap1 on V188 cap5",
            "server": "R121",
        },
        "iteration": {
            "id": "v266_iter001_server_microblend",
            "hypothesis": "Tiny clean server microblends can improve ranking without touching action or point.",
            "generated_candidates": len(search),
            "recommended_safe_candidate": best_safe["candidate"],
            "recommended_exploratory_candidate": best_explore["candidate"],
        },
    }
    (OUTDIR / "clean_autoresearch_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    report = [
        "# V266 Clean Autoresearch Loop",
        "",
        "This branch is clean-only: no TTMATCH, no old-server, and no automatic upload.",
        "",
        "## Current Anchor",
        "",
        "```text",
        "action = V173",
        "point  = V261 cap1 on V188 cap5",
        "server = R121",
        f"PL     = {CURRENT_PUBLIC_PL:.7f}",
        "```",
        "",
        "## Iteration 001: Server Microblend",
        "",
        f"Teacher source: `{V263C_W02_PATH}`",
        f"Derived teacher clipped rows: `{teacher_clipped}`",
        "",
        "Generated candidates:",
        "",
    ]
    for row in rows:
        report.append(
            f"- `{row['candidate']}`: weight={row['server_weight']:.3f}, "
            f"MAD={row['server_mad_vs_anchor']:.6f}, "
            f"corr={row['server_corr_vs_anchor']:.6f}, verdict={row['verdict']}"
        )
    report.extend(
        [
            "",
            "## Recommendation",
            "",
            f"Safe review candidate: `{best_safe['candidate']}`",
            f"Exploratory review candidate: `{best_explore['candidate']}`",
            "",
            "This is a clean server-only micro-probe. Expected effect is small; action and point remain unchanged.",
        ]
    )
    (OUTDIR / "v266_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR),
                "candidates": len(rows),
                "recommended_safe": str(best_safe["candidate"]),
                "recommended_exploratory": str(best_explore["candidate"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
