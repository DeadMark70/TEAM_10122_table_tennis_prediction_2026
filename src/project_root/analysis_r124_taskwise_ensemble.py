"""R124 task-wise ensemble search over validated submissions.

This script does not retrain models.  It combines previously generated
submission columns by task:

  action from one candidate,
  point from another candidate,
  server from another candidate.

The local score is estimated from the OOF component metrics already recorded
for each branch.  This is valid for the competition score because action,
point, and server are scored independently.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


OUTDIR = Path("r124_taskwise_ensemble")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")


def read_submission(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"rally_uid", "actionId", "pointId", "serverGetPoint"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df[["rally_uid", "actionId", "pointId", "serverGetPoint"]].copy()


def combine(action_key: str, point_key: str, server_key: str, action_pool: dict, point_pool: dict, server_pool: dict) -> pd.DataFrame:
    a = read_submission(action_pool[action_key]["path"])
    p = read_submission(point_pool[point_key]["path"])
    s = read_submission(server_pool[server_key]["path"])
    if not a["rally_uid"].equals(p["rally_uid"]) or not a["rally_uid"].equals(s["rally_uid"]):
        raise ValueError(f"rally_uid alignment failed for {action_key}/{point_key}/{server_key}")
    return pd.DataFrame(
        {
            "rally_uid": a["rally_uid"].astype(int),
            "actionId": a["actionId"].astype(int),
            "pointId": p["pointId"].astype(int),
            "serverGetPoint": s["serverGetPoint"].clip(1e-6, 1.0 - 1e-6),
        }
    )


def write_combo(name: str, df: pd.DataFrame) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / name
    df.to_csv(path, index=False, float_format="%.8f")
    upload_path = UPLOAD_DIR / name
    selected_path = SELECTED_DIR / name
    upload_path.write_bytes(path.read_bytes())
    selected_path.write_bytes(path.read_bytes())
    return {"path": str(path), "upload_path": str(upload_path), "selected_path": str(selected_path)}


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    # OOF metrics below are from the corresponding experiment logs/reports.
    action_pool = {
        "r67_public_anchor": {
            "path": UPLOAD_DIR / "submission_r67_r63_blend_w0p2_current_point_server.csv",
            "action": 0.29700342743872726,
            "pl": 0.3518207,
        },
        "r120_local_motif": {
            "path": UPLOAD_DIR / "submission_r120_motif_aw0p03_pw0p075.csv",
            "action": 0.3383336195575144,
            "pl": 0.3470782,
        },
        "r120_conservative_motif": {
            "path": UPLOAD_DIR / "submission_r120_motif_aw0p02_pw0p075.csv",
            "action": 0.3380864157028848,
            "pl": None,
        },
        "r111_anchor": {
            "path": UPLOAD_DIR / "submission_r111_remaining_moe_gru.csv",
            "action": 0.33804228763899724,
            "pl": None,
        },
    }
    point_pool = {
        "r67_v3_point": {
            "path": UPLOAD_DIR / "submission_r67_r63_blend_w0p2_current_point_server.csv",
            "point": 0.20465533648195663,
        },
        "r119_point_w0p05": {
            "path": UPLOAD_DIR / "submission_r119_point_w0p05.csv",
            "point": 0.21324624407399356,
            "pl": 0.3485626,
        },
        "r120_motif_point": {
            "path": UPLOAD_DIR / "submission_r120_motif_aw0p03_pw0p075.csv",
            "point": 0.21364455631376855,
            "pl": 0.3470782,
        },
        "r120_conservative_point": {
            "path": UPLOAD_DIR / "submission_r120_motif_aw0p02_pw0p075.csv",
            "point": 0.21364455631376855,
        },
    }
    server_pool = {
        "r67_current_server": {
            "path": UPLOAD_DIR / "submission_r67_r63_blend_w0p2_current_point_server.csv",
            "server": 0.6188096352755093,
        },
        "r118_server_w0p2": {
            "path": UPLOAD_DIR / "submission_r118_server_w0p2.csv",
            "server": 0.621093765582253,
        },
        "r121_min_w0p2": {
            "path": UPLOAD_DIR / "submission_r121_traj_min_w0p2.csv",
            "server": 0.6225400681884332,
            "pl": 0.3476130,
        },
        "r121_mean_w0p35": {
            "path": UPLOAD_DIR / "submission_r121_traj_mean_w0p35.csv",
            "server": 0.6228601630402966,
        },
    }

    rows = []
    generated = []
    for ak, av in action_pool.items():
        for pk, pv in point_pool.items():
            for sk, sv in server_pool.items():
                overall = 0.4 * av["action"] + 0.4 * pv["point"] + 0.2 * sv["server"]
                name = f"submission_r124_{ak}__{pk}__{sk}.csv"
                rows.append(
                    {
                        "candidate": name,
                        "action_source": ak,
                        "point_source": pk,
                        "server_source": sk,
                        "action_macro_f1": av["action"],
                        "point_macro_f1": pv["point"],
                        "server_auc": sv["server"],
                        "overall_local_est": overall,
                        "known_action_pl": av.get("pl"),
                        "known_point_branch_pl": pv.get("pl"),
                        "known_server_branch_pl": sv.get("pl"),
                    }
                )

    search = pd.DataFrame(rows).sort_values("overall_local_est", ascending=False).reset_index(drop=True)
    search.to_csv(OUTDIR / "r124_taskwise_search.csv", index=False)

    # Generate both local-optimal and public-anchor candidates.
    selected_names = []
    selected_names.extend(search.head(6)["candidate"].tolist())
    public_anchor = search[
        search["action_source"].eq("r67_public_anchor")
        & search["point_source"].isin(["r119_point_w0p05", "r120_motif_point"])
        & search["server_source"].isin(["r67_current_server", "r121_min_w0p2", "r118_server_w0p2"])
    ].head(8)
    selected_names.extend(public_anchor["candidate"].tolist())
    selected_names = list(dict.fromkeys(selected_names))

    for _, row in search[search["candidate"].isin(selected_names)].iterrows():
        df = combine(row["action_source"], row["point_source"], row["server_source"], action_pool, point_pool, server_pool)
        info = write_combo(row["candidate"], df)
        rec = row.to_dict()
        rec.update(info)
        generated.append(rec)

    report = {
        "best_local": search.head(20).to_dict(orient="records"),
        "best_public_anchor": public_anchor.to_dict(orient="records"),
        "generated": generated,
        "notes": [
            "Local estimates are task-wise sums from existing OOF metrics.",
            "R67 public-anchor rows have lower OOF action but are public-validated.",
            "Generated candidates combine columns only; no retraining.",
        ],
    }
    (OUTDIR / "r124_taskwise_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2("analysis_r124_taskwise_ensemble.py", "src/analysis/analysis_r124_taskwise_ensemble.py")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
