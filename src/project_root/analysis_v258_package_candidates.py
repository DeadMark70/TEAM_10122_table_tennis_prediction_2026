from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUTDIR = Path("v258_true_encoder_finetune")
CANDIDATE_PATH = OUTDIR / "v258_candidate_test_actions.npz"
ANCHOR_PATH = Path("upload_candidates_20260519") / "submission_v188_r186_w005_a0p05_cap0p05__v173action_r121server.csv"

NAME_MAP = {
    "v258_raw_action": "submission_v258_raw_action__pv188cap5__sr121.csv",
    "v258_v173blend_w0p05": "submission_v258_v173blend_w0p05__pv188cap5__sr121.csv",
    "v258_v173blend_w0p10": "submission_v258_v173blend_w0p10__pv188cap5__sr121.csv",
    "v258_v173blend_w0p20": "submission_v258_v173blend_w0p20__pv188cap5__sr121.csv",
    "v258_classgate": "submission_v258_classgate__pv188cap5__sr121.csv",
}


def main() -> None:
    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(f"Missing V258 candidate actions: {CANDIDATE_PATH}")
    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(f"Missing current no-old anchor submission: {ANCHOR_PATH}")
    OUTDIR.mkdir(exist_ok=True)
    anchor = pd.read_csv(ANCHOR_PATH)
    data = np.load(CANDIDATE_PATH)
    written = []
    for key, filename in NAME_MAP.items():
        if key not in data:
            continue
        out = pd.DataFrame(
            {
                "rally_uid": anchor["rally_uid"].astype(int),
                "actionId": np.asarray(data[key], dtype=int),
                "pointId": anchor["pointId"].astype(int),
                "serverGetPoint": anchor["serverGetPoint"].astype(float),
            }
        )
        if len(out) != 1845:
            raise RuntimeError(f"{filename} row count mismatch: {len(out)}")
        path = OUTDIR / filename
        out.to_csv(path, index=False, float_format="%.8f")
        written.append(str(path))
    print(json.dumps({"outdir": str(OUTDIR), "generated": len(written), "files": written}))


if __name__ == "__main__":
    main()
