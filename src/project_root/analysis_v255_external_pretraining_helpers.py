from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FAMILY_COLUMNS = ["Zero", "Attack", "Control", "Defensive", "Serve"]
FAMILY_TO_ACTIONS = {
    "Zero": [0],
    "Attack": list(range(1, 8)),
    "Control": list(range(8, 12)),
    "Defensive": list(range(12, 15)),
    "Serve": list(range(15, 19)),
}


def parse_vector_string(value: Any) -> list[float]:
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return []
    inner = text[1:-1].replace(",", " ").split()
    try:
        return [float(x) for x in inner]
    except ValueError:
        return []


def action_family_from_id(action_id: int) -> str:
    a = int(action_id)
    if a == 0:
        return "Zero"
    if 1 <= a <= 7:
        return "Attack"
    if 8 <= a <= 11:
        return "Control"
    if 12 <= a <= 14:
        return "Defensive"
    if 15 <= a <= 18:
        return "Serve"
    return "Zero"


def canonical_phase_from_event(event_type: str, sequence_index: int) -> str:
    text = str(event_type).lower()
    if "net" in text or "terminal" in text or "ending" in text or "end" == text:
        return "terminal_like"
    idx = int(sequence_index)
    if idx <= 1:
        return "receive_like"
    if idx == 2:
        return "rally_like"
    if idx == 3:
        return "third_ball_like"
    if idx == 4:
        return "fourth_ball_like"
    return "rally_like"


def normalize_rows_safe(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("matrix must be 2D")
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    sums = arr.sum(axis=1, keepdims=True)
    out = arr.copy()
    zero = sums[:, 0] <= 0
    if np.any(~zero):
        out[~zero] = out[~zero] / sums[~zero]
    if np.any(zero):
        out[zero] = 1.0 / arr.shape[1]
    return out


def safe_family_prior_to_action_prob(family_prob: pd.DataFrame) -> np.ndarray:
    fam = pd.DataFrame(index=family_prob.index)
    for col in FAMILY_COLUMNS:
        fam[col] = pd.to_numeric(family_prob[col], errors="coerce") if col in family_prob.columns else 0.0
    fam_arr = normalize_rows_safe(fam.to_numpy(dtype=float))
    out = np.zeros((len(fam_arr), 19), dtype=float)
    for j, family in enumerate(FAMILY_COLUMNS):
        actions = FAMILY_TO_ACTIONS[family]
        out[:, actions] = fam_arr[:, [j]] / len(actions)
    return normalize_rows_safe(out)
