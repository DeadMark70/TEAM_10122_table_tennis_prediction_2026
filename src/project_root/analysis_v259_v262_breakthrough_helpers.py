from __future__ import annotations

import numpy as np

ACTION_FAMILY = {
    "Zero": 0,
    "Attack": 1,
    "Control": 2,
    "Defensive": 3,
    "Serve": 4,
}


def action_family(action_id: int) -> int:
    action = int(action_id)
    if action == 0:
        return ACTION_FAMILY["Zero"]
    if 1 <= action <= 7:
        return ACTION_FAMILY["Attack"]
    if 8 <= action <= 11:
        return ACTION_FAMILY["Control"]
    if 12 <= action <= 14:
        return ACTION_FAMILY["Defensive"]
    if 15 <= action <= 18:
        return ACTION_FAMILY["Serve"]
    return ACTION_FAMILY["Zero"]


def point_depth(point_id: int) -> int:
    point = int(point_id)
    if point == 0:
        return 0
    if 1 <= point <= 3:
        return 1
    if 4 <= point <= 6:
        return 2
    if 7 <= point <= 9:
        return 3
    return 0


def point_side(point_id: int) -> int:
    point = int(point_id)
    if point == 0:
        return 0
    if 1 <= point <= 9:
        return ((point - 1) % 3) + 1
    return 0


def normalize_rows_safe(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    row_sum = arr.sum(axis=1, keepdims=True)
    zero = row_sum[:, 0] <= 0.0
    if zero.any():
        arr[zero, :] = 1.0 / arr.shape[1]
        row_sum = arr.sum(axis=1, keepdims=True)
    return arr / row_sum


def verdict_from_deltas(delta: float, public_like_delta: float, strong_delta: float = 0.003, strong_public: float = 0.001) -> str:
    if float(delta) >= strong_delta and float(public_like_delta) >= strong_public:
        return "CANDIDATE_FOR_PUBLIC_PROBE"
    if float(delta) > 0 and float(public_like_delta) >= 0:
        return "LOCAL_WEAK_POSITIVE_NEEDS_REVIEW"
    return "LOCAL_NEGATIVE_DO_NOT_SUBMIT"
