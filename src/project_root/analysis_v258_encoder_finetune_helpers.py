from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

ACTION_FAMILY_ID = {
    "Zero": 0,
    "Attack": 1,
    "Control": 2,
    "Defensive": 3,
    "Serve": 4,
}


def action_family_id(action_id: int) -> int:
    action = int(action_id)
    if action == 0:
        return ACTION_FAMILY_ID["Zero"]
    if 1 <= action <= 7:
        return ACTION_FAMILY_ID["Attack"]
    if 8 <= action <= 11:
        return ACTION_FAMILY_ID["Control"]
    if 12 <= action <= 14:
        return ACTION_FAMILY_ID["Defensive"]
    if 15 <= action <= 18:
        return ACTION_FAMILY_ID["Serve"]
    return ACTION_FAMILY_ID["Zero"]


def pad_sequence(values: list[int] | np.ndarray, max_len: int, pad: int = 0) -> np.ndarray:
    out = np.full(max_len, pad, dtype=np.int64)
    arr = np.asarray(values, dtype=np.int64)[:max_len]
    out[: len(arr)] = arr
    return out


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


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    out = np.zeros((len(labels), n_classes), dtype=float)
    out[np.arange(len(labels)), labels] = 1.0
    return out


def blend_probabilities(anchor_labels: np.ndarray, teacher_prob: np.ndarray, weight: float) -> np.ndarray:
    anchor = one_hot(anchor_labels, teacher_prob.shape[1])
    return normalize_rows_safe((1.0 - float(weight)) * anchor + float(weight) * teacher_prob)


def kd_cross_entropy(student_logits: torch.Tensor, teacher_prob: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    teacher = teacher_prob.clamp_min(1e-8)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    return -(teacher * log_student).sum(dim=1).mean() * (temperature * temperature)
