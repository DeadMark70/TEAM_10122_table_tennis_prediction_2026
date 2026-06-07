"""Small attribution helpers for V248 V173 decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd


def component_weight_grid() -> list[dict]:
    """Original V173 action schedules plus a few decomposition schedules."""
    specs = [
        ("ext05_int10_teacher85", 0.05, 0.10, 0.85),
        ("ext10_int10_teacher80", 0.10, 0.10, 0.80),
        ("ext10_int20_teacher70", 0.10, 0.20, 0.70),
        ("ext15_int15_teacher70", 0.15, 0.15, 0.70),
        ("ext20_int10_teacher70", 0.20, 0.10, 0.70),
        ("ext20_int20_teacher60", 0.20, 0.20, 0.60),
        ("ext30_int20_teacher50", 0.30, 0.20, 0.50),
        ("teacher_only", 0.00, 0.00, 1.00),
        ("external_teacher_no_internal", 0.20, 0.00, 0.80),
        ("internal_teacher_no_external", 0.00, 0.20, 0.80),
        ("external_internal_no_teacher", 0.50, 0.50, 0.00),
    ]
    return [{"name": name, "external": float(we), "internal": float(wi), "teacher": float(wt)} for name, we, wi, wt in specs]


def acceptance_mask_by_phase(rows: pd.DataFrame, changed: np.ndarray, phases: list[str], phase_col: str = "r184_phase") -> np.ndarray:
    phase = rows[phase_col].astype(str).to_numpy()
    return np.asarray(changed, dtype=bool) & np.isin(phase, list(phases))


def transition_counts(frame: pd.DataFrame, phase_col: str, base_col: str, teacher_col: str) -> pd.DataFrame:
    return (
        frame.groupby([phase_col, base_col, teacher_col], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
        .reset_index(drop=True)
    )
