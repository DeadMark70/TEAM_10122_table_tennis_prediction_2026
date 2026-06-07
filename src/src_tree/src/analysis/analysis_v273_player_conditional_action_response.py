"""V273 player-conditional action response teacher.

Builds fold-safe smoothed response-style/backoff tables and uses them only for
trust-gated, cap-limited action replacements.  Point and server columns remain
fixed to the V261 cap1 / R121 anchor.
"""

from __future__ import annotations

import __main__
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analysis_v209_action_selector_reranker import GrUTuning, TransformerTuning, V3Tuning
from analysis_v243_v247_action_experiment_common import ACTION_CLASSES, WEAK_ACTIONS, context_weights, load_action_context


OUTDIR = Path("v273_player_conditional_action_response")
SEARCH_PATH = OUTDIR / "v273_action_search.csv"
REPORT_PATH = OUTDIR / "v273_report.md"
EXPECTED_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
CAPS = [0.005, 0.010, 0.020, 0.050]
SUPPORT_ALPHA = 25.0
SMOOTHING = 8.0


def no_ttmatch_path_guard(paths: list[str | Path]) -> None:
    bad = [str(path) for path in paths if "TTMATCH" in str(path).upper()]
    if bad:
        raise ValueError(f"TTMATCH is banned from clean branch: {bad}")


def validate_submission(df: pd.DataFrame, path: Path) -> None:
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{path} columns={list(df.columns)} expected={EXPECTED_COLUMNS}")
    if len(df) != 1845:
        raise ValueError(f"{path} rows={len(df)} expected 1845")


def action_family(action_id: int) -> int:
    action = int(action_id)
    if action == 0:
        return 0
    if 1 <= action <= 7:
        return 1
    if 8 <= action <= 11:
        return 2
    if 12 <= action <= 14:
        return 3
    if 15 <= action <= 18:
        return 4
    return 0


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    row_sum = arr.sum(axis=1, keepdims=True)
    zero = row_sum[:, 0] <= 0.0
    if zero.any():
        arr[zero] = 1.0 / arr.shape[1]
        row_sum = arr.sum(axis=1, keepdims=True)
    return arr / row_sum


def safe_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["next_hitter_id", "audit_phase", "audit_lag0_action_family", "audit_lag0_depth", "lag0_spinId"]:
        if col not in out:
            out[col] = -1 if col in {"next_hitter_id", "lag0_spinId"} else "missing"
    out["next_hitter_id"] = pd.to_numeric(out["next_hitter_id"], errors="coerce").fillna(-1).astype(int).astype(str)
    out["lag0_spinId"] = pd.to_numeric(out["lag0_spinId"], errors="coerce").fillna(-1).astype(int).astype(str)
    for col in ["audit_phase", "audit_lag0_action_family", "audit_lag0_depth"]:
        out[col] = out[col].fillna("missing").astype(str)
    return out


def global_action_prior(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=19).astype(float)
    return normalize_rows((counts + 1.0)[None, :])[0]


def global_family_prior(y: np.ndarray) -> np.ndarray:
    fam = np.asarray([action_family(v) for v in y], dtype=int)
    counts = np.bincount(fam, minlength=5).astype(float)
    return normalize_rows((counts + 1.0)[None, :])[0]


def action_given_family_prior(y: np.ndarray) -> np.ndarray:
    prior = np.zeros((5, 19), dtype=float)
    global_prior = global_action_prior(y)
    for family in range(5):
        members = [action for action in range(19) if action_family(action) == family]
        mask = np.isin(np.asarray(y, dtype=int), members)
        counts = np.bincount(np.asarray(y, dtype=int)[mask], minlength=19).astype(float)
        smooth = np.zeros(19, dtype=float)
        smooth[members] = global_prior[members]
        if smooth.sum() <= 0:
            smooth[members] = 1.0 / max(len(members), 1)
        smooth = smooth / smooth.sum()
        prior[family] = normalize_rows((counts + SMOOTHING * smooth)[None, :])[0]
    return prior


def style_clusters(train_frame: pd.DataFrame, y: np.ndarray, apply_frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    train = safe_labels(train_frame)
    apply = safe_labels(apply_frame)
    fam = np.asarray([action_family(v) for v in y], dtype=int)
    tmp = pd.DataFrame({"hitter": train["next_hitter_id"].to_numpy(), "family": fam})
    global_fam = global_family_prior(y)
    mapping: dict[str, str] = {}
    for hitter, grp in tmp.groupby("hitter", sort=False):
        counts = np.bincount(grp["family"].to_numpy(dtype=int), minlength=5).astype(float)
        prob = (counts + 4.0 * global_fam) / (counts.sum() + 4.0)
        dom = int(np.argmax(prob))
        attack_bin = "hiA" if prob[1] >= global_fam[1] else "loA"
        control_bin = "hiC" if prob[2] >= global_fam[2] else "loC"
        defense_bin = "hiD" if prob[3] >= global_fam[3] else "loD"
        mapping[str(hitter)] = f"dom{dom}_{attack_bin}_{control_bin}_{defense_bin}"
    train_cluster = train["next_hitter_id"].map(mapping).fillna("global")
    apply_cluster = apply["next_hitter_id"].map(mapping).fillna("global")
    return train_cluster, apply_cluster


def table_counts(frame: pd.DataFrame, labels: np.ndarray, key_cols: list[str], n_classes: int) -> dict[tuple[str, ...], tuple[np.ndarray, int]]:
    tmp = frame.loc[:, key_cols].astype(str).copy()
    tmp["_label"] = np.asarray(labels, dtype=int)
    table: dict[tuple[str, ...], tuple[np.ndarray, int]] = {}
    for key, grp in tmp.groupby(key_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        counts = np.bincount(grp["_label"].to_numpy(dtype=int), minlength=n_classes).astype(float)
        table[tuple(map(str, key))] = (counts, int(len(grp)))
    return table


def apply_action_table(
    frame: pd.DataFrame,
    key_cols: list[str],
    table: dict[tuple[str, ...], tuple[np.ndarray, int]],
    prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = frame.loc[:, key_cols].astype(str).to_numpy()
    prob = np.zeros((len(frame), 19), dtype=float)
    support = np.zeros(len(frame), dtype=float)
    for i, row in enumerate(values):
        counts, n = table.get(tuple(row.tolist()), (np.zeros(19, dtype=float), 0))
        prob[i] = counts + SMOOTHING * prior
        support[i] = float(n)
    return normalize_rows(prob), support


def apply_family_table(
    frame: pd.DataFrame,
    key_cols: list[str],
    table: dict[tuple[str, ...], tuple[np.ndarray, int]],
    family_prior: np.ndarray,
    action_by_family: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = frame.loc[:, key_cols].astype(str).to_numpy()
    prob = np.zeros((len(frame), 19), dtype=float)
    support = np.zeros(len(frame), dtype=float)
    for i, row in enumerate(values):
        counts, n = table.get(tuple(row.tolist()), (np.zeros(5, dtype=float), 0))
        fam_prob = (counts + SMOOTHING * family_prior) / (counts.sum() + SMOOTHING)
        prob[i] = fam_prob @ action_by_family
        support[i] = float(n)
    return normalize_rows(prob), support


def build_teacher(train_frame: pd.DataFrame, y: np.ndarray, apply_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    train = safe_labels(train_frame)
    apply = safe_labels(apply_frame)
    train_cluster, apply_cluster = style_clusters(train, y, apply)
    train = train.assign(style_cluster=train_cluster.to_numpy())
    apply = apply.assign(style_cluster=apply_cluster.to_numpy())

    y = np.asarray(y, dtype=int)
    y_family = np.asarray([action_family(v) for v in y], dtype=int)
    a_prior = global_action_prior(y)
    f_prior = global_family_prior(y)
    a_by_f = action_given_family_prior(y)

    specs = [
        ("action_hitter_phase_spin_depth", "action", ["next_hitter_id", "audit_phase", "lag0_spinId", "audit_lag0_depth"], 0.40),
        ("family_hitter_phase_lag_family", "family", ["next_hitter_id", "audit_phase", "audit_lag0_action_family"], 0.25),
        ("family_style_phase_lag_family", "family", ["style_cluster", "audit_phase", "audit_lag0_action_family"], 0.15),
        ("action_global_phase_lag", "action", ["audit_phase", "audit_lag0_action_family", "audit_lag0_depth"], 0.20),
    ]

    posterior = np.zeros((len(apply), 19), dtype=float)
    weighted_support = np.zeros(len(apply), dtype=float)
    metrics = []
    for name, kind, cols, weight in specs:
        if kind == "action":
            table = table_counts(train, y, cols, 19)
            prob, support = apply_action_table(apply, cols, table, a_prior)
        else:
            table = table_counts(train, y_family, cols, 5)
            prob, support = apply_family_table(apply, cols, table, f_prior, a_by_f)
        posterior += float(weight) * prob
        weighted_support += float(weight) * support
        metrics.append(
            {
                "teacher": name,
                "kind": kind,
                "keys": int(len(table)),
                "apply_mean_support": float(support.mean()) if len(support) else 0.0,
                "weight": float(weight),
            }
        )
    posterior = normalize_rows(posterior)
    trust = weighted_support / (weighted_support + SUPPORT_ALPHA)
    return posterior, weighted_support, trust, metrics


def anchor_margin(prob: np.ndarray) -> np.ndarray:
    sorted_prob = np.sort(np.asarray(prob, dtype=float), axis=1)
    return sorted_prob[:, -1] - sorted_prob[:, -2]


def eligible_replacements(
    teacher_prob: np.ndarray,
    anchor_prob: np.ndarray,
    anchor_action: np.ndarray,
    support: np.ndarray,
    trust: np.ndarray,
) -> dict[str, np.ndarray]:
    candidate = teacher_prob.argmax(axis=1).astype(int)
    anchor_action = np.asarray(anchor_action, dtype=int)
    margin = anchor_margin(anchor_prob)
    row = np.arange(len(anchor_action))
    teacher_gain = teacher_prob[row, candidate] - teacher_prob[row, anchor_action]
    compatible = np.asarray([action_family(v) != 4 for v in candidate], dtype=bool)
    eligible = (
        (candidate != anchor_action)
        & compatible
        & (np.asarray(candidate) < 15)
        & (np.asarray(support, dtype=float) >= 20.0)
        & (np.asarray(trust, dtype=float) >= 0.25)
        & (margin <= 0.75)
        & (teacher_gain > 0.0)
    )
    score = teacher_gain * np.maximum(0.0, 1.0 - margin) * np.asarray(trust, dtype=float)
    score = np.where(eligible, score, -np.inf)
    return {
        "candidate": candidate,
        "eligible": eligible,
        "score": score,
        "margin": margin,
        "teacher_gain": teacher_gain,
    }


def apply_cap(anchor: np.ndarray, repl: dict[str, np.ndarray], cap: float) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(anchor, dtype=int).copy()
    budget = int(np.floor(len(pred) * float(cap)))
    changed = np.zeros(len(pred), dtype=bool)
    if budget <= 0:
        return pred, changed
    score = np.asarray(repl["score"], dtype=float)
    eligible_idx = np.flatnonzero(np.isfinite(score))
    if len(eligible_idx) == 0:
        return pred, changed
    order = eligible_idx[np.argsort(-score[eligible_idx], kind="mergesort")[:budget]]
    pred[order] = np.asarray(repl["candidate"], dtype=int)[order]
    changed[order] = pred[order] != np.asarray(anchor, dtype=int)[order]
    return pred, changed


def weighted_macro_f1_safe(y: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    scores = []
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    weights = np.asarray(weights, dtype=float)
    for cls in ACTION_CLASSES:
        true_pos = weights[(y == cls) & (pred == cls)].sum()
        false_pos = weights[(y != cls) & (pred == cls)].sum()
        false_neg = weights[(y == cls) & (pred != cls)].sum()
        denom = 2.0 * true_pos + false_pos + false_neg
        scores.append(0.0 if denom <= 0 else float(2.0 * true_pos / denom))
    return float(np.mean(scores))


def evaluate_oof(y: np.ndarray, pred: np.ndarray, anchor: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    ordinary = f1_score(y, pred, labels=ACTION_CLASSES, average="macro", zero_division=0)
    base = f1_score(y, anchor, labels=ACTION_CLASSES, average="macro", zero_division=0)
    weak = f1_score(y, pred, labels=WEAK_ACTIONS, average="macro", zero_division=0)
    weighted = weighted_macro_f1_safe(y, pred, weights)
    weighted_base = weighted_macro_f1_safe(y, anchor, weights)
    return {
        "ordinary_action_macro_f1": float(ordinary),
        "ordinary_action_delta_vs_anchor": float(ordinary - base),
        "weak_action_mean_f1": float(weak),
        "iw_action_delta_vs_anchor": float(weighted - weighted_base),
    }


def cap_name(cap: float) -> str:
    return f"{cap:.3f}".replace(".", "p")


def verdict_for(rec: dict[str, object]) -> str:
    if (
        float(rec["ordinary_action_delta_vs_anchor"]) >= 0.0015
        and float(rec["action_churn"]) <= 0.02
        and int(rec["serve_15_18_count"]) <= 2
        and float(rec["mean_support"]) >= 20.0
    ):
        return "CANDIDATE_FOR_REVIEW"
    if float(rec["ordinary_action_delta_vs_anchor"]) > 0 and float(rec["mean_support"]) >= 20.0:
        return "DIAGNOSTIC_ONLY_LOCAL_POSITIVE"
    return "LOCAL_NEGATIVE_DO_NOT_SUBMIT"


def write_submission(name: str, action: np.ndarray, point: pd.DataFrame, server: pd.DataFrame) -> str:
    out = pd.DataFrame(
        {
            "rally_uid": point["rally_uid"].astype(int).to_numpy(),
            "actionId": np.asarray(action, dtype=int),
            "pointId": point["pointId"].astype(int).to_numpy(),
            "serverGetPoint": server["serverGetPoint"].astype(float).to_numpy(),
        }
    )
    path = OUTDIR / name
    validate_submission(out, path)
    out.to_csv(path, index=False, float_format="%.8f")
    return str(path)


def write_report(search: pd.DataFrame, metrics: list[dict[str, object]], anchor_score: float) -> None:
    lines = [
        "# V273 Player-Conditional Action Response Teacher",
        "",
        "Fold-safe smoothed response-style/backoff tables; point and server fixed to the anchor.",
        "",
        "## Policy",
        "",
        "- No TTMATCH input.",
        "- No old-server or old-test labels.",
        "- No raw player lookup full replacement.",
        "- Replacements require support >= 20, trust >= 0.25, weak anchor margin, non-serve candidate action, and cap limits.",
        "",
        "## OOF Anchor",
        "",
        f"- V173 anchor ordinary action macro F1: `{anchor_score:.6f}`",
        "",
        "## Candidates",
        "",
        "| candidate | delta | churn | changed | serve15_18 | support | trust | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in search.to_dict("records"):
        lines.append(
            f"| `{row['candidate']}` | {row['ordinary_action_delta_vs_anchor']:.6f} | "
            f"{row['action_churn']:.6f} | {int(row['changed_rows'])} | "
            f"{int(row['serve_15_18_count'])} | {row['mean_support']:.2f} | "
            f"{row['mean_trust']:.4f} | `{row['verdict']}` |"
        )
    lines.extend(["", "## Teacher Tables", ""])
    for row in metrics:
        lines.append(
            f"- `{row['teacher']}`: kind=`{row['kind']}`, keys=`{row['keys']}`, "
            f"apply_mean_support=`{row['apply_mean_support']:.2f}`, weight=`{row['weight']:.2f}`"
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    __main__.V3Tuning = V3Tuning
    __main__.GrUTuning = GrUTuning
    __main__.TransformerTuning = TransformerTuning
    OUTDIR.mkdir(parents=True, exist_ok=True)
    no_ttmatch_path_guard(["train.csv", "test_new.csv"])

    ctx = load_action_context()
    rows = ctx["rows"].reset_index(drop=True)
    test_rows = ctx["test_rows"].reset_index(drop=True)
    y = np.asarray(ctx["y"], dtype=int)
    v173_oof = np.asarray(ctx["v173_oof"], dtype=int)
    v173_test = np.asarray(ctx["v173_test"], dtype=int)
    weights = context_weights(rows, test_rows)

    oof_prob = np.zeros((len(rows), 19), dtype=float)
    oof_support = np.zeros(len(rows), dtype=float)
    oof_trust = np.zeros(len(rows), dtype=float)
    fold_metrics: list[dict[str, object]] = []
    for fold in sorted(rows["fold"].astype(int).unique()):
        valid = rows["fold"].astype(int).eq(int(fold)).to_numpy()
        train = ~valid
        prob, support, trust, metrics = build_teacher(rows.loc[train].reset_index(drop=True), y[train], rows.loc[valid].reset_index(drop=True))
        oof_prob[valid] = prob
        oof_support[valid] = support
        oof_trust[valid] = trust
        for metric in metrics:
            metric = dict(metric)
            metric["fold"] = int(fold)
            fold_metrics.append(metric)

    test_prob, test_support, test_trust, test_metrics = build_teacher(rows, y, test_rows)
    pd.DataFrame(fold_metrics).to_csv(OUTDIR / "v273_fold_teacher_metrics.csv", index=False)
    pd.DataFrame(test_metrics).to_csv(OUTDIR / "v273_test_teacher_metrics.csv", index=False)

    oof_repl = eligible_replacements(oof_prob, ctx["v173_prob_oof"], v173_oof, oof_support, oof_trust)
    test_repl = eligible_replacements(test_prob, ctx["v173_prob_test"], v173_test, test_support, test_trust)
    anchor_score = f1_score(y, v173_oof, labels=ACTION_CLASSES, average="macro", zero_division=0)

    records = []
    for cap in CAPS:
        name = f"submission_v273_action_style_cap{cap_name(cap)}__pv261cap1__sr121.csv"
        cand_oof, changed_oof = apply_cap(v173_oof, oof_repl, cap)
        cand_test, changed_test = apply_cap(v173_test, test_repl, cap)
        metrics = evaluate_oof(y, cand_oof, v173_oof, weights)
        changed_idx = np.flatnonzero(changed_test)
        mean_support = float(np.mean(test_support[changed_idx])) if len(changed_idx) else 0.0
        mean_trust = float(np.mean(test_trust[changed_idx])) if len(changed_idx) else 0.0
        rec: dict[str, object] = {
            "candidate": name,
            "path": write_submission(name, cand_test, ctx["point"], ctx["server"]),
            **metrics,
            "action_churn": float(np.mean(changed_test)),
            "changed_rows": int(changed_test.sum()),
            "oof_changed_rows": int(changed_oof.sum()),
            "serve_15_18_count": int(np.isin(cand_test[changed_test], [15, 16, 17, 18]).sum()),
            "mean_support": mean_support,
            "mean_trust": mean_trust,
            "eligible_test_rows": int(np.isfinite(test_repl["score"]).sum()),
            "eligible_oof_rows": int(np.isfinite(oof_repl["score"]).sum()),
        }
        rec["verdict"] = verdict_for(rec)
        records.append(rec)

    search = pd.DataFrame(records).sort_values(
        ["ordinary_action_delta_vs_anchor", "action_churn"],
        ascending=[False, True],
    )
    search.to_csv(SEARCH_PATH, index=False)
    write_report(search, test_metrics, float(anchor_score))
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR),
                "anchor_action_macro_f1": float(anchor_score),
                "best_candidate": search.iloc[0]["candidate"],
                "best_delta": float(search.iloc[0]["ordinary_action_delta_vs_anchor"]),
                "generated": len(records),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
