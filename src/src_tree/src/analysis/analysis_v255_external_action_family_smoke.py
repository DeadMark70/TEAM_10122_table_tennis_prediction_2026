"""V255 external action-family smoke proxy.

Uses V255 canonical external corpus to build coarse family priors. External
rows never supervise exact AICUP actionId. The only exported candidates are
V173-centered smoke probes with point fixed to V188 cap5 and server fixed to
R121.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_v238_v242_action_model_helpers import blend_probabilities, normalize_probability_rows
from analysis_v243_v247_action_experiment_common import context_weights, evaluate_action, load_action_context, write_submission
from analysis_v255_external_pretraining_helpers import FAMILY_COLUMNS, safe_family_prior_to_action_prob


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "v255_clean_external_pretraining_corpus" / "v255_canonical_external_events.csv"
OUTDIR = ROOT / "v255_external_action_family_smoke"
SRC_DEST = ROOT / "src" / "analysis" / "analysis_v255_external_action_family_smoke.py"


def phase_from_aicup(rows: pd.DataFrame) -> pd.Series:
    if "phase" in rows.columns:
        raw = rows["phase"].astype(str).str.lower()
        return raw.map(
            {
                "receive": "receive_like",
                "third_ball": "third_ball_like",
                "fourth_ball": "fourth_ball_like",
                "rally": "rally_like",
                "serve": "serve_like",
            }
        ).fillna("rally_like")
    prefix = pd.to_numeric(rows.get("prefix_len", 0), errors="coerce").fillna(0).astype(int)
    return pd.Series(
        np.select(
            [prefix <= 1, prefix == 2, prefix == 3, prefix == 4],
            ["receive_like", "rally_like", "third_ball_like", "fourth_ball_like"],
            default="rally_like",
        ),
        index=rows.index,
    )


def smoothed_family_table(corpus: pd.DataFrame, subset: pd.Series | None = None, alpha: float = 20.0) -> pd.DataFrame:
    data = corpus.copy()
    if subset is not None:
        data = data.loc[subset].copy()
    data = data[data["coarse_family"].isin(FAMILY_COLUMNS)]
    base = data["coarse_family"].value_counts().reindex(FAMILY_COLUMNS, fill_value=0).astype(float) + alpha
    global_prob = base / base.sum()
    rows = []
    for phase, g in data.groupby("phase"):
        counts = g["coarse_family"].value_counts().reindex(FAMILY_COLUMNS, fill_value=0).astype(float) + alpha * global_prob
        prob = counts / counts.sum()
        rec = {"phase": phase}
        rec.update(prob.to_dict())
        rows.append(rec)
    if not rows:
        rec = {"phase": "rally_like"}
        rec.update(global_prob.to_dict())
        rows.append(rec)
    return pd.DataFrame(rows)


def priors_for_rows(rows: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    phase = phase_from_aicup(rows)
    indexed = table.set_index("phase")
    fallback = indexed.mean(numeric_only=True).reindex(FAMILY_COLUMNS).fillna(1.0 / len(FAMILY_COLUMNS))
    out = []
    for p in phase:
        if p in indexed.index:
            out.append(indexed.loc[p, FAMILY_COLUMNS].astype(float).to_numpy())
        else:
            out.append(fallback.to_numpy(dtype=float))
    return pd.DataFrame(out, columns=FAMILY_COLUMNS, index=rows.index)


def build_external_action_prob(rows: pd.DataFrame, table: pd.DataFrame) -> np.ndarray:
    fam = priors_for_rows(rows, table)
    return safe_family_prior_to_action_prob(fam)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Missing V255 corpus: {CORPUS_PATH}")
    corpus = pd.read_csv(CORPUS_PATH)
    if (corpus["source_dataset"] == "TTMATCH").any():
        raise RuntimeError("V255 corpus unexpectedly contains TTMATCH rows")

    ctx = load_action_context()
    rows = ctx["rows"]
    test_rows = ctx["test_rows"]
    y = ctx["y"]
    weights = context_weights(rows, test_rows)

    all_table = smoothed_family_table(corpus)
    greenish = corpus["source_dataset"].isin(["openttgames", "DeepMindrobottabletennis", "sonytabletennis"])
    table_sony_opentt = smoothed_family_table(corpus, greenish)
    phase_prior_oof = build_external_action_prob(rows, all_table)
    phase_prior_test = build_external_action_prob(test_rows, all_table)
    sony_opentt_oof = build_external_action_prob(rows, table_sony_opentt)
    sony_opentt_test = build_external_action_prob(test_rows, table_sony_opentt)

    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "v255_external_family_raw": (phase_prior_oof, phase_prior_test),
        "v255_external_family_v173blend_w0p05": (
            blend_probabilities(ctx["v173_prob_oof"], phase_prior_oof, 0.05),
            blend_probabilities(ctx["v173_prob_test"], phase_prior_test, 0.05),
        ),
        "v255_external_family_v173blend_w0p10": (
            blend_probabilities(ctx["v173_prob_oof"], phase_prior_oof, 0.10),
            blend_probabilities(ctx["v173_prob_test"], phase_prior_test, 0.10),
        ),
        "v255_external_family_v173blend_w0p20": (
            blend_probabilities(ctx["v173_prob_oof"], phase_prior_oof, 0.20),
            blend_probabilities(ctx["v173_prob_test"], phase_prior_test, 0.20),
        ),
        "v255_sony_opentt_family_v173blend_w0p10": (
            blend_probabilities(ctx["v173_prob_oof"], sony_opentt_oof, 0.10),
            blend_probabilities(ctx["v173_prob_test"], sony_opentt_test, 0.10),
        ),
    }

    records = [evaluate_action("v173_anchor", y, ctx["v173_oof"], ctx["v173_oof"], weights)]
    generated = []
    for name, (prob_oof, prob_test) in variants.items():
        prob_oof = normalize_probability_rows(prob_oof)
        prob_test = normalize_probability_rows(prob_test)
        pred = prob_oof.argmax(axis=1).astype(int)
        test_pred = prob_test.argmax(axis=1).astype(int)
        rec = evaluate_action(name, y, pred, ctx["v173_oof"], weights)
        rec["test_churn_vs_v173"] = float(np.mean(test_pred != ctx["v173_test"]))
        rec["test_changed_rows"] = int(np.sum(test_pred != ctx["v173_test"]))
        records.append(rec)
        np.save(OUTDIR / f"{name}_oof_action_prob.npy", prob_oof)
        np.save(OUTDIR / f"{name}_test_action_prob.npy", prob_test)
        generated.append(write_submission(OUTDIR, f"submission_{name}__pv188cap5__sr121.csv", test_pred, ctx["point"], ctx["server"]))

    search = pd.DataFrame(records).sort_values(["delta_vs_v173_anchor", "iw_delta_vs_v173", "weak_delta_vs_v173"], ascending=[False, False, False])
    search.to_csv(OUTDIR / "v255_action_search.csv", index=False)
    all_table.to_csv(OUTDIR / "v255_family_prior_by_phase.csv", index=False)
    table_sony_opentt.to_csv(OUTDIR / "v255_sony_opentt_family_prior_by_phase.csv", index=False)
    best_delta = float(search[search["candidate"].ne("v173_anchor")]["delta_vs_v173_anchor"].max())
    verdict = "GENERATED_LOCAL_POSITIVE" if best_delta > 0 else "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"
    report = {
        "verdict": verdict,
        "best_delta_vs_v173_anchor": best_delta,
        "best": search.head(8).to_dict(orient="records"),
        "generated": generated,
        "external_rows": int(len(corpus)),
        "sources": sorted(corpus["source_dataset"].unique().tolist()),
        "ttmatch_rows": int((corpus["source_dataset"] == "TTMATCH").sum()),
    }
    (OUTDIR / "v255_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# V255 External Action-Family Smoke",
        "",
        f"Verdict: `{verdict}`",
        f"Best delta vs V173: `{best_delta:.6f}`",
        f"External rows: `{len(corpus)}`",
        "TTMATCH rows: `0`",
        "",
        "Top candidates:",
        "",
    ]
    for _, r in search.head(6).iterrows():
        lines.append(
            f"- `{r['candidate']}`: action `{r['action_macro_f1']:.6f}`, delta `{r['delta_vs_v173_anchor']:.6f}`, IW `{r['iw_delta_vs_v173']:.6f}`, test changed `{r.get('test_changed_rows', np.nan)}`"
        )
    (OUTDIR / "v255_report.md").write_text("\n".join(lines), encoding="utf-8")
    SRC_DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), SRC_DEST)
    print(json.dumps({"verdict": verdict, "generated": len(generated), "outdir": str(OUTDIR.relative_to(ROOT)), "best_delta": best_delta}, indent=2))


if __name__ == "__main__":
    main()
