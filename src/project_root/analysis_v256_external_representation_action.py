"""V256 external representation action experiment.

Stage 1 trains lightweight heads on the V255 clean external corpus. Stage 2
transfers coarse external priors into a fold-safe AICUP exact action teacher
while keeping the V173/V188/R121 anchor fixed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis_v243_v247_action_experiment_common import (
    context_weights,
    evaluate_action,
    feature_columns,
    load_action_context,
)
from analysis_v256_external_representation_helpers import (
    BIN_CLASSES,
    FAMILY_CLASSES,
    PHASE_CLASSES,
    external_target_frame,
    normalize_rows_safe,
)


ROOT = Path(".")
OUTDIR = ROOT / "v256_external_representation_action"
CORPUS_PATH = ROOT / "v255_clean_external_pretraining_corpus" / "v255_canonical_external_events.csv"
RANDOM_STATE = 256


def load_v255_corpus() -> pd.DataFrame:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Missing V255 corpus: {CORPUS_PATH}")
    corpus = pd.read_csv(CORPUS_PATH, low_memory=False)
    source_text = " ".join(
        str(value).lower()
        for value in corpus.get("source_dataset", pd.Series(dtype=str)).dropna().unique().tolist()
    )
    if "ttmatch" in source_text:
        raise RuntimeError("V256 must not train on TTMATCH content.")
    return corpus


def build_external_features(corpus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = external_target_frame(corpus)
    features = pd.DataFrame(index=corpus.index)
    for col in ["source_dataset", "event_type", "phase"]:
        features[col] = corpus[col].fillna("missing").astype(str) if col in corpus.columns else "missing"
    for col in ["terminal_like", "landing_x", "landing_y", "landing_z", "speed", "spin"]:
        features[col] = pd.to_numeric(corpus[col], errors="coerce") if col in corpus.columns else np.nan
        features[f"{col}_missing"] = features[col].isna().astype(int)
        features[col] = features[col].fillna(0.0)
    return features, targets


def fit_external_head(features: pd.DataFrame, target: pd.Series, classes: list[str], name: str) -> dict:
    y = target.astype(str)
    if y.nunique() < 2:
        return {
            "name": name,
            "head": name,
            "classes": classes,
            "status": "skipped_single_class",
            "macro_f1": 0.0,
        }
    encoder = LabelEncoder()
    encoder.fit(classes)
    y_encoded = pd.Series(encoder.transform(y.where(y.isin(classes), classes[0])), index=y.index)
    categorical = ["source_dataset", "event_type", "phase"]
    numeric = [col for col in features.columns if col not in categorical]
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )
    model = MLPClassifier(
        hidden_layer_sizes=(48,),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=70,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    pipe = Pipeline([("pre", pre), ("model", model)])
    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded if stratify is not None else None,
    )
    pipe.fit(x_train, y_train)
    pred = encoder.inverse_transform(pipe.predict(x_valid))
    y_valid_labels = encoder.inverse_transform(y_valid)
    macro = f1_score(y_valid_labels, pred, average="macro", labels=classes, zero_division=0)
    return {
        "name": name,
        "head": name,
        "classes": classes,
        "status": "trained",
        "rows": int(len(features)),
        "macro_f1": float(macro),
        "iterations": int(pipe.named_steps["model"].n_iter_),
    }


def train_external_heads(corpus: pd.DataFrame) -> dict[str, dict]:
    features, targets = build_external_features(corpus)
    heads = {
        "family": fit_external_head(features, targets["family"], FAMILY_CLASSES, "family"),
        "phase": fit_external_head(features, targets["phase"], PHASE_CLASSES, "phase"),
        "terminal": fit_external_head(features, targets["terminal"].astype(str), ["0", "1"], "terminal"),
        "speed_bin": fit_external_head(features, targets["speed_bin"], BIN_CLASSES, "speed_bin"),
        "spin_bin": fit_external_head(features, targets["spin_bin"], BIN_CLASSES, "spin_bin"),
        "depth_bin": fit_external_head(features, targets["depth_bin"], BIN_CLASSES, "depth_bin"),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "head": key,
                "status": value["status"],
                "macro_f1": value["macro_f1"],
                "classes": "|".join(value["classes"]),
            }
            for key, value in heads.items()
        ]
    ).to_csv(OUTDIR / "v256_external_head_metrics.csv", index=False)
    return heads


def external_priors_for_aicup(corpus: pd.DataFrame) -> pd.DataFrame:
    targets = external_target_frame(corpus)
    numeric = pd.DataFrame(index=corpus.index)
    for col in ["speed", "spin", "landing_y"]:
        numeric[col] = pd.to_numeric(corpus[col], errors="coerce") if col in corpus.columns else np.nan
    frame = targets.join(numeric)
    grouped = frame.groupby("phase", dropna=False)
    rows = []
    for phase, group in grouped:
        row = {"phase": phase, "count": int(len(group))}
        for family in FAMILY_CLASSES:
            row[f"prior_family_{family}"] = float((group["family"] == family).mean())
        for col in ["speed", "spin", "landing_y"]:
            row[f"prior_{col}_mean"] = float(group[col].mean()) if group[col].notna().any() else 0.0
            row[f"prior_{col}_std"] = float(group[col].std()) if group[col].notna().sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


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
    values = np.where(prefix <= 1, "receive_like", np.where(prefix == 3, "third_ball_like", "rally_like"))
    return pd.Series(values, index=rows.index)


def aicup_external_prior_features(rows: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    phase = phase_from_aicup(rows)
    prior = priors.set_index("phase")
    numeric_cols = [col for col in prior.columns if col != "count"]
    fallback = prior[numeric_cols].mean(numeric_only=True).fillna(0.0)
    records = []
    for item in phase:
        if item in prior.index:
            records.append(prior.loc[item, numeric_cols].fillna(fallback).to_dict())
        else:
            records.append(fallback.to_dict())
    return pd.DataFrame(records, index=rows.index).add_prefix("ext_").fillna(0.0)


def aicup_feature_frame(rows: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    cols = feature_columns(rows, drop_keywords=("next_",))
    base = rows.loc[:, cols].copy().fillna(0.0)
    ext = aicup_external_prior_features(rows, priors)
    return pd.concat([base.reset_index(drop=True), ext.reset_index(drop=True)], axis=1).fillna(0.0)


def align_frames(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = sorted(set(train.columns) | set(test.columns))
    return train.reindex(columns=cols, fill_value=0.0), test.reindex(columns=cols, fill_value=0.0)


def train_fold_safe_action_teacher(
    x: pd.DataFrame, y: np.ndarray, groups: np.ndarray, x_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros((len(x), 19), dtype=float)
    test = np.zeros((len(x_test), 19), dtype=float)
    splitter = GroupKFold(n_splits=5)
    for fold, (trn, val) in enumerate(splitter.split(x, y, groups)):
        model = ExtraTreesClassifier(
            n_estimators=180,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=RANDOM_STATE + fold,
            n_jobs=1,
        )
        model.fit(x.iloc[trn], y[trn])
        val_prob = np.zeros((len(val), 19), dtype=float)
        val_raw = model.predict_proba(x.iloc[val])
        for idx, cls in enumerate(model.classes_):
            val_prob[:, int(cls)] = val_raw[:, idx]
        oof[val] = val_prob
        test_raw = model.predict_proba(x_test)
        test_prob = np.zeros((len(x_test), 19), dtype=float)
        for idx, cls in enumerate(model.classes_):
            test_prob[:, int(cls)] = test_raw[:, idx]
        test += test_prob / 5.0
    return normalize_rows_safe(oof), normalize_rows_safe(test)


def one_hot(labels: np.ndarray, n_classes: int = 19) -> np.ndarray:
    out = np.zeros((len(labels), n_classes), dtype=float)
    out[np.arange(len(labels)), labels.astype(int)] = 1.0
    return out


def blend_teacher(anchor_labels: np.ndarray, teacher_prob: np.ndarray, weight: float) -> np.ndarray:
    anchor = one_hot(anchor_labels, teacher_prob.shape[1])
    return normalize_rows_safe((1.0 - weight) * anchor + weight * teacher_prob)


def public_like_weights(rows: pd.DataFrame) -> np.ndarray:
    prefix_len = pd.to_numeric(rows.get("prefix_len", 0), errors="coerce").fillna(0)
    phase = rows.get("phase", pd.Series("", index=rows.index)).astype(str)
    lag0_depth = rows.get("lag0_depth", pd.Series("", index=rows.index)).astype(str)
    lag0_family = rows.get("lag0_family", pd.Series("", index=rows.index)).astype(str)
    weight = np.ones(len(rows), dtype=float)
    weight += 0.5 * (prefix_len <= 3).to_numpy(dtype=float)
    weight += 0.5 * phase.isin(["receive", "third_ball", "rally"]).to_numpy(dtype=float)
    weight += 0.25 * lag0_depth.eq("long").to_numpy(dtype=float)
    weight += 0.25 * lag0_family.eq("Attack").to_numpy(dtype=float)
    return weight


def submission_frame(action: np.ndarray, point_src: pd.DataFrame, server_src: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rally_uid": point_src["rally_uid"].astype(int),
            "actionId": np.asarray(action, dtype=int),
            "pointId": point_src["pointId"].astype(int),
            "serverGetPoint": server_src["serverGetPoint"].astype(float),
        }
    )


def write_local_submission(name: str, action: np.ndarray, point_src: pd.DataFrame, server_src: pd.DataFrame) -> dict:
    out = submission_frame(action, point_src, server_src)
    if len(out) != 1845:
        raise RuntimeError(f"{name} has {len(out)} rows; expected 1845")
    expected = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    if list(out.columns) != expected:
        raise RuntimeError(f"{name} has invalid columns: {list(out.columns)}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"submission_{name}__pv188cap5__sr121.csv"
    out.to_csv(path, index=False, float_format="%.8f")
    return {"submission": path.name, "path": str(path), "rows": int(len(out)), "do_not_submit": True}


def action_distribution(labels: np.ndarray) -> str:
    counts = pd.Series(labels.astype(int)).value_counts().sort_index()
    return json.dumps({str(int(k)): int(v) for k, v in counts.items()}, sort_keys=True)


def add_candidate_metrics(record: dict, test_pred: np.ndarray, v173_test: np.ndarray) -> dict:
    rec = dict(record)
    rec["public_like_weighted_delta"] = rec["iw_delta_vs_v173"]
    rec["test_action_churn_vs_v173"] = float(np.mean(test_pred != v173_test))
    rec["test_action_distribution"] = action_distribution(test_pred)
    rec["serve_15_18_count"] = int(np.isin(test_pred, [15, 16, 17, 18]).sum())
    return rec


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    corpus = load_v255_corpus()
    heads = train_external_heads(corpus)

    ctx = load_action_context()
    priors = external_priors_for_aicup(corpus)
    x_train, x_test = align_frames(
        aicup_feature_frame(ctx["rows"], priors),
        aicup_feature_frame(ctx["test_rows"], priors),
    )
    groups = ctx["rows"]["fold"].astype(int).to_numpy()
    teacher_oof, teacher_test = train_fold_safe_action_teacher(x_train, ctx["y"], groups, x_test)
    weights = context_weights(ctx["rows"], ctx["test_rows"])

    variants = {
        "v256_external_repr_raw": (teacher_oof, teacher_test),
        "v256_external_repr_v173blend_w0p05": (
            blend_teacher(ctx["v173_oof"], teacher_oof, 0.05),
            blend_teacher(ctx["v173_test"], teacher_test, 0.05),
        ),
        "v256_external_repr_v173blend_w0p10": (
            blend_teacher(ctx["v173_oof"], teacher_oof, 0.10),
            blend_teacher(ctx["v173_test"], teacher_test, 0.10),
        ),
        "v256_external_repr_v173blend_w0p20": (
            blend_teacher(ctx["v173_oof"], teacher_oof, 0.20),
            blend_teacher(ctx["v173_test"], teacher_test, 0.20),
        ),
        "v256_external_repr_v173blend_w0p35": (
            blend_teacher(ctx["v173_oof"], teacher_oof, 0.35),
            blend_teacher(ctx["v173_test"], teacher_test, 0.35),
        ),
    }

    records = []
    anchor = evaluate_action("v173_anchor", ctx["y"], ctx["v173_oof"], ctx["v173_oof"], weights)
    records.append(add_candidate_metrics(anchor, ctx["v173_test"], ctx["v173_test"]))
    generated = []
    for name, (prob_oof, prob_test) in variants.items():
        prob_oof = normalize_rows_safe(prob_oof)
        prob_test = normalize_rows_safe(prob_test)
        pred = prob_oof.argmax(axis=1).astype(int)
        test_pred = prob_test.argmax(axis=1).astype(int)
        rec = evaluate_action(name, ctx["y"], pred, ctx["v173_oof"], weights)
        records.append(add_candidate_metrics(rec, test_pred, ctx["v173_test"]))
        np.save(OUTDIR / f"{name}_oof_action_prob.npy", prob_oof)
        np.save(OUTDIR / f"{name}_test_action_prob.npy", prob_test)
        generated.append(write_local_submission(name, test_pred, ctx["point"], ctx["server"]))

    search = pd.DataFrame(records).sort_values(
        ["delta_vs_v173_anchor", "public_like_weighted_delta", "weak_delta_vs_v173"],
        ascending=[False, False, False],
    )
    search.to_csv(OUTDIR / "v256_action_search.csv", index=False)

    candidates = search[search["candidate"].ne("v173_anchor")]
    best_delta = float(candidates["delta_vs_v173_anchor"].max()) if len(candidates) else 0.0
    best_public_like_delta = float(candidates["public_like_weighted_delta"].max()) if len(candidates) else 0.0
    if best_delta >= 0.002 and best_public_like_delta >= 0.001:
        verdict = "CANDIDATE_FOR_PUBLIC_PROBE"
    elif best_delta > 0:
        verdict = "LOCAL_WEAK_POSITIVE_NEEDS_REVIEW"
    else:
        verdict = "GENERATED_LOCAL_NEGATIVE_DO_NOT_SUBMIT"

    report = {
        "verdict": verdict,
        "best_delta_vs_v173_anchor": best_delta,
        "best_public_like_weighted_delta": best_public_like_delta,
        "external_rows": int(len(corpus)),
        "ttmatch_guard": "passed",
        "external_head_metrics": list(heads.values()),
        "generated": generated,
        "controller_owned_outputs_written": False,
        "top_candidates": search.head(8).to_dict(orient="records"),
    }
    (OUTDIR / "v256_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# V256 External Representation Action",
        "",
        f"Verdict: `{verdict}`",
        f"Best delta vs V173: `{best_delta:.6f}`",
        f"Best public-like delta: `{best_public_like_delta:.6f}`",
        f"External rows: `{len(corpus)}`",
        "TTMATCH guard: `passed`",
        "Controller-owned upload/log/dashboard outputs written: `False`",
        "",
        "External head metrics:",
        "",
    ]
    for rec in heads.values():
        lines.append(f"- `{rec['head']}`: status `{rec['status']}`, macro-F1 `{rec['macro_f1']:.6f}`")
    lines.extend(["", "Top action candidates:", ""])
    for _, r in search.head(6).iterrows():
        lines.append(
            f"- `{r['candidate']}`: action `{r['action_macro_f1']:.6f}`, "
            f"delta `{r['delta_vs_v173_anchor']:.6f}`, "
            f"public-like `{r['public_like_weighted_delta']:.6f}`, "
            f"test churn `{r['test_action_churn_vs_v173']:.6f}`"
        )
    (OUTDIR / "v256_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "outdir": str(OUTDIR),
                "generated": len(generated),
                "verdict": verdict,
                "best_delta": best_delta,
                "best_public_like_delta": best_public_like_delta,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
