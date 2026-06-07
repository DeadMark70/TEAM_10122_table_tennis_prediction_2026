"""V327 response-style contrastive features.

Builds clean incoming-context -> response-family features from coarse sequence
neighborhoods. V326's canonical event table is used when available; otherwise
the script falls back to the clean V274 canonical sample/audit path. It writes
feature/report artifacts only and never writes submissions or selected uploads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v327_response_style_contrastive"
V326_DIR = ROOT / "v326_masked_family_pretrain"
V274_SAMPLE = ROOT / "v274_clean_external_representation" / "v274_canonical_samples.csv"
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"

FAMILY_CLASSES = ["Zero", "Attack", "Control", "Defensive", "Serve"]
PHASE_CLASSES = [
    "serve_like",
    "receive_like",
    "third_ball_like",
    "fourth_ball_like",
    "rally_like",
    "terminal_like",
]
EVENT_COLUMNS = [
    "corpus",
    "sequence_id",
    "step_idx",
    "phase_code",
    "coarse_family",
    "landing_depth",
    "landing_side",
    "has_spin",
    "has_speed",
    "source_weight",
    "source_path",
]


@dataclass(frozen=True)
class ResponseStyleModel:
    matrix: pd.DataFrame
    counts: pd.DataFrame
    pmi: pd.DataFrame
    log_odds: pd.DataFrame
    embeddings: pd.DataFrame
    global_distribution: pd.Series
    nearest_context_examples: list[dict[str, Any]]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def contains_banned_ttmatch(value: Any) -> bool:
    return "TTMATCH" in str(value).replace("\\", "/").upper()


def forbid_ttmatch_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    text_cols = [c for c in df.columns if c in {"corpus", "source", "source_path", "path", "relative_path"}]
    if not text_cols:
        return df.copy()
    banned = pd.Series(False, index=df.index)
    for col in text_cols:
        banned = banned | df[col].map(contains_banned_ttmatch).fillna(False)
    return df.loc[~banned].copy()


def normalize_family(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return "Zero"
    lowered = text.lower()
    for family in FAMILY_CLASSES:
        if lowered == family.lower():
            return family
    if "serve" in lowered:
        return "Serve"
    if "attack" in lowered or "smash" in lowered or "drive" in lowered:
        return "Attack"
    if "defen" in lowered or "clear" in lowered or "lob" in lowered:
        return "Defensive"
    if "control" in lowered or "drop" in lowered or "net" in lowered:
        return "Control"
    return "Zero"


def normalize_phase(value: Any, step_idx: Any | None = None) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text in PHASE_CLASSES:
        return text
    low = text.lower()
    if "serve" in low:
        return "serve_like"
    if "receive" in low:
        return "receive_like"
    if "third" in low:
        return "third_ball_like"
    if "fourth" in low:
        return "fourth_ball_like"
    if "terminal" in low:
        return "terminal_like"
    try:
        idx = int(step_idx)
    except Exception:
        idx = -1
    if idx == 0:
        return "serve_like"
    if idx == 1:
        return "receive_like"
    if idx == 2:
        return "third_ball_like"
    if idx == 3:
        return "fourth_ball_like"
    return "rally_like"


def normalize_side(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if not text or text in {"nan", "none", "<na>"}:
        return "missing"
    if text in {"left", "wide_left", "l"}:
        return "left"
    if text in {"right", "wide_right", "r"}:
        return "right"
    if text in {"center", "centre", "middle", "mid"}:
        return "center"
    try:
        numeric = float(text)
    except ValueError:
        return text.replace(" ", "_")
    if numeric < 0:
        return "left"
    if numeric > 0:
        return "right"
    return "center"


def normalize_depth(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if not text or text in {"nan", "none", "<na>"}:
        return "missing"
    if text in {"short", "low", "near"}:
        return "short"
    if text in {"mid", "middle", "medium"}:
        return "mid"
    if text in {"long", "deep", "high", "far"}:
        return "long"
    try:
        numeric = float(text)
    except ValueError:
        return text.replace(" ", "_")
    if numeric < 0.33:
        return "short"
    if numeric < 0.66:
        return "mid"
    return "long"


def normalize_event_table(raw: pd.DataFrame) -> pd.DataFrame:
    raw = forbid_ttmatch_rows(raw)
    if raw.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    out = pd.DataFrame(index=raw.index)
    out["corpus"] = raw.get("corpus", raw.get("source", raw.get("source_dataset", "unknown"))).astype(str)
    out["sequence_id"] = raw.get("sequence_id", pd.Series(np.arange(len(raw)), index=raw.index)).astype(str)
    out["step_idx"] = pd.to_numeric(
        raw.get("step_idx", raw.get("event_index", raw.get("source_row", pd.Series(0, index=raw.index)))),
        errors="coerce",
    ).fillna(0).astype(int)
    family = raw.get("coarse_family", raw.get("action_family", pd.Series("Zero", index=raw.index)))
    out["coarse_family"] = family.map(normalize_family)
    phase = raw.get("phase_code", raw.get("phase", pd.Series("", index=raw.index)))
    out["phase_code"] = [normalize_phase(v, i) for v, i in zip(phase, out["step_idx"])]
    out["landing_depth"] = raw.get("landing_depth", raw.get("landing_y", pd.Series(pd.NA, index=raw.index))).map(normalize_depth)
    out["landing_side"] = raw.get("landing_side", raw.get("landing_width_or_side", raw.get("landing_x", pd.Series(pd.NA, index=raw.index)))).map(normalize_side)
    if "has_spin" in raw.columns:
        out["has_spin"] = pd.to_numeric(raw["has_spin"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    else:
        out["has_spin"] = raw["spin"].notna().astype(int) if "spin" in raw.columns else 0
    if "has_speed" in raw.columns:
        out["has_speed"] = pd.to_numeric(raw["has_speed"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    else:
        out["has_speed"] = raw["speed"].notna().astype(int) if "speed" in raw.columns else 0
    out["source_weight"] = pd.to_numeric(raw.get("source_weight", pd.Series(1.0, index=raw.index)), errors="coerce").fillna(1.0).clip(lower=0.0)
    out["source_path"] = raw.get("source_path", raw.get("relative_path", pd.Series("", index=raw.index))).astype(str)
    out = forbid_ttmatch_rows(out)
    return out[EVENT_COLUMNS].sort_values(["corpus", "sequence_id", "step_idx"]).reset_index(drop=True)


def _read_v326_table(root: Path) -> tuple[pd.DataFrame, str] | None:
    v326_dir = root / "v326_masked_family_pretrain"
    parquet_path = v326_dir / "v326_external_event_table.parquet"
    csv_path = v326_dir / "v326_external_event_table.csv"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path), "v326_parquet"
        except Exception:
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False), "v326_csv"
    return None


def _read_v274_or_rebuild(root: Path) -> tuple[pd.DataFrame, str]:
    sample = root / "v274_clean_external_representation" / "v274_canonical_samples.csv"
    if sample.exists():
        return pd.read_csv(sample, low_memory=False), "v274_sample"
    try:
        import analysis_v274_clean_external_representation as v274

        _, canonical = v274.build_audit()
        return canonical, "v274_rebuilt"
    except Exception as exc:
        raise FileNotFoundError("No V326 event table or V274 clean canonical fallback is available") from exc


def load_event_table(root: Path = ROOT) -> tuple[pd.DataFrame, str]:
    loaded = _read_v326_table(root)
    if loaded is None:
        loaded = _read_v274_or_rebuild(root)
    raw, source = loaded
    events = normalize_event_table(raw)
    if events.empty:
        raise ValueError("Clean event table is empty after TTMATCH exclusion and normalization")
    return events, source


def context_key(family: Any, phase: Any, depth: Any, side: Any) -> str:
    return "|".join([normalize_family(family), normalize_phase(phase), normalize_depth(depth), normalize_side(side)])


def build_response_pairs(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "corpus",
                "sequence_id",
                "incoming_family",
                "incoming_phase",
                "incoming_depth",
                "incoming_side",
                "response_family",
                "response_phase",
                "sequence_position",
                "context_key",
                "source_weight",
            ]
        )
    e = normalize_event_table(events)
    e = e.sort_values(["corpus", "sequence_id", "step_idx"]).reset_index(drop=True)
    grouped = e.groupby(["corpus", "sequence_id"], sort=False, dropna=False)
    nxt = grouped[["coarse_family", "phase_code"]].shift(-1)
    same_sequence_next = nxt["coarse_family"].notna()
    pairs = pd.DataFrame(
        {
            "corpus": e.loc[same_sequence_next, "corpus"].to_numpy(),
            "sequence_id": e.loc[same_sequence_next, "sequence_id"].to_numpy(),
            "incoming_family": e.loc[same_sequence_next, "coarse_family"].to_numpy(),
            "incoming_phase": e.loc[same_sequence_next, "phase_code"].to_numpy(),
            "incoming_depth": e.loc[same_sequence_next, "landing_depth"].to_numpy(),
            "incoming_side": e.loc[same_sequence_next, "landing_side"].to_numpy(),
            "response_family": nxt.loc[same_sequence_next, "coarse_family"].map(normalize_family).to_numpy(),
            "response_phase": nxt.loc[same_sequence_next, "phase_code"].map(normalize_phase).to_numpy(),
            "sequence_position": e.loc[same_sequence_next, "step_idx"].astype(int).to_numpy(),
            "source_weight": e.loc[same_sequence_next, "source_weight"].astype(float).to_numpy(),
        }
    )
    pairs["context_key"] = [
        context_key(f, p, d, s)
        for f, p, d, s in zip(
            pairs["incoming_family"],
            pairs["incoming_phase"],
            pairs["incoming_depth"],
            pairs["incoming_side"],
        )
    ]
    return pairs.reset_index(drop=True)


def _weighted_counts(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=FAMILY_CLASSES)
    counts = (
        pairs.pivot_table(
            index="context_key",
            columns="response_family",
            values="source_weight",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(columns=FAMILY_CLASSES, fill_value=0.0)
        .sort_index()
    )
    return counts.astype(float)


def _svd_embeddings(matrix: pd.DataFrame, n_components: int) -> pd.DataFrame:
    max_components = max(1, min(int(n_components), matrix.shape[0], matrix.shape[1]))
    centered = matrix.to_numpy(dtype=float) - matrix.to_numpy(dtype=float).mean(axis=0, keepdims=True)
    try:
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        emb = u[:, :max_components] * s[:max_components]
    except np.linalg.LinAlgError:
        emb = np.zeros((len(matrix), max_components), dtype=float)
    if emb.shape[1] < int(n_components):
        pad = np.zeros((emb.shape[0], int(n_components) - emb.shape[1]), dtype=float)
        emb = np.hstack([emb, pad])
    return pd.DataFrame(
        emb,
        index=matrix.index,
        columns=[f"v327_ctx_emb_{i}" for i in range(int(n_components))],
    )


def _nearest_context_examples(embeddings: pd.DataFrame, matrix: pd.DataFrame, k: int = 8) -> list[dict[str, Any]]:
    if len(embeddings) <= 1:
        return [
            {
                "context_key": str(idx),
                "nearest_context_key": "",
                "distance": None,
                "top_response_family": str(matrix.loc[idx].idxmax()),
            }
            for idx in embeddings.index[:k]
        ]
    values = embeddings.to_numpy(dtype=float)
    out: list[dict[str, Any]] = []
    for i, idx in enumerate(embeddings.index[:k]):
        d = np.sqrt(((values - values[i]) ** 2).sum(axis=1))
        d[i] = np.inf
        j = int(np.argmin(d))
        out.append(
            {
                "context_key": str(idx),
                "nearest_context_key": str(embeddings.index[j]),
                "distance": float(d[j]),
                "top_response_family": str(matrix.loc[idx].idxmax()),
                "nearest_top_response_family": str(matrix.iloc[j].idxmax()),
            }
        )
    return out


def build_response_style_model(pairs: pd.DataFrame, n_components: int = 8, alpha: float = 0.5) -> ResponseStyleModel:
    counts = _weighted_counts(pairs)
    if counts.empty:
        counts = pd.DataFrame([[0.0] * len(FAMILY_CLASSES)], columns=FAMILY_CLASSES, index=["missing|rally_like|missing|missing"])
    smoothed = counts + float(alpha)
    matrix = smoothed.div(smoothed.sum(axis=1), axis=0)
    global_counts = counts.sum(axis=0) + float(alpha)
    global_distribution = global_counts / max(float(global_counts.sum()), 1e-12)
    pmi = np.log((matrix + 1e-12).div(global_distribution + 1e-12, axis=1))
    log_odds = np.log((matrix + 1e-6) / (1.0 - matrix + 1e-6))
    embeddings = _svd_embeddings(pmi, n_components=n_components)
    nearest = _nearest_context_examples(embeddings, matrix)
    return ResponseStyleModel(
        matrix=matrix,
        counts=counts,
        pmi=pmi,
        log_odds=log_odds,
        embeddings=embeddings,
        global_distribution=global_distribution,
        nearest_context_examples=nearest,
    )


def aicup_family_from_action(action_id: Any) -> str:
    try:
        value = int(action_id)
    except Exception:
        return "Zero"
    if value in {0, 15, 16, 17, 18}:
        return "Serve"
    if value in {1, 3, 8, 9, 12, 14}:
        return "Attack"
    if value in {2, 4, 5, 10, 11}:
        return "Control"
    if value in {6, 7, 13}:
        return "Defensive"
    return "Zero"


def phase_from_prefix_len(prefix_len: Any) -> str:
    try:
        n = int(prefix_len)
    except Exception:
        return "rally_like"
    return normalize_phase("", max(n - 1, 0))


def depth_from_point(point_id: Any) -> str:
    try:
        value = int(point_id)
    except Exception:
        return "missing"
    if value in {1, 2, 3}:
        return "short"
    if value in {4, 5, 6}:
        return "mid"
    if value in {7, 8, 9}:
        return "long"
    return "missing"


def side_from_position(position_id: Any) -> str:
    try:
        value = int(position_id)
    except Exception:
        return "missing"
    if value == 1:
        return "left"
    if value == 2:
        return "center"
    if value == 3:
        return "right"
    return "missing"


def _empty_counts() -> dict[str, dict[int, int]]:
    return {
        "actionId": {i: 0 for i in range(19)},
        "pointId": {i: 0 for i in range(10)},
        "spinId": {i: 0 for i in range(6)},
        "strengthId": {i: 0 for i in range(6)},
        "positionId": {i: 0 for i in range(5)},
    }


def _increment_counts(counts: dict[str, dict[int, int]], row: pd.Series) -> None:
    for field in counts:
        if field in row:
            value = int(row[field])
            counts[field][value] = counts[field].get(value, 0) + 1


def _prefix_base(group: pd.DataFrame, row_idx: int, counts: dict[str, dict[int, int]]) -> dict[str, Any]:
    current = group.iloc[row_idx]
    feats: dict[str, Any] = {
        "rally_uid": int(current["rally_uid"]),
        "match": int(current["match"]) if "match" in current else -1,
        "prefix_len": int(current["strikeNumber"]),
        "lag0_actionId": int(current["actionId"]),
        "lag0_pointId": int(current["pointId"]) if "pointId" in current else -1,
        "lag0_spinId": int(current["spinId"]) if "spinId" in current else -1,
        "lag0_strengthId": int(current["strengthId"]) if "strengthId" in current else -1,
        "lag0_positionId": int(current["positionId"]) if "positionId" in current else -1,
    }
    for field, values in counts.items():
        for value in sorted(values):
            feats[f"count_{field}_{value}"] = int(values[value])
    return feats


def build_train_prefix_table(train: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, group in train.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue
        counts = _empty_counts()
        for row_idx in range(len(group) - 1):
            _increment_counts(counts, group.iloc[row_idx])
            feats = _prefix_base(group, row_idx, counts)
            rows.append(feats)
    return pd.DataFrame(rows)


def build_test_prefix_table(test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, group in test.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        group = group.reset_index(drop=True)
        counts = _empty_counts()
        for row_idx in range(len(group)):
            _increment_counts(counts, group.iloc[row_idx])
        rows.append(_prefix_base(group, len(group) - 1, counts))
    return pd.DataFrame(rows)


def load_aicup_prefixes(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(root / "train.csv", low_memory=False)
    test = pd.read_csv(root / "test_new.csv", low_memory=False)
    return build_train_prefix_table(train), build_test_prefix_table(test)


def _fallback_feature_row(model: ResponseStyleModel) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    return model.global_distribution, pd.Series(0.0, index=FAMILY_CLASSES), pd.Series(0.0, index=FAMILY_CLASSES), pd.Series(0.0, index=model.embeddings.columns)


def project_prefix_features(prefixes: pd.DataFrame, model: ResponseStyleModel, split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in prefixes.itertuples(index=False):
        data = row._asdict()
        incoming_family = aicup_family_from_action(data.get("lag0_actionId"))
        incoming_phase = phase_from_prefix_len(data.get("prefix_len"))
        incoming_depth = depth_from_point(data.get("lag0_pointId"))
        incoming_side = side_from_position(data.get("lag0_positionId"))
        key = context_key(incoming_family, incoming_phase, incoming_depth, incoming_side)
        covered = key in model.matrix.index
        if covered:
            dist = model.matrix.loc[key]
            pmi = model.pmi.loc[key]
            odds = model.log_odds.loc[key]
            emb = model.embeddings.loc[key]
        else:
            dist, pmi, odds, emb = _fallback_feature_row(model)
        out: dict[str, Any] = {
            "split": split,
            "rally_uid": int(data.get("rally_uid")),
            "match": int(data.get("match", -1)),
            "prefix_len": int(data.get("prefix_len")),
            "v327_context_key": key,
            "v327_context_covered": int(covered),
            "v327_incoming_family": incoming_family,
            "v327_incoming_phase": incoming_phase,
            "v327_incoming_depth": incoming_depth,
            "v327_incoming_side": incoming_side,
        }
        for family in FAMILY_CLASSES:
            out[f"v327_resp_prob_{family}"] = float(dist[family])
            out[f"v327_resp_pmi_{family}"] = float(pmi[family])
            out[f"v327_resp_logodds_{family}"] = float(odds[family])
        for col, value in emb.items():
            out[col] = float(value)
        rows.append(out)
    features = pd.DataFrame(rows)
    coverage = {
        "split": split,
        "rows": int(len(features)),
        "covered_rows": int(features["v327_context_covered"].sum()) if not features.empty else 0,
        "coverage_rate": float(features["v327_context_covered"].mean()) if not features.empty else 0.0,
        "unique_contexts": int(features["v327_context_key"].nunique()) if not features.empty else 0,
        "covered_unique_contexts": int(features.loc[features["v327_context_covered"].eq(1), "v327_context_key"].nunique()) if not features.empty else 0,
    }
    return features, coverage


def build_report(
    *,
    events: pd.DataFrame,
    pairs: pd.DataFrame,
    model: ResponseStyleModel,
    source: str,
    train_coverage: dict[str, Any],
    test_coverage: dict[str, Any],
) -> dict[str, Any]:
    source_counts = events["corpus"].value_counts().sort_index().to_dict() if not events.empty else {}
    top_contexts = []
    if not model.counts.empty:
        totals = model.counts.sum(axis=1).sort_values(ascending=False).head(12)
        for key, total in totals.items():
            row = model.matrix.loc[key]
            top_contexts.append(
                {
                    "context_key": str(key),
                    "pair_count_weighted": float(total),
                    "top_response_family": str(row.idxmax()),
                    "top_response_prob": float(row.max()),
                }
            )
    return {
        "version": "v327_response_style_contrastive",
        "event_table_source": source,
        "event_rows": int(len(events)),
        "response_pairs": int(len(pairs)),
        "contexts": int(len(model.matrix)),
        "response_families": FAMILY_CLASSES,
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "train_coverage": train_coverage,
        "test_coverage": test_coverage,
        "nearest_context_examples": model.nearest_context_examples,
        "top_contexts": top_contexts,
        "artifacts": {
            "features": "v327_response_style_contrastive/v327_aicup_response_style_features.csv",
            "context_response_matrix": "v327_response_style_contrastive/v327_context_response_matrix.csv",
            "report_json": "v327_response_style_contrastive/v327_report.json",
            "report_md": "v327_response_style_contrastive/v327_report.md",
        },
        "submissions_written": 0,
        "ttmatch_content_rows_read": 0,
    }


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V327 Response-Style Contrastive Features",
        "",
        f"- Event table source: `{report['event_table_source']}`",
        f"- Event rows: `{report['event_rows']}`",
        f"- Response pairs: `{report['response_pairs']}`",
        f"- Contexts: `{report['contexts']}`",
        f"- Train coverage: `{report['train_coverage']['covered_rows']}/{report['train_coverage']['rows']}` ({report['train_coverage']['coverage_rate']:.4f})",
        f"- Test coverage: `{report['test_coverage']['covered_rows']}/{report['test_coverage']['rows']}` ({report['test_coverage']['coverage_rate']:.4f})",
        "- Submission files written: `0`",
        "- TTMATCH content rows read: `0`",
        "",
        "## Top Contexts",
        "",
    ]
    for item in report["top_contexts"][:10]:
        lines.append(
            f"- `{item['context_key']}`: pairs={item['pair_count_weighted']:.1f}, "
            f"top={item['top_response_family']} ({item['top_response_prob']:.4f})"
        )
    lines.extend(["", "## Nearest Context Examples", ""])
    for item in report["nearest_context_examples"][:8]:
        nearest = item.get("nearest_context_key", "")
        distance = item.get("distance")
        distance_text = "NA" if distance is None else f"{float(distance):.4f}"
        lines.append(f"- `{item['context_key']}` -> `{nearest}` distance={distance_text}")
    return "\n".join(lines) + "\n"


def write_outputs(
    outdir: Path,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    model: ResponseStyleModel,
    report: dict[str, Any],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    features = pd.concat([train_features, test_features], ignore_index=True)
    features.to_csv(outdir / "v327_aicup_response_style_features.csv", index=False)
    matrix = model.matrix.copy()
    matrix.insert(0, "context_key", matrix.index)
    for col in model.embeddings.columns:
        matrix[col] = model.embeddings[col].to_numpy(dtype=float)
    matrix.to_csv(outdir / "v327_context_response_matrix.csv", index=False)
    (outdir / "v327_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    (outdir / "v327_report.md").write_text(report_markdown(report), encoding="utf-8")


def main() -> None:
    events, source = load_event_table(ROOT)
    pairs = build_response_pairs(events)
    model = build_response_style_model(pairs)
    train_prefixes, test_prefixes = load_aicup_prefixes(ROOT)
    train_features, train_coverage = project_prefix_features(train_prefixes, model, split="train")
    test_features, test_coverage = project_prefix_features(test_prefixes, model, split="test")
    report = build_report(
        events=events,
        pairs=pairs,
        model=model,
        source=source,
        train_coverage=train_coverage,
        test_coverage=test_coverage,
    )
    write_outputs(OUTDIR, train_features, test_features, model, report)
    print(
        json.dumps(
            {
                "outdir": rel(OUTDIR),
                "event_rows": report["event_rows"],
                "response_pairs": report["response_pairs"],
                "contexts": report["contexts"],
                "train_coverage": train_coverage["coverage_rate"],
                "test_coverage": test_coverage["coverage_rate"],
                "submissions_written": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
