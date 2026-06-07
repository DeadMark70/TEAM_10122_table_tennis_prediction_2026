"""V415 clean external representation trainer.

Builds deterministic coarse token and sequence embeddings from V414 outputs
without exposing exact AICUP labels or excluded external sources.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "v414_masked_pretraining_inputs"
OUTDIR = ROOT / "v415_clean_external_representation"

RANDOM_STATE = 415
MAX_COMPONENTS = 32
TOKEN_COLUMNS = {
    "fam": "token_family",
    "phase": "phase",
    "depth": "landing_depth_bin",
    "side": "landing_side_bin",
    "speed": "speed_bin",
    "spin": "spin_bin",
}
FORBIDDEN_COLUMNS = {"actionId", "pointId", "serverGetPoint", "spinId", "strengthId", "positionId"}
EXCLUDED_SOURCES = {"TTMATCH", "TT-MatchDynamics", "sonytabletennis"}


def _normalize_text(value: Any, default: str = "unknown") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return default
    return text.replace(" ", "_")


def _source_mask(frame: pd.DataFrame) -> pd.Series:
    if "source_dataset" not in frame.columns:
        return pd.Series([True] * len(frame), index=frame.index)
    source = frame["source_dataset"].map(_normalize_text)
    mask = ~source.isin(EXCLUDED_SOURCES)
    mask &= ~source.str.contains("ttmatch", case=False, na=False)
    mask &= ~source.str.contains("sony", case=False, na=False)
    return mask


def _clean_sources(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    mask = _source_mask(frame)
    clean = frame.loc[mask].copy()
    if "source_dataset" in clean.columns:
        clean["source_dataset"] = clean["source_dataset"].map(_normalize_text)
    return clean.reset_index(drop=True), int((~mask).sum())


def _assert_no_forbidden_columns(outputs: dict[str, pd.DataFrame]) -> None:
    for name, frame in outputs.items():
        overlap = FORBIDDEN_COLUMNS & set(frame.columns)
        if overlap:
            raise ValueError(f"{name} contains forbidden exact AICUP columns: {sorted(overlap)}")


def _event_tokens(row: pd.Series) -> list[str]:
    tokens: list[str] = []
    for prefix, column in TOKEN_COLUMNS.items():
        tokens.append(f"{prefix}={_normalize_text(row.get(column, 'unknown'))}")
    return tokens


def _build_sequence_documents(pretrain_sequences: pd.DataFrame) -> pd.DataFrame:
    if pretrain_sequences.empty:
        return pd.DataFrame(columns=["source_dataset", "sequence_id", "event_count", "document"])

    frame = pretrain_sequences.copy()
    for column in ["source_dataset", "sequence_id"]:
        frame[column] = frame[column].map(_normalize_text)
    if "event_index" in frame.columns:
        frame["_event_index"] = pd.to_numeric(frame["event_index"], errors="coerce").fillna(0)
    else:
        frame["_event_index"] = np.arange(len(frame))
    frame["_tokens"] = frame.apply(_event_tokens, axis=1)
    frame = frame.sort_values(["source_dataset", "sequence_id", "_event_index"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for (source, sequence_id), group in frame.groupby(["source_dataset", "sequence_id"], sort=True):
        tokens = [token for tokens_for_event in group["_tokens"] for token in tokens_for_event]
        rows.append(
            {
                "source_dataset": source,
                "sequence_id": sequence_id,
                "event_count": int(len(group)),
                "document": " ".join(tokens),
            }
        )
    return pd.DataFrame(rows)


def _fit_embeddings(documents: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    vectorizer = CountVectorizer(lowercase=False, token_pattern=r"(?u)[^\s]+")
    matrix = vectorizer.fit_transform(documents.tolist())
    tokens = vectorizer.get_feature_names_out()
    n_docs, n_features = matrix.shape

    if n_features == 0:
        raise ValueError("No token features were built from pretrain_sequences")

    usable_components = min(MAX_COMPONENTS, max(1, n_features - 1), max(1, n_docs - 1))
    embedding_cols = [f"svd_{idx:02d}" for idx in range(usable_components)]

    if n_features > 1 and n_docs > 1:
        svd = TruncatedSVD(n_components=usable_components, random_state=RANDOM_STATE)
        sequence_values = svd.fit_transform(matrix)
        token_values = svd.components_.T
        explained = [float(value) for value in svd.explained_variance_ratio_]
    else:
        dense = matrix.toarray().astype(float)
        sequence_values = dense[:, :1]
        token_values = np.ones((n_features, 1), dtype=float)
        explained = [1.0]

    token_embeddings = pd.DataFrame({"token": tokens})
    for idx, col in enumerate(embedding_cols):
        token_embeddings[col] = token_values[:, idx]
    token_embeddings = token_embeddings.sort_values("token").reset_index(drop=True)

    sequence_embeddings = pd.DataFrame(sequence_values, columns=embedding_cols)
    fit_report = {
        "n_docs": int(n_docs),
        "n_features": int(n_features),
        "svd_components": int(usable_components),
        "embedding_columns": embedding_cols,
        "explained_variance_ratio": explained,
    }
    return token_embeddings, sequence_embeddings, embedding_cols, fit_report


def _source_summary(sequence_embeddings: pd.DataFrame, embedding_cols: list[str]) -> pd.DataFrame:
    if sequence_embeddings.empty:
        return pd.DataFrame(columns=["source_dataset", "sequence_count", "event_count", "unique_token_count", *embedding_cols])

    rows: list[dict[str, Any]] = []
    for source, group in sequence_embeddings.groupby("source_dataset", sort=True):
        unique_tokens = set()
        for document in group["document"]:
            unique_tokens.update(str(document).split())
        row: dict[str, Any] = {
            "source_dataset": source,
            "sequence_count": int(len(group)),
            "event_count": int(group["event_count"].sum()),
            "unique_token_count": int(len(unique_tokens)),
        }
        for col in embedding_cols:
            row[col] = float(group[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_clean_representations(
    pretrain_sequences: pd.DataFrame,
    masked_event_examples: pd.DataFrame,
    landing_intent_examples: pd.DataFrame,
    physics_reconstruction_examples: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build V415 representation artifacts from V414 dataframes."""

    clean_sequences, excluded_sequences = _clean_sources(pretrain_sequences)
    clean_masked, excluded_masked = _clean_sources(masked_event_examples)
    clean_landing, excluded_landing = _clean_sources(landing_intent_examples)
    clean_physics, excluded_physics = _clean_sources(physics_reconstruction_examples)

    documents = _build_sequence_documents(clean_sequences)
    if documents.empty:
        raise ValueError("pretrain_sequences has no clean rows after source exclusion")

    token_embeddings, sequence_values, embedding_cols, fit_report = _fit_embeddings(documents["document"])
    sequence_embeddings = pd.concat(
        [documents[["source_dataset", "sequence_id", "event_count", "document"]].reset_index(drop=True), sequence_values],
        axis=1,
    )
    source_summary = _source_summary(sequence_embeddings, embedding_cols)

    sequence_embeddings = sequence_embeddings.drop(columns=["document"])
    outputs = {
        "token_embeddings": token_embeddings,
        "sequence_embeddings": sequence_embeddings,
        "source_embedding_summary": source_summary,
    }
    _assert_no_forbidden_columns(outputs)

    report = {
        "version": "V415",
        "random_state": RANDOM_STATE,
        "token_prefixes": list(TOKEN_COLUMNS.keys()),
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
        "excluded_sources": sorted(EXCLUDED_SOURCES),
        "input_rows": {
            "pretrain_sequences": int(len(pretrain_sequences)),
            "masked_event_examples": int(len(masked_event_examples)),
            "landing_intent_examples": int(len(landing_intent_examples)),
            "physics_reconstruction_examples": int(len(physics_reconstruction_examples)),
        },
        "clean_rows": {
            "pretrain_sequences": int(len(clean_sequences)),
            "masked_event_examples": int(len(clean_masked)),
            "landing_intent_examples": int(len(clean_landing)),
            "physics_reconstruction_examples": int(len(clean_physics)),
        },
        "excluded_rows": {
            "pretrain_sequences": excluded_sequences,
            "masked_event_examples": excluded_masked,
            "landing_intent_examples": excluded_landing,
            "physics_reconstruction_examples": excluded_physics,
        },
        "output_rows": {name: int(len(frame)) for name, frame in outputs.items()},
        "source_counts": clean_sequences.groupby("source_dataset").size().to_dict() if not clean_sequences.empty else {},
        **fit_report,
    }
    return outputs, report


def run_pipeline(*, input_dir: Path = INPUT_DIR, outdir: Path = OUTDIR) -> dict[str, Any]:
    input_dir = Path(input_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pretrain_sequences = pd.read_csv(input_dir / "pretrain_sequences.csv", low_memory=False)
    masked_event_examples = pd.read_csv(input_dir / "masked_event_examples.csv", low_memory=False)
    landing_intent_examples = pd.read_csv(input_dir / "landing_intent_examples.csv", low_memory=False)
    physics_reconstruction_examples = pd.read_csv(input_dir / "physics_reconstruction_examples.csv", low_memory=False)

    outputs, report = build_clean_representations(
        pretrain_sequences,
        masked_event_examples,
        landing_intent_examples,
        physics_reconstruction_examples,
    )
    output_paths: dict[str, str] = {}
    for name, frame in outputs.items():
        path = outdir / f"{name}.csv"
        frame.to_csv(path, index=False)
        output_paths[name] = str(path)

    report_path = outdir / "pretraining_report.json"
    report["outputs"] = {**output_paths, "pretraining_report": str(report_path)}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
