from __future__ import annotations

import json
from pathlib import Path

from analysis_v257_coachai_schema_helpers import forbid_ttmatch_path


ROOT = Path(".")
COACHAI_ROOT = ROOT / "external_data" / "CoachAI-Projects-main"
OUTDIR = ROOT / "v257_coachai_shuttlenet_audit"


def find_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            forbid_ttmatch_path(path)
            paths.append(path)
    return sorted(paths)


def find_extensionless_checkpoints(root: Path) -> list[Path]:
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        forbid_ttmatch_path(path)
        if path.name in {"encoder", "decoder", "config"}:
            candidates.append(path)
    return sorted(candidates)


def filter_shuttlenet_checkpoint_like(paths: list[Path]) -> list[Path]:
    shuttle_paths = []
    for path in paths:
        parts = {part.lower() for part in path.parts}
        if "shuttlenet" not in parts and "stroke forecasting" not in str(path).replace("\\", "/").lower():
            continue
        siblings = {p.name for p in path.parent.iterdir() if p.is_file()}
        if {"encoder", "decoder", "config"}.issubset(siblings):
            shuttle_paths.append(path)
    return sorted(shuttle_paths)


def main() -> None:
    if not COACHAI_ROOT.exists():
        raise FileNotFoundError(f"Missing CoachAI root: {COACHAI_ROOT}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    stroke_root = COACHAI_ROOT / "Stroke Forecasting"
    shuttlenet_root = stroke_root / "ShuttleNet"
    extensionless = find_extensionless_checkpoints(COACHAI_ROOT)
    shuttle_extensionless = filter_shuttlenet_checkpoint_like(extensionless)
    torch_checkpoints = find_files(COACHAI_ROOT, (".pt", ".pth", ".ckpt"))
    shuttle_torch_checkpoints = [path for path in torch_checkpoints if "shuttlenet" in str(path).replace("\\", "/").lower()]
    report = {
        "coachai_root": str(COACHAI_ROOT),
        "stroke_forecasting_exists": stroke_root.exists(),
        "shuttlenet_code_exists": shuttlenet_root.exists(),
        "torch_checkpoints": [str(path) for path in torch_checkpoints],
        "shuttlenet_torch_checkpoints": [str(path) for path in shuttle_torch_checkpoints],
        "extensionless_checkpoint_like": [str(path) for path in extensionless],
        "shuttlenet_extensionless_checkpoint_like": [str(path) for path in shuttle_extensionless],
        "csv_files": [str(path) for path in find_files(COACHAI_ROOT, (".csv",))[:200]],
        "parquet_files": [str(path) for path in find_files(COACHAI_ROOT, (".parquet",))[:200]],
    }
    report["has_direct_checkpoint"] = bool(report["shuttlenet_torch_checkpoints"] or report["shuttlenet_extensionless_checkpoint_like"])
    report["default_path"] = "load_checkpoint" if report["has_direct_checkpoint"] else "retrain_from_coachai_data"

    (OUTDIR / "v257_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTDIR / "v257_audit_report.md").write_text(
        "# V257 CoachAI ShuttleNet Audit\n\n"
        f"- ShuttleNet code exists: {report['shuttlenet_code_exists']}\n"
        f"- Direct checkpoint found: {report['has_direct_checkpoint']}\n"
        f"- Default path: {report['default_path']}\n"
        f"- Torch checkpoint count: {len(report['torch_checkpoints'])}\n"
        f"- Extensionless checkpoint-like count: {len(report['extensionless_checkpoint_like'])}\n",
        encoding="utf-8",
    )
    print(json.dumps({"outdir": str(OUTDIR), "default_path": report["default_path"]}))


if __name__ == "__main__":
    main()
