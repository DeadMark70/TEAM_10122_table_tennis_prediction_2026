from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "external_sources.yaml"


def main() -> None:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    sources = data.get("external_sources", {})
    rows = []
    for key, item in sorted(sources.items()):
        local_path = ROOT / item["local_path"]
        rows.append(
            {
                "key": key,
                "display_name": item["display_name"],
                "source_url": item["source_url"],
                "license": item["license"],
                "local_path": item["local_path"],
                "present_locally": local_path.exists(),
                "redistribution_in_repo": bool(item["redistribution_in_repo"]),
                "exact_label_mapping_to_aicup": bool(item["exact_label_mapping_to_aicup"]),
            }
        )
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
