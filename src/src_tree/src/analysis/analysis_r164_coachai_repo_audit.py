from pathlib import Path
import runpy


ROOT_SCRIPT = Path(__file__).resolve().parents[2] / "analysis_r164_coachai_repo_audit.py"


if __name__ == "__main__":
    runpy.run_path(str(ROOT_SCRIPT), run_name="__main__")
