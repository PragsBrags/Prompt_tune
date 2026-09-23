"""Launch the configured pipeline with one fixed Hugging Face model name."""

import os
import sys
from pathlib import Path


def run_model(model_name: str) -> None:
    """Run ``src/cli.py`` while preserving every config value except the name."""
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"

    # config.yaml's local paths are intentionally relative to src.
    os.chdir(src_dir)
    sys.path.insert(0, str(src_dir))
    sys.argv.append(f"model.name={model_name}")

    from cli import main

    main()
