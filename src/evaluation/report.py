import json
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf


def write_evaluation_report(cfg, model_name: str, results: dict, started_at: datetime) -> Path:
    finished_at = datetime.now(timezone.utc)
    run_timestamp = started_at.strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    output_dir = Path(cfg.run.output_dir) / "evaluations" / f"{run_timestamp}_{cfg.run.run_name}"
    output_dir.mkdir(parents=True, exist_ok=False)

    report = {
        "run_name": cfg.run.run_name,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 3),
        "model": model_name,
        "dataset": {
            "name": cfg.data.dataset_name,
            "config_name": cfg.data.config_name,
            "split": cfg.data.split,
            "source_column": cfg.data.source_column,
        },
        "prompt": OmegaConf.to_container(cfg.prompt, resolve=True),
        "max_samples": cfg.data.max_samples,
        "results": results,
    }

    report_path = output_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path
