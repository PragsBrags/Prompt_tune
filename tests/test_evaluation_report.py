import json
from datetime import datetime, timezone

from omegaconf import OmegaConf

from evaluation.report import write_evaluation_report


def test_write_evaluation_report_creates_timestamped_json(tmp_path):
    cfg = OmegaConf.create(
        {
            "run": {"output_dir": str(tmp_path), "run_name": "baseline"},
            "data": {
                "dataset_name": "example/dataset",
                "config_name": "default",
                "split": "test",
                "source_column": "eng_Latn",
                "max_samples": 1,
            },
            "prompt": {"strategy": "zero_shot"},
        }
    )
    results = {"Nepali": {"metrics": {"bleu": 20.0}, "translations": []}}

    report_path = write_evaluation_report(
        cfg,
        "qwen2.5:7b",
        results,
        datetime.now(timezone.utc),
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_path.parent.parent.name == "evaluations"
    assert report["model"] == "qwen2.5:7b"
    assert report["results"] == results
