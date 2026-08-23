from datetime import datetime, timezone

from inference.generator import translate
from evaluation.metrics import compute_all_metrics
from prompting.shot_prompts import build_messages_zero, build_messages_3
from evaluation.report import write_evaluation_report


def run_evaluation(cfg, dataset, model_name):

    started_at = datetime.now(timezone.utc)
    all_results = {}

    for target in cfg.data.target_languages:
        sources = []
        predictions = []
        references = []

        for i in range(cfg.data.max_samples):
            sample = dataset[i]
            english = sample[cfg.data.source_column]
            reference = sample[target.column]

            if cfg.prompt.strategy == "zero_shot":
                messages = build_messages_zero(english, target.name)

            elif cfg.prompt.strategy == "3_shot":
                messages = build_messages_3(english, target.name)

            generated = translate(model_name, messages)

            print(f"<{english}><{generated}><{reference}>")

            sources.append(english)
            predictions.append(generated)
            references.append(reference)

        scores = compute_all_metrics(sources, predictions, references)

        print(f"\n=== English -> {target.name} | strategy={cfg.prompt.strategy} | n={cfg.data.max_samples} ===")
        for metric_name, score in scores.items():
            print(f"<{metric_name}><{score}>")

        all_results[target.name] = {
            "target_column": target.column,
            "translation_count": len(predictions),
            "metrics": scores,
            "translations": [
                {
                    "source": source,
                    "prediction": prediction,
                    "reference": reference,
                }
                for source, prediction, reference in zip(sources, predictions, references)
            ],
        }

    report_path = write_evaluation_report(cfg, model_name, all_results, started_at)
    print(f"Evaluation report written to {report_path}")
    return all_results
