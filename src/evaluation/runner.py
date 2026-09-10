from datetime import datetime, timezone

from inference.generator import translate
from evaluation.metrics import compute_all_metrics
from prompting.shot_prompts import (
    build_messages_3,
    build_messages_back_translation,
    build_messages_consistency_review,
    build_messages_cot_translation,
    build_messages_zero,
    extract_final_translation,
)
from evaluation.report import write_evaluation_report


def run_evaluation(cfg, dataset, model_name):

    started_at = datetime.now(timezone.utc)
    all_results = {}
    source_lang = cfg.data.get("source_language", "English")

    for target in cfg.data.target_languages:
        sources = []
        predictions = []
        references = []
        candidate_predictions = []
        back_translations = []

        for i in range(cfg.data.max_samples):
            sample = dataset[i]
            source = sample[cfg.data.source_column]
            reference = sample[target.column]

            if cfg.prompt.strategy == "zero_shot":
                messages = build_messages_zero(source, target.name, source_lang)

            elif cfg.prompt.strategy == "3_shot":
                messages = build_messages_3(source, target.name, source_lang)

            elif cfg.prompt.strategy == "decomposed_translation":
                messages = build_messages_cot_translation(source, source_lang, target.name)

            elif cfg.prompt.strategy == "back_translation":
                messages = build_messages_zero(source, target.name, source_lang)

            else:
                raise ValueError(f"Unsupported prompt strategy: {cfg.prompt.strategy}")

            generated = translate(model_name, messages)
            if cfg.prompt.strategy == "decomposed_translation":
                generated = extract_final_translation(generated)

            candidate_prediction = None
            
            back_translation = None
            if cfg.prompt.strategy == "back_translation":
                candidate_prediction = generated
                back_messages = build_messages_back_translation(generated, source_lang, target.name)
                back_translation = translate(model_name, back_messages)
                review_messages = build_messages_consistency_review(
                    source,
                    generated,
                    back_translation,
                    source_lang,
                    target.name,
                )
                generated = translate(model_name, review_messages)

            print(f"<{source}><{generated}><{reference}>")

            sources.append(source)
            predictions.append(generated)
            references.append(reference)
            candidate_predictions.append(candidate_prediction)
            back_translations.append(back_translation)

        scores = compute_all_metrics(sources, predictions, references)

        print(f"\n=== {source_lang} -> {target.name} | strategy={cfg.prompt.strategy} | n={cfg.data.max_samples} ===")
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
                    **({"candidate_prediction": candidate_prediction} if candidate_prediction is not None else {}),
                    **({"back_translation": back_translation} if back_translation is not None else {}),
                }
                for source, prediction, reference, candidate_prediction, back_translation in zip(
                    sources,
                    predictions,
                    references,
                    candidate_predictions,
                    back_translations,
                )
            ],
        }

    report_path = write_evaluation_report(cfg, model_name, all_results, started_at)
    print(f"Evaluation report written to {report_path}")
    return all_results
