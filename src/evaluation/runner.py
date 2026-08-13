from inference.generator import translate
from evaluation.metrics import compute_all_metrics
from prompting.shot_prompts import build_messages_zero, build_messages_3


def run_evaluation(cfg, dataset, model_name):

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

        all_results[target.name] = scores

    return all_results