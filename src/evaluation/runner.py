from inference.generator import translate
from evaluation.metrics import compute_all_metrics
from data.data_loader import load_translation_data
from inference.model_loader import load_model
from prompting.shot_prompts import build_messages_zero, build_messages_3


def run_evaluation(cfg):

    prediction = []
    references = []
    sources = []

    dataset = load_translation_data(
        cfg.eval_data,
        cfg.run.seed
        )

    tokenizer, model = load_model(cfg.model)

    batch_size = cfg.eval_data.batch_size

    for i in range(0, len(dataset), batch_size):

        message_batch = []

        for j in range(i, min(i + batch_size, len(dataset))):

            sample = dataset[j]

            source = sample["source"]
            target = sample["target"]

            if cfg.prompt.strategy == "zero_shot":
                messages = build_messages_zero(
                    source,
                    sample["source_language"],
                    sample["target_language"]
                )

            elif cfg.prompt.strategy == "3_shot":
                messages = build_messages_3(source)

            else:
                raise ValueError(
                    f"Unknown prompt strategy: {cfg.prompt.strategy}"
                )

            message_batch.append(messages)
            sources.append(source)
            references.append(target)

        generated = translate(
            model,
            tokenizer,
            message_batch,
            cfg.model
            )
        
        prediction.extend(generated)
        print(f"Processed {min(i + batch_size, len(dataset))}/{len(dataset)}")

    score = compute_all_metrics(sources, prediction, references)

    return score