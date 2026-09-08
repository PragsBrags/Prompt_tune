from data.data_loader import load_translation_data
from evaluation.metrics import compute_all_metrics
from inference.generator import translate
from inference.model_loader import load_model
from prompting.strategies import get_prompt_strategy


def run_evaluation(cfg):
    predictions = []
    references = []
    sources = []

    dataset = load_translation_data(
        cfg.eval_data,
        cfg.run.seed
        )

    tokenizer, model = load_model(cfg.model)
    strategy = get_prompt_strategy(cfg.prompt.strategy)
    batch_size = cfg.eval_data.batch_size

    for start in range(0, len(dataset), batch_size):
        message_batch = []
        target_languages = []

        for index in range(start, min(start + batch_size, len(dataset))):
            sample = dataset[index]
            messages = strategy.builder(
                sample["source"],
                sample["source_language"],
                sample["target_language"],
            )
            message_batch.append(messages)
            target_languages.append(sample["target_language"])
            sources.append(sample["source"])
            references.append(sample["target"])

        generated = translate(
            model=model,
            tokenizer=tokenizer,
            message_batch=message_batch,
            cfg_model=cfg.model,
            strategy=strategy,
            target_languages=target_languages,
        )

        predictions.extend(generated)
        print(f"Processed {min(start + batch_size, len(dataset))}/{len(dataset)}")

    return compute_all_metrics(sources, predictions, references)