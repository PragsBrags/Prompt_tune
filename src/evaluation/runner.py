import csv
import json
import re
from pathlib import Path

import wandb
from omegaconf import OmegaConf

from data.data_loader import load_translation_data_for_direction
from evaluation.metrics import compute_all_metrics
from inference.generator import translate
from inference.model_loader import load_model
from prompting.shot_prompts import (
    build_messages_3,
    build_messages_back_translation,
    build_messages_consistency_review,
    build_messages_cot_translation,
    build_messages_rag,
    build_messages_zero,
    extract_final_translation,
    get_few_shot_examples,
)
from retrieval.retriever import TranslationRetriever
from retrieval.index import ensure_index


PROMPT_STRATEGIES = (
    "zero_shot",
    "few_shot",
    "rag_few_shot",
    "cot_translation",
    "back_translation",
)


def direction_id(direction) -> str:
    """Return a stable, filename-safe identifier for one language direction."""
    def slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    return f"{slug(direction.source_language)}_to_{slug(direction.target_language)}"


def _write_direction_artifacts(cfg, direction, sources, predictions, references, scores):
    """Persist predictions and metrics for one direction without overwriting peers."""
    safe_model_name = cfg.model.name.replace("/", "__")
    name = "_".join(
        [
            safe_model_name,
            cfg.model.source,
            cfg.prompt.strategy,
            direction_id(direction),
        ]
    )
    output_dir = Path(cfg.eval_data.model_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{name}.csv"
    scores_path = output_dir / f"{name}.metrics.json"

    with prediction_path.open(mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["source", "prediction", "reference"])
        writer.writerows(zip(sources, predictions, references))

    with scores_path.open(mode="w", encoding="utf-8") as file:
        json.dump(
            {
                "strategy": cfg.prompt.strategy,
                "direction": direction_id(direction),
                "source_language": direction.source_language,
                "target_language": direction.target_language,
                "source_column": direction.source_column,
                "target_column": direction.target_column,
                "scores": scores,
            },
            file,
            indent=2,
        )

    return prediction_path, scores_path


def _build_messages(cfg, sample, retriever):
    source = sample["source"]
    source_language = sample["source_language"]
    target_language = sample["target_language"]

    if cfg.prompt.strategy == "zero_shot":
        return build_messages_zero(source, source_language, target_language)
    if cfg.prompt.strategy == "few_shot":
        examples = get_few_shot_examples(
            cfg.prompt.examples,
            source_language,
            target_language,
        )
        return build_messages_3(examples, source_language, target_language, source)
    if cfg.prompt.strategy == "rag_few_shot":
        if retriever is None:
            raise RuntimeError("Retriever is required for rag_few_shot strategy")
        examples = retriever.retrieve(source, source_language, target_language)
        return build_messages_rag(examples, source_language, target_language, source)
    if cfg.prompt.strategy == "cot_translation":
        return build_messages_cot_translation(source, source_language, target_language)
    if cfg.prompt.strategy == "back_translation":
        return build_messages_zero(source, source_language, target_language)
    raise ValueError(f"Unknown prompt strategy: {cfg.prompt.strategy}")


def _evaluate_direction(cfg, direction, model, tokenizer, retriever):
    """Generate, score, and persist one configured direction."""
    dataset = load_translation_data_for_direction(cfg.eval_data, direction, cfg.run.seed)
    predictions = []
    references = []
    sources = []
    batch_size = cfg.eval_data.batch_size

    for start in range(0, len(dataset), batch_size):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        message_batch = [_build_messages(cfg, sample, retriever) for sample in batch]
        generated = translate(model, tokenizer, message_batch, cfg.model)

        if cfg.prompt.strategy == "cot_translation":
            generated = [extract_final_translation(text) for text in generated]
        elif cfg.prompt.strategy == "back_translation":
            reviewed = []
            for sample, candidate in zip(batch, generated):
                back_translation = translate(
                    model,
                    tokenizer,
                    [
                        build_messages_back_translation(
                            candidate,
                            sample["source_language"],
                            sample["target_language"],
                        )
                    ],
                    cfg.model,
                )[0]
                review = translate(
                    model,
                    tokenizer,
                    [
                        build_messages_consistency_review(
                            sample["source"],
                            candidate,
                            back_translation,
                            sample["source_language"],
                            sample["target_language"],
                        )
                    ],
                    cfg.model,
                )[0]
                reviewed.append(extract_final_translation(review))
            generated = reviewed

        predictions.extend(generated)
        print(generated)
        sources.extend(batch["source"])
        references.extend(batch["target"])
        print(batch["target"])
        print(
            f"[{direction_id(direction)}] "
            f"Processed {min(start + batch_size, len(dataset))}/{len(dataset)}"
        )

    scores = compute_all_metrics(sources, predictions, references)
    prediction_path, scores_path = _write_direction_artifacts(
        cfg,
        direction,
        sources,
        predictions,
        references,
        scores,
    )

    name = direction_id(direction)
    wandb.log(
        {
            f"eval/{cfg.prompt.strategy}/{name}/{metric}": value
            for metric, value in scores.items()
            if value is not None
        }
    )
    return {
        "direction": name,
        "dataset_config": direction.dataset_config,
        "source_column": direction.source_column,
        "target_column": direction.target_column,
        "source_language": direction.source_language,
        "target_language": direction.target_language,
        "scores": scores,
        "prediction_file": str(prediction_path),
        "scores_file": str(scores_path),
    }


def _strategy_config(cfg, strategy):
    """Return a resolved config copy for one automatic prompting-strategy pass."""
    strategy_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    strategy_cfg.prompt.strategy = strategy
    return strategy_cfg


def run_evaluation(cfg):
    """Evaluate every prompting strategy and direction using one loaded model."""
    tokenizer, model = load_model(cfg.model)

    results = {}
    for strategy in PROMPT_STRATEGIES:
        strategy_cfg = _strategy_config(cfg, strategy)
        if strategy == "rag_few_shot":
            ensure_index(strategy_cfg)
            retriever = TranslationRetriever(strategy_cfg.rag)
        else:
            retriever = None

        strategy_results = {}
        for direction in strategy_cfg.eval_data.directions:
            name = direction_id(direction)
            if name in strategy_results:
                raise ValueError(
                    f"Duplicate evaluation direction {name!r}; each direction must be unique."
                )
            result = _evaluate_direction(
                strategy_cfg,
                direction,
                model,
                tokenizer,
                retriever,
            )
            result["strategy"] = strategy
            strategy_results[name] = result
        results[strategy] = strategy_results

    return results
