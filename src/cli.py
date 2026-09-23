import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from transformers import set_seed
from tracking.experiment import ExpLogger


VALID_RUN_MODES = {"evaluate", "train", "index"}
VALID_MODEL_SOURCES = {"base", "merged"}
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def validate_config(cfg: DictConfig) -> None:
    """Fail before creating external runs or loading models for invalid dispatch."""
    if cfg.run.mode not in VALID_RUN_MODES:
        raise ValueError(
            f"Unsupported run.mode {cfg.run.mode!r}. "
            f"Expected one of: {sorted(VALID_RUN_MODES)}"
        )
    if cfg.model.source not in VALID_MODEL_SOURCES:
        raise ValueError(
            f"Unsupported model.source {cfg.model.source!r}. "
            f"Expected one of: {sorted(VALID_MODEL_SOURCES)}"
        )
    if cfg.run.mode in {"evaluate", "train"}:
        if not cfg.eval_data.directions:
            raise ValueError("Evaluation requires at least one eval_data.directions entry.")
        # Few-shot is always part of automatic evaluation. Validate every
        # direction before training or model loading so a missing map cannot
        # leave the pipeline half-finished.
        from prompting.shot_prompts import get_few_shot_examples

        for direction in cfg.eval_data.directions:
            get_few_shot_examples(
                cfg.prompt.examples,
                direction.source_language,
                direction.target_language,
            )


def log_evaluation_results(logruns, cfg, results) -> None:
    """Write one run record and per-direction records for every strategy."""
    evaluation_directions = [
        {
            "strategy": strategy,
            "dataset_config": result["dataset_config"],
            "direction": result["direction"],
        }
        for strategy, strategy_results in results.items()
        for result in strategy_results.values()
    ]
    logruns.log_run(
        cfg.model.name,
        "evaluate",
        cfg.eval_data.dataset_name,
        cfg.run.seed,
        evaluation_directions,
        None,
    )

    for strategy, strategy_results in results.items():
        for result in strategy_results.values():
            logruns.log_eval(
                cfg.model.name,
                result["source_column"],
                result["target_column"],
                strategy,
                cfg.model.source,
                result["scores"],
                direction_name=result["direction"],
                prediction_file=result["prediction_file"],
                scores_file=result["scores_file"],
            )

@hydra.main(
    version_base=None,
    config_path=str(CONFIG_DIR),
    config_name="config",
)
def main(cfg: DictConfig):
    validate_config(cfg)

    run = wandb.init(
        project = cfg.wandb.project,
        entity = cfg.wandb.entity,
        mode=cfg.wandb.mode,
        group=cfg.wandb.group,
        job_type=cfg.run.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[
            cfg.model.source,
            cfg.training.method,
            "all_prompt_strategies",
        ],
    )
    
    try:
        if cfg.run.mode == "evaluate":
            from evaluation.runner import run_evaluation

            logruns = ExpLogger("run_experiments")
            set_seed(cfg.run.seed, deterministic=True)
            results = run_evaluation(cfg)

            log_evaluation_results(logruns, cfg, results)

            print(results)

        elif cfg.run.mode == "train":
            from training.sft import train_model
            from training.save import save_model

            logruns = ExpLogger("train_experiments")
            set_seed(cfg.run.seed, deterministic=True)

            logruns.log_run(
                cfg.model.name,
                cfg.run.mode,
                cfg.train_data.dataset_name,
                cfg.run.seed,
                None,
                OmegaConf.to_container(cfg.training, resolve=True),
                )

            model, tokenizer = train_model(cfg)

            print("training complete")

            save_model(model, tokenizer, cfg.run)

            print("training complete and model saved")

            # The evaluation loader must read the exported merged model, not
            # the base model that was used to start fine-tuning. Release the
            # training model first so evaluation can reclaim GPU memory.
            del model, tokenizer
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            from evaluation.runner import run_evaluation

            evaluation_cfg = OmegaConf.create(
                OmegaConf.to_container(cfg, resolve=True)
            )
            evaluation_cfg.model.source = "merged"
            results = run_evaluation(evaluation_cfg)
            log_evaluation_results(
                ExpLogger("run_experiments"),
                evaluation_cfg,
                results,
            )
            print("fine-tuned model evaluation complete")

        elif cfg.run.mode == "index":
            from retrieval.index import build_index

            set_seed(cfg.run.seed, deterministic=True)

            build_index(cfg)

            print("indexing complete")

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
