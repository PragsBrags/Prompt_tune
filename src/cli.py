import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed
from tracking.experiment import ExpLogger


VALID_RUN_MODES = {"evaluate", "train", "index"}
VALID_MODEL_SOURCES = {"base", "merged"}
VALID_PROMPT_STRATEGIES = {
    "zero_shot",
    "few_shot",
    "rag_few_shot",
    "cot_translation",
    "back_translation",
}


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
    if cfg.prompt.strategy not in VALID_PROMPT_STRATEGIES:
        raise ValueError(
            f"Unsupported prompt.strategy {cfg.prompt.strategy!r}. "
            f"Expected one of: {sorted(VALID_PROMPT_STRATEGIES)}"
        )

@hydra.main(
    version_base=None,
    config_path="../configs",
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
            cfg.prompt.strategy,
        ],
    )
    
    try:
        if cfg.run.mode == "evaluate":
            from evaluation.runner import run_evaluation

            logruns = ExpLogger("run_experiments")
            set_seed(cfg.run.seed, deterministic=True)
            results = run_evaluation(cfg)

            logruns.log_run(
                cfg.model.name,
                cfg.run.mode,
                cfg.eval_data.dataset_name,
                cfg.eval_data.directions[0].dataset_config,
                None,
                )
            
            print(results)

            logruns.log_eval(
                cfg.model.name,
                results
            )

        elif cfg.run.mode == "train":
            from training.sft import train_model
            from training.save import save_model

            logruns = ExpLogger("train_experiments")
            set_seed(cfg.run.seed, deterministic=True)

            model, tokenizer = train_model(cfg)

            print("training complete")

            save_model(model, tokenizer, cfg.run)

            logruns.log_run(
                        cfg.model.name,
                        cfg.run.mode,
                        cfg.train_data.dataset_name,
                        None,
                        cfg.training,
                                )

            print("training complete and model saved")

        elif cfg.run.mode == "index":
            from retrieval.index import build_index

            set_seed(cfg.run.seed, deterministic=True)

            build_index(cfg)

            print("indexing complete")

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
