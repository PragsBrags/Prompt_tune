import hydra
from omegaconf import DictConfig
from transformers import set_seed


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig):

    if cfg.run.mode == "evaluate":
        from evaluation.runner import run_evaluation

        set_seed(cfg.run.seed, deterministic=True)
        results = run_evaluation(cfg)
        print(results)

    elif cfg.run.mode == "train":
        from training.sft import train_model
        from training.save import save_model

        set_seed(cfg.run.seed, deterministic=True)

        model, tokenizer = train_model(cfg)
        print("training complete")

        save_model(model, tokenizer, cfg.run)

        print("training complete and model saved")


if __name__ == "__main__":
    main()