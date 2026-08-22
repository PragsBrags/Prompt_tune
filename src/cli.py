import hydra
from omegaconf import DictConfig
from transformers import set_seed
from tracking.experiment import ExpLogger

@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig):
    
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


if __name__ == "__main__":
    main()