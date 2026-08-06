from evaluation.runner import run_evaluation
from training.sft import train_model
from training.save import save_model

from omegaconf import DictConfig, OmegaConf
import hydra



@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.run.mode == "evaluate":
        results = run_evaluation(cfg)
        print(results)

    if cfg.run.mode == "train":
        model, tokenizer = train_model(cfg)
        save_model(model, tokenizer, cfg.run)

if __name__ == "__main__":
    main()