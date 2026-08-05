from data.data_loader import load_translation_data
from inference.model_loader import load_model
from evaluation.runner import run_evaluation

from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    dataset = load_translation_data(cfg.data)
    tokenizer, model = load_model(cfg.model)
    results = run_evaluation(cfg,dataset,tokenizer,model)

    print(results)

if __name__ == "__main__":
    main()