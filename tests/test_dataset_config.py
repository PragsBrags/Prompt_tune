from pathlib import Path

from omegaconf import OmegaConf


def test_default_dataset_config_matches_hf_schema():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    cfg = OmegaConf.load(config_path)

    assert cfg.data.dataset_name == "ai4bharat/IN22-Gen"
    assert cfg.data.config_name == "default"
    assert cfg.data.split == "test"
    assert cfg.data.source_column == "eng_Latn"
    assert cfg.data.target_languages[0].column == "mai_Deva"
    assert cfg.data.target_languages[1].column == "npi_Deva"
