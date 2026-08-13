from datasets import load_dataset


def load_translation_data(dataset_cfg):
    load_kwargs = {"split": dataset_cfg.split}

    if getattr(dataset_cfg, "config_name", None):
        dataset = load_dataset(
            dataset_cfg.dataset_name,
            dataset_cfg.config_name,
            **load_kwargs,
        )
    else:
        dataset = load_dataset(dataset_cfg.dataset_name, **load_kwargs)

    return dataset