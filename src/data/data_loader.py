from datasets import load_dataset

def load_translation_data(dataset_cfg,seed):

    dataset = load_dataset(
    dataset_cfg.dataset_name,
    dataset_cfg.language_pair,
    split=dataset_cfg.split,
    revision=dataset_cfg.revision
    )

    if dataset_cfg.shuffle:
        dataset=dataset.shuffle(seed=seed)

    return dataset