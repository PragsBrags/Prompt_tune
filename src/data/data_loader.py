from datasets import load_dataset

def load_translation_data(dataset_cfg):

    dataset = load_dataset(
    dataset_cfg.dataset_name,
    dataset_cfg.language_pair,
    split=dataset_cfg.split
    )

    return dataset