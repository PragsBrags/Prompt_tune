from datasets import load_dataset, concatenate_datasets

def load_translation_data(dataset_cfg,seed):
    datasets = []
    max_samples = getattr(dataset_cfg, "max_samples", None)

    for direction in dataset_cfg.directions:    
        ds = load_dataset(
        "csv",
        data_files = dataset_cfg.dataset_name,
        name=direction.dataset_config,
        split=dataset_cfg.split,
        revision=dataset_cfg.revision
        )

        if max_samples is not None and len(ds) > max_samples:
            ds = ds.shuffle(seed=seed).select(range(max_samples))

        ds = ds.map(
            lambda row:{
                "source": row[direction.source_column],
                "target": row[direction.target_column],
                "source_language": direction.source_language,
                "target_language": direction.target_language,
            },
            remove_columns=ds.column_names,
        )
        datasets.append(ds)

    dataset = concatenate_datasets(datasets)

    if dataset_cfg.shuffle:
        dataset=dataset.shuffle(seed=seed)

    return dataset