from datasets import load_dataset, concatenate_datasets


def _load_translation_dataset(dataset_cfg, dataset_path, seed, directions=None):
    """Load one CSV path using one or more mappings from ``dataset_cfg``."""
    datasets = []
    max_samples = getattr(dataset_cfg, "max_samples", None)
    directions = dataset_cfg.directions if directions is None else directions

    for direction in directions:
        ds = load_dataset(
            "csv",
            data_files=dataset_path,
            name=direction.dataset_config,
            split=dataset_cfg.split,
            revision=dataset_cfg.revision,
        )

        if max_samples is not None and len(ds) > max_samples:
            ds = ds.shuffle(seed=seed).select(range(max_samples))

        ds = ds.map(
            lambda row: {
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
        dataset = dataset.shuffle(seed=seed)

    return dataset


def load_translation_data(dataset_cfg, seed):
    """Load every configured direction from the dataset path.

    Indexing and training deliberately use the concatenated dataset. Evaluation
    uses :func:`load_translation_data_for_direction` so each direction can have
    independent generation artifacts and metrics.
    """
    return _load_translation_dataset(dataset_cfg, dataset_cfg.dataset_name, seed)


def load_translation_data_for_direction(dataset_cfg, direction, seed):
    """Load exactly one configured direction from an evaluation CSV."""
    return _load_translation_dataset(
        dataset_cfg,
        dataset_cfg.dataset_name,
        seed,
        directions=[direction],
    )


def load_train_and_validation_data(train_cfg, seed):
    """Return normalized train and validation datasets from ``train_data`` paths.

    Both files use the same direction mappings, split, revision, sampling, and
    shuffle settings. They must therefore expose the same configured columns.
    """
    train_dataset = _load_translation_dataset(
        train_cfg,
        train_cfg.dataset_name,
        seed,
    )
    validation_dataset = _load_translation_dataset(
        train_cfg,
        train_cfg.valid_name,
        seed,
    )
    return train_dataset, validation_dataset
