import chromadb
from hydra.utils import to_absolute_path
from sentence_transformers import SentenceTransformer

from data.data_loader import load_translation_data


def build_index(cfg):
    """
    Index every directed pair defined in cfg.train_data.directions.

    Required stored fields:
    - document: source sentence vector is based on this
    - target: paired translation for the few-shot example
    - source_language / target_language: retrieval filters
    """
    dataset = load_translation_data(
        cfg.train_data,
        cfg.run.seed,
    )

    index_path = to_absolute_path(cfg.rag.index_path)

    client = chromadb.PersistentClient(path=index_path)

    if cfg.rag.rebuild:
        try:
            client.delete_collection(cfg.rag.collection_name)
        except (ValueError, chromadb.errors.NotFoundError):
            pass

    collection = client.get_or_create_collection(
        name=cfg.rag.collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Must be exactly the same model used in retriever.py.
    embedder = SentenceTransformer(cfg.rag.embedding_model)

    batch_size = cfg.rag.index_batch_size

    for start in range(0, len(dataset), batch_size):
        batch = dataset.select(
            range(start, min(start + batch_size, len(dataset)))
        )

        sources = list(batch["source"])
        targets = list(batch["target"])
        source_languages = list(batch["source_language"])
        target_languages = list(batch["target_language"])

        # E5 uses "passage:" for indexed texts.
        embeddings = embedder.encode(
            [f"passage: {source}" for source in sources],
            normalize_embeddings=True,
        ).tolist()

        ids = [
            f"{start + i}:{source_lang}:{target_lang}"
            for i, (source_lang, target_lang) in enumerate(
                zip(source_languages, target_languages)
            )
        ]

        metadatas = [
            {
                "target": target,
                "source_language": source_lang,
                "target_language": target_lang,
            }
            for target, source_lang, target_lang in zip(
                targets,
                source_languages,
                target_languages,
            )
        ]

        collection.upsert(
            ids=ids,
            documents=sources,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"Indexed {min(start + batch_size, len(dataset))}/{len(dataset)}")

    print(f"Index complete: {collection.count()} records")