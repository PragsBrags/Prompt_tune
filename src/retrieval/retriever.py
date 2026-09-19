from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from hydra.utils import to_absolute_path
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedExample:
    id: str
    source: str
    target: str
    score: float

class TranslationRetriever:
    def __init__(self, rag_cfg):
        self.top_k = rag_cfg.top_k
        self.candidate_k = rag_cfg.candidate_k
        self.client = chromadb.PersistentClient(path=to_absolute_path(rag_cfg.index_path))
        self.collection = self.client.get_collection(name=rag_cfg.collection_name)
        self.embedding = SentenceTransformer(rag_cfg.embedding_model)

    def retrieve(
            self,
            source_text,
            source_lang,
            target_lang,
            ) -> list[RetrievedExample]:
        
        query_embedding = self.embedding.encode(
            [f"query: {source_text}"],
            normalize_embeddings=True,
        ).tolist()

        results = self.collection.query(
            query_embeddings = query_embedding,
            n_results = self.candidate_k,
            where ={
                "$and": [
                    {"source_language": source_lang},
                    {"target_language": target_lang},
                ]
            },
            include = ["documents", "metadatas", "distances"],
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        examples : list[RetrievedExample] = []

        for record_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            score = 1.0 - distance
            retrieved_source = document
            retrieved_target = metadata["target"]

            examples.append(
                RetrievedExample(
                    id = record_id,
                    source = retrieved_source,
                    target = retrieved_target,
                    score = score
                )
            )

            if len(examples) == self.top_k:
                break
        return examples