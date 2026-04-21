from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

try:
    import chromadb
except ImportError:
    chromadb = None


ROOT_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT_DIR / "data" / "chroma_facebook_policy"
CHUNKS_JSONL = ROOT_DIR / "data" / "facebook_policy_chunks.jsonl"
COLLECTION_NAME = "facebook_meta_policy_vi"
EMBEDDING_MODEL = "text-embedding-3-large"

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 40
RRF_K = 60
DENSE_WEIGHT = 1.0
SPARSE_WEIGHT = 1.0
SPARSE_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "ban",
    "bang",
    "bi",
    "cac",
    "cach",
    "cho",
    "co",
    "cua",
    "duoc",
    "dung",
    "gi",
    "la",
    "lam",
    "meta",
    "facebook",
    "instagram",
    "messenger",
    "nao",
    "nay",
    "nhu",
    "noi",
    "trong",
    "tren",
    "ve",
    "va",
    "voi",
}


class FacebookPolicyRetriever:
    """Hybrid retriever: BM25 sparse + Chroma dense + Reciprocal Rank Fusion."""

    def __init__(
        self,
        chroma_dir: Path | str = CHROMA_DIR,
        chunks_jsonl: Path | str = CHUNKS_JSONL,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        rrf_k: int = RRF_K,
        dense_weight: float = DENSE_WEIGHT,
        sparse_weight: float = SPARSE_WEIGHT,
    ) -> None:
        load_dotenv(ROOT_DIR / ".env")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env before querying the retriever.")

        self.embedding_model = embedding_model
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.openai_client = OpenAI()
        self.chroma_client = None
        self.collection = None

        self.documents = self._load_chunks(Path(chunks_jsonl))
        self.doc_by_id = {doc["chunk_id"]: doc for doc in self.documents}
        self.bm25 = BM25Okapi([self._tokenize(doc["bm25_text"]) for doc in self.documents])

        if chromadb is not None:
            try:
                self.chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
                self.collection = self.chroma_client.get_collection(collection_name)
            except Exception:
                self.chroma_client = None
                self.collection = None

    @staticmethod
    def _strip_accents(text: str) -> str:
        decomposed = unicodedata.normalize("NFD", text)
        stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
        return stripped.replace("đ", "d").replace("Đ", "D")

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", text).lower()
        folded = cls._strip_accents(normalized)
        tokens = re.findall(r"[0-9a-zA-ZÀ-ỹ]+", normalized)
        folded_tokens = re.findall(r"[0-9a-zA-Z]+", folded)
        return [
            token
            for token in tokens + folded_tokens
            if len(token) > 1 and cls._strip_accents(token).lower() not in SPARSE_STOPWORDS
        ]

    @staticmethod
    def _load_chunks(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Missing chunks file: {path}")

        documents: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                metadata = dict(row["metadata"])
                text = row["text"]
                chunk_id = metadata.get("chunk_id") or row["id"]
                bm25_text = "\n".join(
                    [
                        metadata.get("title", ""),
                        metadata.get("heading_path", ""),
                        metadata.get("source_type", ""),
                        text,
                    ]
                )
                documents.append(
                    {
                        "chunk_id": chunk_id,
                        "text": text,
                        "metadata": metadata,
                        "bm25_text": bm25_text,
                    }
                )
        if not documents:
            raise RuntimeError(f"No chunks found in {path}")
        return documents

    def _embed_query(self, query: str) -> list[float]:
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=query,
        )
        return response.data[0].embedding

    def _dense_search(self, query: str, candidate_k: int) -> dict[str, dict[str, Any]]:
        if self.collection is None:
            return {}

        n_results = min(candidate_k, self.collection.count())
        if n_results <= 0:
            return {}

        results = self.collection.query(
            query_embeddings=[self._embed_query(query)],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        dense_hits: dict[str, dict[str, Any]] = {}
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for rank, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances), start=1):
            chunk_id = metadata.get("chunk_id", "")
            dense_hits[chunk_id] = {
                "chunk_id": chunk_id,
                "text": document,
                "metadata": metadata,
                "dense_rank": rank,
                "dense_distance": float(distance),
                "dense_score": 1.0 - float(distance),
            }
        return dense_hits

    def _sparse_search(self, query: str, candidate_k: int) -> dict[str, dict[str, Any]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {}

        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)[:candidate_k]

        sparse_hits: dict[str, dict[str, Any]] = {}
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if score <= 0.0:
                continue
            doc = self.documents[idx]
            sparse_hits[doc["chunk_id"]] = {
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
                "sparse_rank": rank,
                "sparse_score": score,
            }
        return sparse_hits

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
    ) -> list[dict[str, Any]]:
        candidate_k = max(candidate_k, top_k * 8)
        dense_hits = self._dense_search(query, candidate_k)
        sparse_hits = self._sparse_search(query, candidate_k)

        fused: dict[str, dict[str, Any]] = {}
        for chunk_id in set(dense_hits) | set(sparse_hits):
            base = dense_hits.get(chunk_id) or sparse_hits.get(chunk_id)
            if base is None:
                continue
            metadata = dict(base["metadata"])
            dense_rank = dense_hits.get(chunk_id, {}).get("dense_rank")
            sparse_rank = sparse_hits.get(chunk_id, {}).get("sparse_rank")

            rrf_score = 0.0
            if dense_rank is not None:
                rrf_score += self.dense_weight / (self.rrf_k + int(dense_rank))
            if sparse_rank is not None:
                rrf_score += self.sparse_weight / (self.rrf_k + int(sparse_rank))

            fused[chunk_id] = {
                "text": base["text"],
                "chunk_id": chunk_id,
                "title": metadata.get("title", ""),
                "canonical_url": metadata.get("canonical_url", ""),
                "source_doc_id": metadata.get("source_doc_id", ""),
                "source_file": metadata.get("source_file", ""),
                "heading_path": metadata.get("heading_path", ""),
                "score": rrf_score,
                "rrf_score": rrf_score,
                "dense_rank": dense_rank,
                "dense_score": dense_hits.get(chunk_id, {}).get("dense_score"),
                "dense_distance": dense_hits.get(chunk_id, {}).get("dense_distance"),
                "sparse_rank": sparse_rank,
                "sparse_score": sparse_hits.get(chunk_id, {}).get("sparse_score"),
                "retrieval_method": "hybrid_bm25_dense_rrf" if self.collection is not None else "bm25_sparse_only",
                "metadata": metadata,
            }

        return sorted(
            fused.values(),
            key=lambda item: (
                item["rrf_score"],
                item["dense_score"] if item["dense_score"] is not None else -1.0,
                item["sparse_score"] if item["sparse_score"] is not None else -1.0,
            ),
            reverse=True,
        )[:top_k]
