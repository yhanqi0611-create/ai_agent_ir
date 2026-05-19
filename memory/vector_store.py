from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _simple_tokenize(text: str) -> List[str]:
    import re

    return [t for t in re.findall(r"[A-Za-z0-9_]+", text.lower()) if t]


class HashEmbedding:
    """
    Pure-Python, deterministic embedding that avoids native deps.
    Not as strong as transformer embeddings, but robust and good enough for follow-up Q&A bootstrapping.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def encode(self, texts: List[str], *, normalize: bool = True) -> np.ndarray:
        import hashlib

        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, txt in enumerate(texts):
            toks = _simple_tokenize(txt)
            for tok in toks:
                h = hashlib.sha256(tok.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if (h[4] & 1) == 0 else -1.0
                mat[i, idx] += sign
        if normalize:
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
        return mat


@dataclass(frozen=True)
class MemoryConfig:
    persist_dir: str = os.getenv("MEMORY_PERSIST_DIR", "memory/store")
    collection_name: str = os.getenv("MEMORY_COLLECTION", "research_items")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "hash-1024")
    backend: str = os.getenv("MEMORY_BACKEND", "auto")  # auto|chroma|local


class VectorMemory:
    def __init__(self, cfg: Optional[MemoryConfig] = None):
        self.cfg = cfg or MemoryConfig()
        os.makedirs(self.cfg.persist_dir, exist_ok=True)

        self._backend = (self.cfg.backend or "auto").lower()
        # Default to hash embeddings (no native deps). If you want transformer embeddings,
        # switch to your preferred embedding stack in a later iteration.
        if self.cfg.embedding_model.startswith("hash-"):
            dim = int(self.cfg.embedding_model.split("-", 1)[1])
            self._embedder = HashEmbedding(dim=dim)
        else:
            # Backwards-compatible: treat as hash dims if misconfigured.
            self._embedder = HashEmbedding(dim=1024)

        self._local_dir = Path(self.cfg.persist_dir)
        self._local_docs_path = self._local_dir / f"{self.cfg.collection_name}.jsonl"
        self._local_emb_path = self._local_dir / f"{self.cfg.collection_name}.emb.npy"
        self._local_ids_path = self._local_dir / f"{self.cfg.collection_name}.ids.json"

        self._chroma_collection = None
        if self._backend in {"auto", "chroma"}:
            try:
                import chromadb  # noqa
                # For Chroma backend, we store documents and let Chroma handle embeddings if configured.
                # To avoid native deps, we do not attach embedding_function here.
                client = chromadb.PersistentClient(path=str(self._local_dir / "chroma"))
                self._chroma_collection = client.get_or_create_collection(
                    name=self.cfg.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                self._backend = "chroma"
            except Exception:
                if self._backend == "chroma":
                    raise
                self._backend = "local"

    def add_texts(
        self,
        *,
        ids: List[str],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if len(ids) != len(texts):
            raise ValueError("ids and texts length mismatch")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas and texts length mismatch")

        if self._backend == "chroma" and self._chroma_collection is not None:
            self._chroma_collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
            return

        # Local backend: append JSONL + maintain embedding matrix
        metas = metadatas or [{} for _ in texts]
        self._local_docs_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing ids + embeddings (if any)
        if self._local_ids_path.exists():
            existing_ids = json.loads(self._local_ids_path.read_text(encoding="utf-8"))
        else:
            existing_ids = []

        existing_emb = None
        if self._local_emb_path.exists():
            existing_emb = np.load(self._local_emb_path)

        new_emb = self._embedder.encode(texts, normalize=True)

        # Upsert semantics: if id exists, replace its doc+meta+embedding; else append
        id_to_idx = {i: idx for idx, i in enumerate(existing_ids)}

        # Load all docs into memory only if we need to replace; else append-only
        docs: List[Dict[str, Any]] = []
        if self._local_docs_path.exists():
            with self._local_docs_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        docs.append(json.loads(line))

        for i, (doc_id, text, meta, emb_row) in enumerate(zip(ids, texts, metas, new_emb)):
            rec = {"id": doc_id, "text": text, "metadata": meta}
            if doc_id in id_to_idx:
                idx = id_to_idx[doc_id]
                docs[idx] = rec
                if existing_emb is not None and idx < existing_emb.shape[0]:
                    existing_emb[idx] = emb_row
            else:
                id_to_idx[doc_id] = len(existing_ids)
                existing_ids.append(doc_id)
                docs.append(rec)
                if existing_emb is None:
                    existing_emb = emb_row.reshape(1, -1)
                else:
                    existing_emb = np.vstack([existing_emb, emb_row.reshape(1, -1)])

        # Persist
        self._local_docs_path.write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in docs) + "\n",
            encoding="utf-8",
        )
        self._local_ids_path.write_text(json.dumps(existing_ids, ensure_ascii=False), encoding="utf-8")
        np.save(self._local_emb_path, existing_emb if existing_emb is not None else np.zeros((0, 1)))

    def query(
        self,
        *,
        query_text: str,
        n_results: int = 6,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._backend == "chroma" and self._chroma_collection is not None:
            res = self._chroma_collection.query(query_texts=[query_text], n_results=n_results, where=where)
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            return [{"text": d, "metadata": (m or {}), "distance": dist} for d, m, dist in zip(docs, metas, dists)]

        # Local backend cosine distance
        if not self._local_docs_path.exists() or not self._local_emb_path.exists():
            return []
        emb = np.load(self._local_emb_path)
        if emb.size == 0:
            return []
        qv = self._embedder.encode([query_text], normalize=True)[0]

        # cosine similarity since vectors normalized
        sims = emb @ qv
        top_idx = np.argsort(-sims)[: max(1, n_results)]

        docs: List[Dict[str, Any]] = []
        with self._local_docs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))

        out: List[Dict[str, Any]] = []
        for idx in top_idx:
            rec = docs[int(idx)]
            meta = rec.get("metadata") or {}
            if where:
                # minimal filter: all key==value must match
                ok = True
                for k, v in where.items():
                    if meta.get(k) != v:
                        ok = False
                        break
                if not ok:
                    continue
            out.append({"text": rec.get("text", ""), "metadata": meta, "distance": float(1.0 - sims[int(idx)])})
        return out

