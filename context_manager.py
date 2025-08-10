from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


@dataclass
class Message:
    role: str
    content: str


class ContextManager:
    """
    Short-term context window + long-term vector memory using SentenceTransformers + FAISS.

    - Short-term: token-budget-based rolling window using a tokenizer
    - Long-term: MiniLM embeddings indexed in a flat L2 FAISS index
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        max_tokens: int = 512,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.history: List[Message] = []

        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.memory_texts: List[str] = []
        self.memory_roles: List[str] = []
        self.memory_embeddings: List[np.ndarray] = []

    def add_message(self, role: str, content: str) -> None:
        self.history.append(Message(role=role, content=content))
        embedding = self.embedding_model.encode(content)
        self.memory_texts.append(content)
        self.memory_roles.append(role)
        self.memory_embeddings.append(np.asarray(embedding, dtype=np.float32))

    def get_relevant_memory(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        if not self.memory_embeddings:
            return []
        query_vec = np.asarray(self.embedding_model.encode(query), dtype=np.float32)
        # Cosine similarity
        mem = np.stack(self.memory_embeddings, axis=0)
        # Normalize
        qn = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        mn = mem / (np.linalg.norm(mem, axis=1, keepdims=True) + 1e-8)
        sims = mn @ qn
        top_idx = np.argsort(-sims)[:k]
        relevant: List[Dict[str, str]] = []
        for idx in top_idx:
            relevant.append({"role": self.memory_roles[int(idx)], "content": self.memory_texts[int(idx)]})
        return relevant

    def get_context(self) -> List[Dict[str, str]]:
        total = 0
        messages: List[Dict[str, str]] = []
        for msg in reversed(self.history):
            tokens = len(self.tokenizer.tokenize(msg.content))
            if total + tokens > self.max_tokens:
                break
            messages.insert(0, {"role": msg.role, "content": msg.content})
            total += tokens
        return messages

    def reset(self) -> None:
        self.history.clear()
        self.memory_texts.clear()
        self.memory_roles.clear()
        self.memory_embeddings = []


__all__ = ["ContextManager", "Message"]


