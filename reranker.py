"""
reranker.py
Cross-encoder re-ranking layer for the manufacturing RAG pipeline.

Drop this file next to app.py and import it there.
Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~80 MB, pre-trained on MS MARCO (Bing search relevance)
  - Returns a float relevance score per (query, doc) pair
  - Higher score = more relevant
"""

from __future__ import annotations
from functools import lru_cache
from typing import List, Tuple

from langchain_core.documents import Document

try:
    from sentence_transformers import CrossEncoder
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)   # load once per process, reuse across Streamlit reruns
def _load_model() -> "CrossEncoder":
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "sentence-transformers is not installed.\n"
            "Run:  pip install sentence-transformers"
        )
    return CrossEncoder(RERANKER_MODEL)


def rerank(
    query: str,
    doc_score_pairs: List[Tuple[Document, float]],
    top_n: int = 5,
) -> List[Tuple[Document, float, float]]:
    """
    Re-rank FAISS results using a cross-encoder.

    Args:
        query:           The user's question.
        doc_score_pairs: Output of FAISS similarity_search_with_score —
                         list of (Document, faiss_distance).
        top_n:           How many re-ranked results to return.

    Returns:
        List of (Document, faiss_distance, rerank_score) sorted by
        rerank_score descending, truncated to top_n.
    """
    model  = _load_model()
    docs   = [doc for doc, _ in doc_score_pairs]
    scores = [s   for _, s  in doc_score_pairs]
    texts  = [doc.page_content for doc in docs]

    rerank_scores: List[float] = model.predict(
        [(query, t) for t in texts]
    ).tolist()

    combined = list(zip(docs, scores, rerank_scores))
    combined.sort(key=lambda x: x[2], reverse=True)

    return combined[:top_n]
