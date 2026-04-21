"""
finbert_features.py

FinBERT-based semantic similarity between consecutive filings.
Uses mean-pooled embeddings from ProsusAI/finbert (via sentence-transformers).

Device auto-detection: MPS (Apple M1/M2) > CUDA > CPU.
Embeddings are cached to data/raw/finbert_cache/{ticker}_embeddings.npy
to avoid recomputation on re-runs.
"""
import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        logger.info(f"Loading ProsusAI/finbert on device={device}")
        _model = SentenceTransformer("ProsusAI/finbert", device=device)
        return _model
    except Exception as e:
        logger.warning(f"Could not load FinBERT: {e}. finbert_cosine_sim will be NaN.")
        return None


def _embed_text(model, text, chunk_words=390):
    """
    Chunk text into word-level windows (~390 words ≈ 512 FinBERT tokens),
    embed each chunk, and return the mean-pooled document embedding.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    words = text.split()
    chunks = [
        " ".join(words[i: i + chunk_words])
        for i in range(0, len(words), chunk_words)
        if words[i: i + chunk_words]
    ]
    if not chunks:
        return None
    try:
        embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.mean(axis=0)
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


def compute_finbert_similarity(df, text_col, ticker, cache_dir="data/raw/finbert_cache"):
    """
    Compute FinBERT cosine similarity between consecutive filings for one ticker.

    Parameters
    ----------
    df : pd.DataFrame  (sorted by filed_at, single ticker)
    text_col : str     column name containing the filing text
    ticker : str       used to name the cache file
    cache_dir : str    directory for embedding cache files

    Returns
    -------
    pd.Series of finbert_cosine_sim, aligned to df.index.
    First row is always NaN (no previous filing).
    """
    model = _load_model()
    if model is None:
        return pd.Series([np.nan] * len(df), index=df.index)

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{ticker}_embeddings.npy")

    texts = df[text_col].fillna("").tolist()
    n = len(texts)

    # Load existing cache
    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=True).item()
        except Exception:
            cached = {}
    else:
        cached = {}

    embeddings = []
    cache_updated = False
    for text in texts:
        # Use first 500 chars as a fast cache key
        key = str(hash(text[:500]))
        if key in cached:
            embeddings.append(cached[key])
        else:
            emb = _embed_text(model, text)
            cached[key] = emb
            embeddings.append(emb)
            cache_updated = True

    if cache_updated:
        np.save(cache_path, cached)

    # Compute consecutive cosine similarities
    sims = [np.nan]
    for i in range(1, n):
        e_prev, e_curr = embeddings[i - 1], embeddings[i]
        if e_prev is None or e_curr is None:
            sims.append(np.nan)
            continue
        try:
            norm = np.linalg.norm(e_prev) * np.linalg.norm(e_curr)
            cos = float(np.dot(e_prev, e_curr) / (norm + 1e-10))
            sims.append(cos)
        except Exception:
            sims.append(np.nan)

    return pd.Series(sims, index=df.index)
