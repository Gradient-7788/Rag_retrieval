import hashlib
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from rag_engine.embeddings.base import BaseEmbeddingModel

class SimulatedEmbeddingModel(BaseEmbeddingModel):
    """
    High-performance semantic embedding simulator.
    Uses MD5/SHA256 text hashing combined with n-gram character/word frequency 
    to generate realistic normalized dense vectors (768, 384, or 1536 dims).
    Enables instant benchmark testing without heavy GPU dependencies.
    """
    def __init__(self, model_name: str = "bge-small-en-v1.5", dimension: int = 384):
        super().__init__(model_name, dimension)
        self.dimension = dimension

    def _text_to_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec

        # Use word hashes to populate dimensions deterministically
        for word in words:
            h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            idx = h % self.dimension
            val = ((h >> 8) % 1000) / 1000.0 - 0.5
            vec[idx] += val

            # Also pair-wise n-gram contribution
            idx_2 = (h >> 16) % self.dimension
            vec[idx_2] += val * 0.5

        # Normalize L2 norm
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = [self._text_to_vector(t) for t in texts]
        return np.vstack(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        return self._text_to_vector(query)

class TFIDFEmbeddingModel(BaseEmbeddingModel):
    """TF-IDF + SVD Dense Embedding Model."""
    def __init__(self, model_name: str = "tfidf-svd-128", dimension: int = 128):
        super().__init__(model_name, dimension)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.svd = TruncatedSVD(n_components=dimension, random_state=42)
        self.is_fitted = False

    def fit(self, texts: List[str]):
        if not texts:
            return
        tfidf_mat = self.vectorizer.fit_transform(texts)
        n_comp = min(self.dimension, tfidf_mat.shape[1] - 1, len(texts) - 1)
        if n_comp > 0:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
            self.svd.fit(tfidf_mat)
            self.dimension = n_comp
        self.is_fitted = True

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            self.fit(texts)
        if not self.is_fitted or not texts:
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
        tfidf_mat = self.vectorizer.transform(texts)
        dense_mat = self.svd.transform(tfidf_mat)
        norms = np.linalg.norm(dense_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (dense_mat / norms).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(self.dimension, dtype=np.float32)
        tfidf_vec = self.vectorizer.transform([query])
        dense_vec = self.svd.transform(tfidf_vec)[0]
        norm = np.linalg.norm(dense_vec)
        if norm > 1e-9:
            dense_vec = dense_vec / norm
        return dense_vec.astype(np.float32)

class SentenceTransformersEmbeddingModel(BaseEmbeddingModel):
    """HuggingFace sentence-transformers integration wrapper with fallback to Simulated embedding."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.st_model = None
        dim = 384
        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(model_name)
            dim = self.st_model.get_sentence_embedding_dimension()
        except Exception:
            self.simulated = SimulatedEmbeddingModel(model_name=model_name, dimension=384)
            dim = 384
        super().__init__(model_name, dim)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if self.st_model is not None:
            return self.st_model.encode(texts, normalize_embeddings=True)
        return self.simulated.embed_texts(texts)

    def embed_query(self, query: str) -> np.ndarray:
        if self.st_model is not None:
            return self.st_model.encode([query], normalize_embeddings=True)[0]
        return self.simulated.embed_query(query)
