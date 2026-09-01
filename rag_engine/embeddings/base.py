from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseEmbeddingModel(ABC):
    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Returns 2D numpy array of shape (len(texts), dimension) normalized."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Returns 1D numpy array of shape (dimension,) normalized."""
        pass
