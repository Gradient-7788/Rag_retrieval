from abc import ABC, abstractmethod
from typing import List, Tuple
from pydantic import BaseModel
from rag_engine.chunking.base import Chunk

class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float
    rank: int

class BaseRetriever(ABC):
    @abstractmethod
    def index(self, chunks: List[Chunk]):
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        pass
