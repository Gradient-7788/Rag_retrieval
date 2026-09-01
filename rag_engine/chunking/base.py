from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = {}
    parent_id: str = None
    start_char: int = 0
    end_char: int = 0

class BaseChunker(ABC):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, text: str, doc_id: str = "doc_0") -> List[Chunk]:
        pass
