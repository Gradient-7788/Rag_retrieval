from typing import List
import re
from rag_engine.chunking.base import BaseChunker, Chunk

class FixedWindowChunker(BaseChunker):
    def chunk(self, text: str, doc_id: str = "doc_0") -> List[Chunk]:
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        idx = 0
        chunk_count = 0
        while idx < len(text):
            chunk_text = text[idx : idx + self.chunk_size]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_count}",
                    doc_id=doc_id,
                    text=chunk_text,
                    metadata={"chunker": "fixed_window", "chunk_size": self.chunk_size, "overlap": self.chunk_overlap},
                    start_char=idx,
                    end_char=idx + len(chunk_text),
                )
            )
            chunk_count += 1
            idx += step
        return chunks

class SentenceChunker(BaseChunker):
    def chunk(self, text: str, doc_id: str = "doc_0") -> List[Chunk]:
        # Split text by sentence delimiters (. ! ?)
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_count = 0

        for sentence in sentences:
            if current_length + len(sentence) > self.chunk_size and current_chunk:
                c_text = " ".join(current_chunk)
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}_sent_{chunk_count}",
                        doc_id=doc_id,
                        text=c_text,
                        metadata={"chunker": "sentence", "sentences_count": len(current_chunk)},
                    )
                )
                chunk_count += 1
                # Handle overlap by keeping the last sentence if possible
                if self.chunk_overlap > 0 and len(current_chunk) > 1:
                    current_chunk = [current_chunk[-1]]
                    current_length = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += len(sentence) + 1

        if current_chunk:
            c_text = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_sent_{chunk_count}",
                    doc_id=doc_id,
                    text=c_text,
                    metadata={"chunker": "sentence", "sentences_count": len(current_chunk)},
                )
            )

        return chunks

class RecursiveChunker(BaseChunker):
    """Recursively splits by paragraphs (\n\n), lines (\n), sentences, then spaces."""
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, separators: List[str] = None):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str, doc_id: str = "doc_0") -> List[Chunk]:
        raw_chunks = self._split_text(text, self.separators)
        chunks = []
        for i, t in enumerate(raw_chunks):
            if t.strip():
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}_rec_{i}",
                        doc_id=doc_id,
                        text=t.strip(),
                        metadata={"chunker": "recursive", "chunk_size": self.chunk_size},
                    )
                )
        return chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)
        good_splits = []

        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if new_separators:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
                else:
                    final_chunks.append(s)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs = []
        current_doc = []
        total = 0
        for s in splits:
            len_s = len(s)
            if total + len_s + (len(separator) if current_doc else 0) > self.chunk_size:
                if total > 0:
                    docs.append(separator.join(current_doc))
                    while total > self.chunk_overlap and current_doc:
                        total -= len(current_doc[0]) + len(separator)
                        current_doc.pop(0)
            current_doc.append(s)
            total += len_s + (len(separator) if len(current_doc) > 1 else 0)
        if current_doc:
            docs.append(separator.join(current_doc))
        return docs

class ParentChildChunker(BaseChunker):
    """Creates large parent chunks for context, and small child chunks for fine-grained retrieval."""
    def __init__(self, parent_chunk_size: int = 1000, child_chunk_size: int = 250, chunk_overlap: int = 30):
        super().__init__(child_chunk_size, chunk_overlap)
        self.parent_chunk_size = parent_chunk_size

    def chunk(self, text: str, doc_id: str = "doc_0") -> List[Chunk]:
        parent_chunker = FixedWindowChunker(chunk_size=self.parent_chunk_size, chunk_overlap=self.chunk_overlap * 2)
        child_chunker = FixedWindowChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        parents = parent_chunker.chunk(text, doc_id=doc_id)
        all_chunks = []

        for p_idx, parent in enumerate(parents):
            parent_id = f"{doc_id}_parent_{p_idx}"
            parent.chunk_id = parent_id
            parent.metadata["is_parent"] = True
            all_chunks.append(parent)

            children = child_chunker.chunk(parent.text, doc_id=parent_id)
            for c_idx, child in enumerate(children):
                child.chunk_id = f"{parent_id}_child_{c_idx}"
                child.parent_id = parent_id
                child.metadata["is_parent"] = False
                all_chunks.append(child)

        return all_chunks

class SemanticChunker(BaseChunker):
    """Groups text by topic continuity using sentence semantic boundary detection."""
    def chunk(self, text: str, doc_id: str = "doc_0") -> List[Chunk]:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]
        if not sentences:
            return []

        chunks = []
        current_chunk = [sentences[0]]
        current_len = len(sentences[0])
        chunk_count = 0

        # Topic boundary heuristics based on paragraph breaks, length, and indicator transition words
        boundary_words = {"however", "furthermore", "moreover", "in conclusion", "on the other hand", "firstly", "secondly", "finally", "next", "in contrast"}

        for i in range(1, len(sentences)):
            sent = sentences[i]
            first_word = sent.split()[0].lower() if sent.split() else ""

            # Check if sentence signals strong topic boundary or size exceeds chunk size
            is_boundary = first_word in boundary_words or current_len >= self.chunk_size

            if is_boundary and current_chunk:
                c_text = " ".join(current_chunk)
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}_semantic_{chunk_count}",
                        doc_id=doc_id,
                        text=c_text,
                        metadata={"chunker": "semantic", "sentence_count": len(current_chunk)},
                    )
                )
                chunk_count += 1
                current_chunk = [sent]
                current_len = len(sent)
            else:
                current_chunk.append(sent)
                current_len += len(sent) + 1

        if current_chunk:
            c_text = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_semantic_{chunk_count}",
                    doc_id=doc_id,
                    text=c_text,
                    metadata={"chunker": "semantic", "sentence_count": len(current_chunk)},
                )
            )

        return chunks
