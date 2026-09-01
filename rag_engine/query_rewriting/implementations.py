from abc import ABC, abstractmethod
from typing import List

class BaseQueryRewriter(ABC):
    @abstractmethod
    def rewrite(self, query: str) -> List[str]:
        """Returns list of queries (including original or expanded/transformed queries)."""
        pass

class DirectQueryRewriter(BaseQueryRewriter):
    """Passes query through unmodified."""
    def rewrite(self, query: str) -> List[str]:
        return [query]

class HyDEQueryRewriter(BaseQueryRewriter):
    """
    Hypothetical Document Embeddings (HyDE).
    Generates hypothetical passage answering the query to retrieve semantically similar passages.
    """
    def __init__(self, llm_generator=None):
        self.llm_generator = llm_generator

    def rewrite(self, query: str) -> List[str]:
        if self.llm_generator:
            prompt = f"Write a brief technical passage that directly answers the question: '{query}'"
            hypothetical_doc = self.llm_generator.generate(prompt)
            return [hypothetical_doc]
        # Algorithmic synthetic fallback for fast offline benchmarking
        hypothetical_doc = f"Detailed explanation regarding {query}. Key technical details, definitions, background, context, and exact mechanics of {query} within the system."
        return [query, hypothetical_doc]

class MultiQueryRewriter(BaseQueryRewriter):
    """
    Generates multi-perspective reformulations of the original query.
    """
    def rewrite(self, query: str) -> List[str]:
        q_lower = query.lower()
        sub_queries = [query]
        sub_queries.append(f"What is the definition and core mechanism of {query}?")
        sub_queries.append(f"How does {query} work in practical implementation?")
        sub_queries.append(f"Key features and specification of {query}")
        return sub_queries

class StepBackQueryRewriter(BaseQueryRewriter):
    """
    Step-Back Prompting: Abstracts specific question to high-level concept query.
    """
    def rewrite(self, query: str) -> List[str]:
        words = query.strip().split()
        if len(words) > 3:
            # Abstract high level step back query
            abstract_q = " ".join(words[:4]) + " fundamental principles and concepts"
            return [query, abstract_q]
        return [query, f"{query} general concept and overview"]

class KeywordExpansionRewriter(BaseQueryRewriter):
    """
    Extracts key terminology and appends synonyms/related terms.
    """
    def rewrite(self, query: str) -> List[str]:
        keywords = [w for w in query.split() if len(w) > 3]
        expanded = f"{query} {' '.join(keywords)} specification architecture parameters"
        return [query, expanded]
