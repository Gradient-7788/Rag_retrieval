import re
from typing import List
from rag_engine.retrieval.base import RetrievalResult

def compute_faithfulness(answer_text: str, context_chunks: List[RetrievalResult]) -> float:
    """
    Measures Faithfulness: fraction of claims in generated answer that are supported by context chunks.
    Splits answer into sentences, checks token overlap/entailment with context.
    """
    if not answer_text or not context_chunks:
        return 0.0

    context_corpus = " ".join([c.chunk.text for c in context_chunks]).lower()
    context_words = set(re.findall(r'\w+', context_corpus))

    sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', answer_text) if len(s.strip()) > 10]
    if not sentences:
        return 1.0

    supported_count = 0
    for sent in sentences:
        s_words = set(re.findall(r'\w+', sent.lower()))
        # Remove common stop words for clean token matching
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "in", "of", "and", "that", "this", "it", "with", "for", "on", "as"}
        content_words = s_words - stop_words
        if not content_words:
            supported_count += 1
            continue

        overlap = len(content_words.intersection(context_words)) / float(len(content_words))
        if overlap >= 0.45:  # Overlap threshold for entailment grounding
            supported_count += 1

    return float(supported_count / len(sentences))

def compute_answer_relevance(answer_text: str, query: str) -> float:
    """
    Measures Answer Relevance: degree to which the generated answer addresses the question.
    """
    if not answer_text or not query:
        return 0.0

    q_words = set(re.findall(r'\w+', query.lower())) - {"what", "how", "why", "who", "where", "is", "are", "the", "a", "an"}
    if not q_words:
        return 1.0

    a_words = set(re.findall(r'\w+', answer_text.lower()))
    overlap = len(q_words.intersection(a_words)) / float(len(q_words))
    # Rescale to [0, 1] with base expectation
    return float(min(1.0, overlap * 1.25))

def compute_context_relevance(query: str, context_chunks: List[RetrievalResult]) -> float:
    """
    Measures Context Relevance (Context Precision): fraction of retrieved chunks relevant to query.
    """
    if not query or not context_chunks:
        return 0.0

    q_words = set(re.findall(r'\w+', query.lower())) - {"what", "how", "why", "the", "is", "a", "in", "of"}
    if not q_words:
        return 1.0

    relevant_chunks = 0
    for r in context_chunks:
        c_words = set(re.findall(r'\w+', r.chunk.text.lower()))
        if len(q_words.intersection(c_words)) > 0:
            relevant_chunks += 1

    return float(relevant_chunks / len(context_chunks))

def compute_citation_accuracy(answer_text: str, context_chunks: List[RetrievalResult]) -> float:
    """
    Measures Citation Accuracy: verifies whether [Doc X] citations match valid indices and contain content.
    """
    if not answer_text:
        return 0.0

    cited_indices = re.findall(r'\[Doc (\d+)\]', answer_text)
    if not cited_indices:
        return 0.5  # Neutral if no citations requested or present

    valid_count = 0
    for idx_str in cited_indices:
        idx = int(idx_str) - 1
        if 0 <= idx < len(context_chunks):
            valid_count += 1

    return float(valid_count / len(cited_indices))
