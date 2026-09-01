import numpy as np
from typing import List, Set

def recall_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """Recall@K = (retrieved_in_top_k ∩ ground_truth) / len(ground_truth)."""
    if not ground_truth_ids:
        return 0.0
    top_k_retrieved = set(retrieved_ids[:k])
    gt_set = set(ground_truth_ids)
    hits = len(top_k_retrieved.intersection(gt_set))
    return float(hits / len(gt_set))

def precision_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """Precision@K = (retrieved_in_top_k ∩ ground_truth) / k."""
    if k <= 0:
        return 0.0
    top_k_retrieved = retrieved_ids[:k]
    gt_set = set(ground_truth_ids)
    hits = sum(1 for r_id in top_k_retrieved if r_id in gt_set)
    return float(hits / k)

def mrr(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
    """MRR = 1 / rank of first relevant item (1-indexed)."""
    gt_set = set(ground_truth_ids)
    for rank, r_id in enumerate(retrieved_ids, start=1):
        if r_id in gt_set:
            return float(1.0 / rank)
    return 0.0

def hit_rate_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """Hit Rate@K = 1.0 if any ground_truth item is in top_k retrieved, else 0.0."""
    top_k_set = set(retrieved_ids[:k])
    gt_set = set(ground_truth_ids)
    return 1.0 if len(top_k_set.intersection(gt_set)) > 0 else 0.0

def dcg_at_k(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain@K."""
    relevances = np.asarray(relevances, dtype=np.float64)[:k]
    if not relevances.size:
        return 0.0
    return float(np.sum((2**relevances - 1) / np.log2(np.arange(2, relevances.size + 2))))

def ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """NDCG@K = DCG@K / IDCG@K."""
    gt_set = set(ground_truth_ids)
    relevances = [1.0 if r_id in gt_set else 0.0 for r_id in retrieved_ids[:k]]
    actual_dcg = dcg_at_k(relevances, k)

    # Ideal DCG: perfect ranking with all relevant documents first
    ideal_relevances = [1.0] * min(len(gt_set), k)
    ideal_dcg = dcg_at_k(ideal_relevances, k)

    if ideal_dcg == 0.0:
        return 0.0
    return float(actual_dcg / ideal_dcg)

def average_precision(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 10) -> float:
    """Average Precision@K for MAP computation."""
    gt_set = set(ground_truth_ids)
    if not gt_set:
        return 0.0

    score = 0.0
    num_hits = 0.0
    for rank, r_id in enumerate(retrieved_ids[:k], start=1):
        if r_id in gt_set:
            num_hits += 1.0
            score += num_hits / rank
    return float(score / min(len(gt_set), k))
