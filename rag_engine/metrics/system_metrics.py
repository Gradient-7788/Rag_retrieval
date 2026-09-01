import psutil
import tracemalloc
import time
import numpy as np
from typing import List, Dict, Any
from pydantic import BaseModel

class SystemTelemetry(BaseModel):
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    total_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    cost_per_1000_queries_usd: float
    throughput_qps: float
    tokens_per_second: float
    memory_rss_mb: float
    peak_memory_mb: float

class SystemTracker:
    def __init__(self):
        self.process = psutil.Process()
        tracemalloc.start()

    def profile_execution(
        self,
        latencies_ms: List[float],
        retrieval_latencies_ms: List[float],
        generation_latencies_ms: List[float],
        total_prompt_tokens: int,
        total_completion_tokens: int,
        estimated_cost: float,
        num_queries: int,
    ) -> SystemTelemetry:
        if not latencies_ms:
            latencies_ms = [0.0]

        p50 = float(np.percentile(latencies_ms, 50))
        p90 = float(np.percentile(latencies_ms, 90))
        p99 = float(np.percentile(latencies_ms, 99))
        total_time_sec = sum(latencies_ms) / 1000.0 if sum(latencies_ms) > 0 else 0.001

        qps = num_queries / total_time_sec if total_time_sec > 0 else 0.0
        tps = (total_prompt_tokens + total_completion_tokens) / total_time_sec if total_time_sec > 0 else 0.0

        rss_mb = self.process.memory_info().rss / (1024.0 * 1024.0)
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        peak_mb = peak_mem / (1024.0 * 1024.0)

        cost_per_1k = (estimated_cost / max(1, num_queries)) * 1000.0

        return SystemTelemetry(
            latency_p50_ms=round(p50, 2),
            latency_p90_ms=round(p90, 2),
            latency_p99_ms=round(p99, 2),
            total_latency_ms=round(sum(latencies_ms), 2),
            retrieval_latency_ms=round(float(np.mean(retrieval_latencies_ms or [0])), 2),
            generation_latency_ms=round(float(np.mean(generation_latencies_ms or [0])), 2),
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            estimated_cost_usd=round(estimated_cost, 6),
            cost_per_1000_queries_usd=round(cost_per_1k, 4),
            throughput_qps=round(qps, 2),
            tokens_per_second=round(tps, 2),
            memory_rss_mb=round(rss_mb, 2),
            peak_memory_mb=round(peak_mb, 2),
        )
