import re
from typing import List, Dict
from rag_engine.evaluator import BenchmarkQuery

class SyntheticDatasetGenerator:
    """
    Generates synthetic ground-truth Q&A datasets directly from documents.
    Extracts key entity statements, definitions, and technical claims.
    """
    def generate_from_documents(self, documents: List[Dict[str, str]], num_queries_per_doc: int = 2) -> List[BenchmarkQuery]:
        benchmark_queries = []
        q_count = 1

        for doc in documents:
            doc_id = doc.get("doc_id", "doc_0")
            text = doc.get("text", "")
            sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if len(s.strip()) > 30]

            for sent in sentences[:num_queries_per_doc]:
                # Construct synthetic query from sentence
                words = sent.split()
                if len(words) > 6:
                    keywords = [w.strip(",.()") for w in words if len(w) > 4][:3]
                    kw_str = " ".join(keywords)
                    query_text = f"What are the details regarding {kw_str}?"
                else:
                    query_text = f"What is described about {doc_id}?"

                benchmark_queries.append(
                    BenchmarkQuery(
                        query_id=f"synth_q_{q_count}",
                        query=query_text,
                        ground_truth_doc_ids=[doc_id],
                        target_answer=sent,
                    )
                )
                q_count += 1

        return benchmark_queries
