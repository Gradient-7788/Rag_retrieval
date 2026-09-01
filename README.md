# RAG Retrieval & Optimization Engine

### Systematic Retrieval-Augmented Generation Benchmarking, Evaluation & Optimization

A modular **Retrieval-Augmented Generation (RAG) benchmarking and optimization engine** designed to answer a practical question:

> **Which RAG architecture actually performs best for a given knowledge base and query distribution?**

Instead of implementing a single fixed RAG pipeline, this project provides a configurable framework for systematically comparing and optimizing the major components of a modern RAG system:

* document chunking
* embedding-based retrieval
* BM25 sparse retrieval
* hybrid retrieval
* reciprocal rank fusion (RRF)
* reranking
* query rewriting
* HyDE
* multi-query retrieval
* step-back prompting
* keyword expansion
* answer generation
* retrieval quality evaluation
* generation quality evaluation
* latency measurement
* hyperparameter sweeps
* Pareto-frontier analysis

The system exposes both a **Python CLI** and a **FastAPI backend**, with a frontend for interacting with and benchmarking different RAG configurations.

---

# 🚀 Why This Project?

A naïve RAG implementation usually looks like:

```text
Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Search
    ↓
LLM
    ↓
Answer
```

However, production RAG quality depends on many interacting design decisions.

For example:

* Is semantic retrieval better than BM25?
* Does hybrid retrieval improve recall?
* Does reranking improve NDCG enough to justify its latency?
* Is HyDE useful for the query distribution?
* Does parent-child chunking outperform fixed-size chunks?
* What chunk size produces the best retrieval quality?
* How much latency does each additional retrieval stage introduce?
* Which configuration provides the best quality/latency trade-off?

This project treats RAG as an **experimentally measurable information-retrieval system**, rather than a black-box LLM wrapper.

---

# 🧠 System Architecture

```text
                           ┌──────────────────────┐
                           │      Documents       │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │     Chunking Layer   │
                           │                      │
                           │ Fixed Window         │
                           │ Sentence             │
                           │ Recursive            │
                           │ Parent-Child         │
                           │ Semantic             │
                           └──────────┬───────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │     Embedding Layer     │
                         │                         │
                         │ TF-IDF                  │
                         │ Sentence Transformers   │
                         │ Configurable embeddings │
                         └────────────┬────────────┘
                                      │
                         ┌────────────┴────────────┐
                         │                         │
                         ▼                         ▼
                ┌────────────────┐       ┌────────────────┐
                │ Dense Retrieval│       │ BM25 Retrieval │
                └───────┬────────┘       └───────┬────────┘
                        │                        │
                        └───────────┬────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │   Hybrid Retrieval  │
                         │  RRF / Weighted     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Reranker        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Context Builder   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    LLM Generator    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                               Final Answer


      ┌──────────────────────────────────────────────────────┐
      │                 Evaluation Layer                     │
      │                                                      │
      │ Recall@K • Precision@K • MRR • NDCG@K • Hit Rate    │
      │ Faithfulness • Answer Relevance • Context Relevance │
      │ Citation Accuracy • Latency • Composite Score       │
      └──────────────────────────┬───────────────────────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │ Optimization Engine  │
                      │                      │
                      │ Chunk Size           │
                      │ Retriever             │
                      │ Hybrid Alpha          │
                      │ Query Rewriter        │
                      │ Reranking             │
                      │ Top-K                 │
                      └──────────┬───────────┘
                                 │
                                 ▼
                       Best / Pareto Configs
```

The pipeline implementation explicitly separates chunking, embeddings, retrieval, reranking, query rewriting and generation, while also measuring retrieval, generation and total latency independently.

---

# ✨ Key Features

## 1. Configurable RAG Pipeline

The entire RAG architecture is represented through a typed `PipelineConfig`.

The current configuration space includes:

```text
Chunking
├── fixed_window
├── sentence
├── recursive
├── parent_child
└── semantic

Retrieval
├── dense
├── bm25
└── hybrid

Hybrid Fusion
├── RRF
└── weighted

Query Rewriting
├── none
├── HyDE
├── multi-query
├── step-back
└── keyword expansion

Reranking
└── enabled / disabled
```

This allows different RAG architectures to be evaluated without rewriting the underlying pipeline.

The current default configuration uses:

```text
Chunk size       : 500
Chunk overlap    : 50
Retriever        : Hybrid
Fusion           : RRF
Hybrid alpha     : 0.5
Top-K            : 5
Reranking        : enabled
Rerank Top-N     : 3
Query rewriting  : none
```

These configuration parameters are defined directly in the pipeline abstraction.

---

# 📚 2. Multiple Chunking Strategies

Chunking is one of the most important and frequently overlooked components of RAG.

The framework supports:

### Fixed Window

Splits documents into fixed-size overlapping chunks.

```text
Document
──────────────────────────────────────────

[ Chunk 1            ]
             [ Chunk 2            ]
                       [ Chunk 3            ]
```

### Sentence Chunking

Attempts to preserve sentence-level semantic boundaries.

### Recursive Chunking

Recursively divides text into smaller units while attempting to preserve meaningful structure.

### Parent-Child Chunking

Maintains larger parent chunks while retrieving smaller child chunks.

This allows retrieval to operate at a fine-grained level while preserving larger contextual units.

### Semantic Chunking

Provides a semantic-oriented chunking abstraction for experimenting with meaning-aware segmentation.

The pipeline dynamically instantiates the selected chunker based on configuration.

---

# 🔎 3. Dense Retrieval

The dense retrieval component represents text as continuous vector embeddings and retrieves semantically similar chunks.

The architecture provides an embedding abstraction supporting:

* simulated embeddings
* TF-IDF embeddings
* Sentence Transformers-compatible embeddings

This abstraction makes the retrieval layer independent of a specific embedding implementation.

The pipeline can therefore be configured to experiment with different embedding backends without changing the retrieval API.

---

# 🔤 4. BM25 Sparse Retrieval

Dense retrieval is powerful for semantic similarity, but lexical retrieval can be superior when exact terminology matters.

The project therefore includes a BM25 retriever.

BM25 provides:

```text
Query
  ↓
Term Matching
  ↓
Term Frequency
  ↓
Inverse Document Frequency
  ↓
Document Relevance Score
```

This makes it possible to directly compare semantic and lexical retrieval.

---

# 🔀 5. Hybrid Retrieval

The framework combines dense and sparse retrieval through a hybrid retriever.

Two fusion modes are supported:

```text
RRF
Weighted Fusion
```

The hybrid configuration exposes an `alpha` parameter controlling the contribution of the retrieval signals.

Conceptually:

```text
Dense Score ───────┐
                   ├──→ Fusion ──→ Final Ranking
BM25 Score ────────┘
```

This allows the system to investigate whether combining semantic and lexical signals improves retrieval robustness.

---

# 🏆 6. Reranking

Initial retrieval is optimized for speed, while reranking can improve the quality of the final candidate set.

The pipeline therefore supports an optional reranking stage:

```text
Query
  ↓
Candidate Retrieval
  ↓
Top-K Candidates
  ↓
Reranker
  ↓
Top-N Context
  ↓
Generation
```

The reranker can be enabled or disabled through configuration, making it possible to quantify the quality gain versus additional latency.

---

# 🔄 7. Advanced Query Rewriting

The framework implements several query transformation strategies.

### Direct Query

Uses the original user query.

### HyDE

**Hypothetical Document Embeddings (HyDE)** transforms a query into a hypothetical answer/document representation before retrieval.

```text
Original Query
      ↓
Hypothetical Answer
      ↓
Embedding
      ↓
Retrieval
```

### Multi-Query

Generates multiple alternative formulations of the same information need.

```text
                 ┌── Query A ──┐
Original Query ──┼── Query B ──┼── Retrieval
                 └── Query C ──┘
```

Retrieved results are merged and deduplicated before ranking.

### Step-Back

Reformulates a specific question into a more general conceptual query.

### Keyword Expansion

Extracts or expands important lexical terms to improve retrieval coverage.

All of these strategies share a common query-rewriter abstraction.

---

# 🤖 8. Generation Layer

The pipeline separates retrieval from generation.

The generation interface provides an abstraction for an LLM-backed answer generator.

The current implementation includes:

```text
BaseLLMGenerator
MockLLMGenerator
OpenAICompatibleLLMGenerator
```

This makes the generation layer replaceable without modifying retrieval, evaluation or optimization logic.

> **Implementation note:** the default pipeline currently uses the mock generator, making the repository runnable without requiring an external LLM API. The OpenAI-compatible generator interface can be used when connecting the system to an actual model provider.

---

# 📊 9. Retrieval Evaluation

A major goal of this project is to avoid evaluating RAG systems solely by looking at generated answers.

The evaluator explicitly measures retrieval quality.

Implemented retrieval metrics include:

* **Recall@K**
* **Precision@K**
* **MRR**
* **NDCG@K**
* **Hit Rate@K**

These metrics allow the system to determine whether relevant information is actually being retrieved before generation occurs.

---

# 🧪 10. Generation Evaluation

The framework also evaluates the generation stage.

Metrics include:

* Faithfulness
* Answer relevance
* Context relevance
* Citation accuracy

This separates two common failure modes:

```text
                         RAG Failure
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          Retrieval Failure          Generation Failure
                 │                         │
         Wrong context              Bad answer
         Missing context            Hallucination
         Low recall                 Poor relevance
```

This distinction is important because a poor answer does not necessarily mean the retriever is responsible.

The evaluation report explicitly stores retrieval and generation metrics alongside a composite quality score.

---

# ⚡ 11. System Telemetry

RAG quality is not the only objective.

A retrieval system that is extremely accurate but prohibitively slow may not be useful in production.

The pipeline therefore tracks:

```text
Retrieval latency
Generation latency
Total latency
```

The CLI comparison view also exposes **P50 latency** alongside retrieval and generation quality metrics.

This enables analysis such as:

```text
                    Quality
                       ▲
                       │
                       │        ●
                       │   ●
                       │
                       │ ●
                       └──────────────────► Latency
```

---

# 🏁 12. Automated Architecture Comparison

The project includes a built-in benchmark comparing several RAG architectures.

The current comparison includes:

| Strategy                           | Retrieval    | Reranker | Query Strategy |
| ---------------------------------- | ------------ | -------- | -------------- |
| Dense Vector                       | Dense        | ❌        | Direct         |
| Sparse Keyword                     | BM25         | ❌        | Direct         |
| Hybrid RRF                         | Dense + BM25 | ❌        | Direct         |
| Hybrid + Reranker                  | Dense + BM25 | ✅        | Direct         |
| Hybrid + Reranker + HyDE           | Dense + BM25 | ✅        | HyDE           |
| Parent-Child + Hybrid + MultiQuery | Hybrid       | ✅        | MultiQuery     |

The same benchmark evaluates each architecture and reports:

```text
NDCG@5
Recall@5
Faithfulness
P50 latency
Composite Quality Score
```

This comparison is implemented both through the CLI and the FastAPI backend.

---

# 🔬 13. RAG Hyperparameter Optimization

The project includes a dedicated optimization engine for searching across RAG configurations.

The current sweep space includes:

```text
Chunk sizes:
    250
    500
    1000

Retriever:
    dense
    bm25
    hybrid

Hybrid alpha:
    0.3
    0.5
    0.7

Query rewriting:
    none
    hyde
    multi_query

Reranking:
    enabled
    disabled

Top-K:
    3
    5
```

The optimizer evaluates candidate configurations and identifies:

* best configuration by NDCG
* best configuration by composite quality score
* Pareto-frontier configurations

The optimization report explicitly stores these outputs.

---

# 📈 Pareto Optimization

RAG systems typically involve a trade-off:

```text
Higher Retrieval Quality
        ↑
        │
        │       ●
        │    ●
        │  ●
        │
        └────────────────────→
                    Latency
```

A configuration may improve NDCG while increasing latency.

Instead of assuming that one metric should always be maximized, this project identifies **Pareto-efficient configurations**.

A configuration belongs to the Pareto frontier when another configuration cannot improve its objectives without sacrificing another objective.

This makes the optimizer useful for practical system design rather than merely leaderboard optimization.

---

# 🧬 End-to-End Query Flow

For a user query:

```text
"What are the main benefits of retrieval-augmented generation?"
```

the pipeline executes:

```text
                 User Query
                     │
                     ▼
              Query Rewriting
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
       Query 1    Query 2    Query 3
          │          │          │
          └──────────┼──────────┘
                     ▼
             Retrieval Layer
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
       Dense                  BM25
          │                     │
          └──────────┬──────────┘
                     ▼
               Hybrid Fusion
                     │
                     ▼
                 Top-K
                     │
                     ▼
                 Reranker
                     │
                     ▼
                Top-N Context
                     │
                     ▼
              LLM Generation
                     │
                     ▼
                 Response
```

The pipeline also records the rewritten queries, retrieved chunks and timing information in the final `RAGPipelineResult`.

---

# 🏗️ Repository Structure

```text
Rag_retrieval/
│
├── rag_engine/
│   │
│   ├── chunking/
│   │   ├── base.py
│   │   └── ...
│   │
│   ├── datasets/
│   │   ├── loader.py
│   │   └── synthetic.py
│   │
│   ├── embeddings/
│   │   ├── base.py
│   │   └── implementations.py
│   │
│   ├── generation/
│   │   └── ...
│   │
│   ├── metrics/
│   │   ├── retrieval_metrics.py
│   │   ├── generation_metrics.py
│   │   └── system_metrics.py
│   │
│   ├── query_rewriting/
│   │   └── ...
│   │
│   ├── retrieval/
│   │   └── ...
│   │
│   ├── evaluator.py
│   ├── optimizer.py
│   └── pipeline.py
│
├── frontend/
│   ├── src/
│   ├── dist/
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── cli.py
├── server.py
└── README.md
```

The core engine is deliberately modular, separating chunking, datasets, embeddings, generation, metrics, query rewriting and retrieval into independent components.

---

# 🖥️ Web Application

The project includes a frontend application served by the FastAPI backend when the compiled frontend assets are available.

The frontend is built with a modern JavaScript toolchain including:

* Vite
* React
* Tailwind CSS

The backend can serve the generated frontend assets directly.

This gives the project two interfaces:

```text
                RAG Engine
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
        CLI              Web Interface
          │                   │
          └─────────┬─────────┘
                    ▼
              Core Pipeline
```

---

# 🌐 FastAPI API

The backend exposes the RAG engine through a REST API.

## Health Check

```http
GET /api/health
```

Returns service status and the number of loaded documents and benchmark queries.

---

## Get Dataset

```http
GET /api/dataset
```

Returns the currently loaded documents and benchmark queries.

---

## Upload Dataset

```http
POST /api/dataset/upload
```

Accepts documents and can automatically generate synthetic benchmark queries.

Example:

```json
{
  "documents": [
    {
      "doc_id": "doc_1",
      "text": "Retrieval-Augmented Generation combines retrieval with language generation..."
    }
  ],
  "generate_synthetic_queries": true
}
```

The backend then updates the active document/query dataset.

---

## Single Query

```http
POST /api/query
```

Executes a single query using a supplied `PipelineConfig`.

The returned result contains:

```text
Original query
Rewritten queries
Retrieved chunks
Generated response
Retrieval latency
Generation latency
Total latency
Pipeline configuration
```

---

## Evaluate Pipeline

```http
POST /api/evaluate
```

Runs the selected RAG architecture against the benchmark query set.

Returns the full evaluation report, including retrieval, generation and system metrics.

---

## Compare Architectures

```http
GET /api/comparison
```

Runs the built-in architecture comparison benchmark.

---

## Hyperparameter Sweep

```http
POST /api/sweep
```

Runs the RAG optimization engine over the selected search space.

The backend supports a configurable maximum number of evaluations.

---

# 💻 CLI Usage

The project provides a dedicated command-line interface.

## Evaluate a Pipeline

```bash
python cli.py evaluate
```

Custom retrieval strategy:

```bash
python cli.py evaluate \
    --retriever hybrid \
    --chunker recursive \
    --rewriter hyde \
    --reranker
```

Supported retrievers:

```text
dense
bm25
hybrid
```

Supported chunkers:

```text
fixed_window
sentence
recursive
parent_child
semantic
```

Supported query rewriting strategies:

```text
none
hyde
multi_query
step_back
keyword
```

These options are directly exposed by the CLI argument parser.

---

# ⚔️ Compare RAG Architectures

```bash
python cli.py compare
```

The comparison benchmark evaluates multiple architectures side-by-side and prints:

```text
Strategy Name
NDCG@5
Recall@5
Faithfulness
P50 latency
Composite Score
```

This is useful for identifying whether additional complexity actually improves system quality.

---

# 🔬 Run an Optimization Sweep

```bash
python cli.py sweep
```

Specify the maximum number of configurations:

```bash
python cli.py sweep --evals 20
```

The optimizer reports:

```text
Configurations evaluated
Best configuration by NDCG
Best configuration by composite score
Pareto frontier size
```

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Gradient-7788/Rag_retrieval.git
cd Rag_retrieval
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the project's Python dependencies according to the repository's dependency configuration.

Then run:

```bash
python cli.py compare
```

or:

```bash
python server.py
```

The FastAPI application runs on:

```text
http://localhost:8000
```

The server is configured to listen on `0.0.0.0:8000`.

---

# 🧪 Benchmarking Methodology

A benchmark query contains:

```text
query_id
query
ground_truth_doc_ids
ground_truth_chunk_ids
target_answer
```

This enables the evaluator to distinguish between:

### Retrieval correctness

Did the system retrieve the relevant documents/chunks?

### Answer quality

Did the generator produce a relevant and faithful answer?

### System efficiency

How much latency was introduced by the retrieval and generation pipeline?

The evaluator combines these into a structured evaluation report.

---

# 📊 Evaluation Framework

| Category   | Metrics                 |
| ---------- | ----------------------- |
| Retrieval  | Recall@K                |
| Retrieval  | Precision@K             |
| Retrieval  | MRR                     |
| Retrieval  | NDCG@K                  |
| Retrieval  | Hit Rate                |
| Generation | Faithfulness            |
| Generation | Answer Relevance        |
| Generation | Context Relevance       |
| Generation | Citation Accuracy       |
| System     | Retrieval Latency       |
| System     | Generation Latency      |
| System     | Total Latency           |
| Overall    | Composite Quality Score |

This makes the project more than a RAG demo: it provides a framework for **quantitatively analyzing retrieval architectures**.

---

# 🧠 Design Principles

## 1. Modular Components

Every major RAG component is abstracted behind an interface.

```text
BaseChunker
BaseEmbeddingModel
BaseRetriever
BaseQueryRewriter
BaseLLMGenerator
```

This allows implementations to be swapped without changing the pipeline orchestration layer.

---

## 2. Configuration-Driven Experiments

Rather than creating separate scripts for every architecture, the system represents architecture choices through configuration.

For example:

```python
PipelineConfig(
    chunker_type="parent_child",
    retriever_type="hybrid",
    fusion_mode="rrf",
    enable_reranker=True,
    query_rewriter_type="multi_query"
)
```

This makes experiments reproducible and enables automated comparison.

---

## 3. Quality + Latency

The project does not assume that the most accurate system is automatically the best system.

Instead:

```text
RAG Quality
     +
System Latency
     +
Retrieval Robustness
     ↓
Practical System Selection
```

This is particularly important when deploying retrieval systems at scale.

---

## 4. Retrieval-First Evaluation

The system explicitly measures retrieval quality before generation quality.

This allows failures to be localized:

```text
Low NDCG
     ↓
Retrieval problem

High NDCG
+
Low Faithfulness
     ↓
Generation / grounding problem
```

This decomposition makes debugging RAG systems substantially easier.

---

# 🔥 What Makes This Project Different?

A typical portfolio RAG project:

```text
PDF
 ↓
Chunk
 ↓
Vector DB
 ↓
LLM
 ↓
Chatbot
```

This project instead focuses on **RAG as an information retrieval engineering problem**:

```text
                  RAG SYSTEM
                      │
       ┌──────────────┼───────────────┐
       │              │               │
       ▼              ▼               ▼
   Chunking       Retrieval       Query Reformulation
       │              │               │
       │       ┌──────┴──────┐         │
       │       │             │         │
       │     Dense         BM25       HyDE
       │       │             │       MultiQuery
       │       └──────┬──────┘       StepBack
       │              │
       │          Hybrid RRF
       │              │
       │          Reranking
       │              │
       └──────────────┼────────────────┘
                      ▼
                  Generation
                      │
                      ▼
                 Evaluation
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Quality      Quality      Latency
       Retrieval   Generation     Metrics
          │           │           │
          └───────────┼───────────┘
                      ▼
                 Optimization
                      │
                      ▼
              Pareto-efficient
                 RAG configs
```

The project therefore demonstrates understanding of **information retrieval, LLM systems, experimentation, benchmarking and optimization**, rather than simply demonstrating that a vector database can be connected to an LLM.

---

# 🛠️ Technology Stack

| Layer           | Technology                                      |
| --------------- | ----------------------------------------------- |
| Language        | Python                                          |
| API             | FastAPI                                         |
| Validation      | Pydantic                                        |
| CLI             | argparse                                        |
| Retrieval       | Dense + BM25 + Hybrid                           |
| Fusion          | Reciprocal Rank Fusion / Weighted               |
| Embeddings      | TF-IDF / Sentence Transformers abstraction      |
| Query Rewriting | HyDE / Multi-Query / Step-Back / Keyword        |
| Generation      | Mock + OpenAI-compatible abstraction            |
| Frontend        | React / Vite                                    |
| Styling         | Tailwind CSS                                    |
| Evaluation      | Custom retrieval + generation metrics           |
| Optimization    | Configuration sweep + Pareto analysis           |
| Data            | Custom benchmark + synthetic dataset generation |

---

# 📁 Core Modules

### `pipeline.py`

Central orchestration layer connecting:

```text
Chunking
→ Embeddings
→ Retrieval
→ Reranking
→ Query Rewriting
→ Generation
```

It also records latency and returns structured pipeline results.

### `evaluator.py`

Runs benchmark queries and aggregates:

```text
Retrieval Metrics
Generation Metrics
System Telemetry
Composite Quality
```

### `optimizer.py`

Searches the RAG configuration space and returns:

```text
Best NDCG configuration
Best Composite configuration
Pareto frontier
```

### `cli.py`

Provides command-line experimentation:

```text
evaluate
compare
sweep
```

### `server.py`

Provides the REST API for:

```text
Dataset management
Single-query inference
Pipeline evaluation
Architecture comparison
Optimization sweeps
```

---

# 🎯 Skills Demonstrated

## Retrieval & NLP

* Information Retrieval
* Dense Retrieval
* Sparse Retrieval
* BM25
* Semantic Search
* Embeddings
* Reranking
* Query Expansion
* HyDE
* Multi-Query Retrieval
* Step-Back Querying

## LLM Engineering

* Retrieval-Augmented Generation
* Context Construction
* Grounded Generation
* LLM evaluation
* Answer relevance
* Faithfulness analysis
* Citation accuracy

## Machine Learning / AI Systems

* Model abstraction
* Benchmark design
* Hyperparameter search
* Multi-objective optimization
* Pareto-frontier analysis
* Latency/quality trade-offs

## Software Engineering

* Modular architecture
* Python abstractions
* Typed configuration
* REST APIs
* CLI tooling
* Frontend integration
* Experiment reproducibility

---

# 🚧 Current Limitations

The project is intentionally designed as a **RAG research/benchmarking engine**, rather than a fully productionized enterprise RAG platform.

Current limitations include:

* default generation uses a mock LLM
* active datasets are held in memory
* retrieval indexes are rebuilt when a pipeline is instantiated/indexed
* authentication is not implemented
* persistent vector database integration is not included
* distributed serving is outside the current scope

These trade-offs keep the benchmark framework lightweight and make experimentation straightforward.

---

# 🔮 Future Work

Potential extensions include:

### Retrieval

* FAISS / Qdrant / Milvus integration
* ANN indexing
* HNSW retrieval
* hybrid sparse-dense indexes
* ColBERT-style late interaction

### Embeddings

* BGE
* E5
* GTE
* multilingual embedding models
* embedding fine-tuning

### Reranking

* cross-encoder rerankers
* BGE rerankers
* ColBERT
* LLM-based reranking

### Generation

* OpenAI
* local LLMs
* vLLM
* Ollama
* Hugging Face Transformers

### Evaluation

* RAGAS integration
* BEIR benchmarks
* MTEB retrieval benchmarks
* larger domain-specific evaluation sets
* human preference evaluation

### Infrastructure

* Docker
* Kubernetes
* Redis caching
* persistent vector stores
* observability dashboards
* experiment tracking
* distributed evaluation workers

---

# 📌 Project Takeaway

The central idea behind this project is simple:

> **RAG quality should be measured and optimized, not assumed.**

A RAG application is a combination of multiple interacting information-retrieval and generation decisions.

This project provides a framework for answering those decisions empirically:

```text
       Architecture
            │
            ▼
        Benchmark
            │
            ▼
       Quantify Quality
            │
            ▼
       Measure Latency
            │
            ▼
      Compare Systems
            │
            ▼
     Optimize Parameters
            │
            ▼
    Select Best Trade-off
```

The result is a modular **RAG experimentation and optimization platform** capable of comparing retrieval strategies, evaluating answer quality, measuring system performance and searching the architecture space for high-quality configurations.

---

# 👨‍💻 Author

**Yesh Lohchab**

Machine Learning • AI • LLM Systems • Information Retrieval • MLOps

---

## ⭐ Resume-Level Summary

**RAG Retrieval & Optimization Engine** — Built a modular RAG benchmarking platform comparing dense, BM25 and hybrid retrieval with RRF fusion, reranking, HyDE/multi-query/step-back rewriting and multiple chunking strategies; implemented retrieval (Recall@K, MRR, NDCG@K), generation (faithfulness, relevance, citation) and latency evaluation, plus configurable architecture sweeps and Pareto-frontier optimization via FastAPI/CLI.
