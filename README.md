# Institutional-Grade MLOps Trading Engine (Architecture Showcase)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2.svg?style=for-the-badge&logo=MLflow&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Polars](https://img.shields.io/badge/Polars-CD792C.svg?style=for-the-badge&logo=Polars&logoColor=white)
![Prefect](https://img.shields.io/badge/Prefect-ffffff.svg?style=for-the-badge&logo=Prefect&logoColor=blue)


Welcome to the architectural showcase of my proprietary Quantitative Trading Engine. 

Due to the sensitive nature of the Alpha-generating strategies and proprietary signals, the source code is kept in a private repository (`britis-investing`). This public repository serves as a **technical showcase** of the data engineering, machine learning pipelines, and Site Reliability Engineering (SRE) practices used to build the platform.

---

## System Architecture

The engine is built on a **Lakehouse Medallion Architecture**, optimizing high-frequency tick data ingestion, cleaning, and model serving.

```mermaid
graph TD
    classDef source fill:#440154,color:#fff,stroke:#fff,stroke-width:2px;
    classDef medallion fill:#3B528B,color:#fff,stroke:#fff,stroke-width:2px;
    classDef ml fill:#21918C,color:#fff,stroke:#fff,stroke-width:2px;
    classDef sre fill:#5EC962,color:#000,stroke:#fff,stroke-width:2px;

    subgraph Data Sources
        B3[B3 Futures Market]:::source --> |Tick Data| Ingest[MT5 Data Loader]:::source
    end

    subgraph Medallion Lakehouse
        Ingest --> Bronze[(Bronze Layer)]:::medallion
        Bronze --> |Raw Splicing| Silver[(Silver Layer)]:::medallion
        Silver --> |Feature Eng & RAG| Gold[(Gold Layer)]:::medallion
    end

    subgraph Machine Learning Pipeline
        Gold --> DRL[Deep Reinforcement Learning]:::ml
        Gold --> Semantic[SignalCortexRAG]:::ml
        DRL --> Inference[Real-Time Inference Engine]:::ml
        Semantic --> Inference
    end

    subgraph SRE & Infrastructure
        Inference --> Exec[Order Execution]:::sre
        Exec --> Monitor[SRE Obsidian Loop]:::sre
    end
```

## Core Technologies
- **Data Engineering:** Polars (for ultra-fast dataframe manipulation), Lakehouse architecture.
- **Orchestration:** Prefect (for modern, dynamic dataflow orchestration instead of legacy Airflow).
- **Machine Learning:** Deep Reinforcement Learning (DRL) for dynamic position sizing, Retrieval-Augmented Generation (RAG) for semantic market analysis.
- **Package Management:** `uv` for high-performance dependency resolution.
- **Infrastructure:** Systemd tunings (timeout control), strict Linux environments (Manjaro).

## Mathematical Highlight: Futures Splicing & Rollovers

One of the biggest challenges in Quant Engineering is dealing with continuous futures contracts. When rolling over from an expiring contract to a new one, artificial price gaps occur, destroying ML model training.

To solve this, the **Silver Layer** implements a mathematically robust **Backward Difference Splicing**:

$$ P_{adjusted}(t) = P_{raw}(t) - \sum_{i=t}^{T} \Delta Gap_i $$

*Where:*
- $P_{raw}$ is the raw price from the Bronze layer.
- $\Delta Gap$ is the exact price difference recorded at the rollover boundary.
- The cumulative backward shift ensures that historical volatilities and returns are preserved, preventing the DRL model from learning fake "jumps".

## About the Author

I am a Data Science and Analytics student at USP (Universidade de São Paulo) with a deep passion for quantitative finance, MLOps, and Data Engineering. My work focuses on building robust, highly available systems that bridge the gap between complex mathematical models and real-time execution.

*Feel free to reach out to discuss Data Engineering, MLOps, or Quant Finance!*

## Documentation
- [B3 Futures Mechanics & The Rollover Problem](docs/domain_knowledge/futures_mechanics.md) - A deep dive into the mathematical anomalies caused by B3 contract rollovers and how our MLOps pipeline solves them.
