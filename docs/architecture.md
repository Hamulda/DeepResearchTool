# Architecture Documentation

## System Overview

Deep Research Tool v2.0 implements a sophisticated DAG-based research pipeline optimized for local execution on M1 Macs. The system follows a modular architecture with clear separation of concerns across retrieval, re-ranking, synthesis, and verification stages.

## Core Components

### 1. DAG Workflow Orchestrator
- **Purpose**: Orchestrates the entire research pipeline
- **Location**: `src/core/enhanced_orchestrator.py`
- **Key Features**:
  - Evidence-based synthesis with ≥2 citations per claim
  - Parallel fan-out for subdotazů with intelligent fan-in
  - Human-in-the-loop checkpoints
  - Deterministic seeding and checkpoint support

### 2. Hierarchical Retrieval Engine
- **Purpose**: Multi-level retrieval from metadata to passages
- **Location**: `src/core/hybrid_retrieval_engine.py`
- **Architecture**:
  - Level 1: Document/section metadata retrieval
  - Level 2: Passage-level retrieval within top sections
  - Qdrant vector DB + BM25 sparse retrieval
  - RRF fusion for optimal result combination

### 3. Contextual Compression
- **Purpose**: Intelligent context filtering before LLM processing
- **Strategy**: Salience + novelty + redundancy filtering
- **Token Budget**: 2k-4k tokens depending on profile
- **Location**: To be implemented in `src/core/context_compressor.py`

### 4. Claim Graph System
- **Purpose**: Track claim relationships and contradictions
- **Location**: `src/graph/claim_graph.py`
- **Features**:
  - Support/contradict relationship tracking
  - Conflict set identification
  - Confidence penalization for disputed claims

### 5. Enhanced Evidence Binding
- **Purpose**: Rich metadata and temporal tracking
- **Schema Extensions**:
  - `timestamp`: ISO8601 capture time
  - `memento_datetime`: Archive timestamp
  - `snapshot_hash`: Content integrity hash
  - `persistent_id`: DOI/ECLI/CIK identifiers

## Data Flow

```
Query → Hierarchical Retrieval → Re-ranking → Contextual Compression → Synthesis → Verification → Claims
```

### Stage Details

1. **Retrieval Stage**
   - Metadata-first retrieval identifies relevant documents/sections
   - Passage-level retrieval within top sections
   - RRF fusion combines vector and sparse results
   - Deduplication across connectors

2. **Re-ranking Stage**
   - Multi-criteria scoring: relevance, authority, recency
   - Cross-encoder or LLM-as-rater implementation
   - Audit logging of ranking decisions

3. **Compression Stage**
   - Salience filtering removes low-value passages
   - Novelty detection prevents redundant content
   - Token budget management for LLM limits

4. **Synthesis Stage**
   - Claim-by-claim generation with inline citations
   - Evidence binding with rich metadata
   - Confidence scoring per claim

5. **Verification Stage**
   - Hallucination detection
   - Contradiction pass with counter-evidence search
   - Confidence calibration

## M1 Optimization

### Metal Performance Shaders (MPS)
- Embedding generation accelerated via Metal
- Batch size optimization for 8GB/16GB configurations
- Memory-efficient processing pipelines

### Ollama Integration
- Local LLM serving with Q4_K_M quantization
- Model switching based on verification confidence
- Streaming inference for large contexts

### Qdrant Configuration
- Hierarchical indexing: FP32 metadata + PQ passages
- Per-profile ef_search tuning
- Memory usage optimization

## Configuration Profiles

### Quick Profile
- Target: <60s execution time
- Basic hierarchical retrieval (2 levels)
- 2k token compression budget
- Primary model: qwen2.5:7b-q4_K_M

### Thorough Profile
- Target: 2-5min execution time
- Full hierarchical retrieval (3 levels)
- 4k token compression budget
- Enhanced contradiction detection
- Primary model: llama3.2:8b-q4_K_M

## Evaluation Framework

### Metrics Categories
1. **Retrieval Metrics**: recall@k, precision@k, nDCG@k
2. **Synthesis Metrics**: groundedness, citation precision, context efficiency
3. **Verification Metrics**: hallucination rate, confidence calibration

### CI Gate Thresholds
- Quick profile: groundedness ≥0.85, hallucination ≤0.10
- Thorough profile: groundedness ≥0.90, hallucination ≤0.06

## Specialized Source Connectors

### Academic Sources
- OpenAlex → Crossref → Unpaywall → Europe PMC pipeline
- DOI resolution and peer-review preference
- Citation graph integration

### Legal Sources
- CourtListener/RECAP for case law
- SEC EDGAR for corporate filings
- Exact string matching for legal citations

### Archive Sources
- Common Crawl WARC positioning
- Memento TimeGate/TimeMap access
- ArchiveBox local archiving integration

## Security & Compliance

### Local-First Architecture
- No external data uploads
- All processing on-device
- Configurable external connector policies

### Evidence Integrity
- Cryptographic hashes for content verification
- Temporal metadata for change detection
- Audit trails for all ranking decisions

## Future Extensions

### Planned Enhancements
- Multi-modal document processing (PDF, images)
- Graph-based query expansion
- Cross-lingual retrieval capabilities
- Advanced quantization strategies
