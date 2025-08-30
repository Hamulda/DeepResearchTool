# Deep Research Tool - Documentation

## Overview

Deep Research Tool je pokročilý research agent optimalizovaný pro lokální běh na Apple Silicon (M1/M2) procesorech. Kombinuje hybridní retrieval, evidence-based synthesis a kontinuální evaluaci.

## Quick Start

### 1. Installation

```bash
git clone https://github.com/your-org/DeepResearchTool.git
cd DeepResearchTool
make setup
```

### 2. Verification

```bash
make smoke-test
```

Očekávaný výstup:
```
🚀 Starting smoke test with profile: quick
Configuration Loading PASSED ✅
Retrieval System PASSED ✅  
Evaluation System PASSED ✅
Basic Query PASSED ✅
📊 Tests passed: 4/4 (100.0%)
🎉 All smoke tests PASSED!
```

### 3. Basic Usage

```bash
# CLI interface
python cli.py search "What are quantum computing breakthroughs in 2024?"

# Direct Python
python main.py --query "Your research question" --profile thorough
```

## Architecture

### DAG Workflow Pipeline

```
Query → Retrieval → Re-ranking → Synthesis → Verification → Output
```

1. **Retrieval**: Hybrid dense + sparse search
2. **Re-ranking**: Cross-encoder nebo LLM-based ranking  
3. **Synthesis**: Claim generation s evidence binding
4. **Verification**: Contradiction checking a confidence scoring

### Key Components

- `src/core/dag_workflow_orchestrator.py` - Main pipeline orchestrator
- `src/core/hybrid_retrieval_engine.py` - Retrieval system
- `src/core/reranking_engine.py` - Re-ranking logic
- `src/core/synthesis_engine.py` - Claim synthesis
- `src/core/verification_engine.py` - Evidence verification

## Configuration

### Profiles

**Quick Profile** (rychlé dotazy):
- Model: qwen2.5:3b-q4_K_M
- Max docs: 20
- RRF k: 40

**Thorough Profile** (detailní research):
- Model: qwen2.5:7b-q4_K_M  
- Max docs: 50
- RRF k: 60

### Custom Configuration

```yaml
profiles:
  custom:
    llm:
      model: "llama3.2:8b-q4_K_M"
      temperature: 0.1
    retrieval:
      max_docs: 30
      rrf_k: 50
      dedup: true
    verification:
      confidence_threshold: 0.7
```

## Development

### Running Tests

```bash
make test          # Unit tests
make smoke-test    # End-to-end test
make eval          # Full evaluation
```

### Code Quality

```bash
make format        # Format with black + isort
make lint          # Lint with flake8 + mypy
```

### Performance Tuning

```bash
make sweep-rrf     # Tune RRF parameters
make bench-qdrant  # Benchmark vector database
```

## Troubleshooting

### Common Issues

1. **Ollama not found**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull qwen2.5:7b-q4_K_M
   ```

2. **Memory issues**
   - Use `quick` profile on 8GB systems
   - Reduce `max_docs` in config

3. **Slow performance**
   - Check M1 Metal acceleration: `python -c "import torch; print(torch.backends.mps.is_available())"`
   - Monitor CPU/Memory usage during queries

### Debug Logs

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python main.py --query "test" --verbose
```

## API Reference

See [API Documentation](api.md) for detailed endpoint descriptions.

## Examples

See [examples/](../examples/) directory for:
- Basic usage patterns
- Custom connector implementation
- Evaluation setup
- Performance optimization
