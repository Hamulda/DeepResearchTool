# Configuration Guide

## Overview

Deep Research Tool uses YAML configuration files with profile-based settings. The main configuration file is `config_m1_local.yaml` with M1-optimized defaults.

## Configuration Structure

```yaml
# Global settings
globals:
  seeds: {python: 1337, numpy: 1337, torch: 1337}
  checkpoints: {enabled: true, path: "./.checkpoints", save_after: ["retrieval","synthesis"]}

# Profile-specific configurations
profiles:
  quick:
    # ... quick profile settings
  thorough:
    # ... thorough profile settings

# Source connectors
specialized_sources:
  # ... connector configurations
```

## Profile Configuration

### Quick Profile (Default)
Optimized for fast results (<60s) with basic hierarchical retrieval:

```yaml
quick:
  retrieval:
    hierarchical: {enabled: true, levels: 2}
    rrf_k: 40
    dedup: true
    compression: {enabled: true, budget_tokens: 2000, strategy: salience}
  qdrant: 
    ef_search: 64
    index_tier: {meta: "fp32", passage: "pq"}
  llm:
    verification: 
      primary_model: "qwen2.5:7b-q4_K_M"
      fallback_model: "llama3.2:8b-q4_K_M"
      confidence_threshold: 0.6
      top_k: 4
```

### Thorough Profile
Enhanced analysis (2-5min) with full claim graph and contradiction detection:

```yaml
thorough:
  retrieval:
    hierarchical: {enabled: true, levels: 3}
    rrf_k: 60
    dedup: true
    compression: {enabled: true, budget_tokens: 4000, strategy: salience+novelty}
  qdrant: 
    ef_search: 96
    index_tier: {meta: "fp32", passage: "pq"}
  llm:
    verification: 
      primary_model: "llama3.2:8b-q4_K_M"
      fallback_model: "qwen2.5:7b-q4_K_M"
      confidence_threshold: 0.7
      top_k: 6
```

## Hierarchical Retrieval Settings

### Levels Configuration
- **Level 1**: Document/section metadata retrieval
- **Level 2**: Passage retrieval within top sections
- **Level 3**: Fine-grained sentence-level retrieval (thorough only)

### Compression Strategies
- **salience**: Remove low-relevance passages
- **novelty**: Detect and remove redundant content
- **redundancy**: Cross-passage deduplication

## Qdrant Configuration

### Index Tiers
- **meta**: Full precision (fp32) for metadata/section embeddings
- **passage**: Product Quantization (pq) for memory efficiency

### Performance Tuning
- **ef_search**: Controls search accuracy vs speed
  - Quick: 64 (faster, adequate accuracy)
  - Thorough: 96 (slower, higher accuracy)

## LLM Configuration

### Model Selection
Primary models for different use cases:
- **qwen2.5:7b-q4_K_M**: Fast inference, good accuracy
- **llama3.2:8b-q4_K_M**: Higher accuracy, slower inference

### Verification Settings
- **confidence_threshold**: Minimum confidence for claims
- **top_k**: Number of passages for verification
- **fallback_model**: Backup model if primary fails

## Source Connector Configuration

### Academic Sources
```yaml
academic:
  openalex:
    enabled: true
    api_key: null  # Open access
    rate_limit: 10
  crossref:
    enabled: true
    mailto: "your@email.com"  # Polite pool access
```

### Legal Sources
```yaml
legal:
  courtlistener:
    enabled: true
    api_key: "your_api_key"
    rate_limit: 5
  sec_edgar:
    enabled: true
    user_agent: "DeepResearchTool/2.0"
```

### Archive Sources
```yaml
archives:
  common_crawl:
    enabled: true
    index_servers: ["http://index.commoncrawl.org/"]
  memento:
    enabled: true
    timegate_url: "http://timetravel.mementoweb.org/timegate/"
```

## Environment Variables

Create `.env` file for sensitive configuration:

```bash
# API Keys
COURTLISTENER_API_KEY=your_key_here
CROSSREF_EMAIL=your@email.com

# Local services
QDRANT_URL=http://localhost:6333
OLLAMA_URL=http://localhost:11434

# Feature flags
ENABLE_TOR_CONNECTOR=false
ENABLE_EXPERIMENTAL_FEATURES=false
```

## Performance Tuning

### Memory Optimization (M1 8GB)
```yaml
quick:
  batch_size: 4
  max_passages: 20
  compression: {budget_tokens: 1500}
  qdrant: {ef_search: 32}
```

### Speed Optimization
```yaml
quick:
  retrieval:
    hierarchical: {enabled: false}  # Single-level retrieval
    rrf_k: 20
  compression: {enabled: false}
```

### Quality Optimization
```yaml
thorough:
  retrieval:
    hierarchical: {levels: 4}  # Maximum granularity
    rrf_k: 80
  verification:
    contradiction_pass: true
    confidence_threshold: 0.8
```

## Troubleshooting

### Common Issues

1. **Memory errors on M1 8GB**
   - Reduce `batch_size` to 2-4
   - Lower `max_passages` to 15-20
   - Use `compression.budget_tokens: 1500`

2. **Slow retrieval performance**
   - Check `qdrant.ef_search` (try lower values)
   - Disable `hierarchical.enabled` for speed
   - Reduce `rrf_k` parameter

3. **Poor result quality**
   - Increase `verification.confidence_threshold`
   - Enable `compression.strategy: salience+novelty`
   - Use thorough profile

### Debug Configuration
```yaml
debug:
  logging:
    level: DEBUG
    save_intermediate: true
  checkpoints:
    enabled: true
    save_retrieval: true
    save_reranking: true
```

## Custom Profiles

Create custom profiles for specific use cases:

```yaml
profiles:
  legal_research:
    base_profile: thorough
    specialized_sources:
      legal: {weight: 0.8}
      academic: {weight: 0.2}
    verification:
      confidence_threshold: 0.9
      exact_citation_matching: true
  
  medical_research:
    base_profile: thorough
    specialized_sources:
      academic: {weight: 0.9, peer_review_only: true}
      legal: {enabled: false}
    verification:
      fact_checking: enhanced
```

## Migration Guide

### From v1.x to v2.0
1. Update config structure to profile-based format
2. Add hierarchical retrieval settings
3. Configure new verification parameters
4. Update source connector settings

### Profile Migration
```python
# Convert old config to new profile format
python scripts/migrate_config.py --old-config config.yaml --profile quick
```
