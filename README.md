# 🔍 Deep Research Tool v2.0 - Advanced Research Agent

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MacBook Air M1 Optimized](https://img.shields.io/badge/M1-optimized-green.svg)](https://developer.apple.com/metal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Evidence-Based Research Agent** s hybridním vyhled��váním, per-claim evidence binding a rozšířenými konektory do specializovaných zdrojů. Optimalizováno pro **čistě lokální provoz** na MacBook Air M1.

## ✨ Klíčové Funkce v2.0

### 🧠 DAG Workflow Orchestrator
- **Evidence-based synthesis** - každý claim musí mít minimálně 2 nezávislé citace
- **Modulární pipeline**: `retrieval → re-ranking → synthesis → verification`
- **Fan-out paralelizace** subdotazů s intelligent fan-in
- **Human-in-the-loop** checkpointy pro critical review

### 🔄 Hybridní Retrieval Engine  
- **Hierarchical retrieval** - metadata-first → section → passage granularity
- **Qdrant** vektorová databáze + **BM25** sparse retrieval
- **Reciprocal Rank Fusion (RRF)** pro optimální kombinaci výsledků
- **M1 Metal** accelerated embeddings pro maximální výkon

### 🎯 Advanced Re-ranking & Compression
- **Cross-encoder** nebo **LLM-as-rater** s multi-kritéria skórováním
- **Contextual compression** - salience filtering + token budgeting
- **Konfigurovatelné váhy**: relevance, autorita, novost, aktualnost
- **Audit logs** s odůvodněním každého přeskupení

### 🕸️ Claim Graph & Contradiction Detection
- **Claim graph** s support/contradict relationships
- **Contradiction verification** - aktivní vyhledávání vyvracející evidence
- **Confidence scoring** s penalizací pro disputed claims
- **Conflict sets** pro sporné oblasti

### 📚 Specialized Source Connectors
- **Common Crawl** - 15+ let web archivů s přesnými WARC pozicemi
- **Memento** - TimeGate/TimeMap time-travel s diff analýzou
- **Ahmia** - Tor OSINT (legal filtering only)
- **Legal APIs** - CourtListener/RECAP + SEC EDGAR
- **Open Access Science** - OpenAlex → Crossref → Unpaywall → Europe PMC

### 📊 Continuous Evaluation & Optimization
- **Regression test set** s 10+ specializovanými tématy
- **Metriky kvality**: recall@k, evidence coverage, answer faithfulness
- **Bayesian hyperparameter optimization** pro per-domain tuning
- **Performance tracking** s JSON/Markdown reporting

### 🍎 MacBook Air M1 Optimalization
- **Ollama modely**: 3B-8B v Q4_K_M kvantizaci pro optimal speed/quality
- **MPS/Metal** acceleration s intelligent batch sizing
- **Memory-efficient** processing s 4k-8k context windows
- **Streaming inference** s progressive context building

## 🚀 Quick Start

### Požadavky
- **Python 3.9+**
- **MacBook Air M1** (doporučeno) nebo Intel Mac/Linux
- **16GB RAM** (minimum 8GB)
- **20GB volného místa** pro modely a cache

### Instalace
```bash
# Klonovat repozitář
git clone https://github.com/vojtechhamada/DeepResearchTool.git
cd DeepResearchTool

# Kompletní setup včetně dependencies a validace
make setup

# Spustit rychlý smoke test (< 60s)
make smoke-test

# Volitelně: spustit kompletní evaluaci
make eval
```

### První dotaz
```bash
# Základní research query
python main.py "What are the latest developments in quantum computing error correction?"

# S konkrétním profilem
python main.py --profile thorough "Legal implications of AI bias in hiring"

# CLI režim pro interaktivní session
python cli.py
```

## 📋 Očekávané výstupy

### Smoke Test (make smoke-test)
```
🧪 Running smoke test with profile: quick
✅ Connection to Ollama: OK (qwen2.5:7b-q4_K_M)
✅ Qdrant health check: OK (3 collections)
✅ Query: "COVID-19 vaccine effectiveness"
✅ Generated 2 claims with evidence:
   - Claim 1: mRNA vaccines show 95% effectiveness (2 citations)
   - Claim 2: Booster shots restore protection (3 citations)
⏱️  Total time: 45.2s
```

### Evaluation Report (make eval)
```json
{
  "timestamp": "2025-08-26T14:30:00Z",
  "profile": "quick",
  "metrics": {
    "retrieval": {
      "recall@10": 0.85,
      "precision@10": 0.72,
      "nDCG@10": 0.78
    },
    "synthesis": {
      "groundedness": 0.91,
      "citation_precision": 0.88,
      "context_usage_efficiency": 0.73
    },
    "verification": {
      "hallucination_rate": 0.06,
      "confidence_calibration": 0.82
    }
  }
}
```

## 🔧 Konfigurace

### Profily
- **quick**: Rychlé výsledky (< 60s), basic hierarchical retrieval
- **thorough**: Detailní analýza (2-5 min), plný claim graph + contradiction detection

### Config file (config_m1_local.yaml)
```yaml
profiles:
  quick:
    retrieval:
      hierarchical: {enabled: true, levels: 2}
      rrf_k: 40
      dedup: true
      compression: {enabled: true, budget_tokens: 2000, strategy: salience}
    qdrant: {ef_search: 64, index_tier: {meta: "fp32", passage: "pq"}}
    llm:
      verification: {primary_model: "qwen2.5:7b-q4_K_M", confidence_threshold: 0.6}
```

## 🛠️ Development

### Běžné příkazy
```bash
# Optimalizace hyperparametrů
make optimize-hparams

# Benchmark různých komponent
make bench-qdrant
make hrag-bench
make compress-bench

# Ladění RRF parametrů
make sweep-rrf

# Kód quality
make format lint test
```

### Přidání nových zdrojů
1. Implement `BaseSpecializedScraper` v `src/scrapers/`
2. Přidat do `config.yaml` sekce `specialized_sources`
3. Spustit `make smoke-test` pro validaci

## 🐛 Troubleshooting

### Časté problémy

**1. Ollama connection error**
```bash
# Zkontrolovat běžící modely
ollama list

# Stáhnout požadovaný model
ollama pull qwen2.5:7b-q4_K_M
```

**2. Qdrant connection refused**
```bash
# Spustit Qdrant lokálně
docker run -p 6333:6333 qdrant/qdrant

# Nebo použít docker-compose
docker-compose up qdrant
```

**3. Memory issues na M1**
```bash
# Použít quick profil
python main.py --profile quick "your query"

# Snížit batch size v config
batch_size: 4
```

**4. Pomalé výsledky**
- Zkontrolovat `make bench-qdrant` - možná potřeba tuning ef_search
- Použít compression: `compression.enabled: true`
- Vypnout contradiction detection: `verification.contradiction_pass: false`

### Debug módy
```bash
# Verbose logging
python main.py --verbose "query"

# Uložit intermediate outputs
python main.py --save-checkpoints "query"

# Audit retrieval decisions
python main.py --audit-retrieval "query"
```

## 📚 Dokumentace

- [Detailed Architecture](docs/architecture.md)
- [Configuration Guide](docs/configuration.md)
- [API Reference](docs/api.md)
- [Evaluation Metrics](docs/evaluation.md)
- [Contributing Guide](docs/contributing.md)

## 📈 Performance Benchmarks

| Komponenta | M1 MacBook Air 8GB | M1 MacBook Air 16GB |
|------------|---------------------|----------------------|
| Smoke test | 45-60s | 35-45s |
| Quick query | 30-45s | 25-35s |
| Thorough query | 2-3 min | 90-120s |
| Eval suite | 8-12 min | 6-9 min |

## 🤝 Contributing

1. Fork repository
2. Vytvořit feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

### CI/CD Gates
- **Lint**: `make lint` musí projít
- **Tests**: `make test` musí projít  
- **Eval**: Metriky nesmí klesnout pod prahy
- **Smoke**: `make smoke-test` musí vytvořit ≥1 claim s 2 citacemi

## 📝 License

MIT License - viz [LICENSE](LICENSE) file pro detaily.

## 🙏 Acknowledgments

- [Qdrant](https://qdrant.tech/) za vector database
- [Ollama](https://ollama.ai/) za local LLM serving
- [ArchiveBox](https://archivebox.io/) za web archiving
- OpenAlex, Crossref, Unpaywall za open science APIs
