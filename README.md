# 🔍 Deep Research Tool v2.0 - Pokročilý Výzkumný Agent

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MacBook Air M1 Optimalizováno](https://img.shields.io/badge/M1-optimalizováno-green.svg)](https://developer.apple.com/metal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Pokročilý výzkumný agent** pro hloubkovou analýzu informací s evidenčními vazbami, hybridním vyhledáváním a rozšířenými konektory do specializovaných zdrojů. Optimalizováno pro **čistě lokální provoz** na MacBook Air M1.

## 📖 Co je Deep Research Tool?

Deep Research Tool je inteligentní výzkumný asistent, který kombinuje nejmodernější technologie umělé inteligence pro provádění hloubkové analýzy informací. Nástroj je navržen tak, aby:

- **Automaticky vyhledával a analyzoval** informace z r��zných zdrojů
- **Ověřoval fakta** pomocí více nezávislých citací
- **Detekoval rozpory** mezi různými zdroji informací
- **Generoval strukturované reporty** s důkazními vazbami
- **Běžel lokálně** bez nutnosti cloudových služeb

## 🎯 Hlavní použití

### Pro výzkumníky a studenty
- Rychlá analýza vědeckých publikací
- Fact-checking a ověřování zdrojů
- Syntéza informací z různých oblastí
- Tracking nejnovějších vývoj v oboru

### Pro novináře a analytiky
- Investigativní žurnalistika s ověřováním faktů
- Analýza trendů a veřejných politik
- Cross-reference různých zpravodajských zdrojů
- Historická analýza událostí

### Pro právníky a compliance
- Research právních precedentů
- Analýza regulatorních změn
- Due diligence dokumentace
- Risk assessment reports

## ✨ Klíčové Funkce v2.0

### 🧠 Inteligentní Workflow Orchestrátor
- **Evidence-based syntéza** - každý claim musí mít minimálně 2 nezávislé citace
- **Modulární pipeline**: vyhledávání → přehodnocení → syntéza → ověření
- **Paralelní zpracování** subdotazů s inteligentním sloučením
- **Human-in-the-loop** kontrolní body pro kritické review

### 🔄 Hybridní Vyhledávací Engine  
- **Hierarchické vyhledávání** - metadata-first → sekce → pasáže
- **Qdrant** vektorová databáze + **BM25** sparse retrieval
- **Reciprocal Rank Fusion (RRF)** pro optimální kombinaci výsledků
- **M1 Metal** akcelerované embeddings pro maximální výkon

### 🎯 Pokročilé Přehodnocení & Komprese
- **Cross-encoder** nebo **LLM-as-rater** s multi-kritéria skórováním
- **Kontextová komprese** - salience filtering + token budgeting
- **Konfigurovatelné váhy**: relevance, autorita, novost, aktualnost
- **Audit logy** s odůvodněním každého přeskupení

### 🕸️ Claim Graph & Detekce Rozpor��
- **Claim graph** s support/contradict vztahy
- **Ověření rozporů** - aktivní vyhledávání protikladné evidence
- **Confidence scoring** s penalizací pro sporné claims
- **Conflict sets** pro sporné oblasti

### 📚 Specializované Konektory Zdrojů
- **Common Crawl** - 15+ let web archivů s přesnými WARC pozicemi
- **Memento** - TimeGate/TimeMap time-travel s diff analýzou
- **Ahmia** - Tor OSINT (pouze legální filtrování)
- **Právní API** - CourtListener/RECAP + SEC EDGAR
- **Open Access věda** - OpenAlex → Crossref → Unpaywall → Europe PMC

### 📊 Kontinuální Hodnocení & Optimalizace
- **Regresní test set** s 10+ specializovanými tématy
- **Metriky kvality**: recall@k, evidence coverage, answer faithfulness
- **Bayesian hyperparameter optimalizace** pro per-domain tuning
- **Performance tracking** s JSON/Markdown reporting

### 🍎 MacBook Air M1 Optimalizace
- **Ollama modely**: 3B-8B v Q4_K_M kvantizaci pro optimální rychlost/kvalitu
- **MPS/Metal** akcelerace s inteligentním batch sizing
- **Memory-efficient** zpracování s 4k-8k context windows
- **Streaming inference** s progresivním context building

## 🚀 Rychlý Start

### Systémové požadavky
- **Python 3.9+**
- **MacBook Air M1** (doporučeno) nebo Intel Mac/Linux
- **16GB RAM** (minimum 8GB)
- **20GB volného místa** pro modely a cache

### Instalace
```bash
# 1. Klonovat repozitář
git clone https://github.com/vojtechhamada/DeepResearchTool.git
cd DeepResearchTool

# 2. Kompletní setup včetně dependencies a validace
make setup

# 3. Spustit rychlý smoke test (< 60s)
make smoke-test

# 4. Volitelně: spustit kompletní evaluaci
make eval
```

### První dotaz
```bash
# Základní research query
python main.py "Jaké jsou nejnovější vývoje v kvantové korekci chyb?"

# S konkrétním profilem
python main.py --profile thorough "Právní důsledky AI bias při náboru zaměstnanců"

# CLI režim pro interaktivní session
python cli.py
```

## 📂 Struktura Projektu

```
DeepResearchTool/
├── main.py                    # Hlavní vstupní bod aplikace
├── cli.py                     # Příkazový řádek interface
├── config.yaml               # Hlavní konfigurace
├── config_m1_local.yaml      # M1 optimalizovaná konfigurace
├── src/                      # Zdrojové kódy
│   ├── core/                 # Základní komponenty
│   ├── agents/               # AI agenti a orchestrátoři
│   ├── retrieval/            # Vyhledávací engine
│   ├── synthesis/            # Syntéza a analýza
│   ├── scrapers/             # Specializované scrapery
│   └── utils/                # Pomocné funkce
├── tests/                    # Testovací suite
├── docs/                     # Dokumentace
├── docker/                   # Docker konfigurace
└── demo_*.py                 # Demo skripty pro různé fáze
```

## 📋 Očekávané Výstupy

### Smoke Test Výsledek
```
🧪 Spouštím smoke test s profilem: quick
✅ Připojení k Ollama: OK (qwen2.5:7b-q4_K_M)
✅ Qdrant health check: OK (3 kolekce)
✅ Dotaz: "Účinnost COVID-19 vakcín"
✅ Vygenerováno 2 claims s důkazy:
   - Claim 1: mRNA vakcíny vykazují 95% účinnost (2 citace)
   - Claim 2: Posilující dávky obnovují ochranu (3 citace)
⏱️  Celkový čas: 45.2s
```

### Evaluation Report
```json
{
  "timestamp": "2025-08-30T14:30:00Z",
  "profile": "quick",
  "metriky": {
    "vyhledávání": {
      "recall@10": 0.85,
      "precision@10": 0.72,
      "nDCG@10": 0.78
    },
    "syntéza": {
      "groundedness": 0.91,
      "citation_precision": 0.88,
      "context_usage_efficiency": 0.73
    },
    "ověření": {
      "hallucination_rate": 0.06,
      "confidence_calibration": 0.82
    }
  }
}
```

## 🔧 Konfigurace

### Profily
- **quick**: Rychlé výsledky (< 60s), základní hierarchické vyhledávání
- **thorough**: Detailní analýza (2-5 min), plný claim graph + detekce rozporů

### Hlavní konfigurační soubor
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

## 🛠️ Vývoj a Údržba

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

## 🐛 Řešení Problémů

### Časté problémy a řešení

**1. Chyba připojení k Ollama**
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

**3. Problémy s pamětí na M1**
```bash
# Použít quick profil
python main.py --profile quick "váš dotaz"

# Snížit batch size v konfiguraci
batch_size: 4
```

## 📊 Performance Benchmarky

| Komponenta | M1 MacBook Air 8GB | M1 MacBook Air 16GB |
|------------|---------------------|----------------------|
| Smoke test | 45-60s | 35-45s |
| Quick dotaz | 30-45s | 25-35s |
| Thorough dotaz | 2-3 min | 90-120s |
| Eval suite | 8-12 min | 6-9 min |

## 📄 Licence

MIT License - viz [LICENSE](LICENSE) soubor pro detaily.

## 🙏 Poděkování

- [Qdrant](https://qdrant.tech/) za vector database
- [Ollama](https://ollama.ai/) za local LLM serving  
- [ArchiveBox](https://archivebox.io/) za web archiving
- OpenAlex, Crossref, Unpaywall za open science APIs
