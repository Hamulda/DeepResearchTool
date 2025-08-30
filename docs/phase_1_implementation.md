# Fáze 1: Základní Architektura a Bezpečnost 🏗️

Tato fáze implementuje bezpečné, kontejnerizované a výkonnostně optimalizované jádro aplikace, které je odolné a připravené na škálování.

## 🎯 Cíle Fáze 1

- ✅ **Kontejnerizace**: Docker orchestrace s izolovanými službami
- ✅ **Bezpečnost**: Šifrování, anonymizace a etické scrapování
- ✅ **ELT Pipeline**: Paměťově efektivní zpracování dat
- ✅ **RAG Systém**: Lokální vektorová databáze s embeddings
- ✅ **Lokální LLM**: Metal-optimalizované inference pro Apple Silicon

## 🏗️ Architektura

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Scraper       │    │   Tor Proxy     │    │  FlareSolverr   │
│   (FastAPI)     │◄──►│  (Anonymizace)  │◄──►│  (Anti-bot)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │
          ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  ELT Pipeline   │    │   Milvus Lite   │    │    DuckDB       │
│  (Parquet)      │◄──►│  (Embeddings)   │◄──►│  (Analytics)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │
          ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  RAG System     │◄──►│  Local LLM      │◄──►│  Prometheus     │
│  (Retrieval)    │    │  (llama.cpp)    │    │  (Monitoring)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Rychlý Start

### Předpoklady

- **Docker & Docker Compose** (20.10+)
- **Python 3.11+** 
- **Apple Silicon** (M1/M2/M3) pro optimální výkon
- **8GB+ RAM** doporučeno
- **20GB+ volného místa** pro modely a data

### 1. Klonování a Příprava

```bash
# Přepnutí na feature větev
git checkout feature/autonomous-platform

# Spuštění celé platformy
./scripts/phase1_launcher.sh start
```

Skript automaticky:
- Zkontroluje systémové požadavky
- Nainstaluje Python závislosti
- Stáhne doporučený LLM model (~4GB)
- Spustí všechny Docker služby
- Ověří zdraví systému

### 2. Manuální Konfigurace (Optional)

```bash
# Kopírování .env template
cp .env.template .env

# Úprava konfigurace
nano .env  # Přidejte API klíče

# Virtuální prostředí
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🔧 Konfigurace

### Klíčové Proměnné (.env)

```bash
# API klíče
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# LLM konfigurace
LLM_MODEL_PATH=./models/mistral-7b-instruct-q4_k_m.gguf
LLM_THREADS=8
LLM_METAL=true

# Performance tuning
CHUNK_SIZE=1000
EMBEDDING_BATCH_SIZE=32
MAX_WORKERS=4
```

### Apple Silicon Optimalizace

```bash
# Metal akcelerace pro LLM
CMAKE_ARGS="-DLLAMA_METAL=on"
FORCE_CMAKE=1
LLAMA_CPP_THREADS=8

# Sentence-transformers optimalizace
MPS_DEVICE=true
```

## 📡 API Endpointy

Po spuštění je dostupné REST API na `http://localhost:8000`:

### Core Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Systémové statistiky
curl http://localhost:8000/stats

# Ingest dokumentů
curl -X POST http://localhost:8000/documents/ingest \
  -H "Content-Type: application/json" \
  -d '[{"id": "1", "title": "Test", "content": "Test content", "url": "http://example.com", "source": "api"}]'

# RAG dotazování
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is artificial intelligence?", "top_k": 5}'

# Chat konverzace
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello, how are you?"}], "use_rag": true}'
```

### Data Pipeline

```bash
# Indexování Parquet dat do vektorové DB
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"table_name": "research_documents", "content_column": "content"}'
```

## 🛠️ Komponenty

### 1. ELT Pipeline (`src/core/elt_pipeline.py`)

**Paměťově efektivní zpracování dat:**

```python
from src.core.elt_pipeline import ELTPipeline

# Inicializace
pipeline = ELTPipeline(data_dir="./data/parquet", chunk_size=1000)

# Streaming ingest
async def data_stream():
    for item in large_dataset:
        yield item

await pipeline.extract_and_load(data_stream(), "documents", "source")

# Analýza bez načítání do RAM
stats = await pipeline.transform_and_analyze("documents")
```

**Výhody:**
- ✅ Konstantní využití RAM bez ohledu na velikost dat
- ✅ Apache Parquet s Snappy kompresí
- ✅ DuckDB pro rychlé analytické dotazy
- ✅ Streaming processing

### 2. RAG Systém (`src/core/rag_system.py`)

**Vektorové vyhledávání s lokálními embeddings:**

```python
from src.core.rag_system import LocalRAGSystem

# Inicializace s Apple Silicon optimalizací
rag = LocalRAGSystem(
    data_dir="./data/parquet",
    model_name="all-MiniLM-L6-v2"  # 384D embeddings
)

# Indexování dokumentů
await rag.index_documents_from_parquet("research_docs")

# Vyhledání relevantního kontextu
context = await rag.search_relevant_context(
    query="machine learning trends",
    top_k=5,
    score_threshold=0.7
)
```

**Komponenty:**
- ✅ **sentence-transformers** s Metal podporou
- ✅ **Milvus Lite** pro vektorovou databázi
- ✅ **COSINE similarity** s IVF_FLAT indexem
- ✅ Batch processing pro efektivitu

### 3. Lokální LLM (`src/core/local_llm.py`)

**Metal-optimalizované LLM inference:**

```python
from src.core.local_llm import RAGLLMPipeline, LLMConfig

# Konfigurace pro Apple Silicon
config = LLMConfig(
    model_path="./models/mistral-7b-instruct-q4_k_m.gguf",
    n_ctx=4096,
    n_threads=8,
    metal=True  # Metal akcelerace
)

# RAG + LLM pipeline
pipeline = RAGLLMPipeline(rag_system=rag, llm_config=config)

# Dotazování s kontextem
result = await pipeline.answer_question(
    "What are the latest AI developments?",
    top_k=5
)
```

**Podporované modely:**
- ✅ **Mistral 7B** (doporučeno)
- ✅ **Llama 2 7B**
- ✅ **CodeLlama 7B**
- ✅ Všechny GGUF formáty

### 4. Autonomní Server (`src/core/autonomous_server.py`)

**Production-ready FastAPI server:**

```python
# Automatické spuštění všech komponent
# Health checks, metrics, background tasks
# Prometheus monitoring
# Error handling a retry logika
```

## 🔒 Bezpečnost

### 1. Anonymizace (Tor Proxy)

- **Rotující IP adresy** přes Tor síť
- **Legal whitelist** pouze pro etické domény
- **Rate limiting** respektující robots.txt

### 2. Anti-Bot Obrana (FlareSolverr)

- **CloudFlare bypass** pro akademické zdroje
- **CAPTCHA handling** 
- **User-Agent rotation**

### 3. Šifrování a Secrets

```bash
# Automatické skenování secrets při buildu
RUN ggshield secret scan path /app --recursive

# Všechny citlivé údaje v .env
# Žádné hardcoded klíče v kódu
# Neprivilegovaný Docker user
```

### 4. Compliance

- ✅ **GDPR** compliance
- ✅ **Academic use** only
- ✅ **Ethical scraping** guidelines
- ✅ **No personal data** collection

## 📊 Monitoring

### Prometheus Metriky (`http://localhost:9090`)

```bash
# Aplikační metriky
autonomous_requests_total
autonomous_request_duration_seconds
autonomous_documents_indexed_total
autonomous_queries_processed_total

# Systémové metriky
milvus_vector_count
duckdb_query_time
llm_inference_duration
```

### Health Checks

```bash
# Kontinuální monitoring
curl http://localhost:8000/health
curl http://localhost:9091/healthz  # Milvus
curl http://localhost:9090/-/healthy  # Prometheus
```

## 🧪 Testování a Validace

### Automatická Validace

```bash
# Kompletní validace všech komponent
./scripts/phase1_launcher.sh validate

# Výsledky ukládány do artifacts/
cat artifacts/phase1_validation_results.json
```

### Manuální Testy

```bash
# Unit testy
pytest tests/test_phase1_integration.py -v

# Integration testy
python scripts/validate_phase1.py

# Performance benchmarky
python scripts/bench_m1_performance.py
```

## 📈 Výkonnostní Charakteristiky

### Apple Silicon (M1/M2/M3)

- **LLM Inference**: ~50-100 tokens/s (Mistral 7B Q4)
- **Embeddings**: ~1000 docs/s (batch 32)
- **Parquet Write**: ~10MB/s streaming
- **Vector Search**: <100ms (10k documents)

### Paměťové Nároky

- **Base System**: ~2GB RAM
- **LLM Model**: ~4GB RAM (Q4 quantization)
- **Embeddings**: ~384MB (1M documents)
- **ELT Pipeline**: <500MB (konstantní)

## 🚨 Troubleshooting

### Časté Problémy

1. **Metal nefunguje**:
   ```bash
   # Fallback na CPU
   export CMAKE_ARGS=""
   pip reinstall llama-cpp-python
   ```

2. **Milvus se nespustí**:
   ```bash
   docker-compose logs milvus
   # Zkontrolujte port 19530
   ```

3. **Model nenalezen**:
   ```bash
   # Manuální stažení
   mkdir -p models/
   wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.q4_k_m.gguf -O models/mistral-7b-instruct-q4_k_m.gguf
   ```

4. **Docker out of space**:
   ```bash
   docker system prune -f
   docker volume prune -f
   ```

### Debug Logging

```bash
# Detailní logy
export LOG_LEVEL=DEBUG
docker-compose -f docker-compose.autonomous.yml logs -f scraper
```

## 🔄 Operace

### Správa Služeb

```bash
# Start platformy
./scripts/phase1_launcher.sh start

# Status check
./scripts/phase1_launcher.sh status

# Graceful stop
./scripts/phase1_launcher.sh stop

# Restart
./scripts/phase1_launcher.sh restart

# Kompletní cleanup
./scripts/phase1_launcher.sh cleanup --full
```

### Data Management

```bash
# Parquet soubory
ls -la data/parquet/

# Vector embeddings  
docker exec autonomous_milvus ./milvus_cli.py show collections

# Monitoring data
docker exec autonomous_prometheus ./promtool query instant 'up'
```

## 🎯 Úspěšná Implementace

Po dokončení Fáze 1 máte funkční:

✅ **Kontejnerizovanou platformu** s Docker orchestrací  
✅ **Bezpečný scraping** s Tor anonymizací  
✅ **ELT pipeline** s Apache Parquet optimalizací  
✅ **RAG systém** s lokálními embeddings  
✅ **Lokální LLM** s Metal akcelerací  
✅ **REST API** pro všechny operace  
✅ **Monitoring** s Prometheus metrikami  
✅ **Automatické testy** a validace  

## 🚀 Další Kroky

Fáze 1 vytváří solidní základ pro:

- **Fáze 2**: Pokročilé RAG techniky (Hybrid search, Re-ranking)
- **Fáze 3**: Multi-agent orchestrace
- **Fáze 4**: Produkční škálování a optimalizace

---

## 📞 Podpora

Pro otázky a problémy:
- Zkontrolujte logy: `docker-compose logs`
- Spusťte validaci: `./scripts/phase1_launcher.sh validate`  
- Zkontrolujte health endpointy
- Použijte debug mód s `LOG_LEVEL=DEBUG`
