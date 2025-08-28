# LangGraph Research Agent - Moderní Stavová Architektura

## Přehled

Tato implementace nahrazuje původní logiku moderní, stavovou a paměťově vybavenou architekturou pomocí **LangGraph**. Systém je navržen jako stavový automat s integrovaným RAG pipeline a pokročilými nástroji.

## Hlavní Komponenty

### 1. Stavový Automat (LangGraph)

**Soubor:** `src/core/langgraph_agent.py`

Implementuje centrální stavový objekt `ResearchAgentState` pomocí TypedDict:

```python
class ResearchAgentState(TypedDict):
    initial_query: str                                    # Vstupní dotaz
    plan: List[str]                                      # Plán výzkumu
    retrieved_docs: List[Dict[str, Any]]                 # Získané dokumenty
    validation_scores: Dict[str, float]                  # Skóre validace
    synthesis: str                                       # Finální syntéza
    messages: Annotated[List[BaseMessage], operator.add] # Zprávy
```

**Uzly grafu:**
- `plan_step`: Generuje strukturovaný plán výzkumu
- `retrieve_step`: Sbírá dokumenty pomocí RAG + web scraping
- `validate_step`: Ověřuje kvalitu získaných dat
- `synthesize_step`: Vytváří finální syntézu

**Tok dat:** START → plan → retrieve → validate → synthesize → END

### 2. RAG Pipeline

**Soubor:** `src/core/rag_pipeline.py`

Kompletní Retrieval-Augmented Generation pipeline:

#### Chunking Strategie
- **RecursiveCharacterTextSplitter** z LangChain
- **chunk_size**: 1000 (dle specifikace)
- **chunk_overlap**: 150 (dle specifikace)
- Automatické rozdělování podle separátorů

#### Embedding Model
- **BAAI/bge-large-en-v1.5** - specializovaný model pro sémantické vyhledávání
- Nahrazuje výchozí OpenAI embedding pro lepší výkon

#### Hybridní Vyhledávání
- **Sémantické vyhledávání** (70% váha)
- **Keyword vyhledávání** (30% váha)
- **RRF (Reciprocal Rank Fusion)** pro kombinaci výsledků

### 3. Vektorová Databáze

**Soubor:** `src/core/memory.py`

Modulární architektura s abstraktní třídou:

```python
class BaseMemoryStore(ABC):
    # Abstraktní rozhraní pro různé typy úložišť
    
class ChromaMemoryStore(BaseMemoryStore):
    # Konkrétní implementace pro ChromaDB
```

**Vlastnosti:**
- **Lokální ChromaDB** - persistentní úložiště
- **Automatické embeddingy** při ukládání
- **Asynchronní operace** pro lepší výkon
- **Snadná výměna** implementace díky abstrakci

### 4. Nástroje (@tool decorator)

**Soubor:** `src/core/tools.py`

Implementované nástroje:

#### Web Scraping Tool
```python
@tool("web_scraping_tool", args_schema=WebScrapingInput)
async def web_scraping_tool(query: str, scrape_type: str = "url", max_pages: int = 1)
```

- **Firecrawl API** integrace pro pokročilý scraping
- **Fallback** na základní scraping při nedostupnosti
- **Strukturované získání** obsahu s metadaty

#### Knowledge Search Tool
- Vyhledávání v lokální knowledge base
- Automatický fallback na web scraping
- Kombinace lokálních a externích zdrojů

#### Document Analysis Tool
- Analýza textů pomocí LLM
- Podpora pro summary, keywords, entities, sentiment

### 5. Konfigurace

**Soubor:** `src/core/config_langgraph.py`

Centralizovaná konfigurace s profily:

#### Profily
- **quick**: Rychlé odpovědi (4k kontext, GPT-4o-mini)
- **thorough**: Důkladná analýza (8k kontext, GPT-4o)
- **academic**: Vysoké nároky na citace (12k kontext, GPT-4o)

#### Environment Variables
```bash
export OPENAI_API_KEY="your-key"
export FIRECRAWL_API_KEY="your-key"
export CHROMA_DB_PATH="./chroma_db"
```

## Použití

### Základní Použití

```python
from src.core.langgraph_agent import ResearchAgentGraph
from src.core.config_langgraph import load_config

# Načtení konfigurace
config = load_config(profile="thorough")

# Inicializace agenta
agent = ResearchAgentGraph(config)

# Spuštění výzkumu
result = await agent.research("Jaké jsou trendy v AI v roce 2024?")
```

### Příkazová Řádka

```bash
# Nová LangGraph architektura (default)
python main.py "Vaš dotaz" --profile thorough

# Legacy architektura
python main.py "Vaš dotaz" --legacy --profile quick

# Uložení výsledků
python main.py "Vaš dotaz" --output results.json

# Audit mode
python main.py "Vaš dotaz" --audit
```

### Demonstrace

```bash
# Kompletní demo všech komponent
python demo_langgraph_research.py
```

## Technické Detaily

### Asynchronní Architektura
- Všechny operace jsou **async/await**
- Paralelní zpracování retrievalů
- Neblokující embedding generování

### Memory Management
- **Chunking** pro velké dokumenty
- **Batch processing** embeddingů
- **Caching** pro opakované dotazy
- **Memory limit** kontrola

### Error Handling
- **Graceful degradation** při chybách nástrojů
- **Fallback mechanismy** pro external APIs
- **Detailed error tracking** v stavu
- **Retry logic** pro network operace

### Performance Optimizations
- **Thread pool** pro CPU-intensive operace
- **Connection pooling** pro HTTP requesty
- **Embedding caching** v ChromaDB
- **Smart chunking** podle typu obsahu

## Kompatibilita

Systém zachovává **plnou zpětnou kompatibilitu** s původním API:

```python
# Stará syntaxe stále funguje
agent = AutomaticResearchAgent("config.yaml", profile="thorough")
result = await agent.research("dotaz")
```

**Migrace:**
1. Nainstalujte nové závislosti: `pip install -r requirements.txt`
2. Nový kód automaticky použije LangGraph architekturu
3. Pro legacy režim použijte `--legacy` flag

## Výhody Nové Architektury

### 🏗️ Stavová Architektura
- **Přehledný tok** dat mezi kroky
- **Sledovatelnost** každého stavu
- **Možnost checkpointů** a recovery
- **Snadné rozšiřování** o nové uzly

### 🧠 Pokročilý RAG
- **Specializované embeddingy** pro lepší vyhledávání
- **Hybridní retrieval** (sémantický + keyword)
- **Modulární úložiště** pro snadnou výměnu
- **Inteligentní chunking** strategií

### 🛠️ Nástroje
- **LangChain @tool** integrace
- **Standardizované rozhraní** pro nástroje
- **Async-first** implementace
- **Robust error handling**

### ⚡ Performance
- **Paralelní zpracování** multiple retrievalů
- **Smart caching** embeddingů a výsledků
- **Memory-efficient** chunking
- **Fast vector search** s ChromaDB

### 🔧 Maintenance
- **Type-safe** implementace s TypedDict
- **Comprehensive logging** pro debugging
- **Modular design** pro snadné úpravy
- **Extensive configuration** možnosti

## Další Kroky

1. **Testování** nové architektury s reálnými dotazy
2. **Optimalizace** embedding modelů pro specifické domény
3. **Rozšíření** nástrojů o další data sources
4. **Implementace** pokročilých RAG technik (HyDE, CoT)
5. **Monitoring** a metrics pro production nasazení
