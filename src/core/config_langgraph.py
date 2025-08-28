"""
Konfigurace pro moderní LangGraph Research Agent
Nastavení pro všechny komponenty nové stavové architektury

Author: Senior Python/MLOps Agent
"""

# Hlavní konfigurace pro LangGraph Research Agent
LANGGRAPH_CONFIG = {
    # LLM konfigurace
    "llm": {
        "model": "gpt-4o-mini",                    # Model pro plánování a validaci
        "synthesis_model": "gpt-4o",               # Pokročilý model pro syntézu
        "temperature": 0.1,                        # Nízká kreativita pro konzistentní plánování
        "synthesis_temperature": 0.2,              # Mírně vyšší kreativita pro syntézu
        "max_tokens": 4000,                        # Maximální délka odpovědi
        "timeout": 60                              # Timeout pro LLM volání (sekundy)
    },

    # RAG Pipeline konfigurace
    "rag": {
        "enabled": True,
        "chunking": {
            "chunk_size": 1000,                    # Velikost chunků (dle specifikace)
            "chunk_overlap": 150,                  # Překryv chunků (dle specifikace)
            "separators": ["\n\n", "\n", " ", ""] # Separátory pro rozdělování
        },
        "retrieval": {
            "k": 5,                                # Počet dokumentů k vrácení při vyhledávání
            "max_context_tokens": 4000,            # Maximální kontext pro LLM
            "similarity_threshold": 0.7             # Prahová hodnota pro similarity
        },
        "hybrid": {
            "enabled": True,
            "semantic_weight": 0.7,                # Váha sémantického vyhledávání
            "keyword_weight": 0.3                  # Váha keyword vyhledávání
        }
    },

    # Memory Store konfigurace (ChromaDB)
    "memory_store": {
        "type": "chroma",                          # Typ úložiště
        "collection_name": "research_documents",   # Název kolekce
        "persist_directory": "./chroma_db",        # Adresář pro persistenci
        "embedding_model": "BAAI/bge-large-en-v1.5" # Specializovaný embedding model
    },

    # Synthesis konfigurace
    "synthesis": {
        "max_docs": 10,                           # Maximální počet dokumentů pro syntézu
        "min_citations_per_claim": 2,             # Minimální citace na tvrzení
        "max_claims": 8,                          # Maximální počet hlavních tvrzení
        "include_metadata": True,                  # Zahrnout metadata zdrojů
        "format": "structured_markdown"            # Formát výstupu
    },

    # Validation konfigurace
    "validation": {
        "min_relevance": 0.7,                     # Minimální relevance dokumentů
        "min_coverage": 0.6,                      # Minimální pokrytí plánu
        "min_quality": 0.5,                       # Minimální kvalita zdrojů
        "enable_llm_validation": True             # Použít LLM pro validaci
    },

    # Tools konfigurace
    "tools": {
        "web_scraping": {
            "enabled": True,
            "max_pages_per_search": 3,             # Maximální počet stránek na vyhledávání
            "timeout": 30,                         # Timeout pro scraping
            "firecrawl_api_key": None              # API klíč pro Firecrawl (env variable)
        },
        "knowledge_search": {
            "enabled": True,
            "fallback_to_web": True,               # Fallback na web pokud není v KB
            "max_local_results": 5                 # Maximální lokální výsledky
        }
    },

    # Performance konfigurace
    "performance": {
        "max_concurrent_retrievals": 3,           # Maximální současné retrievaly
        "cache_embeddings": True,                 # Cachování embeddingů
        "batch_size": 10,                         # Velikost batch pro embedding
        "memory_limit_mb": 1024                   # Limit paměti v MB
    },

    # Logging konfigurace
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "./logs/research_agent.log",
        "max_file_size": "10MB",
        "backup_count": 5
    },

    # Security konfigurace
    "security": {
        "allowed_domains": [],                     # Povolené domény pro scraping ([] = všechny)
        "blocked_domains": [                       # Zakázané domény
            "facebook.com",
            "twitter.com",
            "instagram.com"
        ],
        "max_file_size_mb": 50,                   # Maximální velikost souboru
        "enable_content_filtering": True           # Filtrování nevhodného obsahu
    }
}

# Přednastavené profily pro různé use casy
RESEARCH_PROFILES = {
    "quick": {
        # Rychlý profil pro okamžité odpovědi
        "llm": {
            "model": "gpt-4o-mini",
            "max_tokens": 2000
        },
        "rag": {
            "retrieval": {"k": 3, "max_context_tokens": 2000}
        },
        "synthesis": {
            "max_docs": 5,
            "max_claims": 3
        },
        "validation": {
            "min_relevance": 0.6,
            "min_coverage": 0.5
        }
    },

    "thorough": {
        # Důkladný profil pro kompletní analýzu
        "llm": {
            "model": "gpt-4o",
            "max_tokens": 8000
        },
        "rag": {
            "retrieval": {"k": 10, "max_context_tokens": 8000}
        },
        "synthesis": {
            "max_docs": 15,
            "max_claims": 10
        },
        "validation": {
            "min_relevance": 0.8,
            "min_coverage": 0.7
        }
    },

    "academic": {
        # Akademický profil s vysokými nároky na citace
        "llm": {
            "model": "gpt-4o",
            "synthesis_model": "gpt-4o",
            "temperature": 0.05
        },
        "synthesis": {
            "min_citations_per_claim": 3,
            "max_claims": 12,
            "include_metadata": True
        },
        "validation": {
            "min_relevance": 0.85,
            "min_coverage": 0.8,
            "min_quality": 0.7
        }
    }
}

# Environment variables mapping
ENV_VARIABLES = {
    "OPENAI_API_KEY": "llm.api_key",
    "FIRECRAWL_API_KEY": "tools.web_scraping.firecrawl_api_key",
    "CHROMA_DB_PATH": "memory_store.persist_directory",
    "LOG_LEVEL": "logging.level",
    "MAX_MEMORY_MB": "performance.memory_limit_mb"
}


def load_config(profile: str = "default", config_overrides: dict = None) -> dict:
    """
    Načtení konfigurace s možností přepsání hodnot

    Args:
        profile: Název profilu ("default", "quick", "thorough", "academic")
        config_overrides: Slovník s přepsanými hodnotami

    Returns:
        Kompletní konfigurace
    """
    import os
    from copy import deepcopy

    # Začni se základní konfigurací
    config = deepcopy(LANGGRAPH_CONFIG)

    # Aplikuj profil pokud je zadán
    if profile in RESEARCH_PROFILES:
        profile_config = RESEARCH_PROFILES[profile]
        config = _deep_merge(config, profile_config)

    # Aplikuj environment variables
    for env_var, config_path in ENV_VARIABLES.items():
        env_value = os.getenv(env_var)
        if env_value:
            _set_nested_value(config, config_path, env_value)

    # Aplikuj custom overrides
    if config_overrides:
        config = _deep_merge(config, config_overrides)

    return config


def _deep_merge(base_dict: dict, override_dict: dict) -> dict:
    """Hluboké sloučení slovníků"""
    from copy import deepcopy

    result = deepcopy(base_dict)

    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _set_nested_value(dictionary: dict, path: str, value: str) -> None:
    """Nastavení vnořené hodnoty pomocí dot notation"""
    keys = path.split('.')
    current = dictionary

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Pokus o konverzi typu
    final_key = keys[-1]
    if value.lower() in ('true', 'false'):
        current[final_key] = value.lower() == 'true'
    elif value.isdigit():
        current[final_key] = int(value)
    elif value.replace('.', '').isdigit():
        current[final_key] = float(value)
    else:
        current[final_key] = value


# Validace konfigurace
def validate_config(config: dict) -> list:
    """
    Validace konfigurace a vrácení seznamu chyb

    Args:
        config: Konfigurace k validaci

    Returns:
        Seznam chybových zpráv (prázdný = OK)
    """
    errors = []

    # Kontrola povinných sekcí
    required_sections = ['llm', 'memory_store', 'rag']
    for section in required_sections:
        if section not in config:
            errors.append(f"Chybí povinná sekce: {section}")

    # Kontrola LLM konfigurace
    if 'llm' in config:
        if 'model' not in config['llm']:
            errors.append("Chybí llm.model v konfiguraci")

    # Kontrola memory store
    if 'memory_store' in config:
        if config['memory_store'].get('type') not in ['chroma']:
            errors.append("Nepodporovaný typ memory_store")

    # Kontrola chunking parametrů
    if 'rag' in config and 'chunking' in config['rag']:
        chunk_size = config['rag']['chunking'].get('chunk_size', 0)
        chunk_overlap = config['rag']['chunking'].get('chunk_overlap', 0)

        if chunk_size <= 0:
            errors.append("chunk_size musí být větší než 0")
        if chunk_overlap >= chunk_size:
            errors.append("chunk_overlap musí být menší než chunk_size")

    return errors
