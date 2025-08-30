#!/usr/bin/env python3
"""DuckDB Data Warehouse pro pre-filtraci dat před vektorovým vyhledáváním
Dramaticky snižuje prohledávaný prostor ve vektorové databázi o řády

Author: Senior Python/MLOps Agent
"""

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import re
from typing import Any

# Optional import with fallback
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False
    duckdb = None

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Reprezentace dokumentu v data warehouse"""

    id: str
    content: str
    source: str
    metadata: dict[str, Any]
    timestamp: datetime
    keywords: set[str]


class DataWarehouse:
    """DuckDB-powered data warehouse pro rychlou pre-filtraci dokumentů
    Používá fulltextové vyhledávání a keyword matching pro redukci prohledávaného prostoru
    """

    def __init__(self, db_path: str = "data/warehouse.duckdb", memory_limit: str = "2GB"):
        """Inicializace DuckDB data warehouse
        
        Args:
            db_path: Cesta k DuckDB databázi
            memory_limit: Limit paměti pro DuckDB operace

        """
        self.db_path = Path(db_path)
        self.memory_limit = memory_limit
        self.conn = None

        if not HAS_DUCKDB:
            logger.warning("DuckDB not available, data warehouse disabled")
            return

        # Vytvoření adresáře pokud neexistuje
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Inicializace DuckDB warehouse: {self.db_path}")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self) -> None:
        """Připojení k DuckDB databázi s M1 optimalizacemi"""
        if not HAS_DUCKDB:
            logger.error("DuckDB not available for connection")
            return

        try:
            self.conn = duckdb.connect(str(self.db_path))

            # M1 optimalizace
            self.conn.execute(f"SET memory_limit = '{self.memory_limit}'")
            self.conn.execute("SET threads = 4")  # Optimální pro M1
            self.conn.execute("SET enable_object_cache = true")
            self.conn.execute("SET temp_directory = '/tmp/duckdb'")

            # Instalace rozšíření pro fulltext search
            try:
                self.conn.execute("INSTALL fts")
                self.conn.execute("LOAD fts")
            except Exception as e:
                logger.warning(f"FTS rozšíření není k dispozici: {e}")

            logger.info("✅ DuckDB připojení úspěšné s M1 optimalizacemi")

        except Exception as e:
            logger.error(f"❌ Chyba připojení k DuckDB: {e}")
            raise

    def disconnect(self) -> None:
        """Odpojení od databáze"""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("DuckDB odpojeno")

    def init_db(self) -> None:
        """Inicializace databázového schématu
        Vytvoří tabulky a indexy optimalizované pro rychlé vyhledávání
        """
        if not self.conn:
            raise RuntimeError("Databáze není připojena")

        try:
            # Hlavní tabulka dokumentů
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata JSON,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    keywords TEXT,  -- JSON array jako string pro rychlé vyhledávání
                    content_length INTEGER,
                    source_type TEXT,
                    language TEXT DEFAULT 'en'
                )
            """)

            # Indexy pro rychlé vyhledávání
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_timestamp ON documents(timestamp)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_content_length ON documents(content_length)")

            # Tabulka pro keyword lookup (normalizovaná)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS document_keywords (
                    document_id TEXT,
                    keyword TEXT,
                    frequency INTEGER DEFAULT 1,
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                )
            """)

            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON document_keywords(keyword)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_keywords_doc_id ON document_keywords(document_id)")

            # Materializovaný view pro rychlé statistiky
            self.conn.execute("""
                CREATE OR REPLACE VIEW document_stats AS
                SELECT 
                    source_type,
                    COUNT(*) as doc_count,
                    AVG(content_length) as avg_length,
                    MAX(timestamp) as latest_timestamp
                FROM documents 
                GROUP BY source_type
            """)

            logger.info("✅ DuckDB schéma inicializováno")

        except Exception as e:
            logger.error(f"❌ Chyba inicializace schématu: {e}")
            raise

    def extract_keywords(self, text: str, max_keywords: int = 50) -> set[str]:
        """Extrakce klíčových slov z textu pro rychlé vyhledávání
        
        Args:
            text: Vstupní text
            max_keywords: Maximální počet klíčových slov
            
        Returns:
            Množina klíčových slov

        """
        # Jednoduchá ale efektivní extrakce klíčových slov
        # Normalizace textu
        text = text.lower()

        # Odebrání speciálních znaků a rozdělení na slova
        words = re.findall(r'\b[a-z]{3,}\b', text)

        # Filtrování stop words (základní sada)
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'put', 'say', 'she', 'too', 'use', 'way', 'will', 'with'
        }

        # Počítání frekvence slov
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) >= 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Vrácení nejčastějších slov
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return {word for word, freq in sorted_words[:max_keywords]}

    def add_documents(self, docs: list[dict[str, Any]]) -> None:
        """Hromadné přidání dokumentů do warehouse
        
        Args:
            docs: Seznam dokumentů s poli: id, content, source, metadata

        """
        if not self.conn:
            raise RuntimeError("Databáze není připojena")

        if not docs:
            logger.warning("Žádné dokumenty k přidání")
            return

        try:
            # Příprava dat pro vložení
            document_data = []
            keyword_data = []

            for doc in docs:
                doc_id = doc['id']
                content = doc['content']
                source = doc['source']
                metadata = doc.get('metadata', {})

                # Extrakce klíčových slov
                keywords = self.extract_keywords(content)

                # Určení typu zdroje
                source_type = self._determine_source_type(source, metadata)

                # Příprava dat dokumentu
                document_data.append({
                    'id': doc_id,
                    'content': content,
                    'source': source,
                    'metadata': json.dumps(metadata),
                    'keywords': json.dumps(list(keywords)),
                    'content_length': len(content),
                    'source_type': source_type,
                    'language': metadata.get('language', 'en')
                })

                # Příprava dat klíčových slov
                for keyword in keywords:
                    keyword_data.append({
                        'document_id': doc_id,
                        'keyword': keyword
                    })

            # Hromadné vložení dokumentů
            self.conn.executemany("""
                INSERT OR REPLACE INTO documents 
                (id, content, source, metadata, keywords, content_length, source_type, language)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (d['id'], d['content'], d['source'], d['metadata'],
                 d['keywords'], d['content_length'], d['source_type'], d['language'])
                for d in document_data
            ])

            # Vymazání starých klíčových slov pro tyto dokumenty
            doc_ids = [d['id'] for d in document_data]
            placeholders = ','.join(['?' for _ in doc_ids])
            self.conn.execute(f"DELETE FROM document_keywords WHERE document_id IN ({placeholders})", doc_ids)

            # Vložení nových klíčových slov
            if keyword_data:
                self.conn.executemany("""
                    INSERT INTO document_keywords (document_id, keyword)
                    VALUES (?, ?)
                """, [(k['document_id'], k['keyword']) for k in keyword_data])

            logger.info(f"✅ Přidáno {len(docs)} dokumentů do warehouse")

        except Exception as e:
            logger.error(f"❌ Chyba při přidávání dokumentů: {e}")
            raise

    def query_ids_by_keywords(self, keywords: list[str], limit: int = 1000) -> list[str]:
        """Rychlé vyhledání ID dokumentů podle klíčových slov
        Toto je klíčová optimalizace - redukuje prostor pro vektorové vyhledávání
        
        Args:
            keywords: Seznam klíčových slov pro vyhledání
            limit: Maximální počet výsledků
            
        Returns:
            Seznam ID dokumentů seřazených podle relevance

        """
        if not self.conn:
            raise RuntimeError("Databáze není připojena")

        if not keywords:
            logger.warning("Žádná klíčová slova pro vyhledání")
            return []

        try:
            # Normalizace klíčových slov
            normalized_keywords = [kw.lower().strip() for kw in keywords if len(kw.strip()) >= 3]

            if not normalized_keywords:
                return []

            # Vytvoření SQL dotazu s keyword matchingem
            placeholders = ','.join(['?' for _ in normalized_keywords])

            # Pokročilý dotaz s váženým skórováním
            query = f"""
                WITH keyword_matches AS (
                    SELECT 
                        dk.document_id,
                        COUNT(DISTINCT dk.keyword) as keyword_count,
                        SUM(dk.frequency) as total_frequency,
                        MAX(CASE WHEN dk.keyword IN ({placeholders}) THEN 1 ELSE 0 END) as has_exact_match
                    FROM document_keywords dk
                    WHERE dk.keyword IN ({placeholders})
                       OR dk.keyword LIKE ANY(SELECT '%' || ? || '%' FROM (VALUES {','.join(['(?)' for _ in normalized_keywords])}) AS t(v))
                    GROUP BY dk.document_id
                ),
                scored_documents AS (
                    SELECT 
                        d.id,
                        d.source_type,
                        d.content_length,
                        d.timestamp,
                        km.keyword_count,
                        km.total_frequency,
                        km.has_exact_match,
                        -- Skóre kombinující různé faktory
                        (
                            km.keyword_count * 10 +                    -- Počet matchujících klíčových slov
                            km.total_frequency * 2 +                   -- Celková frekvence
                            km.has_exact_match * 20 +                  -- Bonus za přesný match
                            CASE d.source_type                         -- Bonus podle typu zdroje
                                WHEN 'academic' THEN 15
                                WHEN 'government' THEN 12
                                WHEN 'wikipedia' THEN 8
                                WHEN 'news' THEN 5
                                ELSE 3
                            END +
                            -- Bonus pro novější dokumenty (poslední 2 roky)
                            CASE WHEN d.timestamp > (CURRENT_TIMESTAMP - INTERVAL '2 years') THEN 5 ELSE 0 END
                        ) as relevance_score
                    FROM documents d
                    INNER JOIN keyword_matches km ON d.id = km.document_id
                )
                SELECT id 
                FROM scored_documents
                ORDER BY relevance_score DESC, timestamp DESC
                LIMIT ?
            """

            # Příprava parametrů
            params = (
                normalized_keywords +           # První IN klauzule
                normalized_keywords +           # Druhá IN klauzule
                normalized_keywords +           # LIKE parametry
                normalized_keywords +           # VALUES parametry
                [limit]                         # LIMIT
            )

            # Provedení dotazu
            result = self.conn.execute(query, params).fetchall()

            doc_ids = [row[0] for row in result]

            logger.info(f"🔍 Nalezeno {len(doc_ids)} dokumentů pro klíčová slova: {normalized_keywords[:5]}...")

            return doc_ids

        except Exception as e:
            logger.error(f"❌ Chyba při vyhledávání podle klíčových slov: {e}")
            logger.error(f"Klíčová slova: {keywords}")
            return []

    def _determine_source_type(self, source: str, metadata: dict[str, Any]) -> str:
        """Určení typu zdroje pro prioritizaci
        
        Args:
            source: URL nebo identifikátor zdroje
            metadata: Metadata dokumentu
            
        Returns:
            Typ zdroje (academic, government, wikipedia, news, unknown)

        """
        source_lower = source.lower()

        # Akademické zdroje
        if any(domain in source_lower for domain in [
            'arxiv.org', 'pubmed.gov', 'scholar.google', 'researchgate.net',
            'ieee.org', 'acm.org', 'springer.com', 'nature.com', 'science.org'
        ]):
            return 'academic'

        # Vládní zdroje
        if any(domain in source_lower for domain in [
            '.gov', '.mil', 'europa.eu', 'un.org', 'who.int'
        ]):
            return 'government'

        # Wikipedia
        if 'wikipedia.org' in source_lower:
            return 'wikipedia'

        # Zpravodajské weby
        if any(domain in source_lower for domain in [
            'reuters.com', 'bbc.com', 'cnn.com', 'nytimes.com', 'guardian.com',
            'washingtonpost.com', 'npr.org', 'bloomberg.com'
        ]):
            return 'news'

        # Kontrola metadat
        if metadata.get('type') in ['academic', 'government', 'news']:
            return metadata['type']

        return 'unknown'

    def get_stats(self) -> dict[str, Any]:
        """Získání statistik warehouse
        
        Returns:
            Slovník se statistikami

        """
        if not self.conn:
            raise RuntimeError("Databáze není připojena")

        try:
            # Celkové statistiky
            total_docs = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            total_keywords = self.conn.execute("SELECT COUNT(*) FROM document_keywords").fetchone()[0]

            # Statistiky podle typu zdroje
            source_stats = self.conn.execute("""
                SELECT source_type, COUNT(*) as count, AVG(content_length) as avg_length
                FROM documents 
                GROUP BY source_type
                ORDER BY count DESC
            """).fetchall()

            # Nejčastější klíčová slova
            top_keywords = self.conn.execute("""
                SELECT keyword, COUNT(*) as frequency
                FROM document_keywords
                GROUP BY keyword
                ORDER BY frequency DESC
                LIMIT 20
            """).fetchall()

            return {
                'total_documents': total_docs,
                'total_keywords': total_keywords,
                'source_distribution': {row[0]: {'count': row[1], 'avg_length': row[2]} for row in source_stats},
                'top_keywords': [{'keyword': row[0], 'frequency': row[1]} for row in top_keywords],
                'database_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
            }

        except Exception as e:
            logger.error(f"❌ Chyba při získávání statistik: {e}")
            return {}


# Utility funkce pro snadné použití
def create_warehouse(db_path: str = "data/warehouse.duckdb") -> DataWarehouse:
    """Vytvoření a inicializace nového warehouse
    
    Args:
        db_path: Cesta k databázi
        
    Returns:
        Inicializovaný DataWarehouse

    """
    warehouse = DataWarehouse(db_path)
    warehouse.connect()
    warehouse.init_db()
    return warehouse


if __name__ == "__main__":
    # Testování functionality
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        test_db = os.path.join(temp_dir, "test.duckdb")

        # Test základní funkcionalitu
        with DataWarehouse(test_db) as warehouse:
            warehouse.init_db()

            # Test dokumenty
            test_docs = [
                {
                    'id': 'doc1',
                    'content': 'This is a research paper about artificial intelligence and machine learning algorithms.',
                    'source': 'https://arxiv.org/paper1',
                    'metadata': {'type': 'academic', 'year': 2023}
                },
                {
                    'id': 'doc2',
                    'content': 'Government report on climate change impacts and environmental policy.',
                    'source': 'https://epa.gov/report',
                    'metadata': {'type': 'government', 'year': 2023}
                }
            ]

            warehouse.add_documents(test_docs)

            # Test vyhledávání
            results = warehouse.query_ids_by_keywords(['artificial', 'intelligence'])
            print(f"Nalezeno {len(results)} dokumentů pro AI")

            results = warehouse.query_ids_by_keywords(['climate', 'change'])
            print(f"Nalezeno {len(results)} dokumentů pro climate change")

            # Statistiky
            stats = warehouse.get_stats()
            print(f"Warehouse statistiky: {stats}")

        print("✅ Test DuckDB warehouse úspěšný")
