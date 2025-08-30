"""
Enhanced Processing Worker - Fáze 3
Inteligentní zpracování dat s pokročilým NLP a RAG indexováním
"""

import asyncio
import logging
import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import redis
import dramatiq
from dramatiq.brokers.redis import RedisBroker
import polars as pl
import lancedb
from datetime import datetime, timezone
import hashlib

# Advanced text processing imports (PHASE 3)
import trafilatura
import spacy
from sentence_transformers import SentenceTransformer
from simhash import Simhash
import dateparser
from urllib.parse import urlparse

# Knowledge Graph imports (PHASE 1)
import sys

# Oprav cestu pro lokální prostředí
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent / "src"))

try:
    from knowledge_graph import KnowledgeGraphManager
    from llm_relationship_extractor import LLMRelationshipExtractor

    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Knowledge Graph není dostupný: {e}")
    KNOWLEDGE_GRAPH_AVAILABLE = False

# Multi-Modal Processing imports (PHASE 3)
try:
    from image_processor import ImageProcessor, image_processor

    IMAGE_PROCESSING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Image processing není dostupný: {e}")
    IMAGE_PROCESSING_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis broker setup
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(redis_url)
broker = RedisBroker(url=redis_url)
dramatiq.set_broker(broker)


class AdvancedTextProcessor:
    """Pokročilý procesor textu s NLP analýzou"""

    def __init__(self):
        # Načti spaCy model
        try:
            self.nlp = spacy.load("en_core_web_lg")
            logger.info("✅ spaCy model en_core_web_lg načten")
        except OSError:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model en_core_web_sm načten (fallback)")

        # Sentence transformer pro embeddings
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✅ Sentence transformer model načten")

        # Custom entity patterns
        self._setup_custom_entities()

    def _setup_custom_entities(self):
        """Nastav custom entity patterns pro dark web content"""
        from spacy.lang.en import English
        from spacy.pipeline import EntityRuler

        # Přidej custom patterns pro krypto adresy, PGP klíče, atd.
        patterns = [
            # Bitcoin adresy
            {
                "label": "CRYPTO_ADDRESS",
                "pattern": [{"TEXT": {"REGEX": r"^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$"}}],
            },
            {"label": "CRYPTO_ADDRESS", "pattern": [{"TEXT": {"REGEX": r"^bc1[a-z0-9]{39,59}$"}}]},
            # Ethereum adresy
            {"label": "CRYPTO_ADDRESS", "pattern": [{"TEXT": {"REGEX": r"^0x[a-fA-F0-9]{40}$"}}]},
            # PGP key fingerprints
            {"label": "PGP_KEY", "pattern": [{"TEXT": {"REGEX": r"^[A-F0-9]{40}$"}}]},
            # .onion domény
            {"label": "ONION_ADDRESS", "pattern": [{"TEXT": {"REGEX": r"[a-z2-7]{16,56}\.onion"}}]},
            # Hash hodnoty
            {"label": "HASH", "pattern": [{"TEXT": {"REGEX": r"^[a-f0-9]{32}$"}}]},  # MD5
            {"label": "HASH", "pattern": [{"TEXT": {"REGEX": r"^[a-f0-9]{40}$"}}]},  # SHA1
            {"label": "HASH", "pattern": [{"TEXT": {"REGEX": r"^[a-f0-9]{64}$"}}]},  # SHA256
        ]

        # Přidej EntityRuler do pipeline
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler")
            ruler.add_patterns(patterns)
            logger.info(f"✅ Přidáno {len(patterns)} custom entity patterns")

    def extract_clean_text(self, html_content: str, url: str = "") -> Dict[str, Any]:
        """Extrahuj čistý text pomocí trafilatura"""
        try:
            # Použij trafilatura pro pokročilou extrakci
            extracted = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=True,
                include_images=False,
                include_links=True,
                url=url,
            )

            if not extracted:
                # Fallback na základní regex cleaning
                text = re.sub(r"<[^>]+>", "", html_content)
                text = re.sub(r"\s+", " ", text).strip()
                extracted = text

            # Metadata extrakce
            metadata = trafilatura.extract_metadata(html_content, default_url=url)

            return {
                "clean_text": extracted,
                "text_length": len(extracted),
                "word_count": len(extracted.split()) if extracted else 0,
                "title": metadata.title if metadata else "",
                "author": metadata.author if metadata else "",
                "date": metadata.date if metadata else "",
                "description": metadata.description if metadata else "",
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"❌ Trafilatura extraction error: {e}")
            # Základní fallback
            text = re.sub(r"<[^>]+>", "", html_content)
            text = re.sub(r"\s+", " ", text).strip()

            return {
                "clean_text": text,
                "text_length": len(text),
                "word_count": len(text.split()),
                "title": "",
                "author": "",
                "date": "",
                "description": "",
                "error": str(e),
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            }

    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extrahuj named entities pomocí spaCy"""
        try:
            doc = self.nlp(text)

            entities = {
                "persons": [],
                "organizations": [],
                "locations": [],
                "dates": [],
                "crypto_addresses": [],
                "onion_addresses": [],
                "hashes": [],
                "pgp_keys": [],
                "other": [],
            }

            for ent in doc.ents:
                entity_data = {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 1.0,  # spaCy doesn't provide confidence scores by default
                }

                # Kategorizuj entity
                if ent.label_ == "PERSON":
                    entities["persons"].append(entity_data)
                elif ent.label_ in ["ORG", "COMPANY"]:
                    entities["organizations"].append(entity_data)
                elif ent.label_ in ["GPE", "LOC", "LOCATION"]:
                    entities["locations"].append(entity_data)
                elif ent.label_ in ["DATE", "TIME"]:
                    entities["dates"].append(entity_data)
                elif ent.label_ == "CRYPTO_ADDRESS":
                    entities["crypto_addresses"].append(entity_data)
                elif ent.label_ == "ONION_ADDRESS":
                    entities["onion_addresses"].append(entity_data)
                elif ent.label_ == "HASH":
                    entities["hashes"].append(entity_data)
                elif ent.label_ == "PGP_KEY":
                    entities["pgp_keys"].append(entity_data)
                else:
                    entities["other"].append(entity_data)

            return entities

        except Exception as e:
            logger.error(f"❌ Entity extraction error: {e}")
            return {
                "persons": [],
                "organizations": [],
                "locations": [],
                "dates": [],
                "crypto_addresses": [],
                "onion_addresses": [],
                "hashes": [],
                "pgp_keys": [],
                "other": [],
                "error": str(e),
            }

    def calculate_content_hash(self, text: str) -> str:
        """Vytvoř Simhash pro deduplikaci"""
        try:
            simhash = Simhash(text)
            return str(simhash.value)
        except Exception as e:
            logger.error(f"❌ Simhash error: {e}")
            # Fallback na SHA256
            return hashlib.sha256(text.encode()).hexdigest()

    def semantic_chunking(
        self, text: str, max_chunk_size: int = 512, overlap: int = 50
    ) -> List[Dict[str, Any]]:
        """Pokročilé sémantické dělení textu"""
        try:
            # Rozděl text na věty pomocí spaCy
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]

            chunks = []
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                sentence_length = len(sentence.split())

                # Pokud věta sama je příliš dlouhá, rozděl ji
                if sentence_length > max_chunk_size:
                    if current_chunk:
                        chunks.append(
                            {
                                "text": " ".join(current_chunk),
                                "sentence_count": len(current_chunk),
                                "word_count": current_length,
                            }
                        )
                        current_chunk = []
                        current_length = 0

                    # Rozděl dlouhou větu na slova
                    words = sentence.split()
                    for i in range(0, len(words), max_chunk_size - overlap):
                        chunk_words = words[i : i + max_chunk_size]
                        chunks.append(
                            {
                                "text": " ".join(chunk_words),
                                "sentence_count": 1,
                                "word_count": len(chunk_words),
                            }
                        )
                    continue

                # Pokud přidání věty přesáhne limit, ukonči chunk
                if current_length + sentence_length > max_chunk_size and current_chunk:
                    chunks.append(
                        {
                            "text": " ".join(current_chunk),
                            "sentence_count": len(current_chunk),
                            "word_count": current_length,
                        }
                    )

                    # Začni nový chunk s překryvem
                    if overlap > 0 and len(current_chunk) > 1:
                        overlap_sentences = current_chunk[-overlap:]
                        current_chunk = overlap_sentences + [sentence]
                        current_length = sum(len(s.split()) for s in current_chunk)
                    else:
                        current_chunk = [sentence]
                        current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length

            # Přidej poslední chunk
            if current_chunk:
                chunks.append(
                    {
                        "text": " ".join(current_chunk),
                        "sentence_count": len(current_chunk),
                        "word_count": current_length,
                    }
                )

            return chunks

        except Exception as e:
            logger.error(f"❌ Semantic chunking error: {e}")
            # Fallback na základní word chunking
            words = text.split()
            chunks = []
            for i in range(0, len(words), max_chunk_size - overlap):
                chunk_words = words[i : i + max_chunk_size]
                chunks.append(
                    {
                        "text": " ".join(chunk_words),
                        "sentence_count": 1,
                        "word_count": len(chunk_words),
                    }
                )
            return chunks

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generuj sentence embeddings pro chunky"""
        try:
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.sentence_model.encode(texts, convert_to_tensor=False)

            # Přidej embeddings k chunks
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings[i].tolist()
                chunk["embedding_model"] = "all-MiniLM-L6-v2"
                chunk["embedding_dim"] = len(embeddings[i])

            return chunks

        except Exception as e:
            logger.error(f"❌ Embedding generation error: {e}")
            # Vrať chunks bez embeddings
            for chunk in chunks:
                chunk["embedding"] = None
                chunk["embedding_error"] = str(e)
            return chunks


class EnhancedProcessingWorker:
    """Vylepšený worker pro zpracování dat s pokročilým NLP a Knowledge Graph"""

    def __init__(self):
        # Použij lokální cesty místo Docker cest
        base_dir = Path.cwd() / "data"
        self.data_dir = base_dir
        self.cache_dir = base_dir / "cache"
        self.vector_db_path = self.data_dir / "vector_db"

        # Zajisti že adresáře existují
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        self.vector_db_path.mkdir(exist_ok=True)

        # Inicializuj LanceDB
        self.db = lancedb.connect(str(self.vector_db_path))

        # Inicializuj text processor
        self.text_processor = AdvancedTextProcessor()

        # Inicializuj Knowledge Graph Manager (FÁZE 1)
        try:
            self.kg_manager = KnowledgeGraphManager()
            logger.info("✅ Knowledge Graph Manager inicializován")
        except Exception as e:
            logger.error(f"❌ Chyba při inicializaci Knowledge Graph: {e}")
            self.kg_manager = None

        # Inicializuj LLM Relationship Extractor (FÁZE 1)
        try:
            self.relation_extractor = LLMRelationshipExtractor()
            logger.info("✅ LLM Relationship Extractor inicializován")
        except Exception as e:
            logger.error(f"❌ Chyba při inicializaci LLM Extractor: {e}")
            self.relation_extractor = None

        # Inicializuj Image Processor (FÁZE 3)
        self.image_processor = None
        if IMAGE_PROCESSING_AVAILABLE:
            try:
                self.image_processor = ImageProcessor()
                logger.info("✅ Image Processor inicializován")
            except Exception as e:
                logger.error(f"❌ Chyba při inicializaci Image Processor: {e}")
                self.image_processor = None
        else:
            logger.warning("⚠️ Image processing není dostupný")

        logger.info("✅ Enhanced Processing Worker inicializován")

    def process_scraped_data_enhanced(self, file_path: str, task_id: str) -> Dict[str, Any]:
        """Pokročilé zpracování scraped dat"""
        try:
            logger.info(f"🧠 Enhanced processing: {file_path} (task: {task_id})")

            # Načti surová data
            raw_data = pl.read_parquet(file_path)

            processed_records = []

            for row in raw_data.iter_rows(named=True):
                url = row["url"]
                content = row["content"]
                metadata = (
                    json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"]
                )

                # 1. Extrakce čistého textu pomocí trafilatura
                extracted = self.text_processor.extract_clean_text(content, url)

                if not extracted["clean_text"]:
                    logger.warning(f"⚠️ Prázdný text po extrakci: {url}")
                    continue

                # 2. Named Entity Recognition
                entities = self.text_processor.extract_entities(extracted["clean_text"])

                # 3. Deduplikace pomocí Simhash
                content_hash = self.text_processor.calculate_content_hash(extracted["clean_text"])

                # 4. Sémantické chunking
                chunks = self.text_processor.semantic_chunking(extracted["clean_text"])

                # 5. Generování embeddings
                chunks_with_embeddings = self.text_processor.generate_embeddings(chunks)

                processed_record = {
                    "task_id": task_id,
                    "url": url,
                    "domain": urlparse(url).netloc,
                    "content_hash": content_hash,
                    "clean_text": extracted["clean_text"],
                    "text_length": extracted["text_length"],
                    "word_count": extracted["word_count"],
                    "title": extracted["title"],
                    "author": extracted["author"],
                    "date": extracted["date"],
                    "description": extracted["description"],
                    "entities": entities,
                    "chunk_count": len(chunks_with_embeddings),
                    "chunks": chunks_with_embeddings,
                    "original_metadata": metadata,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                }

                processed_records.append(processed_record)
                logger.info(
                    f"✅ Zpracováno: {url} ({len(chunks_with_embeddings)} chunks, {len(entities.get('persons', []))} persons)"
                )

            # Ulož zpracovaná data
            processed_df = pl.DataFrame(processed_records)
            output_path = self.data_dir / f"processed_enhanced_{task_id}.parquet"
            processed_df.write_parquet(output_path)

            logger.info(
                f"💾 Enhanced processing dokončeno: {len(processed_records)} records → {output_path}"
            )

            # Pošli na RAG indexování
            index_for_rag.send(str(output_path), task_id)

            return {
                "success": True,
                "input_file": file_path,
                "output_file": str(output_path),
                "records_processed": len(processed_records),
                "total_chunks": sum(r["chunk_count"] for r in processed_records),
                "total_entities": sum(
                    len(r["entities"].get("persons", []))
                    + len(r["entities"].get("organizations", []))
                    + len(r["entities"].get("locations", []))
                    for r in processed_records
                ),
                "task_id": task_id,
            }

        except Exception as e:
            logger.error(f"❌ Enhanced processing error {file_path}: {e}")
            return {"success": False, "input_file": file_path, "error": str(e), "task_id": task_id}

    def index_for_rag_enhanced(self, file_path: str, task_id: str) -> Dict[str, Any]:
        """Pokročilé RAG indexování s embeddings"""
        try:
            logger.info(f"🔍 RAG indexing: {file_path} (task: {task_id})")

            # Načti zpracovaná data
            processed_data = pl.read_parquet(file_path)

            # Připrav dokumenty pro LanceDB s embeddings
            documents = []
            table_name = f"rag_documents_{task_id}"

            for row in processed_data.iter_rows(named=True):
                base_doc_info = {
                    "task_id": task_id,
                    "url": row["url"],
                    "domain": row["domain"],
                    "content_hash": row["content_hash"],
                    "title": row["title"],
                    "author": row["author"],
                    "date": row["date"],
                    "description": row["description"],
                    "entities_persons": len(row["entities"]["persons"]) if row["entities"] else 0,
                    "entities_orgs": (
                        len(row["entities"]["organizations"]) if row["entities"] else 0
                    ),
                    "entities_locations": (
                        len(row["entities"]["locations"]) if row["entities"] else 0
                    ),
                    "total_word_count": row["word_count"],
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                }

                # Přidej každý chunk jako samostatný dokument
                for i, chunk in enumerate(row["chunks"]):
                    if chunk.get("embedding"):  # Pouze chunky s embeddings
                        doc = {
                            **base_doc_info,
                            "chunk_id": f"{task_id}_{row['url']}_{i}",
                            "chunk_index": i,
                            "chunk_text": chunk["text"],
                            "chunk_word_count": chunk["word_count"],
                            "chunk_sentence_count": chunk["sentence_count"],
                            "vector": chunk["embedding"],  # LanceDB očekává 'vector' pro embedding
                            "embedding_model": chunk["embedding_model"],
                            "embedding_dim": chunk["embedding_dim"],
                        }
                        documents.append(doc)

            if not documents:
                logger.warning(f"⚠️ Žádné dokumenty s embeddings pro indexování: {task_id}")
                return {"success": False, "reason": "no_embeddings", "task_id": task_id}

            # Vytvoř nebo rozšiř tabulku v LanceDB
            try:
                if table_name in self.db.table_names():
                    table = self.db.open_table(table_name)
                    table.add(documents)
                    logger.info(
                        f"✅ Přidáno {len(documents)} chunks do existující tabulky {table_name}"
                    )
                else:
                    table = self.db.create_table(table_name, documents)
                    logger.info(f"✅ Vytvořena nová tabulka {table_name} s {len(documents)} chunks")

                # Vytvoř index pro rychlé vyhledávání
                table.create_index(
                    "vector",
                    index_type="IVF_FLAT",
                    num_partitions=min(256, len(documents) // 10 + 1),
                )

            except Exception as index_error:
                logger.error(f"❌ LanceDB indexing error: {index_error}")
                # Fallback bez indexu
                table = self.db.create_table(table_name, documents)

            return {
                "success": True,
                "input_file": file_path,
                "table_name": table_name,
                "chunks_indexed": len(documents),
                "task_id": task_id,
                "rag_ready": True,
            }

        except Exception as e:
            logger.error(f"❌ RAG indexing error {file_path}: {e}")
            return {"success": False, "input_file": file_path, "error": str(e), "task_id": task_id}

    def search_rag_documents(
        self, query: str, limit: int = 5, task_id: str = None
    ) -> Dict[str, Any]:
        """Vyhledej relevantní dokumenty pomocí sémantické podobnosti"""
        try:
            logger.info(f"🔍 RAG search: '{query}' (limit: {limit})")

            # Generuj embedding pro query
            query_embedding = self.text_processor.sentence_model.encode([query])[0].tolist()

            # Najdi relevantní tabulky
            if task_id:
                table_names = [f"rag_documents_{task_id}"]
            else:
                table_names = [
                    name for name in self.db.table_names() if name.startswith("rag_documents_")
                ]

            all_results = []

            for table_name in table_names:
                try:
                    table = self.db.open_table(table_name)

                    # Proveď vektorové vyhledávání
                    results = table.search(query_embedding).limit(limit).to_list()

                    for result in results:
                        result["similarity_score"] = float(result.get("_distance", 0))
                        result["table_source"] = table_name
                        all_results.append(result)

                except Exception as table_error:
                    logger.warning(f"⚠️ Chyba při vyhledávání v tabulce {table_name}: {table_error}")
                    continue

            # Seřaď podle similarity score a omez výsledky
            all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            top_results = all_results[:limit]

            return {
                "success": True,
                "query": query,
                "results": top_results,
                "total_found": len(all_results),
                "returned": len(top_results),
            }

        except Exception as e:
            logger.error(f"❌ RAG search error: {e}")
            return {"success": False, "query": query, "error": str(e)}

    async def process_with_knowledge_graph(self, file_path: str, task_id: str) -> Dict[str, Any]:
        """
        Zpracování dat s integrací Knowledge Graph (FÁZE 1)
        Extrahuje entity, vztahy pomocí LLM a ukládá do Neo4j
        """
        try:
            logger.info(f"🧠 Knowledge Graph processing: {file_path} (task: {task_id})")

            # Načti surová data
            raw_data = pl.read_parquet(file_path)

            processed_records = []
            kg_stats = {
                "total_entities_added": 0,
                "total_relations_added": 0,
                "sources_processed": 0,
                "errors": [],
            }

            for row in raw_data.iter_rows(named=True):
                url = row["url"]
                content = row["content"]
                metadata = (
                    json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else row["metadata"]
                )

                # 1. Extrakce čistého textu
                extracted = self.text_processor.extract_clean_text(content, url)
                if not extracted["clean_text"]:
                    logger.warning(f"⚠️ Prázdný text po extrakci: {url}")
                    continue

                # 2. Named Entity Recognition pomocí spaCy
                entities = self.text_processor.extract_entities(extracted["clean_text"])

                # 3. Extrakce vztahů pomocí LLM (FÁZE 1)
                relationships_data = {}
                if self.relation_extractor:
                    try:
                        relationships_data = await self.relation_extractor.extract_relationships(
                            extracted["clean_text"], entities
                        )
                        logger.info(
                            f"🤖 LLM extrahoval {len(relationships_data.get('relations', []))} vztahů pro {url}"
                        )
                    except Exception as e:
                        logger.error(f"❌ LLM extrakce vztahů selhala pro {url}: {e}")
                        # Fallback na heuristické vztahy
                        relationships_data = {
                            "entities": [],
                            "relations": (
                                self.relation_extractor.create_simple_relations_from_entities(
                                    entities, url
                                )
                                if self.relation_extractor
                                else []
                            ),
                        }

                # 4. Ukládání do Knowledge Graph (FÁZE 1)
                if self.kg_manager and relationships_data:
                    try:
                        # Kombinuj entity ze spaCy a LLM
                        final_entities = entities.copy()

                        # Přidej vztahy do finálních entit
                        final_relations = relationships_data.get("relations", [])

                        # Ulož do Neo4j
                        kg_result = await self.kg_manager.add_entities_and_relations(
                            final_entities,
                            final_relations,
                            url,
                            {**extracted, **metadata, "task_id": task_id},
                        )

                        kg_stats["total_entities_added"] += kg_result.get("entities_added", 0)
                        kg_stats["total_relations_added"] += kg_result.get("relations_added", 0)
                        kg_stats["sources_processed"] += 1 if kg_result.get("source_added") else 0

                        if kg_result.get("errors"):
                            kg_stats["errors"].extend(kg_result["errors"])

                        logger.info(
                            f"📊 KG uloženo: {kg_result.get('entities_added', 0)} entit, {kg_result.get('relations_added', 0)} vztahů pro {url}"
                        )

                    except Exception as e:
                        logger.error(f"❌ Chyba při ukládání do KG pro {url}: {e}")
                        kg_stats["errors"].append(f"KG save error for {url}: {e}")

                # 5. Standardní zpracování pro RAG
                content_hash = self.text_processor.calculate_content_hash(extracted["clean_text"])
                chunks = self.text_processor.semantic_chunking(extracted["clean_text"])
                chunks_with_embeddings = self.text_processor.generate_embeddings(chunks)

                processed_record = {
                    "task_id": task_id,
                    "url": url,
                    "domain": urlparse(url).netloc,
                    "content_hash": content_hash,
                    "clean_text": extracted["clean_text"],
                    "text_length": extracted["text_length"],
                    "word_count": extracted["word_count"],
                    "title": extracted["title"],
                    "author": extracted["author"],
                    "date": extracted["date"],
                    "description": extracted["description"],
                    "entities": entities,
                    "relationships": relationships_data,
                    "chunk_count": len(chunks_with_embeddings),
                    "chunks": chunks_with_embeddings,
                    "original_metadata": metadata,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "knowledge_graph_processed": bool(self.kg_manager),
                }

                processed_records.append(processed_record)
                logger.info(
                    f"✅ Kompletní zpracování: {url} ({len(chunks_with_embeddings)} chunks, {len(final_relations)} vztahů)"
                )

            # Ulož zpracovaná data
            if processed_records:
                processed_df = pl.DataFrame(processed_records)
                output_path = self.data_dir / f"processed_kg_{task_id}.parquet"
                processed_df.write_parquet(output_path)

                logger.info(
                    f"💾 Knowledge Graph processing dokončeno: {len(processed_records)} records → {output_path}"
                )

                # Pošli na RAG indexování
                index_for_rag.send(str(output_path), task_id)
            else:
                output_path = None

            return {
                "success": True,
                "input_file": file_path,
                "output_file": str(output_path) if output_path else None,
                "records_processed": len(processed_records),
                "knowledge_graph_stats": kg_stats,
                "total_chunks": sum(r["chunk_count"] for r in processed_records),
                "total_entities": sum(
                    len(r["entities"].get("persons", []))
                    + len(r["entities"].get("organizations", []))
                    + len(r["entities"].get("locations", []))
                    for r in processed_records
                ),
                "total_relations": sum(
                    len(r["relationships"].get("relations", [])) for r in processed_records
                ),
                "task_id": task_id,
            }

        except Exception as e:
            logger.error(f"❌ Knowledge Graph processing error {file_path}: {e}")
            return {"success": False, "input_file": file_path, "error": str(e), "task_id": task_id}

    async def process_images_from_directory(self, images_dir: str, task_id: str) -> Dict[str, Any]:
        """
        Zpracování obrázků z adresáře (FÁZE 3: Multi-Modality)

        Args:
            images_dir: Cesta k adresáři s obrázky
            task_id: ID úlohy

        Returns:
            Výsledky zpracování obrázků
        """
        try:
            if not self.image_processor:
                return {
                    "success": False,
                    "error": "Image processor není dostupný",
                    "images_processed": 0,
                }

            logger.info(f"🖼️ Zpracovávám obrázky z adresáře: {images_dir}")

            images_path = Path(images_dir)
            if not images_path.exists():
                return {
                    "success": False,
                    "error": f"Adresář {images_dir} neexistuje",
                    "images_processed": 0,
                }

            # Najdi všechny obrázky v adresáři
            image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}
            image_files = []

            for ext in image_extensions:
                image_files.extend(list(images_path.glob(f"*{ext}")))
                image_files.extend(list(images_path.glob(f"*{ext.upper()}")))

            if not image_files:
                return {
                    "success": True,
                    "images_processed": 0,
                    "message": "Žádné obrázky nenalezeny",
                }

            logger.info(f"📊 Nalezeno {len(image_files)} obrázků k zpracování")

            # Zpracuj obrázky
            processed_images = []
            image_entities = []
            image_embeddings = []

            for image_file in image_files:
                try:
                    # Zpracuj obrázek pomocí Image Processor
                    image_result = self.image_processor.process_image(
                        str(image_file), f"file://{image_file}"
                    )

                    if not image_result.get("error"):
                        processed_images.append(image_result)

                        # Extrahuj entity z OCR textu
                        ocr_text = image_result.get("ocr_result", {}).get("text", "")
                        if ocr_text.strip():
                            text_entities = self.text_processor.extract_entities(ocr_text)

                            # Přidej image-specific metadata k entitám
                            for entity_type, entities_list in text_entities.items():
                                for entity in entities_list:
                                    entity["source_type"] = "image_ocr"
                                    entity["image_file"] = str(image_file)
                                    entity["task_id"] = task_id

                            image_entities.append(
                                {
                                    "image_file": str(image_file),
                                    "entities": text_entities,
                                    "ocr_text": ocr_text,
                                }
                            )

                        # Uložení embeddings pro cross-modální vyhledávání
                        embeddings = image_result.get("embeddings", {})
                        if embeddings:
                            embeddings["image_file"] = str(image_file)
                            embeddings["task_id"] = task_id
                            image_embeddings.append(embeddings)

                except Exception as e:
                    logger.error(f"❌ Chyba při zpracování {image_file}: {e}")
                    continue

            # Ulož výsledky zpracování obrázků
            if processed_images:
                # Ulož jako JSON pro complex data
                output_path = self.data_dir / f"processed_images_{task_id}.json"

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "task_id": task_id,
                            "images_directory": str(images_dir),
                            "processed_images": processed_images,
                            "image_entities": image_entities,
                            "image_embeddings": image_embeddings,
                            "processed_at": datetime.now(timezone.utc).isoformat(),
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

                # Index embeddings do LanceDB pro cross-modální vyhledávání
                await self._index_image_embeddings(image_embeddings, task_id)

                # Ulož entity do Knowledge Graph
                if self.kg_manager and image_entities:
                    await self._save_image_entities_to_kg(image_entities, task_id)

            return {
                "success": True,
                "images_processed": len(processed_images),
                "total_images_found": len(image_files),
                "entities_extracted": len(image_entities),
                "embeddings_generated": len(image_embeddings),
                "output_file": str(output_path) if processed_images else None,
                "task_id": task_id,
            }

        except Exception as e:
            logger.error(f"❌ Chyba při zpracování obrázků: {e}")
            return {"success": False, "error": str(e), "images_processed": 0}

    async def _index_image_embeddings(self, image_embeddings: List[Dict[str, Any]], task_id: str):
        """Index image embeddings do LanceDB pro cross-modální vyhledávání"""
        try:
            if not image_embeddings:
                return

            # Připrav dokumenty pro image embeddings
            image_documents = []
            table_name = f"image_embeddings_{task_id}"

            for i, embedding_data in enumerate(image_embeddings):
                # Text embedding (z OCR)
                if "text_embedding" in embedding_data:
                    doc = {
                        "doc_id": f"{task_id}_img_text_{i}",
                        "image_file": embedding_data["image_file"],
                        "task_id": task_id,
                        "embedding_type": "text",
                        "vector": embedding_data["text_embedding"],
                        "embedding_model": embedding_data.get("text_embedding_model", "unknown"),
                        "embedding_dim": embedding_data.get("text_embedding_dim", 0),
                        "indexed_at": datetime.now(timezone.utc).isoformat(),
                    }
                    image_documents.append(doc)

                # CLIP image embedding
                if "clip_image_embedding" in embedding_data:
                    doc = {
                        "doc_id": f"{task_id}_img_clip_{i}",
                        "image_file": embedding_data["image_file"],
                        "task_id": task_id,
                        "embedding_type": "clip_image",
                        "vector": embedding_data["clip_image_embedding"],
                        "embedding_model": embedding_data.get("clip_embedding_model", "unknown"),
                        "embedding_dim": embedding_data.get("clip_embedding_dim", 0),
                        "indexed_at": datetime.now(timezone.utc).isoformat(),
                    }
                    image_documents.append(doc)

            if image_documents:
                # Vytvoř tabulku pro image embeddings
                table = self.db.create_table(table_name, image_documents)
                table.create_index(
                    "vector",
                    index_type="IVF_FLAT",
                    num_partitions=min(256, len(image_documents) // 10 + 1),
                )

                logger.info(
                    f"✅ Indexováno {len(image_documents)} image embeddings do {table_name}"
                )

        except Exception as e:
            logger.error(f"❌ Chyba při indexování image embeddings: {e}")

    async def _save_image_entities_to_kg(self, image_entities: List[Dict[str, Any]], task_id: str):
        """Ulož entity z obrázků do Knowledge Graph"""
        try:
            for image_entity_data in image_entities:
                image_file = image_entity_data["image_file"]
                entities = image_entity_data["entities"]
                ocr_text = image_entity_data["ocr_text"]

                # Vytvořit jednoduché vztahy mezi entitami v obrázku
                simple_relations = []
                if self.relation_extractor:
                    simple_relations = (
                        self.relation_extractor.create_simple_relations_from_entities(
                            entities, image_file
                        )
                    )

                # Ulož do Knowledge Graph
                await self.kg_manager.add_entities_and_relations(
                    entities,
                    simple_relations,
                    image_file,
                    {
                        "content_type": "image",
                        "ocr_text": ocr_text,
                        "task_id": task_id,
                        "extracted_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

                logger.info(f"✅ Uloženy entity z obrázku {image_file} do Knowledge Graph")

        except Exception as e:
            logger.error(f"❌ Chyba při ukládání image entit do KG: {e}")

    async def search_cross_modal(
        self, query: str, search_type: str = "text_to_image", limit: int = 5, task_id: str = None
    ) -> Dict[str, Any]:
        """
        Cross-modální vyhledávání mezi textem a obrázky

        Args:
            query: Textový dotaz
            search_type: "text_to_image" nebo "image_to_text"
            limit: Počet výsledků
            task_id: ID úlohy pro filtrování
        """
        try:
            if not self.image_processor:
                return {"success": False, "error": "Image processor není dostupný"}

            # Generuj embedding pro query
            if search_type == "text_to_image":
                # Použij CLIP text encoder
                if hasattr(self.image_processor, "clip_model") and self.image_processor.clip_model:
                    import clip
                    import torch

                    text_input = clip.tokenize([query[:77]]).to(self.image_processor.clip_device)
                    with torch.no_grad():
                        query_embedding = self.image_processor.clip_model.encode_text(text_input)
                        query_embedding = query_embedding.cpu().numpy()[0].tolist()
                else:
                    # Fallback na sentence transformer
                    query_embedding = self.text_processor.sentence_model.encode([query])[0].tolist()
            else:
                # Pro image_to_text použij sentence transformer
                query_embedding = self.text_processor.sentence_model.encode([query])[0].tolist()

            # Najdi relevantní tabulky
            if task_id:
                table_names = [f"image_embeddings_{task_id}"]
            else:
                table_names = [
                    name for name in self.db.table_names() if name.startswith("image_embeddings_")
                ]

            all_results = []

            for table_name in table_names:
                try:
                    table = self.db.open_table(table_name)

                    # Filtruj podle typu embedding
                    if search_type == "text_to_image":
                        # Hledej CLIP image embeddings
                        results = (
                            table.search(query_embedding)
                            .where("embedding_type = 'clip_image'")
                            .limit(limit)
                            .to_list()
                        )
                    else:
                        # Hledej text embeddings z OCR
                        results = (
                            table.search(query_embedding)
                            .where("embedding_type = 'text'")
                            .limit(limit)
                            .to_list()
                        )

                    for result in results:
                        result["similarity_score"] = float(result.get("_distance", 0))
                        result["search_type"] = search_type
                        all_results.append(result)

                except Exception as e:
                    logger.warning(f"⚠️ Chyba při vyhledávání v {table_name}: {e}")
                    continue

            # Seřaď výsledky
            all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            top_results = all_results[:limit]

            return {
                "success": True,
                "query": query,
                "search_type": search_type,
                "results": top_results,
                "total_found": len(all_results),
                "returned": len(top_results),
            }

        except Exception as e:
            logger.error(f"❌ Cross-modal search error: {e}")
            return {"success": False, "error": str(e)}


# Multi-Modal Processing Dramatiq tasks (FÁZE 3)
@dramatiq.actor(queue_name="processing")
def process_images_from_directory(images_dir: str, task_id: str) -> Dict[str, Any]:
    """Dramatiq actor pro zpracování obrázků z adresáře"""
    worker = EnhancedProcessingWorker()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(worker.process_images_from_directory(images_dir, task_id))
    finally:
        loop.close()


@dramatiq.actor(queue_name="processing")
def search_cross_modal(
    query: str, search_type: str = "text_to_image", limit: int = 5, task_id: str = None
) -> Dict[str, Any]:
    """Dramatiq actor pro cross-modální vyhledávání"""
    worker = EnhancedProcessingWorker()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            worker.search_cross_modal(query, search_type, limit, task_id)
        )
    finally:
        loop.close()


# Knowledge Graph Dramatiq tasks (FÁZE 1)
@dramatiq.actor(queue_name="processing")
def process_with_knowledge_graph(file_path: str, task_id: str) -> Dict[str, Any]:
    """Dramatiq actor pro zpracování s Knowledge Graph"""
    worker = EnhancedProcessingWorker()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(worker.process_with_knowledge_graph(file_path, task_id))
    finally:
        loop.close()


# Enhanced Processing Dramatiq tasks (FÁZE 3)
@dramatiq.actor(queue_name="processing")
def process_scraped_data_enhanced(file_path: str, task_id: str) -> Dict[str, Any]:
    """Enhanced processing dramatiq actor"""
    worker = EnhancedProcessingWorker()
    return worker.process_scraped_data_enhanced(file_path, task_id)


@dramatiq.actor(queue_name="processing")
def index_for_rag(file_path: str, task_id: str) -> Dict[str, Any]:
    """RAG indexing dramatiq actor"""
    worker = EnhancedProcessingWorker()
    return worker.index_for_rag_enhanced(file_path, task_id)


@dramatiq.actor(queue_name="processing")
def search_rag_documents(query: str, limit: int = 5, task_id: str = None) -> Dict[str, Any]:
    """RAG search dramatiq actor"""
    worker = EnhancedProcessingWorker()
    return worker.search_rag_documents(query, limit, task_id)


# Global worker instance
enhanced_worker = EnhancedProcessingWorker()

logger.info("🚀 Enhanced Processing Worker s multi-modality připraven!")
