"""
📚 Document Store pro ukládání a správu dokumentů
Poskytuje jednotné rozhraní pro ukládání a vyhledávání dokumentů
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import hashlib
import uuid


class DocumentStore:
    """
    📄 Správce dokumentů pro autonomní agenta

    Poskytuje funkcionalitu pro ukládání, vyhledávání a správu
    dokumentů získaných během výzkumného procesu.
    """

    def __init__(self, storage_path: str = "data/documents"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory index pro rychlé vyhledávání
        self.documents_index: Dict[str, Dict[str, Any]] = {}
        self.metadata_file = self.storage_path / "index.json"

        # Načtení existujícího indexu
        self._load_index()

    async def store_document(self, document: Dict[str, Any]) -> str:
        """
        Uloží dokument do store

        Args:
            document: Dictionary s dokumentem (musí obsahovat 'content')

        Returns:
            str: Unikátní ID dokumentu
        """
        try:
            # Generování ID
            doc_id = str(uuid.uuid4())

            # Příprava metadat
            metadata = {
                "id": doc_id,
                "url": document.get("url", ""),
                "title": document.get("title", ""),
                "content_hash": self._calculate_hash(document.get("content", "")),
                "content_length": len(document.get("content", "")),
                "timestamp": (
                    document.get("timestamp", datetime.now()).isoformat()
                    if isinstance(document.get("timestamp"), datetime)
                    else document.get("timestamp", datetime.now().isoformat())
                ),
                "source": document.get("source", "unknown"),
                "content_type": document.get("content_type", "text/plain"),
                "credibility_score": document.get("credibility_score", 0.0),
                "entities": document.get("entities", []),
                "patterns": document.get("patterns", []),
            }

            # Uložení obsahu do souboru
            content_file = self.storage_path / f"{doc_id}.txt"
            with open(content_file, "w", encoding="utf-8") as f:
                f.write(document.get("content", ""))

            # Uložení metadat do indexu
            self.documents_index[doc_id] = metadata

            # Persist index
            await self._save_index()

            logging.info(f"📚 Dokument uložen: {doc_id}")
            return doc_id

        except Exception as e:
            logging.error(f"❌ Chyba při ukládání dokumentu: {e}")
            raise

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Získá dokument podle ID

        Args:
            doc_id: ID dokumentu

        Returns:
            Dict s dokumentem nebo None
        """
        try:
            if doc_id not in self.documents_index:
                return None

            metadata = self.documents_index[doc_id].copy()

            # Načtení obsahu
            content_file = self.storage_path / f"{doc_id}.txt"
            if content_file.exists():
                with open(content_file, "r", encoding="utf-8") as f:
                    metadata["content"] = f.read()
            else:
                metadata["content"] = ""

            return metadata

        except Exception as e:
            logging.error(f"❌ Chyba při načítání dokumentu {doc_id}: {e}")
            return None

    async def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Vyhledá dokumenty podle query

        Args:
            query: Vyhledávací dotaz
            limit: Max počet výsledků

        Returns:
            List dokumentů seřazených podle relevance
        """
        try:
            results = []
            query_lower = query.lower()

            for doc_id, metadata in self.documents_index.items():
                score = 0.0

                # Vyhledávání v title
                title = metadata.get("title", "").lower()
                if query_lower in title:
                    score += 2.0

                # Vyhledávání v URL
                url = metadata.get("url", "").lower()
                if query_lower in url:
                    score += 1.0

                # Vyhledávání v source
                source = metadata.get("source", "").lower()
                if query_lower in source:
                    score += 0.5

                # Bonus za vysokou důvěryhodnost
                credibility = metadata.get("credibility_score", 0.0)
                score += credibility * 0.5

                if score > 0:
                    result = metadata.copy()
                    result["search_score"] = score
                    results.append(result)

            # Seřazení podle skóre
            results.sort(key=lambda x: x["search_score"], reverse=True)
            return results[:limit]

        except Exception as e:
            logging.error(f"❌ Chyba při vyhledávání: {e}")
            return []

    async def get_documents_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Vrátí všechny dokumenty z konkrétního zdroje"""
        try:
            results = []
            for doc_id, metadata in self.documents_index.items():
                if metadata.get("source") == source:
                    doc = await self.get_document(doc_id)
                    if doc:
                        results.append(doc)
            return results
        except Exception as e:
            logging.error(f"❌ Chyba při načítání dokumentů ze zdroje {source}: {e}")
            return []

    async def get_recent_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Vrátí nejnovější dokumenty"""
        try:
            # Seřazení podle timestamp
            sorted_docs = sorted(
                self.documents_index.items(), key=lambda x: x[1].get("timestamp", ""), reverse=True
            )

            results = []
            for doc_id, _ in sorted_docs[:limit]:
                doc = await self.get_document(doc_id)
                if doc:
                    results.append(doc)

            return results
        except Exception as e:
            logging.error(f"❌ Chyba při načítání nejnovějších dokumentů: {e}")
            return []

    async def update_document_metadata(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """
        Aktualizuje metadata dokumentu

        Args:
            doc_id: ID dokumentu
            updates: Dictionary s aktualizacemi

        Returns:
            bool: True při úspěchu
        """
        try:
            if doc_id not in self.documents_index:
                return False

            # Aktualizace metadat
            self.documents_index[doc_id].update(updates)

            # Persist changes
            await self._save_index()

            logging.info(f"📝 Metadata dokumentu {doc_id} aktualizována")
            return True

        except Exception as e:
            logging.error(f"❌ Chyba při aktualizaci metadat {doc_id}: {e}")
            return False

    async def delete_document(self, doc_id: str) -> bool:
        """Smaže dokument"""
        try:
            if doc_id not in self.documents_index:
                return False

            # Smazání souboru
            content_file = self.storage_path / f"{doc_id}.txt"
            if content_file.exists():
                content_file.unlink()

            # Odstranění z indexu
            del self.documents_index[doc_id]

            # Persist changes
            await self._save_index()

            logging.info(f"🗑️ Dokument {doc_id} smazán")
            return True

        except Exception as e:
            logging.error(f"❌ Chyba při mazání dokumentu {doc_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Vrátí statistiky document store"""
        try:
            total_docs = len(self.documents_index)
            total_size = 0
            sources = {}

            for metadata in self.documents_index.values():
                # Velikost
                total_size += metadata.get("content_length", 0)

                # Zdroje
                source = metadata.get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1

            return {
                "total_documents": total_docs,
                "total_content_size": total_size,
                "sources": sources,
                "storage_path": str(self.storage_path),
                "avg_credibility": self._calculate_avg_credibility(),
            }
        except Exception as e:
            logging.error(f"❌ Chyba při získávání statistik: {e}")
            return {}

    def _calculate_hash(self, content: str) -> str:
        """Vypočítá hash obsahu"""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _calculate_avg_credibility(self) -> float:
        """Vypočítá průměrnou důvěryhodnost dokumentů"""
        if not self.documents_index:
            return 0.0

        total_credibility = sum(
            metadata.get("credibility_score", 0.0) for metadata in self.documents_index.values()
        )
        return total_credibility / len(self.documents_index)

    def _load_index(self):
        """Načte index z souboru"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.documents_index = json.load(f)
                logging.info(f"📚 Načten index s {len(self.documents_index)} dokumenty")
        except Exception as e:
            logging.warning(f"⚠️ Chyba při načítání indexu: {e}")
            self.documents_index = {}

    async def _save_index(self):
        """Uloží index do souboru"""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.documents_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"❌ Chyba při ukládání indexu: {e}")

    async def cleanup_old_documents(self, days: int = 30) -> int:
        """
        Vyčistí staré dokumenty

        Args:
            days: Dokumenty starší než tento počet dní budou smazány

        Returns:
            int: Počet smazaných dokumentů
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            deleted_count = 0

            to_delete = []
            for doc_id, metadata in self.documents_index.items():
                doc_timestamp = datetime.fromisoformat(metadata.get("timestamp", ""))
                if doc_timestamp < cutoff_date:
                    to_delete.append(doc_id)

            for doc_id in to_delete:
                if await self.delete_document(doc_id):
                    deleted_count += 1

            logging.info(f"🧹 Smazáno {deleted_count} starých dokumentů")
            return deleted_count

        except Exception as e:
            logging.error(f"❌ Chyba při čištění dokumentů: {e}")
            return 0
