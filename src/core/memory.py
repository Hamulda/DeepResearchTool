"""
Modulární architektura pro paměťové úložiště s abstraktní základní třídou
a konkrétní implementací pro ChromaDB a memory management

Author: Senior Python/MLOps Agent
"""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
import gc
import os

logger = logging.getLogger(__name__)

# Safe import of psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring will be limited")


@dataclass
class Document:
    """Reprezentace dokumentu v paměťovém úložišti"""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    id: Optional[str] = None


class BaseMemoryStore(ABC):
    """Abstraktní základní třída pro paměťové úložiště"""

    @abstractmethod
    async def initialize(self) -> None:
        """Inicializace úložiště"""
        pass

    @abstractmethod
    async def add_document(self, document: Document) -> str:
        """Přidání dokumentu do úložiště"""
        pass

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Přidání více dokumentů najednou"""
        pass

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[Document]:
        """Vyhledání dokumentů"""
        pass

    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Získání konkrétního dokumentu"""
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """Smazání dokumentu"""
        pass


class MemoryManager:
    """Správce paměti pro optimalizaci výkonu na M1 MacBook"""

    def __init__(self, max_memory_percent: float = 80.0):
        """
        Inicializace memory manageru
        
        Args:
            max_memory_percent: Maximální procento využití paměti
        """
        self.max_memory_percent = max_memory_percent
        self.memory_store = None
        
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None

    def get_memory_info(self) -> Dict[str, Any]:
        """Získání informací o využití paměti"""
        try:
            if not PSUTIL_AVAILABLE:
                return {
                    "system": {"available": "unknown", "used_percent": 0},
                    "process": {"rss_mb": 0, "percent": 0},
                    "psutil_available": False
                }

            # Systémové informace o paměti
            memory = psutil.virtual_memory()
            
            # Informace o procesu
            process_memory = self.process.memory_info()
            
            return {
                "system": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_percent": memory.percent,
                    "free_gb": memory.free / (1024**3)
                },
                "process": {
                    "rss_mb": process_memory.rss / (1024**2),
                    "vms_mb": process_memory.vms / (1024**2),
                    "percent": self.process.memory_percent()
                },
                "psutil_available": True
            }
        except Exception as e:
            logger.error(f"Chyba při získávání informací o paměti: {e}")
            return {"psutil_available": False, "error": str(e)}

    def check_memory_pressure(self) -> Dict[str, Any]:
        """Kontrola tlaku na paměť"""
        try:
            memory_info = self.get_memory_info()
            
            if not memory_info.get("psutil_available", False):
                return {"pressure": False, "psutil_available": False}

            system_pressure = memory_info.get("system", {}).get("used_percent", 0) > self.max_memory_percent
            process_pressure = memory_info.get("process", {}).get("percent", 0) > 50.0

            return {
                "pressure": system_pressure or process_pressure,
                "system_pressure": system_pressure,
                "process_pressure": process_pressure,
                "memory_info": memory_info
            }
        except Exception as e:
            logger.error(f"Chyba při kontrole tlaku na paměť: {e}")
            return {"pressure": False, "error": str(e)}
    
    def force_gc(self) -> None:
        """Vynucené spuštění garbage collectoru"""
        try:
            collected = gc.collect()
            logger.info(f"Garbage collector uvolnil {collected} objektů")
        except Exception as e:
            logger.error(f"Chyba při garbage collection: {e}")
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimalizace využití paměti"""
        try:
            # Před optimalizací
            before_memory = self.get_memory_info()
            
            # Garbage collection
            self.force_gc()
            
            # Po optimalizaci
            after_memory = self.get_memory_info()
            
            # Výpočet úspory pouze pokud je psutil dostupné
            if before_memory.get("psutil_available") and after_memory.get("psutil_available"):
                before_rss = before_memory.get("process", {}).get("rss_mb", 0)
                after_rss = after_memory.get("process", {}).get("rss_mb", 0)
                saved_mb = before_rss - after_rss
            else:
                saved_mb = 0

            return {
                "success": True,
                "memory_saved_mb": saved_mb,
                "before": before_memory,
                "after": after_memory,
                "psutil_available": PSUTIL_AVAILABLE
            }
        except Exception as e:
            logger.error(f"Chyba při optimalizaci paměti: {e}")
            return {"success": False, "error": str(e)}
    
    async def monitor_memory_async(self, interval: int = 30):
        """Asynchronní monitoring paměti"""
        while True:
            try:
                pressure_info = self.check_memory_pressure()
                
                if pressure_info.get("pressure", False):
                    logger.warning("Detekován tlak na paměť, spouštím optimalizaci...")
                    self.optimize_memory()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Chyba v memory monitoru: {e}")
                await asyncio.sleep(interval)
    
    def set_memory_store(self, store: BaseMemoryStore):
        """Nastavení paměťového úložiště pro monitoring"""
        self.memory_store = store


class SimpleMemoryStore(BaseMemoryStore):
    """Jednoduchá in-memory implementace pro testování"""
    
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.counter = 0
    
    async def initialize(self) -> None:
        """Inicializace úložiště"""
        logger.info("SimpleMemoryStore inicializováno")
    
    async def add_document(self, document: Document) -> str:
        """Přidání dokumentu"""
        if document.id is None:
            document.id = f"doc_{self.counter}"
            self.counter += 1
        
        self.documents[document.id] = document
        return document.id
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Přidání více dokumentů"""
        ids = []
        for doc in documents:
            doc_id = await self.add_document(doc)
            ids.append(doc_id)
        return ids
    
    async def search(self, query: str, limit: int = 10) -> List[Document]:
        """Jednoduché textové vyhledání"""
        results = []
        query_lower = query.lower()
        
        for doc in self.documents.values():
            if query_lower in doc.content.lower():
                results.append(doc)
                if len(results) >= limit:
                    break
        
        return results
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Získání dokumentu podle ID"""
        return self.documents.get(doc_id)
    
    async def delete_document(self, doc_id: str) -> bool:
        """Smazání dokumentu"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiky úložiště"""
        return {
            "total_documents": len(self.documents),
            "memory_usage_mb": sum(len(doc.content.encode('utf-8')) for doc in self.documents.values()) / (1024**2)
        }


# Factory pro vytvoření memory store
def create_memory_store(store_type: str = "simple") -> BaseMemoryStore:
    """
    Factory funkce pro vytvoření paměťového úložiště
    
    Args:
        store_type: Typ úložiště ("simple", "chroma", "qdrant")
    
    Returns:
        Instance paměťového úložiště
    """
    if store_type == "simple":
        return SimpleMemoryStore()
    else:
        raise NotImplementedError(f"Store type '{store_type}' není implementován")


# Export hlavních tříd
__all__ = [
    "MemoryManager",
    "BaseMemoryStore", 
    "SimpleMemoryStore",
    "Document",
    "create_memory_store"
]