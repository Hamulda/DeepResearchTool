#!/usr/bin/env python3
"""
MCP Qdrant Integration (Optional Feature)
MCP server client for Qdrant collection inspection and health checks from IDE

Author: Senior IT Specialist
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


@dataclass
class QdrantCollectionInfo:
    """Qdrant collection information"""
    name: str
    vectors_count: int
    points_count: int
    status: str
    config: Dict[str, Any]
    disk_usage: Optional[int] = None
    ram_usage: Optional[int] = None


@dataclass
class QdrantHealthStatus:
    """Qdrant cluster health status"""
    status: str  # "green", "yellow", "red"
    version: str
    collections_count: int
    total_vectors: int
    total_points: int
    disk_usage: int
    memory_usage: int
    uptime: Optional[str] = None


class MCPQdrantClient:
    """MCP client for Qdrant inspection and management"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Feature flag
        self.enabled = config.get("mcp", {}).get("qdrant", {}).get("enabled", False)

        if not self.enabled:
            self.logger.info("MCP Qdrant integration disabled by feature flag")
            return

        if not QDRANT_AVAILABLE:
            self.logger.error("Qdrant client not available - install with: pip install qdrant-client")
            self.enabled = False
            return

        # Qdrant connection
        self.qdrant_url = config.get("qdrant", {}).get("url", "http://localhost:6333")
        self.client = None
        self.connected = False

    async def connect(self) -> bool:
        """Connect to Qdrant instance"""
        if not self.enabled:
            return False

        try:
            self.client = QdrantClient(url=self.qdrant_url)

            # Test connection
            collections = await asyncio.to_thread(self.client.get_collections)
            self.connected = True
            self.logger.info(f"MCP Qdrant client connected to {self.qdrant_url}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            self.connected = False
            return False

    async def get_health_status(self) -> Optional[QdrantHealthStatus]:
        """Get comprehensive Qdrant health status"""
        if not self.connected:
            if not await self.connect():
                return None

        try:
            # Get cluster info
            cluster_info = await asyncio.to_thread(self.client.get_cluster_info)

            # Get collections
            collections_response = await asyncio.to_thread(self.client.get_collections)
            collections = collections_response.collections

            # Calculate totals
            total_vectors = 0
            total_points = 0

            for collection in collections:
                try:
                    collection_info = await asyncio.to_thread(
                        self.client.get_collection,
                        collection_name=collection.name
                    )
                    total_vectors += collection_info.vectors_count or 0
                    total_points += collection_info.points_count or 0
                except Exception as e:
                    self.logger.warning(f"Failed to get info for collection {collection.name}: {e}")

            # Determine status
            status = "green"
            if len(collections) == 0:
                status = "yellow"  # No collections
            elif any(getattr(col, 'status', 'green') != 'green' for col in collections):
                status = "red"  # Collection issues

            health = QdrantHealthStatus(
                status=status,
                version=cluster_info.get("version", "unknown"),
                collections_count=len(collections),
                total_vectors=total_vectors,
                total_points=total_points,
                disk_usage=0,  # Would need additional API calls
                memory_usage=0  # Would need additional API calls
            )

            return health

        except Exception as e:
            self.logger.error(f"Failed to get health status: {e}")
            return QdrantHealthStatus(
                status="red",
                version="unknown",
                collections_count=0,
                total_vectors=0,
                total_points=0,
                disk_usage=0,
                memory_usage=0
            )

    async def list_collections(self) -> List[QdrantCollectionInfo]:
        """List all collections with detailed information"""
        if not self.connected:
            if not await self.connect():
                return []

        collection_infos = []

        try:
            collections_response = await asyncio.to_thread(self.client.get_collections)
            collections = collections_response.collections

            for collection in collections:
                try:
                    # Get detailed collection info
                    collection_info = await asyncio.to_thread(
                        self.client.get_collection,
                        collection_name=collection.name
                    )

                    # Extract configuration
                    config = {}
                    if hasattr(collection_info, 'config'):
                        config = {
                            "distance": getattr(collection_info.config.params.vectors, 'distance', 'unknown'),
                            "size": getattr(collection_info.config.params.vectors, 'size', 0),
                            "hnsw_config": getattr(collection_info.config.hnsw_config, '__dict__', {}) if collection_info.config.hnsw_config else {}
                        }

                    info = QdrantCollectionInfo(
                        name=collection.name,
                        vectors_count=collection_info.vectors_count or 0,
                        points_count=collection_info.points_count or 0,
                        status=getattr(collection_info, 'status', 'active'),
                        config=config
                    )

                    collection_infos.append(info)

                except Exception as e:
                    self.logger.warning(f"Failed to get info for collection {collection.name}: {e}")
                    # Add minimal info
                    info = QdrantCollectionInfo(
                        name=collection.name,
                        vectors_count=0,
                        points_count=0,
                        status="unknown",
                        config={}
                    )
                    collection_infos.append(info)

        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}")

        return collection_infos

    async def inspect_collection(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Inspect specific collection in detail"""
        if not self.connected:
            if not await self.connect():
                return None

        try:
            # Get collection info
            collection_info = await asyncio.to_thread(
                self.client.get_collection,
                collection_name=collection_name
            )

            # Get sample points
            sample_points = []
            try:
                points_response = await asyncio.to_thread(
                    self.client.scroll,
                    collection_name=collection_name,
                    limit=5
                )
                sample_points = [
                    {
                        "id": point.id,
                        "payload_keys": list(point.payload.keys()) if point.payload else [],
                        "vector_size": len(point.vector) if hasattr(point, 'vector') and point.vector else 0
                    }
                    for point in points_response[0]
                ]
            except Exception as e:
                self.logger.warning(f"Failed to get sample points: {e}")

            # Compile inspection results
            inspection = {
                "collection_name": collection_name,
                "status": getattr(collection_info, 'status', 'unknown'),
                "vectors_count": collection_info.vectors_count or 0,
                "points_count": collection_info.points_count or 0,
                "config": {
                    "vector_size": getattr(collection_info.config.params.vectors, 'size', 0) if collection_info.config else 0,
                    "distance_metric": getattr(collection_info.config.params.vectors, 'distance', 'unknown') if collection_info.config else 'unknown',
                    "hnsw_config": getattr(collection_info.config.hnsw_config, '__dict__', {}) if collection_info.config and collection_info.config.hnsw_config else {},
                    "quantization": getattr(collection_info.config.quantization_config, '__dict__', {}) if collection_info.config and collection_info.config.quantization_config else None
                },
                "sample_points": sample_points,
                "inspection_timestamp": datetime.now().isoformat()
            }

            return inspection

        except Exception as e:
            self.logger.error(f"Failed to inspect collection {collection_name}: {e}")
            return None

    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        if not self.enabled:
            return {
                "status": "disabled",
                "message": "MCP Qdrant integration disabled by feature flag"
            }

        health_report = {
            "timestamp": datetime.now().isoformat(),
            "mcp_qdrant_enabled": self.enabled,
            "connection_status": "disconnected",
            "health_status": None,
            "collections": [],
            "issues": [],
            "recommendations": []
        }

        # Test connection
        if await self.connect():
            health_report["connection_status"] = "connected"

            # Get health status
            health_status = await self.get_health_status()
            if health_status:
                health_report["health_status"] = asdict(health_status)

            # Get collections
            collections = await self.list_collections()
            health_report["collections"] = [asdict(col) for col in collections]

            # Analyze issues
            issues = []
            recommendations = []

            if health_status:
                if health_status.status == "red":
                    issues.append("Qdrant cluster status is RED")
                    recommendations.append("Check Qdrant logs and restart if necessary")

                if health_status.collections_count == 0:
                    issues.append("No collections found")
                    recommendations.append("Create collections for document indexing")

                if health_status.total_vectors == 0:
                    issues.append("No vectors indexed")
                    recommendations.append("Index documents to enable vector search")

            # Collection-specific checks
            for collection in collections:
                if collection.status != "active":
                    issues.append(f"Collection {collection.name} status: {collection.status}")

                if collection.vectors_count == 0:
                    issues.append(f"Collection {collection.name} has no vectors")
                    recommendations.append(f"Index documents into collection {collection.name}")

            health_report["issues"] = issues
            health_report["recommendations"] = recommendations

        else:
            health_report["issues"] = ["Failed to connect to Qdrant"]
            health_report["recommendations"] = [
                "Check Qdrant server is running",
                f"Verify connection URL: {self.qdrant_url}",
                "Check network connectivity"
            ]

        return health_report

    def generate_mcp_inspection_report(self, health_report: Dict[str, Any]) -> str:
        """Generate human-readable inspection report for IDE"""
        lines = []
        lines.append("# Qdrant MCP Inspection Report")
        lines.append(f"Generated: {health_report['timestamp']}")
        lines.append("")

        # Connection status
        lines.append("## Connection Status")
        status_icon = "✅" if health_report["connection_status"] == "connected" else "❌"
        lines.append(f"{status_icon} Connection: {health_report['connection_status']}")
        lines.append("")

        # Health status
        if health_report["health_status"]:
            health = health_report["health_status"]
            lines.append("## Cluster Health")
            status_icon = {"green": "✅", "yellow": "⚠️", "red": "❌"}.get(health["status"], "❓")
            lines.append(f"{status_icon} Status: {health['status']}")
            lines.append(f"📊 Collections: {health['collections_count']}")
            lines.append(f"🔢 Total Vectors: {health['total_vectors']:,}")
            lines.append(f"📄 Total Points: {health['total_points']:,}")
            lines.append("")

        # Collections
        if health_report["collections"]:
            lines.append("## Collections")
            for collection in health_report["collections"]:
                status_icon = "✅" if collection["status"] == "active" else "⚠️"
                lines.append(f"{status_icon} **{collection['name']}**")
                lines.append(f"  - Vectors: {collection['vectors_count']:,}")
                lines.append(f"  - Points: {collection['points_count']:,}")
                lines.append(f"  - Status: {collection['status']}")
                if collection["config"]:
                    config = collection["config"]
                    if "size" in config:
                        lines.append(f"  - Vector Size: {config['size']}")
                    if "distance" in config:
                        lines.append(f"  - Distance: {config['distance']}")
                lines.append("")

        # Issues
        if health_report["issues"]:
            lines.append("## Issues Found")
            for issue in health_report["issues"]:
                lines.append(f"❌ {issue}")
            lines.append("")

        # Recommendations
        if health_report["recommendations"]:
            lines.append("## Recommendations")
            for rec in health_report["recommendations"]:
                lines.append(f"💡 {rec}")
            lines.append("")

        lines.append("---")
        lines.append("*Report generated by MCP Qdrant Integration*")

        return "\n".join(lines)


class MCPQdrantIntegration:
    """Main MCP Qdrant integration manager"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.mcp_client = MCPQdrantClient(config)

        # Integration settings
        self.auto_health_check = config.get("mcp", {}).get("auto_health_check", True)
        self.health_check_interval = config.get("mcp", {}).get("health_check_interval", 300)  # 5 minutes

    async def initialize(self) -> bool:
        """Initialize MCP integration"""
        if not self.mcp_client.enabled:
            return False

        self.logger.info("Initializing MCP Qdrant integration...")

        success = await self.mcp_client.connect()

        if success and self.auto_health_check:
            # Run initial health check
            health_report = await self.mcp_client.run_health_check()

            if health_report["issues"]:
                self.logger.warning(f"Qdrant health issues detected: {len(health_report['issues'])} issues")
                for issue in health_report["issues"]:
                    self.logger.warning(f"  - {issue}")

        return success

    async def get_inspection_summary(self) -> Optional[str]:
        """Get inspection summary for IDE display"""
        if not self.mcp_client.enabled:
            return "MCP Qdrant integration disabled"

        try:
            health_report = await self.mcp_client.run_health_check()
            return self.mcp_client.generate_mcp_inspection_report(health_report)
        except Exception as e:
            self.logger.error(f"Failed to generate inspection summary: {e}")
            return f"MCP inspection failed: {e}"

    async def inspect_collection_for_ide(self, collection_name: str) -> Optional[str]:
        """Inspect specific collection and format for IDE"""
        if not self.mcp_client.enabled:
            return "MCP Qdrant integration disabled"

        try:
            inspection = await self.mcp_client.inspect_collection(collection_name)

            if not inspection:
                return f"Failed to inspect collection: {collection_name}"

            # Format for IDE display
            lines = []
            lines.append(f"# Collection Inspection: {collection_name}")
            lines.append(f"Inspected: {inspection['inspection_timestamp']}")
            lines.append("")

            lines.append("## Overview")
            lines.append(f"Status: {inspection['status']}")
            lines.append(f"Vectors: {inspection['vectors_count']:,}")
            lines.append(f"Points: {inspection['points_count']:,}")
            lines.append("")

            lines.append("## Configuration")
            config = inspection['config']
            lines.append(f"Vector Size: {config['vector_size']}")
            lines.append(f"Distance Metric: {config['distance_metric']}")

            if config['hnsw_config']:
                lines.append("HNSW Config:")
                for key, value in config['hnsw_config'].items():
                    lines.append(f"  - {key}: {value}")

            if config['quantization']:
                lines.append("Quantization: Enabled")
            lines.append("")

            if inspection['sample_points']:
                lines.append("## Sample Points")
                for i, point in enumerate(inspection['sample_points'][:3], 1):
                    lines.append(f"Point {i}:")
                    lines.append(f"  - ID: {point['id']}")
                    lines.append(f"  - Vector Size: {point['vector_size']}")
                    lines.append(f"  - Payload Keys: {', '.join(point['payload_keys'])}")

            return "\n".join(lines)

        except Exception as e:
            self.logger.error(f"Failed to inspect collection {collection_name}: {e}")
            return f"Collection inspection failed: {e}"


def create_mcp_qdrant_integration(config: Dict[str, Any]) -> MCPQdrantIntegration:
    """Factory function for MCP Qdrant integration"""
    return MCPQdrantIntegration(config)
