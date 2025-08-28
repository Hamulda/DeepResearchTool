#!/usr/bin/env python3
"""
Cost Tracker
Comprehensive cost tracking and analysis for DeepResearchTool operations

Author: Senior Python/MLOps Agent
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Categories of costs tracked"""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    LLM_INFERENCE = "llm_inference"
    EMBEDDING = "embedding"
    RERANKING = "reranking"
    SCRAPING = "scraping"
    MANUAL_REVIEW = "manual_review"


@dataclass
class CostEvent:
    """Individual cost event"""
    event_id: str
    category: CostCategory
    operation: str
    cost_usd: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "category": self.category.value,
            "operation": self.operation,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ClaimCost:
    """Cost breakdown for a single claim"""
    claim_id: str
    total_cost_usd: float
    cost_breakdown: Dict[CostCategory, float] = field(default_factory=dict)
    events: List[CostEvent] = field(default_factory=list)

    def add_cost(self, event: CostEvent):
        """Add a cost event to this claim"""
        self.events.append(event)
        if event.category not in self.cost_breakdown:
            self.cost_breakdown[event.category] = 0.0
        self.cost_breakdown[event.category] += event.cost_usd
        self.total_cost_usd += event.cost_usd

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "total_cost_usd": self.total_cost_usd,
            "cost_breakdown": {cat.value: cost for cat, cost in self.cost_breakdown.items()},
            "events": [event.to_dict() for event in self.events]
        }


class CostModel:
    """Cost model with pricing for different operations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Default pricing (in USD)
        self.pricing = {
            # LLM costs (per 1K tokens)
            "llm.ollama.input_token": 0.0,  # Local Ollama is free
            "llm.ollama.output_token": 0.0,
            "llm.openai.gpt4.input_token": 0.03,  # If fallback to OpenAI
            "llm.openai.gpt4.output_token": 0.06,

            # Embedding costs (per 1K tokens)
            "embedding.local": 0.0,  # Local embeddings are free
            "embedding.openai": 0.0001,

            # Compute costs (per minute)
            "compute.cpu_minute": 0.002,  # Rough estimate
            "compute.memory_gb_minute": 0.001,

            # Storage costs (per GB per day)
            "storage.disk_gb_day": 0.0001,
            "storage.cache_gb_day": 0.0002,

            # Network costs (per GB)
            "network.transfer_gb": 0.01,
            "network.api_request": 0.001,

            # Specialized operations
            "reranking.cross_encoder_operation": 0.001,
            "scraping.webpage": 0.0001,
            "manual_review.minute": 2.0  # Human reviewer cost
        }

        # Update with config overrides
        pricing_config = config.get("cost_tracking", {}).get("pricing", {})
        self.pricing.update(pricing_config)

        logger.info("Cost model initialized with pricing configuration")

    def calculate_llm_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate LLM inference cost"""

        if "ollama" in model_name.lower():
            return 0.0  # Local Ollama is free

        # Default to GPT-4 pricing
        input_cost = (input_tokens / 1000) * self.pricing.get("llm.openai.gpt4.input_token", 0.03)
        output_cost = (output_tokens / 1000) * self.pricing.get("llm.openai.gpt4.output_token", 0.06)

        return input_cost + output_cost

    def calculate_embedding_cost(self, tokens: int, model_type: str = "local") -> float:
        """Calculate embedding cost"""

        if model_type == "local":
            return 0.0

        price_per_1k = self.pricing.get(f"embedding.{model_type}", 0.0001)
        return (tokens / 1000) * price_per_1k

    def calculate_compute_cost(self, duration_minutes: float, cpu_cores: int = 1, memory_gb: float = 1.0) -> float:
        """Calculate compute cost"""

        cpu_cost = duration_minutes * cpu_cores * self.pricing.get("compute.cpu_minute", 0.002)
        memory_cost = duration_minutes * memory_gb * self.pricing.get("compute.memory_gb_minute", 0.001)

        return cpu_cost + memory_cost

    def calculate_storage_cost(self, size_gb: float, duration_days: float = 1.0, storage_type: str = "disk") -> float:
        """Calculate storage cost"""

        price_per_gb_day = self.pricing.get(f"storage.{storage_type}_gb_day", 0.0001)
        return size_gb * duration_days * price_per_gb_day

    def calculate_network_cost(self, transfer_gb: float = 0.0, api_requests: int = 0) -> float:
        """Calculate network cost"""

        transfer_cost = transfer_gb * self.pricing.get("network.transfer_gb", 0.01)
        api_cost = api_requests * self.pricing.get("network.api_request", 0.001)

        return transfer_cost + api_cost

    def calculate_operation_cost(self, operation: str, quantity: int = 1) -> float:
        """Calculate cost for specific operations"""

        unit_cost = self.pricing.get(operation, 0.0)
        return unit_cost * quantity


class CostTracker:
    """Main cost tracking system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cost_config = config.get("cost_tracking", {})

        # Cost model
        self.cost_model = CostModel(config)

        # Storage
        self.storage_dir = Path(config.get("cost_storage", "cost_data"))
        self.storage_dir.mkdir(exist_ok=True)

        # Tracking state
        self.enabled = self.cost_config.get("enabled", True)
        self.current_session_id = None

        # Cost tracking
        self.claim_costs: Dict[str, ClaimCost] = {}
        self.session_costs: Dict[str, List[CostEvent]] = defaultdict(list)
        self.total_costs_by_category: Dict[CostCategory, float] = defaultdict(float)

        # Performance metrics
        self.cost_tracking_stats = {
            "total_events_tracked": 0,
            "total_cost_usd": 0.0,
            "claims_tracked": 0,
            "sessions_tracked": 0
        }

        logger.info(f"Cost tracker initialized (enabled: {self.enabled})")

    def start_session(self, session_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Start a new cost tracking session"""

        if not self.enabled:
            return

        self.current_session_id = session_id

        if metadata is None:
            metadata = {}

        # Record session start event
        start_event = CostEvent(
            event_id=f"{session_id}_start",
            category=CostCategory.COMPUTE,
            operation="session_start",
            cost_usd=0.0,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "session_id": session_id,
                "session_type": metadata.get("type", "research"),
                **metadata
            }
        )

        self.session_costs[session_id].append(start_event)
        self.cost_tracking_stats["sessions_tracked"] += 1

        logger.info(f"Started cost tracking session: {session_id}")

    def track_llm_cost(
        self,
        claim_id: Optional[str],
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "inference",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track LLM inference cost"""

        if not self.enabled:
            return

        cost = self.cost_model.calculate_llm_cost(model_name, input_tokens, output_tokens)

        event_metadata = {
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            **(metadata or {})
        }

        self._record_cost_event(
            claim_id=claim_id,
            category=CostCategory.LLM_INFERENCE,
            operation=operation,
            cost_usd=cost,
            metadata=event_metadata
        )

    def track_embedding_cost(
        self,
        claim_id: Optional[str],
        tokens: int,
        model_type: str = "local",
        operation: str = "embedding",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track embedding cost"""

        if not self.enabled:
            return

        cost = self.cost_model.calculate_embedding_cost(tokens, model_type)

        event_metadata = {
            "model_type": model_type,
            "tokens": tokens,
            **(metadata or {})
        }

        self._record_cost_event(
            claim_id=claim_id,
            category=CostCategory.EMBEDDING,
            operation=operation,
            cost_usd=cost,
            metadata=event_metadata
        )

    def track_compute_cost(
        self,
        claim_id: Optional[str],
        duration_minutes: float,
        cpu_cores: int = 1,
        memory_gb: float = 1.0,
        operation: str = "processing",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track compute cost"""

        if not self.enabled:
            return

        cost = self.cost_model.calculate_compute_cost(duration_minutes, cpu_cores, memory_gb)

        event_metadata = {
            "duration_minutes": duration_minutes,
            "cpu_cores": cpu_cores,
            "memory_gb": memory_gb,
            **(metadata or {})
        }

        self._record_cost_event(
            claim_id=claim_id,
            category=CostCategory.COMPUTE,
            operation=operation,
            cost_usd=cost,
            metadata=event_metadata
        )

    def track_reranking_cost(
        self,
        claim_id: Optional[str],
        operations: int,
        reranking_type: str = "cross_encoder",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track reranking cost"""

        if not self.enabled:
            return

        cost = self.cost_model.calculate_operation_cost(f"reranking.{reranking_type}_operation", operations)

        event_metadata = {
            "operations": operations,
            "reranking_type": reranking_type,
            **(metadata or {})
        }

        self._record_cost_event(
            claim_id=claim_id,
            category=CostCategory.RERANKING,
            operation=reranking_type,
            cost_usd=cost,
            metadata=event_metadata
        )

    def track_scraping_cost(
        self,
        claim_id: Optional[str],
        pages_scraped: int,
        transfer_gb: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track web scraping cost"""

        if not self.enabled:
            return

        scraping_cost = self.cost_model.calculate_operation_cost("scraping.webpage", pages_scraped)
        network_cost = self.cost_model.calculate_network_cost(transfer_gb=transfer_gb)
        total_cost = scraping_cost + network_cost

        event_metadata = {
            "pages_scraped": pages_scraped,
            "transfer_gb": transfer_gb,
            "scraping_cost": scraping_cost,
            "network_cost": network_cost,
            **(metadata or {})
        }

        self._record_cost_event(
            claim_id=claim_id,
            category=CostCategory.SCRAPING,
            operation="web_scraping",
            cost_usd=total_cost,
            metadata=event_metadata
        )

    def track_storage_cost(
        self,
        claim_id: Optional[str],
        size_gb: float,
        duration_days: float = 1.0,
        storage_type: str = "disk",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track storage cost"""

        if not self.enabled:
            return

        cost = self.cost_model.calculate_storage_cost(size_gb, duration_days, storage_type)

        event_metadata = {
            "size_gb": size_gb,
            "duration_days": duration_days,
            "storage_type": storage_type,
            **(metadata or {})
        }

        self._record_cost_event(
            claim_id=claim_id,
            category=CostCategory.STORAGE,
            operation=f"{storage_type}_storage",
            cost_usd=cost,
            metadata=event_metadata
        )

    def _record_cost_event(
        self,
        claim_id: Optional[str],
        category: CostCategory,
        operation: str,
        cost_usd: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a cost event"""

        event_id = f"{self.current_session_id}_{category.value}_{len(self.session_costs[self.current_session_id])}"

        event = CostEvent(
            event_id=event_id,
            category=category,
            operation=operation,
            cost_usd=cost_usd,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "session_id": self.current_session_id,
                "claim_id": claim_id,
                **(metadata or {})
            }
        )

        # Add to session costs
        if self.current_session_id:
            self.session_costs[self.current_session_id].append(event)

        # Add to claim costs if claim_id provided
        if claim_id:
            if claim_id not in self.claim_costs:
                self.claim_costs[claim_id] = ClaimCost(claim_id, 0.0)
                self.cost_tracking_stats["claims_tracked"] += 1

            self.claim_costs[claim_id].add_cost(event)

        # Update totals
        self.total_costs_by_category[category] += cost_usd
        self.cost_tracking_stats["total_cost_usd"] += cost_usd
        self.cost_tracking_stats["total_events_tracked"] += 1

        logger.debug(f"Recorded cost event: {operation} = ${cost_usd:.6f}")

    def get_claim_cost(self, claim_id: str) -> Optional[ClaimCost]:
        """Get cost breakdown for a specific claim"""
        return self.claim_costs.get(claim_id)

    def get_session_cost(self, session_id: str) -> float:
        """Get total cost for a session"""
        events = self.session_costs.get(session_id, [])
        return sum(event.cost_usd for event in events)

    def get_cost_analysis(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get comprehensive cost analysis"""

        if since is None:
            since = datetime.now(timezone.utc) - timedelta(days=7)

        # Filter events by time
        recent_events = []
        for session_events in self.session_costs.values():
            for event in session_events:
                if event.timestamp >= since:
                    recent_events.append(event)

        # Analyze by category
        costs_by_category = defaultdict(float)
        events_by_category = defaultdict(int)

        for event in recent_events:
            costs_by_category[event.category] += event.cost_usd
            events_by_category[event.category] += 1

        # Analyze by claim
        claim_cost_analysis = {}
        for claim_id, claim_cost in self.claim_costs.items():
            recent_claim_events = [e for e in claim_cost.events if e.timestamp >= since]
            if recent_claim_events:
                recent_cost = sum(e.cost_usd for e in recent_claim_events)
                claim_cost_analysis[claim_id] = {
                    "total_cost": recent_cost,
                    "events": len(recent_claim_events),
                    "cost_per_event": recent_cost / len(recent_claim_events),
                    "primary_category": max(claim_cost.cost_breakdown.items(), key=lambda x: x[1])[0].value
                }

        # Calculate efficiency metrics
        total_recent_cost = sum(costs_by_category.values())
        total_claims = len([c for c in claim_cost_analysis.values() if c["total_cost"] > 0])

        analysis = {
            "analysis_period": {
                "start": since.isoformat(),
                "end": datetime.now(timezone.utc).isoformat()
            },
            "summary": {
                "total_cost_usd": total_recent_cost,
                "total_events": len(recent_events),
                "claims_with_costs": total_claims,
                "avg_cost_per_claim": total_recent_cost / max(total_claims, 1),
                "avg_cost_per_event": total_recent_cost / max(len(recent_events), 1)
            },
            "costs_by_category": {cat.value: cost for cat, cost in costs_by_category.items()},
            "events_by_category": {cat.value: count for cat, count in events_by_category.items()},
            "top_claims_by_cost": sorted(
                claim_cost_analysis.items(),
                key=lambda x: x[1]["total_cost"],
                reverse=True
            )[:10],
            "efficiency_metrics": {
                "cost_per_claim": total_recent_cost / max(total_claims, 1),
                "llm_cost_ratio": costs_by_category[CostCategory.LLM_INFERENCE] / max(total_recent_cost, 0.001),
                "compute_cost_ratio": costs_by_category[CostCategory.COMPUTE] / max(total_recent_cost, 0.001)
            }
        }

        return analysis

    def export_cost_data(self, format: str = "json") -> Dict[str, Any]:
        """Export cost data for external analysis"""

        export_data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "format": format,
            "cost_model_pricing": self.cost_model.pricing,
            "tracking_stats": self.cost_tracking_stats,
            "claims": {claim_id: claim.to_dict() for claim_id, claim in self.claim_costs.items()},
            "sessions": {
                session_id: [event.to_dict() for event in events]
                for session_id, events in self.session_costs.items()
            },
            "category_totals": {cat.value: cost for cat, cost in self.total_costs_by_category.items()}
        }

        return export_data

    def save_cost_data(self, filepath: Optional[Path] = None):
        """Save cost data to file"""

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.storage_dir / f"cost_export_{timestamp}.json"

        export_data = self.export_cost_data()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Cost data saved to: {filepath}")
        return filepath

    def get_cost_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions for cost optimization"""

        suggestions = []
        analysis = self.get_cost_analysis()

        # High LLM costs
        if analysis["efficiency_metrics"]["llm_cost_ratio"] > 0.8:
            suggestions.append({
                "type": "llm_optimization",
                "priority": "high",
                "suggestion": "LLM costs are >80% of total. Consider using smaller models or caching more aggressively.",
                "potential_savings": "20-50%"
            })

        # High per-claim costs
        avg_cost_per_claim = analysis["summary"]["avg_cost_per_claim"]
        if avg_cost_per_claim > 0.10:  # $0.10 per claim threshold
            suggestions.append({
                "type": "claim_efficiency",
                "priority": "medium",
                "suggestion": f"Average cost per claim (${avg_cost_per_claim:.4f}) is high. Review claim generation efficiency.",
                "potential_savings": "10-30%"
            })

        # Expensive claims
        expensive_claims = [c for c in analysis["top_claims_by_cost"] if c[1]["total_cost"] > 0.05]
        if expensive_claims:
            suggestions.append({
                "type": "expensive_claims",
                "priority": "medium",
                "suggestion": f"Found {len(expensive_claims)} claims with cost >$0.05. Review their processing pipeline.",
                "potential_savings": "15-25%"
            })

        # Compute costs
        if analysis["efficiency_metrics"]["compute_cost_ratio"] > 0.3:
            suggestions.append({
                "type": "compute_optimization",
                "priority": "low",
                "suggestion": "Compute costs are >30% of total. Consider optimizing algorithms or using faster hardware.",
                "potential_savings": "5-15%"
            })

        return suggestions


def create_cost_tracker(config: Dict[str, Any]) -> CostTracker:
    """Factory function for cost tracker"""
    return CostTracker(config)


# Usage example
if __name__ == "__main__":
    config = {
        "cost_tracking": {
            "enabled": True,
            "pricing": {
                "llm.ollama.input_token": 0.0,
                "llm.ollama.output_token": 0.0
            }
        },
        "cost_storage": "test_costs"
    }

    tracker = CostTracker(config)

    # Start session
    tracker.start_session("test_session_001", {"type": "research", "profile": "thorough"})

    # Track various costs
    tracker.track_llm_cost("claim_001", "ollama/llama2", 500, 200, "synthesis")
    tracker.track_embedding_cost("claim_001", 1000, "local", "document_embedding")
    tracker.track_compute_cost("claim_001", 2.5, cpu_cores=2, memory_gb=4.0, "retrieval")
    tracker.track_reranking_cost("claim_001", 5, "cross_encoder")
    tracker.track_scraping_cost("claim_001", 10, transfer_gb=0.1)

    # Get claim cost
    claim_cost = tracker.get_claim_cost("claim_001")
    if claim_cost:
        print(f"Claim cost: ${claim_cost.total_cost_usd:.6f}")
        print(f"Cost breakdown: {claim_cost.cost_breakdown}")

    # Get analysis
    analysis = tracker.get_cost_analysis()
    print(f"Total cost: ${analysis['summary']['total_cost_usd']:.6f}")
    print(f"Cost per claim: ${analysis['summary']['avg_cost_per_claim']:.6f}")

    # Get optimization suggestions
    suggestions = tracker.get_cost_optimization_suggestions()
    for suggestion in suggestions:
        print(f"Suggestion ({suggestion['priority']}): {suggestion['suggestion']}")

    # Save data
    saved_file = tracker.save_cost_data()
    print(f"Cost data saved to: {saved_file}")
