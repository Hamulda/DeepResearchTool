#!/usr/bin/env python3
"""Autonomous Research Agent with Adaptive Workflow
Implements self-directed research based on interim findings

Author: Advanced IT Specialist
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ResearchPhase(Enum):
    """Research workflow phases"""

    INITIAL_DISCOVERY = "initial_discovery"
    DEEP_DIVE = "deep_dive"
    CROSS_VALIDATION = "cross_validation"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"
    COMPLETION = "completion"


@dataclass
class ResearchLead:
    """Represents a research lead discovered during investigation"""

    entity: str
    context: str
    confidence: float
    source: str
    follow_up_keywords: list[str] = field(default_factory=list)
    priority: int = 1  # 1=highest, 5=lowest
    phase_discovered: ResearchPhase = ResearchPhase.INITIAL_DISCOVERY
    verification_status: str = "unverified"  # unverified, partial, verified, disputed


@dataclass
class ResearchDecision:
    """Represents an autonomous research decision"""

    action: str  # "search_archive", "cross_reference", "verify_claim", "expand_timeline"
    target: str  # What to search for
    source_type: str  # Which scraper to use
    reasoning: str  # Why this decision was made
    expected_value: float  # Expected information value (0.0-1.0)
    urgency: int = 1  # 1=urgent, 5=low priority


class AutonomousResearchAgent:
    """AI agent that makes autonomous research decisions based on findings"""

    def __init__(self, orchestrator, max_iterations: int = 10):
        self.orchestrator = orchestrator
        self.max_iterations = max_iterations
        self.research_leads: list[ResearchLead] = []
        self.processed_keywords: set[str] = set()
        self.research_decisions: list[ResearchDecision] = []
        self.current_phase = ResearchPhase.INITIAL_DISCOVERY
        self.knowledge_graph = {}  # Entity relationships

    async def autonomous_research(self, initial_topic: str) -> dict[str, Any]:
        """Conduct autonomous research with adaptive workflow"""
        logger.info(f"Starting autonomous research on: {initial_topic}")

        # Initialize research state
        self.processed_keywords.add(initial_topic.lower())
        iteration = 0
        accumulated_findings = []

        # Initial research
        current_results = await self.orchestrator.deep_research(initial_topic)
        accumulated_findings.extend(current_results.unique_sources)

        # Extract initial leads
        await self._extract_research_leads(current_results.unique_sources, initial_topic)

        # Adaptive research loop
        while iteration < self.max_iterations and self._has_promising_leads():
            iteration += 1
            logger.info(f"Autonomous research iteration {iteration}")

            # Decide next research action
            decision = await self._make_research_decision()
            if not decision:
                logger.info("No promising research decisions found, ending research")
                break

            self.research_decisions.append(decision)
            logger.info(
                f"Decision: {decision.action} targeting '{decision.target}' - {decision.reasoning}"
            )

            # Execute research decision
            new_results = await self._execute_research_decision(decision)
            if new_results:
                accumulated_findings.extend(new_results)

                # Extract new leads from results
                await self._extract_research_leads(new_results, decision.target)

                # Update knowledge graph
                await self._update_knowledge_graph(new_results, decision.target)

                # Advance research phase if appropriate
                self._advance_research_phase()

        # Synthesize final results
        final_analysis = await self._synthesize_autonomous_findings(
            initial_topic, accumulated_findings
        )

        return {
            "initial_topic": initial_topic,
            "iterations_completed": iteration,
            "total_sources_found": len(accumulated_findings),
            "research_leads_discovered": len(self.research_leads),
            "autonomous_decisions": self.research_decisions,
            "final_analysis": final_analysis,
            "knowledge_graph": self.knowledge_graph,
            "research_phases_completed": self.current_phase.value,
        }

    async def _extract_research_leads(self, sources: list[dict[str, Any]], context: str):
        """Extract potential research leads from sources using AI analysis"""
        try:
            # Combine source content for analysis
            combined_content = " ".join(
                [source.get("content", "")[:500] for source in sources[-5:]]  # Last 5 sources
            )

            # Use AI to identify research leads
            lead_extraction_prompt = f"""
Analyze this research content and identify specific entities, names, organizations, or events that warrant further investigation.

CONTEXT: Research on "{context}"

CONTENT:
{combined_content[:2000]}

Extract potential research leads in JSON format:
{{
  "leads": [
    {{
      "entity": "specific name or term",
      "context": "why it's relevant",
      "keywords": ["related", "search", "terms"],
      "priority": 1-5,
      "lead_type": "person|organization|event|document|operation"
    }}
  ]
}}

Focus on:
- Names of people, organizations, operations
- Specific dates, locations, or events mentioned
- References to documents or code names
- Connections to other known entities
- Anomalies or inconsistencies

JSON:
"""

            response = await self.orchestrator.ollama_agent._query_ollama_async(
                lead_extraction_prompt
            )

            # Parse leads
            leads_data = self._parse_leads_response(response)

            for lead_data in leads_data.get("leads", []):
                # Check if we haven't already processed this
                entity = lead_data.get("entity", "").lower()
                if entity and entity not in self.processed_keywords:
                    research_lead = ResearchLead(
                        entity=lead_data.get("entity"),
                        context=lead_data.get("context", ""),
                        confidence=0.7,  # Default confidence
                        source=context,
                        follow_up_keywords=lead_data.get("keywords", []),
                        priority=lead_data.get("priority", 3),
                        phase_discovered=self.current_phase,
                    )
                    self.research_leads.append(research_lead)
                    logger.info(f"New research lead discovered: {research_lead.entity}")

        except Exception as e:
            logger.error(f"Error extracting research leads: {e}")

    def _parse_leads_response(self, response: str) -> dict[str, Any]:
        """Parse JSON response for research leads"""
        try:
            # Find JSON in response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)

            return {"leads": []}
        except Exception as e:
            logger.warning(f"Error parsing leads response: {e}")
            return {"leads": []}

    async def _make_research_decision(self) -> ResearchDecision | None:
        """Make autonomous decision about next research step"""
        if not self.research_leads:
            return None

        # Sort leads by priority and confidence
        sorted_leads = sorted(
            [lead for lead in self.research_leads if lead.verification_status == "unverified"],
            key=lambda x: (x.priority, -x.confidence),
        )

        if not sorted_leads:
            return None

        top_lead = sorted_leads[0]

        # Decide on research strategy based on current phase
        if self.current_phase == ResearchPhase.INITIAL_DISCOVERY:
            return await self._decide_initial_discovery(top_lead)
        if self.current_phase == ResearchPhase.DEEP_DIVE:
            return await self._decide_deep_dive(top_lead)
        if self.current_phase == ResearchPhase.CROSS_VALIDATION:
            return await self._decide_cross_validation(top_lead)
        if self.current_phase == ResearchPhase.VERIFICATION:
            return await self._decide_verification(top_lead)

        return None

    async def _decide_initial_discovery(self, lead: ResearchLead) -> ResearchDecision:
        """Make decision for initial discovery phase"""
        # Look for specific entities or operations in archives
        if any(keyword in lead.entity.lower() for keyword in ["operation", "project", "program"]):
            return ResearchDecision(
                action="search_archive",
                target=lead.entity,
                source_type="archive_hunter",
                reasoning=f"'{lead.entity}' appears to be an operation/project name requiring archive search",
                expected_value=0.8,
                urgency=1,
            )

        # Search for person or organization
        if any(
            keyword in lead.context.lower() for keyword in ["cia", "fbi", "government", "agency"]
        ):
            return ResearchDecision(
                action="search_government_docs",
                target=lead.entity,
                source_type="wayback",
                reasoning=f"Government connection detected for '{lead.entity}', searching official sources",
                expected_value=0.7,
                urgency=2,
            )

        # Default to comprehensive search
        return ResearchDecision(
            action="comprehensive_search",
            target=lead.entity,
            source_type="multi_source",
            reasoning=f"Comprehensive search for '{lead.entity}' to establish baseline information",
            expected_value=0.6,
            urgency=3,
        )

    async def _decide_deep_dive(self, lead: ResearchLead) -> ResearchDecision:
        """Make decision for deep dive phase"""
        # Focus on document verification and timeline establishment
        return ResearchDecision(
            action="verify_timeline",
            target=lead.entity,
            source_type="arxiv",
            reasoning=f"Deep dive into '{lead.entity}' to establish verified timeline and cross-references",
            expected_value=0.7,
            urgency=1,
        )

    async def _decide_cross_validation(self, lead: ResearchLead) -> ResearchDecision:
        """Make decision for cross-validation phase"""
        return ResearchDecision(
            action="cross_validate",
            target=lead.entity,
            source_type="rss_feeds",
            reasoning=f"Cross-validating information about '{lead.entity}' across multiple source types",
            expected_value=0.8,
            urgency=1,
        )

    async def _decide_verification(self, lead: ResearchLead) -> ResearchDecision:
        """Make decision for verification phase"""
        return ResearchDecision(
            action="verify_claims",
            target=lead.entity,
            source_type="medical_research_scraper",
            reasoning=f"Final verification of claims about '{lead.entity}' using authoritative sources",
            expected_value=0.9,
            urgency=1,
        )

    async def _execute_research_decision(self, decision: ResearchDecision) -> list[dict[str, Any]]:
        """Execute the autonomous research decision"""
        try:
            # Mark as processed
            self.processed_keywords.add(decision.target.lower())

            # Select appropriate scraper based on decision
            scraper = self._select_scraper(decision.source_type)
            if not scraper:
                logger.warning(f"No scraper available for source type: {decision.source_type}")
                return []

            # Execute search
            if hasattr(scraper, "search_async"):
                results = await scraper.search_async(decision.target)
            else:
                # Run in executor for sync scrapers
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(None, scraper.search, decision.target)

            logger.info(f"Autonomous search for '{decision.target}' found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error executing research decision: {e}")
            return []

    def _select_scraper(self, source_type: str):
        """Select appropriate scraper based on source type"""
        scraper_mapping = {
            "archive_hunter": "archive_hunter",
            "wayback": "wayback",
            "arxiv": "arxiv",
            "rss_feeds": "rss_feeds",
            "medical_research_scraper": "medical_research_scraper",
            "multi_source": "wayback",  # Default for multi-source
        }

        scraper_name = scraper_mapping.get(source_type)
        return self.orchestrator.scrapers.get(scraper_name) if scraper_name else None

    async def _update_knowledge_graph(self, sources: list[dict[str, Any]], query: str):
        """Update knowledge graph with new relationships"""
        try:
            # Extract entities and relationships using AI
            combined_content = " ".join(
                [source.get("content", "")[:300] for source in sources[-3:]]
            )

            entities_data = self.orchestrator.ollama_agent.extract_entities_and_relationships(
                combined_content
            )

            # Update knowledge graph
            if query not in self.knowledge_graph:
                self.knowledge_graph[query] = {"entities": [], "relationships": [], "sources": []}

            # Add entities
            entities = entities_data.get("entities", {})
            for entity_type, entity_list in entities.items():
                self.knowledge_graph[query]["entities"].extend(entity_list)

            # Add relationships
            relationships = entities_data.get("relationships", [])
            self.knowledge_graph[query]["relationships"].extend(relationships)

            # Add source references
            for source in sources:
                self.knowledge_graph[query]["sources"].append(
                    {
                        "title": source.get("title", ""),
                        "url": source.get("url", ""),
                        "relevance": source.get("priority_score", 0.5),
                    }
                )

        except Exception as e:
            logger.error(f"Error updating knowledge graph: {e}")

    def _advance_research_phase(self):
        """Advance to next research phase based on current state"""
        verified_leads = len(
            [lead for lead in self.research_leads if lead.verification_status == "verified"]
        )
        total_leads = len(self.research_leads)

        if self.current_phase == ResearchPhase.INITIAL_DISCOVERY and total_leads >= 5:
            self.current_phase = ResearchPhase.DEEP_DIVE
            logger.info("Advanced to DEEP_DIVE phase")
        elif self.current_phase == ResearchPhase.DEEP_DIVE and total_leads >= 10:
            self.current_phase = ResearchPhase.CROSS_VALIDATION
            logger.info("Advanced to CROSS_VALIDATION phase")
        elif self.current_phase == ResearchPhase.CROSS_VALIDATION and verified_leads >= 3:
            self.current_phase = ResearchPhase.VERIFICATION
            logger.info("Advanced to VERIFICATION phase")
        elif self.current_phase == ResearchPhase.VERIFICATION:
            self.current_phase = ResearchPhase.SYNTHESIS
            logger.info("Advanced to SYNTHESIS phase")

    def _has_promising_leads(self) -> bool:
        """Check if there are still promising research leads to pursue"""
        unverified_leads = [
            lead for lead in self.research_leads if lead.verification_status == "unverified"
        ]
        high_priority_leads = [lead for lead in unverified_leads if lead.priority <= 2]

        return len(high_priority_leads) > 0 and self.current_phase != ResearchPhase.SYNTHESIS

    async def _synthesize_autonomous_findings(
        self, topic: str, sources: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Synthesize findings from autonomous research"""
        try:
            # Create synthesis using enhanced analyzer
            synthesis = self.orchestrator.enhanced_analyzer.analyze_research_data(
                sources, topic, self.orchestrator.priority_scorer
            )

            # Add autonomous research metadata
            autonomous_metadata = {
                "research_methodology": "autonomous_agent",
                "iterations_completed": len(self.research_decisions),
                "leads_discovered": len(self.research_leads),
                "verification_rate": len(
                    [l for l in self.research_leads if l.verification_status == "verified"]
                )
                / max(len(self.research_leads), 1),
                "knowledge_graph_entities": sum(
                    len(kg.get("entities", [])) for kg in self.knowledge_graph.values()
                ),
                "research_phase_reached": self.current_phase.value,
                "decision_chain": [
                    {
                        "action": d.action,
                        "target": d.target,
                        "reasoning": d.reasoning,
                        "expected_value": d.expected_value,
                    }
                    for d in self.research_decisions
                ],
            }

            synthesis.metadata.update(autonomous_metadata)

            return {
                "synthesis": synthesis,
                "knowledge_graph": self.knowledge_graph,
                "research_decisions": self.research_decisions,
                "research_leads": [
                    {
                        "entity": lead.entity,
                        "context": lead.context,
                        "confidence": lead.confidence,
                        "verification_status": lead.verification_status,
                        "phase_discovered": lead.phase_discovered.value,
                    }
                    for lead in self.research_leads
                ],
            }

        except Exception as e:
            logger.error(f"Error synthesizing autonomous findings: {e}")
            return {
                "error": str(e),
                "partial_results": {
                    "leads_count": len(self.research_leads),
                    "decisions_count": len(self.research_decisions),
                },
            }
