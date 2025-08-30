"""
Testy pro LangGraph Research Agent architekturu
Validace všech hlavních komponent nové stavové architektury včetně rozšířených funkcí

Author: Senior Python/MLOps Agent
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from src.core.langgraph_agent import ResearchAgentGraph, ResearchAgentState
from src.core.config_langgraph import load_config, validate_config
from src.core.memory import ChromaMemoryStore, Document
from src.core.rag_pipeline import RAGPipeline, DocumentProcessor
from src.core.tools import web_scraping_tool, knowledge_search_tool
from src.core.enhanced_tools import (
    semantic_scholar_search,
    data_gov_search,
    wayback_machine_search,
    cross_reference_sources,
)


class TestLangGraphAgent:
    """Testy pro hlavní LangGraph agent"""

    @pytest.fixture
    def config(self):
        """Test konfigurace"""
        return {
            "llm": {"model": "gpt-4o-mini", "temperature": 0.1},
            "memory_store": {
                "type": "chroma",
                "collection_name": "test_collection",
                "persist_directory": "./test_chroma_db",
            },
            "rag": {"chunking": {"chunk_size": 500, "chunk_overlap": 50}},
            "validation": {"threshold": 0.7, "max_retries": 2},
        }

    @pytest.fixture
    def agent(self, config):
        """Fixture pro research agenta"""
        return ResearchAgentGraph(config)

    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test inicializace agenta"""
        assert agent is not None
        assert agent.config is not None
        assert agent.graph is not None
        assert len(agent.tools) > 0

    @pytest.mark.asyncio
    async def test_plan_step(self, agent):
        """Test plánovacího kroku"""
        initial_state = ResearchAgentState(
            initial_query="Co je umělá inteligence?",
            plan=[],
            retrieved_docs=[],
            validation_scores={},
            synthesis="",
            messages=[],
            current_step="initialized",
            processing_time=0.0,
            errors=[],
            validation_threshold=0.7,
            retry_count=0,
            human_approval_required=False,
            human_decision=None,
            pending_action=None,
            sources_used=[],
        )

        with patch("langchain_openai.ChatOpenAI") as mock_llm:
            mock_response = Mock()
            mock_response.content = "1. Definice AI\n2. Historie AI\n3. Aplikace AI"
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)

            result = await agent.plan_step(initial_state)

            assert len(result["plan"]) == 3
            assert "Definice AI" in result["plan"][0]
            assert result["current_step"] == "plan_completed"

    @pytest.mark.asyncio
    async def test_validate_sources_step(self, agent):
        """Test validace zdrojů"""
        test_state = ResearchAgentState(
            initial_query="Test query",
            plan=["step1"],
            retrieved_docs=[
                {"content": "Test content 1", "source": "test1.com", "metadata": {}},
                {"content": "Test content 2", "source": "test2.com", "metadata": {}},
            ],
            validation_scores={},
            synthesis="",
            messages=[],
            current_step="retrieve_completed",
            processing_time=0.0,
            errors=[],
            validation_threshold=0.7,
            retry_count=0,
            human_approval_required=False,
            human_decision=None,
            pending_action=None,
            sources_used=[],
        )

        with patch("langchain_openai.ChatOpenAI") as mock_llm:
            mock_response = Mock()
            mock_response.content = "0.8"
            mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)

            result = await agent.validate_sources_step(test_state)

            assert "source_validation_results" in result
            assert len(result["source_validation_results"]) == 2
            assert result["current_step"] == "validate_sources_completed"

    @pytest.mark.asyncio
    async def test_routing_after_validation_high_score(self, agent):
        """Test routingu po validaci s vysokým skóre"""
        test_state = {
            "source_validation_results": [{"score": 0.8}, {"score": 0.9}],
            "validation_threshold": 0.7,
        }

        route = agent._route_after_validation(test_state)
        assert route == "continue_to_synthesis"

    @pytest.mark.asyncio
    async def test_routing_after_validation_low_score_retry(self, agent):
        """Test routingu po validaci s nízkým skóre - retry"""
        test_state = {
            "source_validation_results": [{"score": 0.3}, {"score": 0.4}],
            "validation_threshold": 0.7,
            "retry_count": 0,
        }

        route = agent._route_after_validation(test_state)
        assert route == "retry_planning"
        assert test_state["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_routing_after_validation_need_approval(self, agent):
        """Test routingu po validaci - potřeba schválení"""
        test_state = {
            "source_validation_results": [{"score": 0.3}, {"score": 0.4}],
            "validation_threshold": 0.7,
            "retry_count": 2,
        }

        route = agent._route_after_validation(test_state)
        assert route == "need_approval"
        assert test_state["human_approval_required"] == True
        assert test_state["pending_action"]["type"] == "low_quality_sources"

    @pytest.mark.asyncio
    async def test_human_approval_step(self, agent):
        """Test human approval kroku"""
        test_state = ResearchAgentState(
            initial_query="test",
            plan=[],
            retrieved_docs=[],
            validation_scores={},
            synthesis="",
            messages=[],
            current_step="validate_sources_completed",
            processing_time=0.0,
            errors=[],
            validation_threshold=0.7,
            retry_count=0,
            human_approval_required=True,
            human_decision=None,
            pending_action=None,
            sources_used=[],
        )

        result = await agent.human_approval_step(test_state)

        assert result["human_decision"] in ["approved", "rejected"]
        assert result["current_step"] in ["approved", "rejected"]


class TestEnhancedTools:
    """Testy pro rozšířené nástroje"""

    @pytest.mark.asyncio
    async def test_semantic_scholar_search(self):
        """Test Semantic Scholar vyhledávání"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "data": [
                        {
                            "title": "Test Paper",
                            "abstract": "Test abstract",
                            "authors": [{"name": "Test Author"}],
                            "year": 2023,
                            "citationCount": 10,
                            "url": "https://test.com",
                            "venue": "Test Venue",
                            "publicationTypes": ["JournalArticle"],
                            "paperId": "123",
                        }
                    ],
                    "total": 1,
                }
            )
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await semantic_scholar_search("AI in healthcare", limit=5)

            assert result["success"] == True
            assert len(result["papers"]) == 1
            assert result["papers"][0]["title"] == "Test Paper"
            assert result["query"] == "AI in healthcare"

    @pytest.mark.asyncio
    async def test_data_gov_search(self):
        """Test Data.gov vyhledávání"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "success": True,
                    "result": {
                        "results": [
                            {
                                "title": "Test Dataset",
                                "notes": "Test description",
                                "organization": {"title": "Test Org"},
                                "tags": [{"name": "health"}],
                                "name": "test-dataset",
                                "metadata_modified": "2023-01-01",
                                "resources": [{}],
                                "id": "test-id",
                            }
                        ],
                        "count": 1,
                    },
                }
            )
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await data_gov_search("healthcare data", limit=5)

            assert result["success"] == True
            assert len(result["datasets"]) == 1
            assert result["datasets"][0]["title"] == "Test Dataset"

    @pytest.mark.asyncio
    async def test_wayback_machine_search(self):
        """Test Wayback Machine vyhledávání"""
        with patch("aiohttp.ClientSession.get") as mock_get:
            # Mock pro availability API
            mock_availability_response = Mock()
            mock_availability_response.status = 200
            mock_availability_response.json = AsyncMock(
                return_value={
                    "archived_snapshots": {
                        "closest": {
                            "available": True,
                            "url": "https://web.archive.org/web/20230101/https://example.com",
                            "timestamp": "20230101120000",
                        }
                    }
                }
            )

            # Mock pro content response
            mock_content_response = Mock()
            mock_content_response.status = 200
            mock_content_response.text = AsyncMock(return_value="<html>Test content</html>")

            mock_get.return_value.__aenter__.side_effect = [
                mock_availability_response,
                mock_content_response,
            ]

            result = await wayback_machine_search("https://example.com")

            assert result["success"] == True
            assert "archived_url" in result
            assert result["timestamp"] == "20230101120000"

    @pytest.mark.asyncio
    async def test_cross_reference_sources(self):
        """Test křížové reference zdrojů"""
        # Mock všechny jednotlivé funkce
        with patch("src.core.enhanced_tools.semantic_scholar_search") as mock_scholar:
            with patch("src.core.enhanced_tools.data_gov_search") as mock_datagov:
                mock_scholar.return_value = {"success": True, "papers": [{"title": "Test paper"}]}
                mock_datagov.return_value = {
                    "success": True,
                    "datasets": [{"title": "Test dataset"}],
                }

                result = await cross_reference_sources("test query")

                assert "sources" in result
                assert "summary" in result
                assert result["query"] == "test query"


class TestStreamlitIntegration:
    """Testy pro Streamlit integraci"""

    def test_streamlit_app_exists(self):
        """Test existence Streamlit aplikace"""
        import os

        assert os.path.exists("streamlit_app.py")

    @pytest.mark.asyncio
    async def test_agent_integration_with_streamlit_config(self):
        """Test integrace agenta s Streamlit konfigurací"""
        streamlit_config = {
            "llm": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "synthesis_model": "gpt-4o",
                "synthesis_temperature": 0.2,
            },
            "validation": {"threshold": 0.8, "max_retries": 3},
        }

        agent = ResearchAgentGraph(streamlit_config)
        assert agent.config["validation"]["threshold"] == 0.8
        assert agent.config["validation"]["max_retries"] == 3


class TestCompleteWorkflow:
    """Testy kompletního workflow"""

    @pytest.fixture
    def config(self):
        return {
            "llm": {"model": "gpt-4o-mini", "temperature": 0.1},
            "memory_store": {
                "type": "chroma",
                "collection_name": "test_workflow",
                "persist_directory": "./test_workflow_db",
            },
            "validation": {"threshold": 0.7, "max_retries": 1},
        }

    @pytest.mark.asyncio
    async def test_complete_research_workflow(self, config):
        """Test kompletního research workflow"""
        agent = ResearchAgentGraph(config)

        # Mock všechny LLM volání
        with patch("langchain_openai.ChatOpenAI") as mock_llm:
            mock_responses = [
                Mock(content="1. Definice AI\n2. Historie AI"),  # planning
                Mock(content="0.8"),  # source validation
                Mock(content="0.8"),  # relevance validation
                Mock(content="## Souhrn\nAI je...\n## Závěr\nAI má potenciál..."),  # synthesis
            ]
            mock_llm.return_value.ainvoke = AsyncMock(side_effect=mock_responses)

            # Mock RAG pipeline
            with patch("src.core.rag_pipeline.RAGPipeline") as mock_rag:
                mock_rag.return_value.initialize = AsyncMock()
                mock_rag.return_value.search = AsyncMock(
                    return_value=[Mock(content="Test AI content", metadata={"source": "test"})]
                )

                result = await agent.research("Co je umělá inteligence?")

                assert result["query"] == "Co je umělá inteligence?"
                assert len(result["plan"]) > 0
                assert "synthesis" in result
                assert result["synthesis"].startswith("## Souhrn")
                assert "processing_time" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
