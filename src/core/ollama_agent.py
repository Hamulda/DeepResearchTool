#!/usr/bin/env python3
"""
Ollama Research Agent for Deep Research Tool
Advanced AI integration with context-aware analysis

Author: Advanced IT Specialist
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import yaml
import aiohttp
from dataclasses import dataclass

from .context_manager import ContextManager

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Result of AI analysis"""
    response: str
    confidence: float
    sources_analyzed: int
    key_findings: List[str]
    potential_issues: List[str]
    recommendations: List[str]
    timestamp: datetime

class OllamaResearchAgent:
    """Advanced research agent with Ollama integration"""

    def __init__(self, model_name: str = "llama3.2:8b", host: str = "http://localhost:11434"):
        """
        Initialize the Ollama research agent

        Args:
            model_name: Name of the Ollama model to use
            host: Ollama server host URL
        """
        self.model_name = model_name
        self.host = host.rstrip('/')
        self.context_manager = ContextManager()
        self.conversation_memory = []
        self.system_prompts = self._load_system_prompts()

    def _load_system_prompts(self) -> Dict[str, str]:
        """Load system prompts for different analysis types"""
        return {
            'general_research': """
            Jsi pokročilý expert na analýzu konspirativních teorií, historických výzkumů a archivních materiálů.
            Tvým úkolem je:
            1. Identifikovat klíčové informace a souvislosti v poskytnutých dokumentech
            2. Rozpoznat potenciální dezinformace vs. legitimní výzkum
            3. Poskytovat kritickou analýzu zdrojů a jejich kredibility
            4. Hledat skryté souvislosti mezi dokumenty a událostmi
            5. Upozornit na časové nesrovnalosti nebo faktické chyby
            6. Identifikovat vzorce a trendy napříč materiály
            
            Buď skeptický, ale objektivní. Preferuj ověřitelné zdroje a jasně rozliš mezi fakty a spekulacemi.
            Vždy uveď stupeň jistoty svých závěrů a navrhni další kroky pro ověření.
            """,

            'conspiracy_analysis': """
            Specializuješ se na analýzu konspirativních teorií s vědeckým přístupem.
            Hodnotíš:
            1. Faktickou základnu tvrzení
            2. Kvalitu a původ důkazů
            3. Logické chyby v argumentaci
            4. Motivace a pozadí autorů
            5. Časový kontext a historické souvislosti
            
            Rozliš mezi:
            - Oprávněnými otázkami a pochybnostmi
            - Neopodstatněnými spekulacemi
            - Možnými skutečnými konspiracemi
            - Čistými fantaziemi
            """,

            'document_verification': """
            Specializuješ se na ověřování pravosti a integrity dokumentů.
            Kontroluješ:
            1. Konzistenci stylu a jazyka
            2. Chronologické nesrovnalosti
            3. Anachronismy v obsahu
            4. Formátování a technické aspekty
            5. Cross-reference s dalšími zdroji
            
            Poskytni hodnocení pravděpodobnosti pravosti dokumentu a identifikuj potenciální červené vlajky.
            """,

            'timeline_analysis': """
            Specializuješ se na konstrukci a analýzu časových linií událostí.
            Vytváříš:
            1. Chronologické seřazení událostí
            2. Identifikaci klíčových momentů
            3. Analýzu příčin a následků
            4. Detekci vzorců a cyklů
            5. Srovnání s oficiálními verzemi
            
            Zvýrazni nesrovnalosti a navrhni alternativní interpretace tam, kde je to opodstatněné.
            """
        }

    async def analyze_with_context(self, query: str, documents: List[Dict], 
                                 analysis_type: str = 'general_research') -> AnalysisResult:
        """
        Perform context-aware analysis using Ollama

        Args:
            query: Research query
            documents: List of documents to analyze
            analysis_type: Type of analysis to perform

        Returns:
            AnalysisResult object
        """
        # Prepare context
        prioritized_context = self.context_manager.prepare_context(
            documents, query, max_tokens=3500
        )

        # Get appropriate system prompt
        system_prompt = self.system_prompts.get(analysis_type, self.system_prompts['general_research'])

        # Add memory context if available
        memory_context = self.context_manager.get_memory_context(max_items=3)
        if memory_context:
            system_prompt += f"\n\nKontext předchozí konverzace:\n{memory_context}"

        # Construct the analysis prompt
        analysis_prompt = f"""
        Systémové instrukce: {system_prompt}
        
        Dokumenty k analýze:
        {prioritized_context}
        
        Dotaz: {query}
        
        Proveď důkladnou analýzu a odpověz strukturovaně s následujícími sekcemi:
        1. SHRNUTÍ KLÍČOVÝCH ZJIŠTĚNÍ
        2. ANALÝZA ZDROJŮ A KREDIBILITY  
        3. IDENTIFIKOVANÉ SOUVISLOSTI
        4. POTENCIÁLNÍ PROBLÉMY A ČERVENÉ VLAJKY
        5. DOPORUČENÍ PRO DALŠÍ VÝZKUM
        6. STUPEŇ JISTOTY (1-10) A ZDŮVODNĚNÍ
        """

        try:
            # Make request to Ollama
            response = await self._make_ollama_request(analysis_prompt)

            # Parse response and extract structured information
            parsed_result = self._parse_analysis_response(response, len(documents))

            # Add to conversation memory
            self.context_manager.add_to_memory(
                query, parsed_result.response,
                {'analysis_type': analysis_type, 'timestamp': datetime.now()}
            )

            return parsed_result

        except Exception as e:
            logger.error(f"Error in Ollama analysis: {e}")
            return AnalysisResult(
                response=f"Chyba při analýze: {str(e)}",
                confidence=0.0,
                sources_analyzed=len(documents),
                key_findings=[],
                potential_issues=[f"Technická chyba: {str(e)}"],
                recommendations=["Zkuste analýzu opakovat"],
                timestamp=datetime.now()
            )

    async def chat_with_context(self, message: str, context: Dict[str, Any] = None) -> str:
        """
        Interactive chat with context awareness

        Args:
            message: User message
            context: Optional context from previous research

        Returns:
            AI response string
        """
        # Build chat prompt with context
        chat_prompt = f"""
        Jsi pokročilý research assistant specializující se na konspirativní teorie a historický výzkum.
        
        {'Kontext z předchozího výzkumu:' + str(context) if context else ''}
        
        Uživatel: {message}
        
        Odpověz informativně a buď připraven na následné dotazy. Pokud potřebuješ více informací pro přesnou odpověď, navrhni konkrétní způsoby, jak je získat.
        """

        try:
            response = await self._make_ollama_request(chat_prompt, stream=False)
            return response
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Omlouvám se, došlo k chybě: {str(e)}"

    async def stream_analysis(self, query: str, documents: List[Dict]) -> AsyncGenerator[str, None]:
        """
        Stream analysis results in real-time

        Args:
            query: Research query
            documents: Documents to analyze

        Yields:
            Partial analysis results
        """
        prioritized_context = self.context_manager.prepare_context(
            documents, query, max_tokens=3500
        )

        system_prompt = self.system_prompts['general_research']
        analysis_prompt = f"""
        {system_prompt}
        
        Dokumenty: {prioritized_context}
        Dotaz: {query}
        
        Proveď postupnou analýzu a komentuj své zjištění:
        """

        try:
            async for chunk in self._stream_ollama_request(analysis_prompt):
                yield chunk
        except Exception as e:
            yield f"Chyba při streamování: {str(e)}"

    async def _make_ollama_request(self, prompt: str, stream: bool = False) -> str:
        """Make request to Ollama API"""
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Ollama API error: {response.status}")

                if stream:
                    full_response = ""
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode())
                                if 'response' in data:
                                    full_response += data['response']
                                if data.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                    return full_response
                else:
                    data = await response.json()
                    return data.get('response', '')

    async def _stream_ollama_request(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream request to Ollama API"""
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Ollama API error: {response.status}")

                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode())
                            if 'response' in data:
                                yield data['response']
                            if data.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue

    def _parse_analysis_response(self, response: str, sources_count: int) -> AnalysisResult:
        """Parse structured analysis response from AI"""
        # Extract key findings
        key_findings = self._extract_section(response, "KLÍČOVÝCH ZJIŠTĚNÍ", "ANALÝZA ZDROJŮ")

        # Extract potential issues
        potential_issues = self._extract_section(response, "POTENCIÁLNÍ PROBLÉMY", "DOPORUČENÍ")

        # Extract recommendations
        recommendations = self._extract_section(response, "DOPORUČENÍ", "STUPEŇ JISTOTY")

        # Extract confidence level
        confidence = self._extract_confidence(response)

        return AnalysisResult(
            response=response,
            confidence=confidence,
            sources_analyzed=sources_count,
            key_findings=key_findings,
            potential_issues=potential_issues,
            recommendations=recommendations,
            timestamp=datetime.now()
        )

    def _extract_section(self, text: str, start_marker: str, end_marker: str) -> List[str]:
        """Extract bulleted items from a section"""
        import re

        # Find section content
        start_pattern = rf"{start_marker}[:\s]*"
        end_pattern = rf"\d+\.\s*{end_marker}" if end_marker else r"$"

        section_match = re.search(f"{start_pattern}(.*?)(?={end_pattern})", text, re.DOTALL | re.IGNORECASE)
        if not section_match:
            return []

        section_content = section_match.group(1)

        # Extract bullet points or numbered items
        items = re.findall(r'[-•*]\s*(.+?)(?=\n[-•*]|\n\d+\.|\n[A-Z]+[:.]|\Z)',
                          section_content, re.DOTALL)

        return [item.strip().replace('\n', ' ') for item in items if item.strip()]

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence level from response"""
        import re

        # Look for confidence patterns
        patterns = [
            r"STUPEŇ JISTOTY[:\s]*(\d+(?:\.\d+)?)",
            r"jistota[:\s]*(\d+(?:\.\d+)?)",
            r"confidence[:\s]*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)/10"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    confidence = float(match.group(1))
                    return confidence / 10.0 if confidence > 1.0 else confidence
                except ValueError:
                    continue

        return 0.5  # Default neutral confidence

    async def verify_ollama_connection(self) -> bool:
        """Verify connection to Ollama server"""
        try:
            url = f"{self.host}/api/tags"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model['name'] for model in data.get('models', [])]
                        return self.model_name in models
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False

    async def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            url = f"{self.host}/api/tags"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model['name'] for model in data.get('models', [])]
            return []
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []

    def switch_model(self, model_name: str):
        """Switch to a different Ollama model"""
        self.model_name = model_name
        logger.info(f"Switched to model: {model_name}")

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return list(self.context_manager.conversation_memory)

    def clear_conversation_history(self):
        """Clear conversation history"""
        self.context_manager.conversation_memory.clear()
        logger.info("Conversation history cleared")

    async def analyze_research(self, topic: str, context: str, analysis_type: str = 'general_research') -> AnalysisResult:
        """Enhanced research analysis with structured JSON output"""
        try:
            # Create structured prompt for JSON output
            structured_prompt = self._create_structured_prompt(topic, context, analysis_type)

            response = await self._query_ollama_async(structured_prompt)

            # Try to parse structured JSON response
            structured_data = self._parse_structured_response(response)

            if structured_data:
                # Create analysis result from structured data
                return self._create_analysis_from_structured_data(structured_data, topic)
            else:
                # Fallback to regular text analysis
                logger.warning("Failed to parse structured response, using fallback")
                return await self._fallback_analysis(topic, context, analysis_type)

        except Exception as e:
            logger.error(f"Error in enhanced research analysis: {e}")
            return AnalysisResult(
                response=f"Error during analysis: {str(e)}",
                confidence=0.1,
                key_findings=["Analysis failed due to technical error"],
                potential_issues=[str(e)],
                recommendations=["Please try again or contact support"],
                analysis_type=analysis_type,
                metadata={'error': str(e)}
            )

    def _create_structured_prompt(self, topic: str, context: str, analysis_type: str) -> str:
        """Create prompt that encourages structured JSON output"""

        json_schema = {
            "analysis_result": {
                "confidence_score": "float between 0.0 and 1.0",
                "key_findings": ["list of important discoveries"],
                "entities": {
                    "people": [{"name": "string", "role": "string", "relevance": "float"}],
                    "organizations": [{"name": "string", "type": "string", "relevance": "float"}],
                    "locations": [{"name": "string", "type": "string", "relevance": "float"}],
                    "events": [{"name": "string", "date": "YYYY-MM-DD or null", "relevance": "float"}],
                    "documents": [{"title": "string", "type": "string", "relevance": "float"}]
                },
                "relationships": [
                    {
                        "entity1": "string",
                        "entity2": "string",
                        "relationship_type": "string",
                        "strength": "float",
                        "description": "string"
                    }
                ],
                "timeline": [
                    {
                        "date": "YYYY-MM-DD or YYYY or null",
                        "event": "string",
                        "significance": "string",
                        "sources": ["list of source references"]
                    }
                ],
                "credibility_assessment": {
                    "source_reliability": "float",
                    "evidence_strength": "float",
                    "bias_indicators": ["list of potential biases"],
                    "fact_check_status": "verified|partially_verified|unverified|disputed"
                },
                "anomalies_detected": [
                    {
                        "type": "anachronism|inconsistency|bias|propaganda",
                        "description": "string",
                        "severity": "low|medium|high|critical"
                    }
                ],
                "research_gaps": ["list of missing information areas"],
                "recommendations": ["list of next research steps"],
                "summary": "comprehensive text summary"
            }
        }

        prompt = f"""
You are an advanced research analyst. Analyze the following research data about "{topic}" and provide a comprehensive structured analysis.

ANALYSIS TYPE: {analysis_type}

RESEARCH CONTEXT:
{context[:3000]}  # Limit context to prevent token overflow

INSTRUCTIONS:
1. Provide your analysis in valid JSON format following this exact schema:
{json.dumps(json_schema, indent=2)}

2. Extract and categorize all entities (people, organizations, locations, events, documents)
3. Identify relationships between entities with strength scores (0.0-1.0)
4. Create a chronological timeline of relevant events
5. Assess source credibility and evidence strength
6. Detect anomalies like anachronisms, inconsistencies, bias, or propaganda
7. Identify gaps in current research
8. Provide specific recommendations for further investigation

IMPORTANT:
- Respond ONLY with valid JSON
- Use null for unknown dates
- Assign relevance scores (0.0-1.0) to all entities
- Be objective and evidence-based
- Flag any potential misinformation or bias

JSON Response:
"""
        return prompt

    def _parse_structured_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse structured JSON response from Ollama"""
        try:
            # Clean response - remove any text before/after JSON
            response = response.strip()

            # Find JSON boundaries
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                logger.warning("No JSON found in response")
                return None

            json_str = response[start_idx:end_idx]

            # Parse JSON
            structured_data = json.loads(json_str)

            # Validate required fields
            if 'analysis_result' not in structured_data:
                logger.warning("Missing analysis_result in structured response")
                return None

            return structured_data['analysis_result']

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response content: {response[:500]}...")
            return None
        except Exception as e:
            logger.error(f"Error parsing structured response: {e}")
            return None

    def _create_analysis_from_structured_data(self, data: Dict[str, Any], topic: str) -> AnalysisResult:
        """Create AnalysisResult from structured data"""
        try:
            # Extract basic fields
            confidence = float(data.get('confidence_score', 0.5))
            key_findings = data.get('key_findings', [])
            research_gaps = data.get('research_gaps', [])
            recommendations = data.get('recommendations', [])
            summary = data.get('summary', 'No summary available')

            # Extract entities and relationships for metadata
            entities = data.get('entities', {})
            relationships = data.get('relationships', [])
            timeline = data.get('timeline', [])
            credibility = data.get('credibility_assessment', {})
            anomalies = data.get('anomalies_detected', [])

            # Create potential issues from anomalies and credibility assessment
            potential_issues = []

            # Add anomalies as potential issues
            for anomaly in anomalies:
                issue_desc = f"{anomaly.get('type', 'Unknown')} detected: {anomaly.get('description', '')}"
                if anomaly.get('severity') in ['high', 'critical']:
                    issue_desc = f"⚠️ {issue_desc}"
                potential_issues.append(issue_desc)

            # Add credibility concerns
            bias_indicators = credibility.get('bias_indicators', [])
            if bias_indicators:
                potential_issues.append(f"Bias indicators detected: {', '.join(bias_indicators)}")

            fact_check_status = credibility.get('fact_check_status', 'unverified')
            if fact_check_status in ['disputed', 'unverified']:
                potential_issues.append(f"Fact-check status: {fact_check_status}")

            # Create comprehensive metadata
            metadata = {
                'entities': entities,
                'relationships': relationships,
                'timeline': timeline,
                'credibility_assessment': credibility,
                'anomalies_detected': anomalies,
                'structured_analysis': True,
                'extraction_timestamp': datetime.now().isoformat()
            }

            return AnalysisResult(
                response=summary,
                confidence=min(max(confidence, 0.0), 1.0),  # Clamp between 0 and 1
                key_findings=key_findings[:10],  # Limit to 10 findings
                potential_issues=potential_issues[:10],  # Limit to 10 issues
                recommendations=recommendations[:8],  # Limit to 8 recommendations
                analysis_type='structured_analysis',
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error creating analysis from structured data: {e}")
            return self._create_fallback_analysis(data, topic)

    def _create_fallback_analysis(self, data: Dict[str, Any], topic: str) -> AnalysisResult:
        """Create basic analysis result when structured parsing partially fails"""
        try:
            # Extract what we can
            summary = data.get('summary', f'Analysis of {topic} completed with partial data extraction.')
            confidence = float(data.get('confidence_score', 0.3))
            key_findings = data.get('key_findings', ['Partial analysis completed'])

            return AnalysisResult(
                response=summary,
                confidence=confidence,
                key_findings=key_findings,
                potential_issues=['Structured analysis partially failed'],
                recommendations=['Consider re-running analysis'],
                analysis_type='partial_structured',
                metadata={'structured_analysis': False, 'fallback_used': True}
            )
        except Exception as e:
            logger.error(f"Error in fallback analysis creation: {e}")
            return AnalysisResult(
                response=f"Analysis of {topic} encountered errors during processing.",
                confidence=0.1,
                key_findings=['Analysis failed'],
                potential_issues=[str(e)],
                recommendations=['Please try again'],
                analysis_type='error',
                metadata={'error': str(e)}
            )

    async def _fallback_analysis(self, topic: str, context: str, analysis_type: str) -> AnalysisResult:
        """Fallback to regular text analysis if structured parsing fails"""
        logger.info("Using fallback text analysis")

        prompt = self._create_analysis_prompt(topic, context, analysis_type)
        response = await self._query_ollama_async(prompt)

        # Parse regular text response
        findings = self._extract_key_findings(response)
        issues = self._extract_potential_issues(response)
        recommendations = self._extract_recommendations(response)
        confidence = self._calculate_confidence(response, findings)

        return AnalysisResult(
            response=response,
            confidence=confidence,
            key_findings=findings,
            potential_issues=issues,
            recommendations=recommendations,
            analysis_type=analysis_type,
            metadata={'structured_analysis': False, 'fallback_used': True}
        )

    def extract_entities_and_relationships(self, text: str) -> Dict[str, Any]:
        """Extract structured entities and relationships from text using AI"""
        try:
            entity_prompt = f"""
Extract structured information from this text in JSON format:

TEXT: {text[:2000]}

Provide JSON with:
{{
  "entities": {{
    "people": [{{"name": "string", "role": "string", "relevance": 0.0-1.0}}],
    "organizations": [{{"name": "string", "type": "string", "relevance": 0.0-1.0}}],
    "locations": [{{"name": "string", "type": "string", "relevance": 0.0-1.0}}],
    "events": [{{"name": "string", "date": "YYYY-MM-DD or null", "relevance": 0.0-1.0}}]
  }},
  "relationships": [{{
    "entity1": "string",
    "entity2": "string",
    "relationship_type": "string",
    "strength": 0.0-1.0,
    "description": "string"
  }}]
}}

JSON:
"""

            # This would be called asynchronously in practice
            loop = asyncio.get_event_loop()
            response = loop.run_until_complete(self._query_ollama_async(entity_prompt))

            # Parse response
            extracted_data = self._parse_structured_response(response)
            return extracted_data if extracted_data else {}

        except Exception as e:
            logger.error(f"Error extracting entities and relationships: {e}")
            return {}

