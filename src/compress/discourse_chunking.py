#!/usr/bin/env python3
"""
Discourse-Aware Chunking System
Inteligentní chunking který respektuje diskurzní strukturu dokumentů

Author: Senior Python/MLOps Agent
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import spacy
from spacy.lang.en import English

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Typy chunků podle diskurzní struktury"""
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    CITATION = "citation"
    QUOTE = "quote"
    CODE_BLOCK = "code_block"
    TABLE = "table"
    CAPTION = "caption"
    SPEECH_ACT = "speech_act"


class SpeechActType(Enum):
    """Typy řečových aktů"""
    CLAIM = "claim"
    EVIDENCE = "evidence"
    HYPOTHESIS = "hypothesis"
    CONCLUSION = "conclusion"
    QUESTION = "question"
    DEFINITION = "definition"
    EXAMPLE = "example"
    CONTRAST = "contrast"
    CAUSATION = "causation"


@dataclass
class DiscourseChunk:
    """Chunk s diskurzní anotací"""
    id: str
    text: str
    chunk_type: ChunkType
    speech_act: Optional[SpeechActType]
    start_position: int
    end_position: int
    depth_level: int  # Hierarchical depth (headings)
    parent_chunk_id: Optional[str]
    children_chunk_ids: List[str]
    discourse_markers: List[str]
    entities: List[str]
    claims_density: float
    metadata: Dict[str, Any]


class DiscourseAwareChunker:
    """Diskurz-aware chunking engine"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunking_config = config.get("compression", {}).get("discourse_chunking", {})

        # Chunking parameters
        self.max_chunk_size = self.chunking_config.get("max_chunk_size", 512)
        self.min_chunk_size = self.chunking_config.get("min_chunk_size", 50)
        self.overlap_ratio = self.chunking_config.get("overlap_ratio", 0.1)

        # Discourse analysis settings
        self.detect_speech_acts = self.chunking_config.get("detect_speech_acts", True)
        self.preserve_citations = self.chunking_config.get("preserve_citations", True)
        self.respect_list_structure = self.chunking_config.get("respect_list_structure", True)

        # NLP model
        self.nlp = None

        # Discourse patterns
        self._compile_discourse_patterns()

    async def initialize(self):
        """Inicializace NLP modelů"""

        logger.info("Initializing Discourse-Aware Chunker...")

        try:
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Fallback to basic English model
                self.nlp = English()
                logger.warning("Full spaCy model not available, using basic tokenizer")

            logger.info("✅ Discourse-Aware Chunker initialized")

        except Exception as e:
            logger.error(f"Failed to initialize chunker: {e}")
            raise

    def _compile_discourse_patterns(self):
        """Kompilace regex patterns pro diskurzní struktury"""

        # Heading patterns
        self.heading_patterns = [
            re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE),  # Markdown headings
            re.compile(r'^([A-Z][A-Z\s]{2,})$', re.MULTILINE),  # ALL CAPS headings
            re.compile(r'^\d+\.\s+([A-Z].+)$', re.MULTILINE),  # Numbered headings
            re.compile(r'^([IVX]+)\.\s+([A-Z].+)$', re.MULTILINE),  # Roman numeral headings
        ]

        # List patterns
        self.list_patterns = [
            re.compile(r'^\s*[\*\-\+]\s+(.+)$', re.MULTILINE),  # Bullet lists
            re.compile(r'^\s*\d+\.\s+(.+)$', re.MULTILINE),  # Numbered lists
            re.compile(r'^\s*[a-z]\)\s+(.+)$', re.MULTILINE),  # Lettered lists
        ]

        # Citation patterns
        self.citation_patterns = [
            re.compile(r'\[(\d+)\]'),  # [1], [2], etc.
            re.compile(r'\(([A-Za-z]+,?\s*\d{4})\)'),  # (Author, 2024)
            re.compile(r'([A-Za-z]+\s+et\s+al\.,?\s*\d{4})'),  # Author et al., 2024
            re.compile(r'doi:\s*([^\s]+)'),  # DOI citations
        ]

        # Quote patterns
        self.quote_patterns = [
            re.compile(r'"([^"]+)"'),  # Double quotes
            re.compile(r''([^']+)''),  # Smart quotes
            re.compile(r'^\s*>\s+(.+)$', re.MULTILINE),  # Blockquotes
        ]

        # Speech act indicators
        self.speech_act_patterns = {
            SpeechActType.CLAIM: [
                re.compile(r'\b(we\s+claim|we\s+argue|we\s+propose|we\s+assert)\b', re.IGNORECASE),
                re.compile(r'\b(it\s+is\s+evident|clearly|obviously|undoubtedly)\b', re.IGNORECASE),
            ],
            SpeechActType.EVIDENCE: [
                re.compile(r'\b(evidence\s+shows|data\s+indicates|studies\s+show|research\s+demonstrates)\b', re.IGNORECASE),
                re.compile(r'\b(according\s+to|based\s+on|as\s+shown\s+by)\b', re.IGNORECASE),
            ],
            SpeechActType.HYPOTHESIS: [
                re.compile(r'\b(we\s+hypothesize|we\s+predict|we\s+expect|presumably)\b', re.IGNORECASE),
                re.compile(r'\b(if\s+.+\s+then|assuming\s+that)\b', re.IGNORECASE),
            ],
            SpeechActType.CONCLUSION: [
                re.compile(r'\b(in\s+conclusion|therefore|thus|hence|consequently)\b', re.IGNORECASE),
                re.compile(r'\b(we\s+conclude|our\s+results\s+suggest)\b', re.IGNORECASE),
            ],
            SpeechActType.QUESTION: [
                re.compile(r'\?'),  # Simple question mark
                re.compile(r'\b(what\s+if|how\s+might|why\s+does)\b', re.IGNORECASE),
            ],
            SpeechActType.DEFINITION: [
                re.compile(r'\b(is\s+defined\s+as|refers\s+to|means\s+that)\b', re.IGNORECASE),
                re.compile(r'\b(.+)\s+is\s+(.+)$'),  # X is Y pattern
            ],
            SpeechActType.CAUSATION: [
                re.compile(r'\b(because\s+of|due\s+to|caused\s+by|results\s+in)\b', re.IGNORECASE),
                re.compile(r'\b(leads\s+to|triggers|induces|produces)\b', re.IGNORECASE),
            ]
        }

    async def chunk_document(self, text: str, document_id: str = "doc") -> List[DiscourseChunk]:
        """
        Hlavní chunking funkce s diskurzní analýzou

        Args:
            text: Text dokumentu k rozchunkování
            document_id: ID dokumentu pro chunk identifikaci

        Returns:
            List DiscourseChunk s diskurzní anotacemi
        """

        logger.info(f"Starting discourse-aware chunking for document {document_id}")

        if not self.nlp:
            await self.initialize()

        # STEP 1: Preprocess text
        preprocessed_text = self._preprocess_text(text)

        # STEP 2: Identify discourse structures
        discourse_structures = self._identify_discourse_structures(preprocessed_text)

        # STEP 3: Create initial chunks based on structure
        structural_chunks = self._create_structural_chunks(
            preprocessed_text, discourse_structures, document_id
        )

        # STEP 4: Apply size-based chunking within structures
        sized_chunks = await self._apply_size_based_chunking(structural_chunks)

        # STEP 5: Detect speech acts
        if self.detect_speech_acts:
            sized_chunks = self._detect_speech_acts(sized_chunks)

        # STEP 6: Calculate claims density
        sized_chunks = self._calculate_claims_density(sized_chunks)

        # STEP 7: Build hierarchical relationships
        final_chunks = self._build_chunk_hierarchy(sized_chunks)

        logger.info(f"Created {len(final_chunks)} discourse-aware chunks")

        return final_chunks

    def _preprocess_text(self, text: str) -> str:
        """Předzpracování textu"""

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Preserve paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Clean up common artifacts
        text = re.sub(r'\u00a0', ' ', text)  # Non-breaking spaces
        text = re.sub(r'[\ufeff\u200b]', '', text)  # Zero-width characters

        return text.strip()

    def _identify_discourse_structures(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Identifikace diskurzních struktur v textu"""

        structures = {
            "headings": [],
            "lists": [],
            "citations": [],
            "quotes": [],
            "code_blocks": [],
            "tables": []
        }

        # Find headings
        for pattern in self.heading_patterns:
            for match in pattern.finditer(text):
                structures["headings"].append({
                    "text": match.group().strip(),
                    "start": match.start(),
                    "end": match.end(),
                    "level": self._determine_heading_level(match.group())
                })

        # Find lists
        for pattern in self.list_patterns:
            for match in pattern.finditer(text):
                structures["lists"].append({
                    "text": match.group().strip(),
                    "start": match.start(),
                    "end": match.end(),
                    "item_text": match.group(1) if match.groups() else match.group()
                })

        # Find citations
        all_citations = []
        for pattern in self.citation_patterns:
            for match in pattern.finditer(text):
                all_citations.append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "citation_id": match.group(1) if match.groups() else match.group()
                })

        # Merge overlapping citations
        structures["citations"] = self._merge_overlapping_spans(all_citations)

        # Find quotes
        for pattern in self.quote_patterns:
            for match in pattern.finditer(text):
                structures["quotes"].append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "quote_content": match.group(1) if match.groups() else match.group()
                })

        # Find code blocks (basic detection)
        code_pattern = re.compile(r'```[\s\S]*?```|`[^`]+`')
        for match in code_pattern.finditer(text):
            structures["code_blocks"].append({
                "text": match.group(),
                "start": match.start(),
                "end": match.end()
            })

        # Sort all structures by position
        for key in structures:
            structures[key].sort(key=lambda x: x["start"])

        return structures

    def _determine_heading_level(self, heading_text: str) -> int:
        """Určení úrovně nadpisu"""

        # Markdown style
        if heading_text.startswith('#'):
            return len(heading_text) - len(heading_text.lstrip('#'))

        # All caps - assume level 1
        if heading_text.isupper():
            return 1

        # Numbered - extract level from numbering
        if re.match(r'^\d+\.', heading_text):
            return 2

        # Roman numerals - level 1
        if re.match(r'^[IVX]+\.', heading_text):
            return 1

        return 2  # Default level

    def _merge_overlapping_spans(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sloučení překrývajících se spanů"""

        if not spans:
            return []

        sorted_spans = sorted(spans, key=lambda x: x["start"])
        merged = [sorted_spans[0]]

        for current in sorted_spans[1:]:
            last = merged[-1]

            if current["start"] <= last["end"]:
                # Merge overlapping spans
                merged[-1] = {
                    "text": last["text"] + " " + current["text"],
                    "start": last["start"],
                    "end": max(last["end"], current["end"]),
                    "citation_id": f"{last.get('citation_id', '')},{current.get('citation_id', '')}"
                }
            else:
                merged.append(current)

        return merged

    def _create_structural_chunks(self, text: str, structures: Dict[str, List[Dict[str, Any]]],
                                document_id: str) -> List[DiscourseChunk]:
        """Vytvoření chunků podle struktury"""

        chunks = []
        chunk_id_counter = 0

        # Create sorted list of all structural boundaries
        all_boundaries = []

        for struct_type, items in structures.items():
            for item in items:
                all_boundaries.append({
                    "position": item["start"],
                    "type": struct_type,
                    "data": item,
                    "is_start": True
                })
                all_boundaries.append({
                    "position": item["end"],
                    "type": struct_type,
                    "data": item,
                    "is_start": False
                })

        # Add document boundaries
        all_boundaries.append({"position": 0, "type": "document", "is_start": True})
        all_boundaries.append({"position": len(text), "type": "document", "is_start": False})

        all_boundaries.sort(key=lambda x: (x["position"], not x["is_start"]))

        # Create chunks between boundaries
        last_pos = 0
        current_context = []

        for boundary in all_boundaries:
            pos = boundary["position"]

            # Create chunk if there's content
            if pos > last_pos and last_pos < len(text):
                chunk_text = text[last_pos:pos].strip()

                if len(chunk_text) >= self.min_chunk_size:
                    chunk_type, speech_act = self._determine_chunk_characteristics(
                        chunk_text, current_context
                    )

                    chunk = DiscourseChunk(
                        id=f"{document_id}_chunk_{chunk_id_counter}",
                        text=chunk_text,
                        chunk_type=chunk_type,
                        speech_act=speech_act,
                        start_position=last_pos,
                        end_position=pos,
                        depth_level=self._get_current_depth(current_context),
                        parent_chunk_id=None,  # Will be set in hierarchy building
                        children_chunk_ids=[],
                        discourse_markers=self._extract_discourse_markers(chunk_text),
                        entities=[],  # Will be filled by NLP processing
                        claims_density=0.0,  # Will be calculated later
                        metadata={
                            "structural_context": current_context.copy(),
                            "boundary_types": [b["type"] for b in all_boundaries
                                             if last_pos <= b["position"] <= pos]
                        }
                    )

                    chunks.append(chunk)
                    chunk_id_counter += 1

            # Update context
            if boundary["is_start"]:
                current_context.append(boundary)
            else:
                # Remove corresponding start boundary
                current_context = [ctx for ctx in current_context
                                 if not (ctx["type"] == boundary["type"] and
                                        ctx["data"] == boundary["data"])]

            last_pos = pos

        return chunks

    def _determine_chunk_characteristics(self, text: str,
                                       context: List[Dict[str, Any]]) -> Tuple[ChunkType, Optional[SpeechActType]]:
        """Určení typu chunku a řečového aktu"""

        # Check context for explicit types
        for ctx in context:
            if ctx["type"] == "headings":
                return ChunkType.HEADING, None
            elif ctx["type"] == "lists":
                return ChunkType.LIST_ITEM, None
            elif ctx["type"] == "citations":
                return ChunkType.CITATION, None
            elif ctx["type"] == "quotes":
                return ChunkType.QUOTE, None
            elif ctx["type"] == "code_blocks":
                return ChunkType.CODE_BLOCK, None

        # Detect speech acts in text
        speech_act = self._detect_primary_speech_act(text)

        # Default to paragraph
        return ChunkType.PARAGRAPH, speech_act

    def _detect_primary_speech_act(self, text: str) -> Optional[SpeechActType]:
        """Detekce primárního řečového aktu v textu"""

        act_scores = {}

        for act_type, patterns in self.speech_act_patterns.items():
            score = 0
            for pattern in patterns:
                matches = pattern.findall(text)
                score += len(matches)

            if score > 0:
                act_scores[act_type] = score

        if act_scores:
            return max(act_scores.keys(), key=lambda x: act_scores[x])

        return None

    def _get_current_depth(self, context: List[Dict[str, Any]]) -> int:
        """Získání aktuální hierarchické hloubky"""

        heading_levels = []
        for ctx in context:
            if ctx["type"] == "headings":
                heading_levels.append(ctx["data"].get("level", 1))

        return max(heading_levels) if heading_levels else 0

    def _extract_discourse_markers(self, text: str) -> List[str]:
        """Extrakce diskurzních markerů"""

        markers = []

        # Common discourse markers
        marker_patterns = [
            r'\b(however|nevertheless|furthermore|moreover|therefore|thus|hence)\b',
            r'\b(in contrast|on the other hand|similarly|likewise)\b',
            r'\b(first|second|third|finally|in conclusion)\b',
            r'\b(for example|for instance|such as|namely)\b',
            r'\b(because|since|as a result|consequently)\b'
        ]

        for pattern in marker_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            markers.extend([m.lower() for m in matches])

        return list(set(markers))  # Remove duplicates

    async def _apply_size_based_chunking(self, chunks: List[DiscourseChunk]) -> List[DiscourseChunk]:
        """Aplikace size-based chunkingu v rámci strukturálních chunků"""

        sized_chunks = []

        for chunk in chunks:
            if len(chunk.text) <= self.max_chunk_size:
                # Chunk is already good size
                sized_chunks.append(chunk)
            else:
                # Split large chunk while preserving discourse structure
                sub_chunks = await self._split_large_chunk(chunk)
                sized_chunks.extend(sub_chunks)

        return sized_chunks

    async def _split_large_chunk(self, chunk: DiscourseChunk) -> List[DiscourseChunk]:
        """Rozdělení velkého chunku s ohledem na diskurz"""

        text = chunk.text
        target_size = self.max_chunk_size
        overlap_size = int(target_size * self.overlap_ratio)

        # Try to split on sentence boundaries first
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
        else:
            # Fallback sentence splitting
            sentences = re.split(r'[.!?]+\s+', text)

        sub_chunks = []
        current_text = ""
        current_sentences = []
        sub_chunk_counter = 0

        for sentence in sentences:
            # Check if adding this sentence would exceed target size
            if len(current_text + sentence) > target_size and current_text:
                # Create sub-chunk
                sub_chunk = DiscourseChunk(
                    id=f"{chunk.id}_sub_{sub_chunk_counter}",
                    text=current_text.strip(),
                    chunk_type=chunk.chunk_type,
                    speech_act=chunk.speech_act,
                    start_position=chunk.start_position,  # Approximate
                    end_position=chunk.start_position + len(current_text),
                    depth_level=chunk.depth_level,
                    parent_chunk_id=chunk.id,
                    children_chunk_ids=[],
                    discourse_markers=chunk.discourse_markers,
                    entities=[],
                    claims_density=0.0,
                    metadata={
                        **chunk.metadata,
                        "is_sub_chunk": True,
                        "parent_chunk": chunk.id
                    }
                )

                sub_chunks.append(sub_chunk)
                sub_chunk_counter += 1

                # Start new chunk with overlap
                if overlap_size > 0 and len(current_sentences) > 1:
                    overlap_sentences = current_sentences[-2:]  # Last 2 sentences
                    current_text = " ".join(overlap_sentences) + " " + sentence
                    current_sentences = overlap_sentences + [sentence]
                else:
                    current_text = sentence
                    current_sentences = [sentence]
            else:
                current_text += " " + sentence if current_text else sentence
                current_sentences.append(sentence)

        # Add final sub-chunk
        if current_text.strip():
            sub_chunk = DiscourseChunk(
                id=f"{chunk.id}_sub_{sub_chunk_counter}",
                text=current_text.strip(),
                chunk_type=chunk.chunk_type,
                speech_act=chunk.speech_act,
                start_position=chunk.start_position + len(chunk.text) - len(current_text),
                end_position=chunk.end_position,
                depth_level=chunk.depth_level,
                parent_chunk_id=chunk.id,
                children_chunk_ids=[],
                discourse_markers=chunk.discourse_markers,
                entities=[],
                claims_density=0.0,
                metadata={
                    **chunk.metadata,
                    "is_sub_chunk": True,
                    "parent_chunk": chunk.id
                }
            )

            sub_chunks.append(sub_chunk)

        return sub_chunks

    def _detect_speech_acts(self, chunks: List[DiscourseChunk]) -> List[DiscourseChunk]:
        """Detekce řečových aktů v chuncích"""

        for chunk in chunks:
            if chunk.speech_act is None:  # Only if not already detected
                chunk.speech_act = self._detect_primary_speech_act(chunk.text)

        return chunks

    def _calculate_claims_density(self, chunks: List[DiscourseChunk]) -> List[DiscourseChunk]:
        """Výpočet hustoty tvrzení v chuncích"""

        for chunk in chunks:
            # Simple heuristic for claims density
            claim_indicators = [
                r'\b(we\s+claim|we\s+argue|we\s+propose|we\s+assert)\b',
                r'\b(it\s+is\s+evident|clearly|obviously)\b',
                r'\b(shows?\s+that|indicates?\s+that|suggests?\s+that)\b',
                r'\b(therefore|thus|hence|consequently)\b'
            ]

            total_indicators = 0
            for pattern in claim_indicators:
                matches = re.findall(pattern, chunk.text, re.IGNORECASE)
                total_indicators += len(matches)

            # Normalize by text length
            words_count = len(chunk.text.split())
            chunk.claims_density = total_indicators / max(words_count / 100, 1)  # Per 100 words

        return chunks

    def _build_chunk_hierarchy(self, chunks: List[DiscourseChunk]) -> List[DiscourseChunk]:
        """Vybudování hierarchických vztahů mezi chunky"""

        # Sort chunks by position
        sorted_chunks = sorted(chunks, key=lambda x: x.start_position)

        # Build parent-child relationships based on depth levels
        for i, chunk in enumerate(sorted_chunks):
            # Find parent (previous chunk with lower depth)
            for j in range(i - 1, -1, -1):
                potential_parent = sorted_chunks[j]
                if potential_parent.depth_level < chunk.depth_level:
                    chunk.parent_chunk_id = potential_parent.id
                    potential_parent.children_chunk_ids.append(chunk.id)
                    break

            # Extract entities using NLP if available
            if self.nlp:
                try:
                    doc = self.nlp(chunk.text[:1000])  # Limit for performance
                    chunk.entities = [ent.text for ent in doc.ents]
                except Exception as e:
                    logger.warning(f"Entity extraction failed for chunk {chunk.id}: {e}")

        return sorted_chunks

    def get_chunking_stats(self, chunks: List[DiscourseChunk]) -> Dict[str, Any]:
        """Statistiky chunkingu"""

        chunk_types = {}
        speech_acts = {}

        for chunk in chunks:
            # Count chunk types
            chunk_type = chunk.chunk_type.value
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

            # Count speech acts
            if chunk.speech_act:
                speech_act = chunk.speech_act.value
                speech_acts[speech_act] = speech_acts.get(speech_act, 0) + 1

        # Size statistics
        chunk_sizes = [len(chunk.text) for chunk in chunks]
        claims_densities = [chunk.claims_density for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "chunk_types_distribution": chunk_types,
            "speech_acts_distribution": speech_acts,
            "size_stats": {
                "min": min(chunk_sizes) if chunk_sizes else 0,
                "max": max(chunk_sizes) if chunk_sizes else 0,
                "mean": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
                "median": sorted(chunk_sizes)[len(chunk_sizes)//2] if chunk_sizes else 0
            },
            "claims_density_stats": {
                "min": min(claims_densities) if claims_densities else 0,
                "max": max(claims_densities) if claims_densities else 0,
                "mean": sum(claims_densities) / len(claims_densities) if claims_densities else 0
            },
            "hierarchical_depth": max([chunk.depth_level for chunk in chunks]) if chunks else 0
        }


# Factory function
def create_discourse_chunker(config: Dict[str, Any]) -> DiscourseAwareChunker:
    """Factory function pro Discourse-Aware Chunker"""
    return DiscourseAwareChunker(config)
