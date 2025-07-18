"""
Prompt Parser Agent - Optimized for computational efficiency
Handles natural language parsing with minimal overhead
"""

import asyncio
import re
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from ..core.agent_framework import BaseAgent, AgentMessage, MessageType, Priority

class DesignIntent(Enum):
    SIMPLE_STRUCTURE = "simple_structure"
    COMPLEX_INFRASTRUCTURE = "complex_infrastructure"
    RETROFIT_UPGRADE = "retrofit_upgrade"
    ANALYSIS_ONLY = "analysis_only"
    VALIDATION_CHECK = "validation_check"

@dataclass
class EngineeringParameter:
    name: str
    value: Any
    unit: str
    confidence: float
    source: str  # "explicit" or "inferred"

@dataclass
class DesignConstraint:
    type: str
    value: Any
    priority: Priority
    source: str

@dataclass
class ParsedPrompt:
    intent: DesignIntent
    parameters: List[EngineeringParameter]
    constraints: List[DesignConstraint]
    materials: List[str]
    codes_standards: List[str]
    confidence_score: float
    ambiguities: List[str]

class PromptParserAgent(BaseAgent):
    """
    High-performance prompt parser optimized for engineering language
    Uses regex patterns and lookup tables for maximum efficiency
    """
    
    def __init__(self, agent_id: str = "prompt_parser", max_workers: int = 2):
        super().__init__(agent_id, max_workers)
        
        # Precompiled regex patterns for performance
        self._compile_patterns()
        
        # Lookup tables for fast matching
        self._load_lookup_tables()
        
        # Cache for parsed results
        self.parse_cache = {}
        
        logging.info(f"PromptParserAgent initialized with {len(self.unit_patterns)} unit patterns")
    
    def _compile_patterns(self):
        """Precompile regex patterns for maximum performance"""
        
        # Dimension patterns
        self.dimension_patterns = [
            re.compile(r'(\d+\.?\d*)\s*(?:m|meter|metres?|ft|feet|foot)\s*(?:high|height|tall)', re.IGNORECASE),
            re.compile(r'(\d+\.?\d*)\s*(?:m|meter|metres?|ft|feet|foot)\s*(?:long|length)', re.IGNORECASE),
            re.compile(r'(\d+\.?\d*)\s*(?:m|meter|metres?|ft|feet|foot)\s*(?:wide|width|thick|thickness)', re.IGNORECASE),
            re.compile(r'height\s*(?:of|=|:)?\s*(\d+\.?\d*)\s*(?:m|meter|metres?|ft|feet|foot)', re.IGNORECASE),
            re.compile(r'length\s*(?:of|=|:)?\s*(\d+\.?\d*)\s*(?:m|meter|metres?|ft|feet|foot)', re.IGNORECASE),
        ]
        
        # Material patterns
        self.material_patterns = [
            re.compile(r'\b(reinforced concrete|concrete|steel|timber|wood|masonry|brick)\b', re.IGNORECASE),
            re.compile(r'\b(grade\s*\d+|f\'c\s*=?\s*\d+|fy\s*=?\s*\d+)\b', re.IGNORECASE),
        ]
        
        # Load patterns
        self.load_patterns = [
            re.compile(r'(\d+\.?\d*)\s*(?:kn|kip|ton|tonne|lb|pound)(?:/m|/ft)?\s*(?:load|force)', re.IGNORECASE),
            re.compile(r'(?:live|dead|wind|seismic|earthquake)\s*load', re.IGNORECASE),
        ]
        
        # Code/Standard patterns
        self.code_patterns = [
            re.compile(r'\b(aci\s*\d+|asce\s*\d+|aisc\s*\d+|ibc\s*\d+|fema\s*\d+)\b', re.IGNORECASE),
            re.compile(r'\b(seismic\s*zone\s*\d+|wind\s*zone\s*\d+)\b', re.IGNORECASE),
        ]
        
        # Unit patterns
        self.unit_patterns = {
            'length': re.compile(r'\b(?:m|meter|metres?|ft|feet|foot|in|inch|inches|mm|millimeter|cm|centimeter)\b', re.IGNORECASE),
            'force': re.compile(r'\b(?:kn|kilonewton|kip|ton|tonne|lb|pound|n|newton)\b', re.IGNORECASE),
            'pressure': re.compile(r'\b(?:mpa|psi|psf|kpa|pa|pascal)\b', re.IGNORECASE),
            'area': re.compile(r'\b(?:m2|ft2|in2|mm2|cm2|sq\s*m|sq\s*ft|square\s*meter|square\s*foot)\b', re.IGNORECASE),
        }
    
    def _load_lookup_tables(self):
        """Load lookup tables for fast keyword matching"""
        
        # Structure types
        self.structure_types = {
            'wall': ['wall', 'retaining wall', 'floodwall', 'barrier'],
            'foundation': ['foundation', 'footing', 'pile', 'micropile', 'caisson'],
            'beam': ['beam', 'girder', 'lintel'],
            'column': ['column', 'pillar', 'post'],
            'slab': ['slab', 'deck', 'platform'],
            'culvert': ['culvert', 'pipe', 'conduit'],
            'weir': ['weir', 'gate', 'control structure'],
        }
        
        # Engineering keywords
        self.engineering_keywords = {
            'design': ['design', 'calculate', 'analyze', 'check', 'verify'],
            'safety': ['safety', 'factor of safety', 'ultimate', 'allowable'],
            'seismic': ['seismic', 'earthquake', 'dynamic', 'response spectrum'],
            'foundation': ['bearing capacity', 'settlement', 'soil', 'geotechnical'],
            'hydraulic': ['flow', 'pressure', 'head', 'discharge', 'flood'],
        }
        
        # Material properties
        self.material_properties = {
            'concrete': {'density': 2400, 'unit': 'kg/m3', 'default_strength': 25},
            'steel': {'density': 7850, 'unit': 'kg/m3', 'default_strength': 250},
            'soil': {'density': 1800, 'unit': 'kg/m3', 'default_bearing': 100},
        }
    
    async def handle_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle prompt parsing request"""
        
        if message.payload.get("action") == "parse_prompt":
            prompt = message.payload.get("prompt", "")
            
            # Check cache first
            cache_key = hash(prompt)
            if cache_key in self.parse_cache:
                parsed_result = self.parse_cache[cache_key]
                logging.info(f"Cache hit for prompt parsing")
            else:
                # Parse prompt
                parsed_result = await self._parse_prompt_fast(prompt)
                self.parse_cache[cache_key] = parsed_result
                
                # Limit cache size
                if len(self.parse_cache) > 100:
                    # Remove oldest entries
                    oldest_keys = list(self.parse_cache.keys())[:20]
                    for key in oldest_keys:
                        del self.parse_cache[key]
            
            return AgentMessage(
                sender=self.agent_id,
                receiver=message.sender,
                message_type=MessageType.RESPONSE,
                payload={
                    "parsed_prompt": parsed_result.__dict__,
                    "processing_time": time.time() - message.timestamp
                },
                correlation_id=message.correlation_id
            )
        
        return None
    
    async def _parse_prompt_fast(self, prompt: str) -> ParsedPrompt:
        """Fast prompt parsing using precompiled patterns"""
        
        # Normalize prompt
        prompt_lower = prompt.lower()
        
        # Extract parameters using regex patterns
        parameters = []
        
        # Extract dimensions
        for pattern in self.dimension_patterns:
            matches = pattern.findall(prompt)
            for match in matches:
                if isinstance(match, tuple):
                    value = float(match[0])
                    unit = self._extract_unit(pattern.pattern, prompt)
                else:
                    value = float(match)
                    unit = self._extract_unit(pattern.pattern, prompt)
                
                param_name = self._infer_parameter_name(pattern.pattern)
                parameters.append(EngineeringParameter(
                    name=param_name,
                    value=value,
                    unit=unit,
                    confidence=0.9,
                    source="explicit"
                ))
        
        # Extract materials
        materials = []
        for pattern in self.material_patterns:
            matches = pattern.findall(prompt)
            materials.extend(matches)
        
        # Extract codes/standards
        codes_standards = []
        for pattern in self.code_patterns:
            matches = pattern.findall(prompt)
            codes_standards.extend(matches)
        
        # Determine design intent
        intent = self._classify_intent_fast(prompt_lower)
        
        # Extract constraints
        constraints = self._extract_constraints_fast(prompt)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(parameters, materials, codes_standards)
        
        # Identify ambiguities
        ambiguities = self._identify_ambiguities_fast(prompt)
        
        return ParsedPrompt(
            intent=intent,
            parameters=parameters,
            constraints=constraints,
            materials=materials,
            codes_standards=codes_standards,
            confidence_score=confidence_score,
            ambiguities=ambiguities
        )
    
    def _classify_intent_fast(self, prompt: str) -> DesignIntent:
        """Fast intent classification using keyword matching"""
        
        # Simple keyword-based classification
        if any(word in prompt for word in ['retrofit', 'upgrade', 'existing', 'modify']):
            return DesignIntent.RETROFIT_UPGRADE
        elif any(word in prompt for word in ['analyze', 'check', 'verify', 'validate']):
            return DesignIntent.ANALYSIS_ONLY
        elif any(word in prompt for word in ['complex', 'system', 'integrated', 'multiple']):
            return DesignIntent.COMPLEX_INFRASTRUCTURE
        else:
            return DesignIntent.SIMPLE_STRUCTURE
    
    def _extract_constraints_fast(self, prompt: str) -> List[DesignConstraint]:
        """Fast constraint extraction"""
        constraints = []
        
        # Safety factor constraints
        safety_pattern = re.compile(r'safety\s*factor\s*(?:of|=|:)?\s*(\d+\.?\d*)', re.IGNORECASE)
        matches = safety_pattern.findall(prompt)
        for match in matches:
            constraints.append(DesignConstraint(
                type="safety_factor",
                value=float(match),
                priority=Priority.HIGH,
                source="explicit"
            ))
        
        # Code requirements
        if 'seismic' in prompt.lower():
            constraints.append(DesignConstraint(
                type="seismic_design",
                value=True,
                priority=Priority.HIGH,
                source="inferred"
            ))
        
        return constraints
    
    def _extract_unit(self, pattern: str, prompt: str) -> str:
        """Extract unit from pattern match"""
        if 'm' in pattern or 'meter' in pattern:
            return 'm'
        elif 'ft' in pattern or 'feet' in pattern:
            return 'ft'
        elif 'kn' in pattern:
            return 'kN'
        elif 'psi' in pattern:
            return 'psi'
        else:
            return 'unknown'
    
    def _infer_parameter_name(self, pattern: str) -> str:
        """Infer parameter name from regex pattern"""
        if 'height' in pattern:
            return 'height'
        elif 'length' in pattern:
            return 'length'
        elif 'width' in pattern or 'thick' in pattern:
            return 'width'
        elif 'load' in pattern:
            return 'load'
        else:
            return 'dimension'
    
    def _calculate_confidence(self, parameters: List[EngineeringParameter], 
                            materials: List[str], codes_standards: List[str]) -> float:
        """Calculate overall confidence score"""
        base_confidence = 0.5
        
        # Add confidence based on extracted information
        if parameters:
            base_confidence += 0.3
        if materials:
            base_confidence += 0.1
        if codes_standards:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _identify_ambiguities_fast(self, prompt: str) -> List[str]:
        """Fast ambiguity detection"""
        ambiguities = []
        
        # Check for vague terms
        vague_terms = ['strong', 'large', 'small', 'appropriate', 'suitable', 'adequate']
        for term in vague_terms:
            if term in prompt.lower():
                ambiguities.append(f"Vague term: '{term}' requires clarification")
        
        # Check for missing units
        numbers = re.findall(r'\d+\.?\d*', prompt)
        if numbers and not any(pattern.search(prompt) for pattern in self.unit_patterns.values()):
            ambiguities.append("Numbers found without clear units")
        
        return ambiguities
    
    async def handle_response(self, message: AgentMessage):
        """Handle response messages"""
        logging.info(f"Received response: {message.payload}")
    
    async def handle_notification(self, message: AgentMessage):
        """Handle notification messages"""
        logging.info(f"Received notification: {message.payload}")
    
    async def handle_error(self, message: AgentMessage):
        """Handle error messages"""
        logging.error(f"Received error: {message.payload}")

# Performance optimizations
import time

# Singleton pattern for shared resources
_prompt_parser_instance = None

def get_prompt_parser_agent() -> PromptParserAgent:
    """Get singleton instance of prompt parser agent"""
    global _prompt_parser_instance
    if _prompt_parser_instance is None:
        _prompt_parser_instance = PromptParserAgent()
    return _prompt_parser_instance