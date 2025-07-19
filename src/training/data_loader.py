"""
Data loading and preprocessing for Text-to-CAD training
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class TrainingExample:
    prompt: str
    intent: str
    parameters: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    files: List[str]
    expected_ifc: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class FileExample:
    file_path: str
    file_type: str
    expected_data: Dict[str, Any]
    engineering_domain: str

@dataclass
class IFCExample:
    ifc_content: str
    element_count: int
    spatial_hierarchy: Dict[str, Any]
    properties: List[Dict[str, Any]]
    source_prompt: Optional[str] = None

class DataLoader:
    """
    Comprehensive data loader for training the multi-agent system
    """
    
    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "prompts").mkdir(exist_ok=True)
        (self.data_dir / "files").mkdir(exist_ok=True)
        (self.data_dir / "ifc_models").mkdir(exist_ok=True)
        (self.data_dir / "synthetic").mkdir(exist_ok=True)
        (self.data_dir / "validation").mkdir(exist_ok=True)
        
        self.prompt_examples = []
        self.file_examples = []
        self.ifc_examples = []
        
        logging.info(f"DataLoader initialized with data directory: {self.data_dir}")
    
    def load_prompt_data(self) -> List[TrainingExample]:
        """Load and parse prompt training data"""
        
        prompt_files = list(self.data_dir.glob("**/*.json"))
        
        for prompt_file in prompt_files:
            if "prompt" in prompt_file.name.lower():
                try:
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if "prompts" in data:
                        for prompt_data in data["prompts"]:
                            example = TrainingExample(
                                prompt=prompt_data.get("text", ""),
                                intent=prompt_data.get("intent", "unknown"),
                                parameters=prompt_data.get("parameters", []),
                                constraints=prompt_data.get("constraints", []),
                                files=prompt_data.get("files", []),
                                expected_ifc=prompt_data.get("expected_ifc"),
                                metadata=prompt_data.get("metadata", {})
                            )
                            self.prompt_examples.append(example)
                            
                except Exception as e:
                    logging.warning(f"Failed to load prompt file {prompt_file}: {e}")
        
        logging.info(f"Loaded {len(self.prompt_examples)} prompt examples")
        return self.prompt_examples
    
    def load_file_data(self) -> List[FileExample]:
        """Load engineering file examples"""
        
        file_types = {
            '.xlsx': 'excel',
            '.xls': 'excel', 
            '.csv': 'csv',
            '.pdf': 'pdf',
            '.gsz': 'geostudio',
            '.gsd': 'geostudio',
            '.gp12a': 'staad_pro',
            '.gp12d': 'staad_pro',
            '.txt': 'text'
        }
        
        files_dir = self.data_dir / "files"
        if files_dir.exists():
            for file_path in files_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix in file_types:
                    file_type = file_types[file_path.suffix]
                    
                    # Try to load expected data from companion JSON file
                    expected_file = file_path.with_suffix('.expected.json')
                    expected_data = {}
                    if expected_file.exists():
                        try:
                            with open(expected_file, 'r') as f:
                                expected_data = json.load(f)
                        except Exception as e:
                            logging.warning(f"Failed to load expected data for {file_path}: {e}")
                    
                    example = FileExample(
                        file_path=str(file_path),
                        file_type=file_type,
                        expected_data=expected_data,
                        engineering_domain=self._infer_domain(file_path.name)
                    )
                    self.file_examples.append(example)
        
        logging.info(f"Loaded {len(self.file_examples)} file examples")
        return self.file_examples
    
    def load_ifc_data(self) -> List[IFCExample]:
        """Load IFC model examples"""
        
        ifc_dir = self.data_dir / "ifc_models"
        if ifc_dir.exists():
            for ifc_file in ifc_dir.rglob("*.ifc"):
                try:
                    with open(ifc_file, 'r', encoding='utf-8') as f:
                        ifc_content = f.read()
                    
                    # Enhanced IFC analysis
                    element_count = self._count_ifc_elements(ifc_content)
                    
                    # Determine domain from file path
                    domain = self._determine_ifc_domain(ifc_file)
                    
                    # Load metadata if available
                    metadata_file = ifc_file.with_suffix('.metadata.json')
                    spatial_hierarchy = {}
                    properties = []
                    source_prompt = None
                    
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                                spatial_hierarchy = metadata.get('spatial_hierarchy', {})
                                properties = metadata.get('properties', [])
                                source_prompt = metadata.get('source_prompt')
                        except Exception as e:
                            logging.warning(f"Failed to load IFC metadata for {ifc_file}: {e}")
                    else:
                        # Extract spatial hierarchy and properties for reference files
                        spatial_hierarchy = self._extract_spatial_hierarchy(ifc_content)
                        properties = self._extract_ifc_properties(ifc_content)
                        
                        # Generate reverse-engineered prompt if this is a reference file
                        if "reference" in str(ifc_file):
                            source_prompt = self._generate_prompt_from_ifc(ifc_file.name, ifc_content, domain)
                    
                    example = IFCExample(
                        ifc_content=ifc_content,
                        element_count=element_count,
                        spatial_hierarchy=spatial_hierarchy,
                        properties=properties,
                        source_prompt=source_prompt
                    )
                    self.ifc_examples.append(example)
                    
                except Exception as e:
                    logging.warning(f"Failed to load IFC file {ifc_file}: {e}")
        
        logging.info(f"Loaded {len(self.ifc_examples)} IFC examples")
        return self.ifc_examples
    
    def generate_synthetic_data(self, count: int = 1000) -> List[TrainingExample]:
        """Generate synthetic training data"""
        
        synthetic_examples = []
        
        # Templates for different engineering scenarios
        templates = {
            "simple_structure": [
                "Design a {material} {structure} {height}m high and {length}m long",
                "Create a {structure} using {material} with {width}m width",
                "Build a {structure_type} {height}m tall for {purpose}"
            ],
            "complex_infrastructure": [
                "Design a {material} floodwall {height}m high and {length}m long with {foundation_type} foundation for {protection_level} protection",
                "Create a {structure_type} system with {component_count} components for {capacity} capacity",
                "Build a {infrastructure_type} with integrated {systems} for {environment} conditions"
            ],
            "retrofit_upgrade": [
                "Upgrade existing {structure} to meet {code_standard} requirements",
                "Retrofit {structure_type} with {improvement_type} for {performance_goal}",
                "Modify existing {structure} to add {new_feature}"
            ]
        }
        
        # Parameter options
        materials = ["concrete", "steel", "reinforced_concrete", "precast_concrete", "composite"]
        structures = ["wall", "beam", "column", "foundation", "slab", "footing"]
        structure_types = ["retaining_wall", "floodwall", "foundation_system", "pile_system"]
        foundation_types = ["spread_footing", "pile_foundation", "mat_foundation", "micropile"]
        
        for i in range(count):
            intent = random.choice(list(templates.keys()))
            template = random.choice(templates[intent])
            
            # Generate parameters based on template
            parameters = []
            constraints = []
            
            # Fill template with realistic engineering values
            values = {
                'material': random.choice(materials),
                'structure': random.choice(structures),
                'structure_type': random.choice(structure_types),
                'height': round(random.uniform(2.0, 15.0), 1),
                'length': round(random.uniform(10.0, 500.0), 1),
                'width': round(random.uniform(0.3, 5.0), 1),
                'foundation_type': random.choice(foundation_types),
                'protection_level': f"{random.choice([50, 100, 500])}-year",
                'capacity': f"{random.randint(100, 10000)}kN",
                'component_count': random.randint(5, 50),
                'purpose': random.choice(["flood_protection", "earth_retention", "structural_support"]),
                'infrastructure_type': random.choice(["bridge", "culvert", "marina", "retaining_system"]),
                'systems': random.choice(["drainage", "monitoring", "access"]),
                'environment': random.choice(["marine", "seismic", "high_wind"]),
                'code_standard': random.choice(["ACI_318", "ASCE_7", "IBC_2021"]),
                'improvement_type': random.choice(["seismic_upgrade", "waterproofing", "reinforcement"]),
                'performance_goal': random.choice(["increased_capacity", "code_compliance", "durability"]),
                'new_feature': random.choice(["drainage_system", "access_platform", "monitoring"])
            }
            
            # Generate prompt text
            try:
                prompt_text = template.format(**values)
            except KeyError as e:
                # Skip if template has missing keys
                continue
            
            # Create parameters list
            for key, value in values.items():
                if key in prompt_text:
                    if isinstance(value, (int, float)):
                        unit = self._get_unit_for_parameter(key)
                        parameters.append({
                            "name": key,
                            "value": value,
                            "unit": unit,
                            "confidence": round(random.uniform(0.8, 1.0), 2),
                            "source": "synthetic"
                        })
                    else:
                        parameters.append({
                            "name": key,
                            "value": value,
                            "confidence": round(random.uniform(0.8, 1.0), 2),
                            "source": "synthetic"
                        })
            
            # Generate constraints
            if intent == "complex_infrastructure":
                constraints.extend([
                    {"type": "safety", "requirement": "factor_of_safety_2.0"},
                    {"type": "code_compliance", "requirement": "local_building_codes"}
                ])
            elif intent == "retrofit_upgrade":
                constraints.append({
                    "type": "existing_structure", "requirement": "maintain_service_during_construction"
                })
            
            example = TrainingExample(
                prompt=prompt_text,
                intent=intent,
                parameters=parameters,
                constraints=constraints,
                files=[],
                metadata={"synthetic": True, "generator_version": "1.0"}
            )
            synthetic_examples.append(example)
        
        # Save synthetic data
        synthetic_file = self.data_dir / "synthetic" / "generated_prompts.json"
        with open(synthetic_file, 'w', encoding='utf-8') as f:
            json.dump({
                "prompts": [
                    {
                        "text": ex.prompt,
                        "intent": ex.intent,
                        "parameters": ex.parameters,
                        "constraints": ex.constraints,
                        "metadata": ex.metadata
                    } for ex in synthetic_examples
                ]
            }, f, indent=2)
        
        logging.info(f"Generated {len(synthetic_examples)} synthetic examples")
        return synthetic_examples
    
    def split_data(self, test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, List[TrainingExample]]:
        """Split data into train/validation/test sets"""
        
        all_examples = self.prompt_examples.copy()
        random.shuffle(all_examples)
        
        total = len(all_examples)
        test_count = int(total * test_size)
        val_count = int(total * val_size)
        train_count = total - test_count - val_count
        
        splits = {
            "train": all_examples[:train_count],
            "validation": all_examples[train_count:train_count + val_count],
            "test": all_examples[train_count + val_count:]
        }
        
        logging.info(f"Data split: Train={len(splits['train'])}, Val={len(splits['validation'])}, Test={len(splits['test'])}")
        return splits
    
    def create_batches(self, examples: List[TrainingExample], batch_size: int = 8) -> List[List[TrainingExample]]:
        """Create training batches"""
        
        batches = []
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def augment_data(self, examples: List[TrainingExample], augmentation_factor: int = 3) -> List[TrainingExample]:
        """Augment training data with variations"""
        
        augmented = []
        
        for example in examples:
            # Original example
            augmented.append(example)
            
            # Create variations
            for _ in range(augmentation_factor):
                # Vary numerical parameters slightly
                new_parameters = []
                for param in example.parameters:
                    if isinstance(param.get("value"), (int, float)):
                        # Add small random variation
                        original_value = param["value"]
                        variation = original_value * random.uniform(0.9, 1.1)
                        
                        new_param = param.copy()
                        new_param["value"] = round(variation, 2)
                        new_param["confidence"] = max(0.5, param.get("confidence", 1.0) - 0.1)
                        new_parameters.append(new_param)
                    else:
                        new_parameters.append(param.copy())
                
                # Create augmented example
                augmented_example = TrainingExample(
                    prompt=example.prompt,  # Could also vary text slightly
                    intent=example.intent,
                    parameters=new_parameters,
                    constraints=example.constraints.copy(),
                    files=example.files.copy(),
                    metadata={**(example.metadata or {}), "augmented": True}
                )
                augmented.append(augmented_example)
        
        logging.info(f"Augmented data from {len(examples)} to {len(augmented)} examples")
        return augmented
    
    def save_training_data(self, examples: List[TrainingExample], filename: str):
        """Save training data to file"""
        
        output_file = self.data_dir / filename
        data = {
            "prompts": [
                {
                    "text": ex.prompt,
                    "intent": ex.intent,
                    "parameters": ex.parameters,
                    "constraints": ex.constraints,
                    "files": ex.files,
                    "expected_ifc": ex.expected_ifc,
                    "metadata": ex.metadata
                } for ex in examples
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Saved {len(examples)} examples to {output_file}")
    
    def _infer_domain(self, filename: str) -> str:
        """Infer engineering domain from filename"""
        
        filename_lower = filename.lower()
        
        if any(word in filename_lower for word in ['flood', 'water', 'hydro']):
            return 'hydraulic'
        elif any(word in filename_lower for word in ['soil', 'geo', 'foundation']):
            return 'geotechnical'
        elif any(word in filename_lower for word in ['struct', 'beam', 'column']):
            return 'structural'
        elif any(word in filename_lower for word in ['bridge', 'road', 'transportation']):
            return 'transportation'
        else:
            return 'general'
    
    def _get_unit_for_parameter(self, param_name: str) -> str:
        """Get appropriate unit for parameter"""
        
        unit_map = {
            'height': 'm',
            'length': 'm', 
            'width': 'm',
            'thickness': 'm',
            'diameter': 'm',
            'capacity': 'kN',
            'load': 'kN',
            'pressure': 'kPa',
            'flow_rate': 'gpm',
            'volume': 'm³',
            'area': 'm²',
            'force': 'kN',
            'moment': 'kN·m',
            'stress': 'MPa'
        }
        
        return unit_map.get(param_name, 'dimensionless')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        
        stats = {
            "total_prompt_examples": len(self.prompt_examples),
            "total_file_examples": len(self.file_examples),
            "total_ifc_examples": len(self.ifc_examples),
            "intent_distribution": {},
            "domain_distribution": {},
            "parameter_types": set(),
            "constraint_types": set()
        }
        
        # Analyze prompt examples
        for example in self.prompt_examples:
            # Intent distribution
            intent = example.intent
            stats["intent_distribution"][intent] = stats["intent_distribution"].get(intent, 0) + 1
            
            # Parameter types
            for param in example.parameters:
                stats["parameter_types"].add(param.get("name", "unknown"))
            
            # Constraint types
            for constraint in example.constraints:
                stats["constraint_types"].add(constraint.get("type", "unknown"))
        
        # Analyze file examples
        for example in self.file_examples:
            domain = example.engineering_domain
            stats["domain_distribution"][domain] = stats["domain_distribution"].get(domain, 0) + 1
        
        # Convert sets to lists for JSON serialization
        stats["parameter_types"] = list(stats["parameter_types"])
        stats["constraint_types"] = list(stats["constraint_types"])
        
        return stats
    
    def _count_ifc_elements(self, ifc_content: str) -> int:
        """Count IFC elements in content"""
        element_types = [
            'IFCWALL', 'IFCBEAM', 'IFCCOLUMN', 'IFCSLAB', 'IFCDOOR', 'IFCWINDOW',
            'IFCFOOTING', 'IFCPILE', 'IFCBRIDGE', 'IFCROAD', 'IFCRAIL', 'IFCDUCT',
            'IFCPIPE', 'IFCPUMP', 'IFCFLOWCONTROLLER', 'IFCSPACE', 'IFCZONE'
        ]
        
        count = 0
        for element_type in element_types:
            count += ifc_content.count(element_type)
        
        return count
    
    def _determine_ifc_domain(self, file_path: Path) -> str:
        """Determine engineering domain from file path"""
        path_str = str(file_path).lower()
        
        if 'building' in path_str or 'architecture' in path_str or 'hvac' in path_str:
            return 'building'
        elif 'infra' in path_str or 'bridge' in path_str or 'road' in path_str or 'rail' in path_str:
            return 'infrastructure'
        elif 'structural' in path_str:
            return 'structural'
        elif 'landscaping' in path_str or 'plumbing' in path_str:
            return 'utilities'
        else:
            return 'general'
    
    def _extract_spatial_hierarchy(self, ifc_content: str) -> Dict[str, Any]:
        """Extract basic spatial hierarchy from IFC content"""
        hierarchy = {
            'project': None,
            'sites': [],
            'buildings': [],
            'storeys': [],
            'spaces': []
        }
        
        # Simple regex-based extraction (would be enhanced with proper IFC parsing)
        import re
        
        # Extract project
        project_match = re.search(r'IFCPROJECT\(.*?\)', ifc_content)
        if project_match:
            hierarchy['project'] = project_match.group(0)
        
        # Extract sites
        site_matches = re.findall(r'IFCSITE\(.*?\)', ifc_content)
        hierarchy['sites'] = site_matches[:5]  # Limit to first 5
        
        # Extract buildings
        building_matches = re.findall(r'IFCBUILDING\(.*?\)', ifc_content)
        hierarchy['buildings'] = building_matches[:5]
        
        # Extract storeys/levels
        storey_matches = re.findall(r'IFCBUILDINGSTOREY\(.*?\)', ifc_content)
        hierarchy['storeys'] = storey_matches[:10]
        
        return hierarchy
    
    def _extract_ifc_properties(self, ifc_content: str) -> List[Dict[str, Any]]:
        """Extract basic properties from IFC content"""
        properties = []
        
        # Extract property sets
        import re
        prop_matches = re.findall(r'IFCPROPERTYSET\(.*?\)', ifc_content)
        
        for i, prop_match in enumerate(prop_matches[:10]):  # Limit to first 10
            properties.append({
                'id': f'prop_{i}',
                'type': 'property_set',
                'content': prop_match[:200]  # Truncate long content
            })
        
        return properties
    
    def _generate_prompt_from_ifc(self, filename: str, ifc_content: str, domain: str) -> str:
        """Generate a reverse-engineered prompt from IFC file analysis"""
        
        # Analyze filename for clues
        filename_lower = filename.lower()
        
        # Base prompts by domain and filename analysis
        if 'building-architecture' in filename_lower:
            return "Design a multi-story building with architectural elements including walls, doors, windows, and spaces"
        elif 'building-structural' in filename_lower:
            return "Create a structural building framework with beams, columns, slabs, and foundation elements"
        elif 'building-hvac' in filename_lower:
            return "Design an HVAC system for a building including ducts, air handling units, and ventilation spaces"
        elif 'infra-bridge' in filename_lower:
            return "Design a bridge structure with deck, supports, abutments, and approach elements"
        elif 'infra-road' in filename_lower:
            return "Create a road infrastructure with roadway elements, intersections, and traffic management"
        elif 'infra-rail' in filename_lower:
            return "Design a railway infrastructure with tracks, platforms, signals, and support structures"
        elif 'landscaping' in filename_lower:
            return "Design landscaping elements including planted areas, paths, and outdoor spaces"
        elif 'plumbing' in filename_lower:
            return "Create a plumbing system with pipes, fixtures, pumps, and water management elements"
        else:
            # Analyze content for element types
            wall_count = ifc_content.count('IFCWALL')
            beam_count = ifc_content.count('IFCBEAM')
            column_count = ifc_content.count('IFCCOLUMN')
            
            if wall_count > beam_count and wall_count > column_count:
                return f"Design a structure with {wall_count} walls and associated architectural elements"
            elif beam_count > 0 and column_count > 0:
                return f"Create a structural framework with {beam_count} beams and {column_count} columns"
            else:
                return f"Design a {domain} structure with appropriate engineering elements"