"""
IFC Generator Agent - Computationally efficient IFC file generation
Uses templates and optimized geometry generation for maximum performance
"""

import asyncio
import uuid
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import math
import json
from pathlib import Path

from ..core.agent_framework import BaseAgent, AgentMessage, MessageType, Priority
from ..training.ids_parser import IDSParser
from ..training.ids_validator import IDSValidator

class IFCElementType(Enum):
    WALL = "IfcWall"
    BEAM = "IfcBeam"
    COLUMN = "IfcColumn"
    SLAB = "IfcSlab"
    FOOTING = "IfcFooting"
    PILE = "IfcPile"
    BUILDING_ELEMENT_PROXY = "IfcBuildingElementProxy"

@dataclass
class IFCGeometry:
    representation_type: str
    coordinates: List[List[float]]
    extrusion_direction: Optional[List[float]] = None
    extrusion_depth: Optional[float] = None

@dataclass
class IFCProperty:
    name: str
    value: Any
    unit: str
    property_set: str

@dataclass
class IFCElement:
    global_id: str
    element_type: IFCElementType
    name: str
    geometry: IFCGeometry
    properties: List[IFCProperty]
    material: Optional[str] = None
    location: Optional[List[float]] = None

@dataclass
class IFCModel:
    project_name: str
    site_name: str
    building_name: str
    elements: List[IFCElement]
    spatial_hierarchy: Dict[str, Any]
    metadata: Dict[str, Any]

class IFCGeneratorAgent(BaseAgent):
    """
    High-performance IFC generator using templates and optimized algorithms
    """
    
    def __init__(self, agent_id: str = "ifc_generator", max_workers: int = 2):
        super().__init__(agent_id, max_workers)
        
        # Load IFC templates for common elements
        self._load_ifc_templates()
        
        # Geometry optimization cache
        self.geometry_cache = {}
        
        # IDS integration for compliance validation
        self.ids_parser = IDSParser()
        self.ids_validator = IDSValidator()
        self.ids_specifications = []
        self._load_ids_specifications()
        
        # IFC generation statistics
        self.generation_stats = {
            'models_generated': 0,
            'elements_created': 0,
            'avg_generation_time': 0.0,
            'ids_validations': 0,
            'ids_compliance_rate': 0.0
        }
        
        logging.info(f"IFCGeneratorAgent initialized with {len(self.ifc_templates)} templates")
    
    def _load_ifc_templates(self):
        """Load optimized IFC templates for common structural elements"""
        
        self.ifc_templates = {
            'wall': {
                'type': IFCElementType.WALL,
                'default_properties': [
                    IFCProperty('LoadBearing', True, 'boolean', 'Pset_WallCommon'),
                    IFCProperty('IsExternal', False, 'boolean', 'Pset_WallCommon'),
                    IFCProperty('FireRating', 'None', 'string', 'Pset_WallCommon'),
                    IFCProperty('ThermalTransmittance', 0.0, 'W/m²K', 'Pset_WallCommon')
                ],
                'geometry_type': 'ExtrudedAreaSolid',
                'base_quantities': ['Length', 'Width', 'Height', 'GrossFootprintArea', 'NetFootprintArea']
            },
            'footing': {
                'type': IFCElementType.FOOTING,
                'default_properties': [
                    IFCProperty('PredefinedType', 'STRIP_FOOTING', 'string', 'Pset_FootingCommon'),
                    IFCProperty('LoadBearing', True, 'boolean', 'Pset_FootingCommon'),
                    IFCProperty('BearingCapacity', 0.0, 'kN/m²', 'Pset_FoundationCommon'),
                    IFCProperty('DesignLoad', 0.0, 'kN', 'Pset_FoundationCommon')
                ],
                'geometry_type': 'ExtrudedAreaSolid',
                'base_quantities': ['Length', 'Width', 'Height', 'Volume', 'Weight']
            },
            'pile': {
                'type': IFCElementType.PILE,
                'default_properties': [
                    IFCProperty('PredefinedType', 'BORED', 'string', 'Pset_PileCommon'),
                    IFCProperty('Diameter', 0.0, 'mm', 'Pset_PileCommon'),
                    IFCProperty('Length', 0.0, 'mm', 'Pset_PileCommon'),
                    IFCProperty('UltimateCapacity', 0.0, 'kN', 'Pset_PileCommon'),
                    IFCProperty('WorkingLoad', 0.0, 'kN', 'Pset_PileCommon')
                ],
                'geometry_type': 'ExtrudedAreaSolid',
                'base_quantities': ['Length', 'CrossSectionArea', 'Volume']
            },
            'floodwall': {
                'type': IFCElementType.BUILDING_ELEMENT_PROXY,
                'default_properties': [
                    IFCProperty('ElementType', 'FLOODWALL', 'string', 'Pset_FloodwallCommon'),
                    IFCProperty('DesignFloodLevel', 0.0, 'm', 'Pset_FloodwallCommon'),
                    IFCProperty('FloodProtectionLevel', '100-year', 'string', 'Pset_FloodwallCommon'),
                    IFCProperty('LoadBearing', True, 'boolean', 'Pset_WallCommon'),
                    IFCProperty('SeismicDesign', False, 'boolean', 'Pset_StructuralDesign')
                ],
                'geometry_type': 'ExtrudedAreaSolid',
                'base_quantities': ['Length', 'Width', 'Height', 'Volume', 'SurfaceArea']
            }
        }
    
    def _load_ids_specifications(self):
        """Load available IDS specifications for compliance validation"""
        
        self.ids_specifications = []
        
        # Load from buildingSMART examples
        ids_dir = Path("buildingsmart_ids/Documentation/Examples")
        if ids_dir.exists():
            for ids_file in ids_dir.glob("*.ids"):
                try:
                    document = self.ids_parser.parse_ids_file(ids_file)
                    self.ids_specifications.append({
                        'file': str(ids_file),
                        'document': document,
                        'name': document.info.title,
                        'domain': self.ids_parser._classify_domain(document),
                        'complexity': self.ids_parser._calculate_complexity_score(document)
                    })
                except Exception as e:
                    logging.warning(f"Failed to load IDS specification {ids_file}: {e}")
        
        # Load from project-specific IDS files
        project_ids_dir = Path("training_data/ids_specifications")
        project_ids_dir.mkdir(exist_ok=True)
        
        for ids_file in project_ids_dir.glob("*.ids"):
            try:
                document = self.ids_parser.parse_ids_file(ids_file)
                self.ids_specifications.append({
                    'file': str(ids_file),
                    'document': document,
                    'name': document.info.title,
                    'domain': self.ids_parser._classify_domain(document),
                    'complexity': self.ids_parser._calculate_complexity_score(document)
                })
            except Exception as e:
                logging.warning(f"Failed to load project IDS specification {ids_file}: {e}")
        
        logging.info(f"Loaded {len(self.ids_specifications)} IDS specifications for validation")
    
    async def handle_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle IFC generation request"""
        
        if message.payload.get("action") == "generate_ifc":
            design_data = message.payload.get("design_data", {})
            context = message.payload.get("context", {})
            
            start_time = time.time()
            
            # Generate IFC model
            ifc_model = await self._generate_ifc_model_optimized(design_data, context)
            
            # Convert to IFC format
            ifc_content = await self._convert_to_ifc_format(ifc_model)
            
            # Validate against IDS specifications if available
            ids_validation_results = await self._validate_against_ids_specifications(ifc_content, design_data)
            
            generation_time = time.time() - start_time
            
            # Update statistics
            self.generation_stats['models_generated'] += 1
            self.generation_stats['elements_created'] += len(ifc_model.elements)
            self.generation_stats['avg_generation_time'] = (
                (self.generation_stats['avg_generation_time'] * (self.generation_stats['models_generated'] - 1) + 
                 generation_time) / self.generation_stats['models_generated']
            )
            
            return AgentMessage(
                sender=self.agent_id,
                receiver=message.sender,
                message_type=MessageType.RESPONSE,
                payload={
                    "ifc_model": ifc_model.__dict__,
                    "ifc_content": ifc_content,
                    "generation_time": generation_time,
                    "element_count": len(ifc_model.elements),
                    "model_size": len(ifc_content),
                    "ids_validation": ids_validation_results
                },
                correlation_id=message.correlation_id
            )
        
        return None
    
    async def _generate_ifc_model_optimized(self, design_data: Dict[str, Any], 
                                          context: Dict[str, Any]) -> IFCModel:
        """Generate IFC model using optimized algorithms"""
        
        # Extract design parameters
        parsed_prompt = design_data.get('parsed_prompt', {})
        file_data = design_data.get('file_data', [])
        
        # Create spatial hierarchy
        spatial_hierarchy = self._create_spatial_hierarchy_fast(parsed_prompt, context)
        
        # Generate elements based on design intent
        elements = await self._generate_elements_from_design(parsed_prompt, file_data)
        
        # Create IFC model
        ifc_model = IFCModel(
            project_name=context.get('project_name', 'Civil Infrastructure Project'),
            site_name=context.get('site_name', 'Project Site'),
            building_name=context.get('building_name', 'Infrastructure'),
            elements=elements,
            spatial_hierarchy=spatial_hierarchy,
            metadata={
                'generated_by': 'Text-to-CAD Multi-Agent System',
                'generation_time': time.time(),
                'schema_version': 'IFC4',
                'design_parameters': parsed_prompt
            }
        )
        
        return ifc_model
    
    def _create_spatial_hierarchy_fast(self, parsed_prompt: Dict[str, Any], 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Create spatial hierarchy efficiently"""
        
        return {
            'project': {
                'global_id': self._generate_guid(),
                'name': context.get('project_name', 'Civil Infrastructure Project'),
                'description': 'AI-generated civil infrastructure model'
            },
            'site': {
                'global_id': self._generate_guid(),
                'name': context.get('site_name', 'Project Site'),
                'ref_latitude': context.get('latitude', 0.0),
                'ref_longitude': context.get('longitude', 0.0),
                'ref_elevation': context.get('elevation', 0.0)
            },
            'building': {
                'global_id': self._generate_guid(),
                'name': context.get('building_name', 'Infrastructure'),
                'building_type': 'CIVIL_INFRASTRUCTURE'
            },
            'storey': {
                'global_id': self._generate_guid(),
                'name': 'Ground Level',
                'elevation': 0.0
            }
        }
    
    async def _generate_elements_from_design(self, parsed_prompt: Dict[str, Any], 
                                           file_data: List[Dict[str, Any]]) -> List[IFCElement]:
        """Generate IFC elements from design data"""
        
        elements = []
        
        # Extract parameters from parsed prompt
        parameters = parsed_prompt.get('parameters', [])
        intent = parsed_prompt.get('intent', 'simple_structure')
        materials = parsed_prompt.get('materials', [])
        
        # Generate elements based on intent
        if intent == 'simple_structure':
            elements.extend(await self._generate_simple_structure(parameters, materials))
        elif intent == 'complex_infrastructure':
            elements.extend(await self._generate_complex_infrastructure(parameters, materials, file_data))
        elif intent == 'retrofit_upgrade':
            elements.extend(await self._generate_retrofit_elements(parameters, materials, file_data))
        
        return elements
    
    async def _generate_simple_structure(self, parameters: List[Dict[str, Any]], 
                                       materials: List[str]) -> List[IFCElement]:
        """Generate simple structural elements"""
        
        elements = []
        
        # Extract dimensions
        height = self._extract_parameter_value(parameters, 'height', 3.0)
        length = self._extract_parameter_value(parameters, 'length', 10.0)
        width = self._extract_parameter_value(parameters, 'width', 0.3)
        
        # Determine primary material
        primary_material = materials[0] if materials else 'concrete'
        
        # Generate wall
        wall_element = await self._create_wall_element(
            name="Main Wall",
            length=length,
            height=height,
            thickness=width,
            material=primary_material
        )
        elements.append(wall_element)
        
        # Generate foundation
        footing_element = await self._create_footing_element(
            name="Wall Footing",
            length=length,
            width=width * 2,  # Footing wider than wall
            height=0.5,  # Default footing depth
            material=primary_material
        )
        elements.append(footing_element)
        
        return elements
    
    async def _generate_complex_infrastructure(self, parameters: List[Dict[str, Any]], 
                                             materials: List[str], 
                                             file_data: List[Dict[str, Any]]) -> List[IFCElement]:
        """Generate complex infrastructure elements"""
        
        elements = []
        
        # Extract engineering data from files
        engineering_data = self._extract_engineering_data_from_files(file_data)
        
        # Generate floodwall system
        if any('flood' in str(param).lower() for param in parameters):
            floodwall_elements = await self._generate_floodwall_system(parameters, materials, engineering_data)
            elements.extend(floodwall_elements)
        
        # Generate pile foundation system
        if any('pile' in str(param).lower() for param in parameters):
            pile_elements = await self._generate_pile_system(parameters, materials, engineering_data)
            elements.extend(pile_elements)
        
        return elements
    
    async def _generate_retrofit_elements(self, parameters: List[Dict[str, Any]], 
                                        materials: List[str], 
                                        file_data: List[Dict[str, Any]]) -> List[IFCElement]:
        """Generate retrofit and upgrade elements"""
        
        elements = []
        
        # For retrofit, create modification elements
        retrofit_wall = await self._create_wall_element(
            name="Retrofit Wall Section",
            length=self._extract_parameter_value(parameters, 'length', 5.0),
            height=self._extract_parameter_value(parameters, 'height', 2.0),
            thickness=self._extract_parameter_value(parameters, 'width', 0.2),
            material=materials[0] if materials else 'steel'
        )
        elements.append(retrofit_wall)
        
        return elements
    
    async def _generate_floodwall_system(self, parameters: List[Dict[str, Any]], 
                                       materials: List[str], 
                                       engineering_data: Dict[str, Any]) -> List[IFCElement]:
        """Generate floodwall system elements"""
        
        elements = []
        
        # Extract floodwall parameters
        height = self._extract_parameter_value(parameters, 'height', 4.2)
        length = self._extract_parameter_value(parameters, 'length', 850.0)
        thickness = self._extract_parameter_value(parameters, 'width', 0.6)
        
        # Create floodwall segments (break into manageable pieces)
        segment_length = 50.0  # 50m segments
        num_segments = int(length / segment_length)
        
        for i in range(num_segments):
            segment_name = f"Floodwall Segment {i+1}"
            
            # Create floodwall element
            floodwall = IFCElement(
                global_id=self._generate_guid(),
                element_type=IFCElementType.BUILDING_ELEMENT_PROXY,
                name=segment_name,
                geometry=self._create_wall_geometry(segment_length, height, thickness),
                properties=self._create_floodwall_properties(height, engineering_data),
                material=materials[0] if materials else 'reinforced_concrete',
                location=[i * segment_length, 0.0, 0.0]
            )
            elements.append(floodwall)
        
        return elements
    
    async def _generate_pile_system(self, parameters: List[Dict[str, Any]], 
                                  materials: List[str], 
                                  engineering_data: Dict[str, Any]) -> List[IFCElement]:
        """Generate pile foundation system"""
        
        elements = []
        
        # Extract pile parameters
        diameter = self._extract_parameter_value(parameters, 'diameter', 0.3)
        length = self._extract_parameter_value(parameters, 'length', 15.0)
        spacing = self._extract_parameter_value(parameters, 'spacing', 2.0)
        
        # Generate pile grid
        pile_count = 20  # Default pile count
        
        for i in range(pile_count):
            pile_name = f"Micropile {i+1}"
            
            # Calculate pile position
            x_pos = (i % 5) * spacing
            y_pos = (i // 5) * spacing
            
            pile = IFCElement(
                global_id=self._generate_guid(),
                element_type=IFCElementType.PILE,
                name=pile_name,
                geometry=self._create_pile_geometry(diameter, length),
                properties=self._create_pile_properties(diameter, length, engineering_data),
                material=materials[0] if materials else 'steel',
                location=[x_pos, y_pos, 0.0]
            )
            elements.append(pile)
        
        return elements
    
    async def _create_wall_element(self, name: str, length: float, height: float, 
                                 thickness: float, material: str) -> IFCElement:
        """Create optimized wall element"""
        
        # Check geometry cache first
        cache_key = f"wall_{length}_{height}_{thickness}"
        if cache_key in self.geometry_cache:
            geometry = self.geometry_cache[cache_key]
        else:
            geometry = self._create_wall_geometry(length, height, thickness)
            self.geometry_cache[cache_key] = geometry
        
        # Create properties from template
        template = self.ifc_templates['wall']
        properties = template['default_properties'].copy()
        
        # Add specific properties
        properties.extend([
            IFCProperty('Height', height, 'm', 'Qto_WallBaseQuantities'),
            IFCProperty('Length', length, 'm', 'Qto_WallBaseQuantities'),
            IFCProperty('Width', thickness, 'm', 'Qto_WallBaseQuantities'),
            IFCProperty('GrossFootprintArea', length * thickness, 'm²', 'Qto_WallBaseQuantities'),
            IFCProperty('NetSideArea', length * height, 'm²', 'Qto_WallBaseQuantities'),
            IFCProperty('GrossVolume', length * height * thickness, 'm³', 'Qto_WallBaseQuantities')
        ])
        
        return IFCElement(
            global_id=self._generate_guid(),
            element_type=IFCElementType.WALL,
            name=name,
            geometry=geometry,
            properties=properties,
            material=material,
            location=[0.0, 0.0, 0.0]
        )
    
    async def _create_footing_element(self, name: str, length: float, width: float, 
                                    height: float, material: str) -> IFCElement:
        """Create optimized footing element"""
        
        geometry = self._create_footing_geometry(length, width, height)
        
        # Create properties from template
        template = self.ifc_templates['footing']
        properties = template['default_properties'].copy()
        
        # Add specific properties
        properties.extend([
            IFCProperty('Length', length, 'm', 'Qto_FootingBaseQuantities'),
            IFCProperty('Width', width, 'm', 'Qto_FootingBaseQuantities'),
            IFCProperty('Height', height, 'm', 'Qto_FootingBaseQuantities'),
            IFCProperty('Volume', length * width * height, 'm³', 'Qto_FootingBaseQuantities'),
            IFCProperty('Weight', length * width * height * 2400, 'kg', 'Qto_FootingBaseQuantities')  # Concrete density
        ])
        
        return IFCElement(
            global_id=self._generate_guid(),
            element_type=IFCElementType.FOOTING,
            name=name,
            geometry=geometry,
            properties=properties,
            material=material,
            location=[0.0, 0.0, -height]  # Below ground level
        )
    
    def _create_wall_geometry(self, length: float, height: float, thickness: float) -> IFCGeometry:
        """Create optimized wall geometry"""
        
        # Create rectangular profile
        profile_coords = [
            [0.0, 0.0],
            [length, 0.0],
            [length, thickness],
            [0.0, thickness],
            [0.0, 0.0]  # Close the profile
        ]
        
        return IFCGeometry(
            representation_type="ExtrudedAreaSolid",
            coordinates=profile_coords,
            extrusion_direction=[0.0, 0.0, 1.0],
            extrusion_depth=height
        )
    
    def _create_footing_geometry(self, length: float, width: float, height: float) -> IFCGeometry:
        """Create optimized footing geometry"""
        
        # Create rectangular profile
        profile_coords = [
            [0.0, 0.0],
            [length, 0.0],
            [length, width],
            [0.0, width],
            [0.0, 0.0]
        ]
        
        return IFCGeometry(
            representation_type="ExtrudedAreaSolid",
            coordinates=profile_coords,
            extrusion_direction=[0.0, 0.0, 1.0],
            extrusion_depth=height
        )
    
    def _create_pile_geometry(self, diameter: float, length: float) -> IFCGeometry:
        """Create optimized pile geometry"""
        
        # Create circular profile
        radius = diameter / 2
        num_points = 16  # 16-sided polygon approximation
        profile_coords = []
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            profile_coords.append([x, y])
        
        profile_coords.append(profile_coords[0])  # Close the profile
        
        return IFCGeometry(
            representation_type="ExtrudedAreaSolid",
            coordinates=profile_coords,
            extrusion_direction=[0.0, 0.0, -1.0],  # Downward
            extrusion_depth=length
        )
    
    def _create_floodwall_properties(self, height: float, engineering_data: Dict[str, Any]) -> List[IFCProperty]:
        """Create floodwall-specific properties"""
        
        template = self.ifc_templates['floodwall']
        properties = template['default_properties'].copy()
        
        # Add calculated properties
        properties.extend([
            IFCProperty('WallHeight', height, 'm', 'Pset_FloodwallCommon'),
            IFCProperty('DesignFloodLevel', height - 0.5, 'm', 'Pset_FloodwallCommon'),
            IFCProperty('Freeboard', 0.5, 'm', 'Pset_FloodwallCommon'),
            IFCProperty('SeismicZone', engineering_data.get('seismic_zone', 'Zone 1'), 'string', 'Pset_StructuralDesign')
        ])
        
        return properties
    
    def _create_pile_properties(self, diameter: float, length: float, 
                              engineering_data: Dict[str, Any]) -> List[IFCProperty]:
        """Create pile-specific properties"""
        
        template = self.ifc_templates['pile']
        properties = template['default_properties'].copy()
        
        # Update with actual values
        for prop in properties:
            if prop.name == 'Diameter':
                prop.value = diameter * 1000  # Convert to mm
            elif prop.name == 'Length':
                prop.value = length * 1000  # Convert to mm
        
        # Add calculated properties
        cross_section_area = math.pi * (diameter / 2) ** 2
        properties.extend([
            IFCProperty('CrossSectionArea', cross_section_area, 'm²', 'Qto_PileBaseQuantities'),
            IFCProperty('Volume', cross_section_area * length, 'm³', 'Qto_PileBaseQuantities'),
            IFCProperty('SoilType', engineering_data.get('soil_type', 'Clay'), 'string', 'Pset_FoundationCommon')
        ])
        
        return properties
    
    async def _convert_to_ifc_format(self, ifc_model: IFCModel) -> str:
        """Convert IFC model to standard IFC format"""
        
        # This is a simplified IFC format generation
        # In production, would use IfcOpenShell library
        
        ifc_lines = []
        
        # IFC Header
        ifc_lines.append("ISO-10303-21;")
        ifc_lines.append("HEADER;")
        ifc_lines.append("FILE_DESCRIPTION((''), '2;1');")
        ifc_lines.append("FILE_NAME('', '', '', '', '', '', '');")
        ifc_lines.append("FILE_SCHEMA(('IFC4'));")
        ifc_lines.append("ENDSEC;")
        ifc_lines.append("DATA;")
        
        # Generate IFC entities
        entity_id = 1
        
        # Project
        ifc_lines.append(f"#{entity_id}=IFCPROJECT('{ifc_model.spatial_hierarchy['project']['global_id']}',#2,$,'{ifc_model.project_name}',$,$,$,$,#3);")
        entity_id += 1
        
        # Add simplified entities for each element
        for element in ifc_model.elements:
            ifc_lines.append(f"#{entity_id}={element.element_type.value}('{element.global_id}',#2,$,'{element.name}',$,$,$,$,$);")
            entity_id += 1
        
        # IFC Footer
        ifc_lines.append("ENDSEC;")
        ifc_lines.append("END-ISO-10303-21;")
        
        return '\n'.join(ifc_lines)
    
    def _extract_parameter_value(self, parameters: List[Dict[str, Any]], param_name: str, default: float) -> float:
        """Extract parameter value with fallback to default"""
        
        for param in parameters:
            if param.get('name', '').lower() == param_name.lower():
                return float(param.get('value', default))
        
        return default
    
    def _extract_engineering_data_from_files(self, file_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract engineering data from analyzed files"""
        
        engineering_data = {
            'seismic_zone': 'Zone 1',
            'soil_type': 'Clay',
            'design_loads': [],
            'material_properties': {}
        }
        
        for file_info in file_data:
            data = file_info.get('data', {})
            
            # Extract from Excel files
            if 'engineering_data' in data:
                eng_data = data['engineering_data']
                if 'loads' in eng_data:
                    engineering_data['design_loads'].extend(eng_data['loads'])
                if 'materials' in eng_data:
                    for material in eng_data['materials']:
                        mat_name = material.get('value', '').lower()
                        if 'concrete' in mat_name:
                            engineering_data['material_properties']['concrete'] = material
        
        return engineering_data
    
    def _generate_guid(self) -> str:
        """Generate IFC-compatible GUID"""
        return str(uuid.uuid4()).upper()
    
    async def handle_response(self, message: AgentMessage):
        """Handle response messages"""
        logging.info(f"IFC generator received response: {message.payload}")
    
    async def _validate_against_ids_specifications(self, ifc_content: str, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generated IFC content against available IDS specifications"""
        
        validation_results = {
            'validated': False,
            'total_specifications': 0,
            'passed_specifications': 0,
            'failed_specifications': 0,
            'compliance_rate': 0.0,
            'applicable_specs': [],
            'validation_details': []
        }
        
        if not self.ids_specifications:
            validation_results['message'] = "No IDS specifications available for validation"
            return validation_results
        
        try:
            # Save IFC content to temporary file for validation
            temp_ifc_path = Path("temp_validation_model.ifc")
            with open(temp_ifc_path, 'w', encoding='utf-8') as f:
                f.write(ifc_content)
            
            applicable_specs = self._find_applicable_ids_specifications(design_data)
            validation_results['applicable_specs'] = [spec['name'] for spec in applicable_specs]
            
            if not applicable_specs:
                # Use first few general specifications if no specific matches
                applicable_specs = self.ids_specifications[:3]
            
            validation_results['total_specifications'] = len(applicable_specs)
            passed_count = 0
            
            for spec_info in applicable_specs:
                spec_file = Path(spec_info['file'])
                
                try:
                    # Validate against this specification
                    result = self.ids_validator.validate_ifc_against_ids(temp_ifc_path, spec_file)
                    
                    validation_detail = {
                        'specification': spec_info['name'],
                        'passed': result.overall_passed,
                        'error_count': result.summary.get('total_errors', 0),
                        'warning_count': result.summary.get('total_warnings', 0),
                        'validated_entities': result.summary.get('validated_entities', 0)
                    }
                    
                    validation_results['validation_details'].append(validation_detail)
                    
                    if result.overall_passed:
                        passed_count += 1
                    
                except Exception as e:
                    logging.warning(f"IDS validation failed for {spec_info['name']}: {e}")
                    validation_results['validation_details'].append({
                        'specification': spec_info['name'],
                        'passed': False,
                        'error': str(e)
                    })
            
            validation_results['passed_specifications'] = passed_count
            validation_results['failed_specifications'] = len(applicable_specs) - passed_count
            validation_results['compliance_rate'] = passed_count / max(1, len(applicable_specs))
            validation_results['validated'] = True
            
            # Update statistics
            self.generation_stats['ids_validations'] += 1
            current_compliance = self.generation_stats['ids_compliance_rate']
            self.generation_stats['ids_compliance_rate'] = (
                (current_compliance * (self.generation_stats['ids_validations'] - 1) + 
                 validation_results['compliance_rate']) / self.generation_stats['ids_validations']
            )
            
            # Cleanup
            if temp_ifc_path.exists():
                temp_ifc_path.unlink()
            
            logging.info(f"IDS validation complete: {passed_count}/{len(applicable_specs)} specifications passed")
            
        except Exception as e:
            logging.error(f"IDS validation error: {e}")
            validation_results['error'] = str(e)
        
        return validation_results
    
    def _find_applicable_ids_specifications(self, design_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find IDS specifications applicable to the current design"""
        
        applicable_specs = []
        
        # Extract design characteristics
        parsed_prompt = design_data.get('parsed_prompt', {})
        intent = parsed_prompt.get('intent', '').lower()
        parameters = parsed_prompt.get('parameters', [])
        
        # Look for domain-specific specifications
        target_domains = []
        
        # Determine target domains from intent and parameters
        if any(keyword in intent for keyword in ['wall', 'building', 'architecture']):
            target_domains.append('architectural')
        if any(keyword in intent for keyword in ['beam', 'column', 'structural', 'foundation']):
            target_domains.append('structural')
        if any(keyword in intent for keyword in ['bridge', 'road', 'infrastructure']):
            target_domains.append('infrastructure_transport')
        if any(keyword in intent for keyword in ['pipe', 'duct', 'hvac', 'mep']):
            target_domains.append('mep')
        
        # If no specific domain found, use general domain
        if not target_domains:
            target_domains.append('general')
        
        # Find specifications matching target domains
        for spec in self.ids_specifications:
            if spec['domain'] in target_domains:
                applicable_specs.append(spec)
        
        # Sort by complexity (simpler first for faster validation)
        applicable_specs.sort(key=lambda x: x['complexity'])
        
        return applicable_specs[:5]  # Limit to 5 most applicable specifications
    
    def get_ids_compliance_summary(self) -> Dict[str, Any]:
        """Get summary of IDS compliance performance"""
        
        return {
            'loaded_specifications': len(self.ids_specifications),
            'total_validations': self.generation_stats['ids_validations'],
            'average_compliance_rate': self.generation_stats['ids_compliance_rate'],
            'specifications_by_domain': {
                spec['domain']: len([s for s in self.ids_specifications if s['domain'] == spec['domain']])
                for spec in self.ids_specifications
            }
        }
    
    async def handle_notification(self, message: AgentMessage):
        """Handle notification messages"""
        logging.info(f"IFC generator received notification: {message.payload}")
    
    async def handle_error(self, message: AgentMessage):
        """Handle error messages"""
        logging.error(f"IFC generator received error: {message.payload}")

# Singleton pattern
_ifc_generator_instance = None

def get_ifc_generator_agent() -> IFCGeneratorAgent:
    """Get singleton instance of IFC generator agent"""
    global _ifc_generator_instance
    if _ifc_generator_instance is None:
        _ifc_generator_instance = IFCGeneratorAgent()
    return _ifc_generator_instance