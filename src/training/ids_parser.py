"""
IDS (Information Delivery Specification) Parser and Validator for Text-to-CAD system
"""

import xml.etree.ElementTree as ET
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

class FacetCardinality(Enum):
    """IDS Facet cardinality options"""
    REQUIRED = "required"
    OPTIONAL = "optional" 
    PROHIBITED = "prohibited"

class DataType(Enum):
    """IFC data types for property validation"""
    IFCLABEL = "IFCLABEL"
    IFCTEXT = "IFCTEXT"
    IFCIDENTIFIER = "IFCIDENTIFIER"
    IFCREAL = "IFCREAL"
    IFCINTEGER = "IFCINTEGER"
    IFCBOOLEAN = "IFCBOOLEAN"
    IFCLOGICAL = "IFCLOGICAL"
    IFCLENGTHLEARMEASURE = "IFCLENGTHLEARMEASURE"
    IFCAREAMEASURE = "IFCAREAMEASURE"
    IFCVOLUMEMEASURE = "IFCVOLUMEMEASURE"

@dataclass
class IDSValue:
    """Represents an IDS value with restrictions"""
    simple_value: Optional[str] = None
    restriction_base: Optional[str] = None
    pattern: Optional[str] = None
    enumeration: Optional[List[str]] = None
    min_inclusive: Optional[Union[int, float]] = None
    max_inclusive: Optional[Union[int, float]] = None
    min_exclusive: Optional[Union[int, float]] = None
    max_exclusive: Optional[Union[int, float]] = None
    length: Optional[int] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None

@dataclass
class EntityFacet:
    """IDS Entity facet for specifying IFC entity requirements"""
    name: IDSValue
    predefined_type: Optional[IDSValue] = None
    cardinality: FacetCardinality = FacetCardinality.REQUIRED

@dataclass
class PropertyFacet:
    """IDS Property facet for specifying property requirements"""
    property_set: IDSValue
    base_name: IDSValue
    value: Optional[IDSValue] = None
    data_type: Optional[DataType] = None
    cardinality: FacetCardinality = FacetCardinality.REQUIRED

@dataclass
class ClassificationFacet:
    """IDS Classification facet for specifying classification requirements"""
    value: Optional[IDSValue] = None
    system: Optional[IDSValue] = None
    cardinality: FacetCardinality = FacetCardinality.REQUIRED

@dataclass
class MaterialFacet:
    """IDS Material facet for specifying material requirements"""
    value: Optional[IDSValue] = None
    cardinality: FacetCardinality = FacetCardinality.REQUIRED

@dataclass
class PartOfFacet:
    """IDS PartOf facet for specifying spatial relationships"""
    entity: EntityFacet
    cardinality: FacetCardinality = FacetCardinality.REQUIRED

@dataclass
class AttributeFacet:
    """IDS Attribute facet for specifying IFC attribute requirements"""
    name: IDSValue
    value: Optional[IDSValue] = None
    cardinality: FacetCardinality = FacetCardinality.REQUIRED

@dataclass
class IDSSpecification:
    """Complete IDS specification with applicability and requirements"""
    name: str
    ifc_version: List[str]
    description: Optional[str] = None
    applicability: List[Dict[str, Any]] = None
    requirements: List[Dict[str, Any]] = None

@dataclass
class IDSInfo:
    """IDS metadata information"""
    title: str
    copyright: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    purpose: Optional[str] = None
    milestone: Optional[str] = None

@dataclass
class IDSDocument:
    """Complete IDS document structure"""
    info: IDSInfo
    specifications: List[IDSSpecification]

class IDSParser:
    """
    Parser for IDS (Information Delivery Specification) XML files
    """
    
    def __init__(self):
        self.namespace = {'ids': 'http://standards.buildingsmart.org/IDS'}
        self.supported_ifc_versions = ['IFC2X3', 'IFC4', 'IFC4X3']
        
        logging.info("IDSParser initialized")
    
    def parse_ids_file(self, ids_path: Path) -> IDSDocument:
        """Parse an IDS XML file and return structured data"""
        
        try:
            tree = ET.parse(ids_path)
            root = tree.getroot()
            
            # Parse info section
            info = self._parse_info_section(root)
            
            # Parse specifications
            specifications = self._parse_specifications_section(root)
            
            document = IDSDocument(info=info, specifications=specifications)
            
            logging.info(f"Successfully parsed IDS file: {ids_path.name} with {len(specifications)} specifications")
            return document
            
        except Exception as e:
            logging.error(f"Failed to parse IDS file {ids_path}: {e}")
            raise
    
    def _parse_info_section(self, root: ET.Element) -> IDSInfo:
        """Parse the info section of an IDS document"""
        
        info_elem = root.find('ids:info', self.namespace)
        if info_elem is None:
            raise ValueError("IDS document missing required info section")
        
        return IDSInfo(
            title=self._get_text_value(info_elem, 'ids:title'),
            copyright=self._get_text_value(info_elem, 'ids:copyright'),
            version=self._get_text_value(info_elem, 'ids:version'),
            description=self._get_text_value(info_elem, 'ids:description'),
            author=self._get_text_value(info_elem, 'ids:author'),
            date=self._get_text_value(info_elem, 'ids:date'),
            purpose=self._get_text_value(info_elem, 'ids:purpose'),
            milestone=self._get_text_value(info_elem, 'ids:milestone')
        )
    
    def _parse_specifications_section(self, root: ET.Element) -> List[IDSSpecification]:
        """Parse all specifications from an IDS document"""
        
        specifications = []
        specs_elem = root.find('ids:specifications', self.namespace)
        
        if specs_elem is not None:
            for spec_elem in specs_elem.findall('ids:specification', self.namespace):
                spec = self._parse_specification(spec_elem)
                specifications.append(spec)
        
        return specifications
    
    def _parse_specification(self, spec_elem: ET.Element) -> IDSSpecification:
        """Parse a single specification element"""
        
        name = spec_elem.get('name', 'Unnamed Specification')
        ifc_version = spec_elem.get('ifcVersion', 'IFC4').split()
        description = spec_elem.get('description')
        
        # Parse applicability
        applicability = []
        applicability_elem = spec_elem.find('ids:applicability', self.namespace)
        if applicability_elem is not None:
            applicability = self._parse_facets(applicability_elem)
        
        # Parse requirements
        requirements = []
        requirements_elem = spec_elem.find('ids:requirements', self.namespace)
        if requirements_elem is not None:
            requirements = self._parse_facets(requirements_elem)
        
        return IDSSpecification(
            name=name,
            ifc_version=ifc_version,
            description=description,
            applicability=applicability,
            requirements=requirements
        )
    
    def _parse_facets(self, parent_elem: ET.Element) -> List[Dict[str, Any]]:
        """Parse facets (entity, property, classification, etc.) from an element"""
        
        facets = []
        
        # Parse entity facets
        for entity_elem in parent_elem.findall('ids:entity', self.namespace):
            facet = self._parse_entity_facet(entity_elem)
            facets.append({'type': 'entity', 'data': facet})
        
        # Parse property facets
        for prop_elem in parent_elem.findall('ids:property', self.namespace):
            facet = self._parse_property_facet(prop_elem)
            facets.append({'type': 'property', 'data': facet})
        
        # Parse classification facets
        for class_elem in parent_elem.findall('ids:classification', self.namespace):
            facet = self._parse_classification_facet(class_elem)
            facets.append({'type': 'classification', 'data': facet})
        
        # Parse material facets
        for mat_elem in parent_elem.findall('ids:material', self.namespace):
            facet = self._parse_material_facet(mat_elem)
            facets.append({'type': 'material', 'data': facet})
        
        # Parse attribute facets
        for attr_elem in parent_elem.findall('ids:attribute', self.namespace):
            facet = self._parse_attribute_facet(attr_elem)
            facets.append({'type': 'attribute', 'data': facet})
        
        # Parse partOf facets
        for partof_elem in parent_elem.findall('ids:partOf', self.namespace):
            facet = self._parse_partof_facet(partof_elem)
            facets.append({'type': 'partOf', 'data': facet})
        
        return facets
    
    def _parse_entity_facet(self, entity_elem: ET.Element) -> EntityFacet:
        """Parse an entity facet"""
        
        name_elem = entity_elem.find('ids:name', self.namespace)
        name_value = self._parse_ids_value(name_elem) if name_elem is not None else None
        
        predefined_type_elem = entity_elem.find('ids:predefinedType', self.namespace)
        predefined_type = self._parse_ids_value(predefined_type_elem) if predefined_type_elem is not None else None
        
        cardinality = self._get_cardinality(entity_elem)
        
        return EntityFacet(
            name=name_value,
            predefined_type=predefined_type,
            cardinality=cardinality
        )
    
    def _parse_property_facet(self, prop_elem: ET.Element) -> PropertyFacet:
        """Parse a property facet"""
        
        property_set_elem = prop_elem.find('ids:propertySet', self.namespace)
        property_set = self._parse_ids_value(property_set_elem) if property_set_elem is not None else None
        
        base_name_elem = prop_elem.find('ids:baseName', self.namespace)
        base_name = self._parse_ids_value(base_name_elem) if base_name_elem is not None else None
        
        value_elem = prop_elem.find('ids:value', self.namespace)
        value = self._parse_ids_value(value_elem) if value_elem is not None else None
        
        data_type_attr = prop_elem.get('dataType')
        data_type = DataType(data_type_attr) if data_type_attr else None
        
        cardinality = self._get_cardinality(prop_elem)
        
        return PropertyFacet(
            property_set=property_set,
            base_name=base_name,
            value=value,
            data_type=data_type,
            cardinality=cardinality
        )
    
    def _parse_classification_facet(self, class_elem: ET.Element) -> ClassificationFacet:
        """Parse a classification facet"""
        
        value_elem = class_elem.find('ids:value', self.namespace)
        value = self._parse_ids_value(value_elem) if value_elem is not None else None
        
        system_elem = class_elem.find('ids:system', self.namespace)
        system = self._parse_ids_value(system_elem) if system_elem is not None else None
        
        cardinality = self._get_cardinality(class_elem)
        
        return ClassificationFacet(
            value=value,
            system=system,
            cardinality=cardinality
        )
    
    def _parse_material_facet(self, mat_elem: ET.Element) -> MaterialFacet:
        """Parse a material facet"""
        
        value_elem = mat_elem.find('ids:value', self.namespace)
        value = self._parse_ids_value(value_elem) if value_elem is not None else None
        
        cardinality = self._get_cardinality(mat_elem)
        
        return MaterialFacet(
            value=value,
            cardinality=cardinality
        )
    
    def _parse_attribute_facet(self, attr_elem: ET.Element) -> AttributeFacet:
        """Parse an attribute facet"""
        
        name_elem = attr_elem.find('ids:name', self.namespace)
        name = self._parse_ids_value(name_elem) if name_elem is not None else None
        
        value_elem = attr_elem.find('ids:value', self.namespace)
        value = self._parse_ids_value(value_elem) if value_elem is not None else None
        
        cardinality = self._get_cardinality(attr_elem)
        
        return AttributeFacet(
            name=name,
            value=value,
            cardinality=cardinality
        )
    
    def _parse_partof_facet(self, partof_elem: ET.Element) -> PartOfFacet:
        """Parse a partOf facet"""
        
        entity_elem = partof_elem.find('ids:entity', self.namespace)
        entity = self._parse_entity_facet(entity_elem) if entity_elem is not None else None
        
        cardinality = self._get_cardinality(partof_elem)
        
        return PartOfFacet(
            entity=entity,
            cardinality=cardinality
        )
    
    def _parse_ids_value(self, value_elem: ET.Element) -> IDSValue:
        """Parse an IDS value with possible restrictions"""
        
        if value_elem is None:
            return None
        
        ids_value = IDSValue()
        
        # Check for simple value
        simple_value_elem = value_elem.find('ids:simpleValue', self.namespace)
        if simple_value_elem is not None:
            ids_value.simple_value = simple_value_elem.text
            return ids_value
        
        # Check for restriction
        restriction_elem = value_elem.find('.//xs:restriction', {'xs': 'http://www.w3.org/2001/XMLSchema'})
        if restriction_elem is not None:
            ids_value.restriction_base = restriction_elem.get('base')
            
            # Parse restriction facets
            for facet in restriction_elem:
                if facet.tag.endswith('pattern'):
                    ids_value.pattern = facet.get('value')
                elif facet.tag.endswith('enumeration'):
                    if ids_value.enumeration is None:
                        ids_value.enumeration = []
                    ids_value.enumeration.append(facet.get('value'))
                elif facet.tag.endswith('minInclusive'):
                    ids_value.min_inclusive = self._convert_numeric(facet.get('value'))
                elif facet.tag.endswith('maxInclusive'):
                    ids_value.max_inclusive = self._convert_numeric(facet.get('value'))
                elif facet.tag.endswith('minExclusive'):
                    ids_value.min_exclusive = self._convert_numeric(facet.get('value'))
                elif facet.tag.endswith('maxExclusive'):
                    ids_value.max_exclusive = self._convert_numeric(facet.get('value'))
                elif facet.tag.endswith('length'):
                    ids_value.length = int(facet.get('value'))
                elif facet.tag.endswith('minLength'):
                    ids_value.min_length = int(facet.get('value'))
                elif facet.tag.endswith('maxLength'):
                    ids_value.max_length = int(facet.get('value'))
        
        return ids_value
    
    def _get_cardinality(self, elem: ET.Element) -> FacetCardinality:
        """Get the cardinality (required/optional/prohibited) of a facet"""
        
        cardinality_attr = elem.get('cardinality', 'required')
        try:
            return FacetCardinality(cardinality_attr)
        except ValueError:
            return FacetCardinality.REQUIRED
    
    def _get_text_value(self, parent: ET.Element, xpath: str) -> Optional[str]:
        """Get text value from an element, returning None if not found"""
        
        elem = parent.find(xpath, self.namespace)
        return elem.text if elem is not None else None
    
    def _convert_numeric(self, value: str) -> Union[int, float]:
        """Convert string value to appropriate numeric type"""
        
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value
    
    def validate_ids_file(self, ids_path: Path) -> Dict[str, Any]:
        """Validate an IDS file structure and content"""
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        try:
            document = self.parse_ids_file(ids_path)
            
            # Basic validation
            if not document.info.title:
                validation_results['errors'].append("Missing required title in info section")
                validation_results['valid'] = False
            
            if not document.specifications:
                validation_results['warnings'].append("No specifications found in IDS document")
            
            # Validate specifications
            for i, spec in enumerate(document.specifications):
                spec_errors = self._validate_specification(spec, i)
                validation_results['errors'].extend(spec_errors)
                if spec_errors:
                    validation_results['valid'] = False
            
            # Summary
            validation_results['summary'] = {
                'total_specifications': len(document.specifications),
                'ifc_versions': list(set([v for spec in document.specifications for v in spec.ifc_version])),
                'facet_types': self._count_facet_types(document)
            }
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Parse error: {str(e)}")
        
        return validation_results
    
    def _validate_specification(self, spec: IDSSpecification, index: int) -> List[str]:
        """Validate a single specification"""
        
        errors = []
        
        if not spec.name:
            errors.append(f"Specification {index}: Missing name")
        
        if not spec.ifc_version:
            errors.append(f"Specification {index}: Missing IFC version")
        else:
            for version in spec.ifc_version:
                if version not in self.supported_ifc_versions:
                    errors.append(f"Specification {index}: Unsupported IFC version '{version}'")
        
        if not spec.applicability:
            errors.append(f"Specification {index}: Missing applicability facets")
        
        return errors
    
    def _count_facet_types(self, document: IDSDocument) -> Dict[str, int]:
        """Count different types of facets in the document"""
        
        facet_counts = {}
        
        for spec in document.specifications:
            for facets in [spec.applicability or [], spec.requirements or []]:
                for facet in facets:
                    facet_type = facet.get('type', 'unknown')
                    facet_counts[facet_type] = facet_counts.get(facet_type, 0) + 1
        
        return facet_counts
    
    def generate_validation_summary(self, document: IDSDocument) -> Dict[str, Any]:
        """Generate a comprehensive summary of an IDS document for training purposes"""
        
        summary = {
            'document_info': {
                'title': document.info.title,
                'version': document.info.version,
                'author': document.info.author,
                'date': document.info.date,
                'purpose': document.info.purpose
            },
            'specifications_count': len(document.specifications),
            'ifc_versions': list(set([v for spec in document.specifications for v in spec.ifc_version])),
            'facet_analysis': self._analyze_facets(document),
            'complexity_score': self._calculate_complexity_score(document),
            'domain_classification': self._classify_domain(document),
            'training_potential': self._assess_training_potential(document)
        }
        
        return summary
    
    def _analyze_facets(self, document: IDSDocument) -> Dict[str, Any]:
        """Analyze facet usage patterns in the document"""
        
        facet_analysis = {
            'types': {},
            'cardinality_distribution': {},
            'complexity_indicators': {}
        }
        
        for spec in document.specifications:
            for facets in [spec.applicability or [], spec.requirements or []]:
                for facet in facets:
                    facet_type = facet.get('type', 'unknown')
                    facet_data = facet.get('data')
                    
                    # Count facet types
                    facet_analysis['types'][facet_type] = facet_analysis['types'].get(facet_type, 0) + 1
                    
                    # Count cardinality
                    if hasattr(facet_data, 'cardinality'):
                        cardinality = facet_data.cardinality.value
                        facet_analysis['cardinality_distribution'][cardinality] = \
                            facet_analysis['cardinality_distribution'].get(cardinality, 0) + 1
        
        return facet_analysis
    
    def _calculate_complexity_score(self, document: IDSDocument) -> float:
        """Calculate complexity score for the IDS document"""
        
        score = 0.0
        
        # Base score from number of specifications
        score += len(document.specifications) * 0.1
        
        # Score from facet complexity
        total_facets = 0
        complex_facets = 0
        
        for spec in document.specifications:
            for facets in [spec.applicability or [], spec.requirements or []]:
                for facet in facets:
                    total_facets += 1
                    facet_data = facet.get('data')
                    
                    # Check for complex restrictions
                    if facet.get('type') == 'property' and facet_data:
                        if (hasattr(facet_data, 'value') and facet_data.value and 
                            (facet_data.value.pattern or facet_data.value.enumeration)):
                            complex_facets += 1
        
        if total_facets > 0:
            score += (complex_facets / total_facets) * 0.5
        
        return min(1.0, score)
    
    def _classify_domain(self, document: IDSDocument) -> str:
        """Classify the engineering domain of the IDS document"""
        
        entity_patterns = {
            'architectural': ['IFCWALL', 'IFCDOOR', 'IFCWINDOW', 'IFCSPACE', 'IFCROOF'],
            'structural': ['IFCBEAM', 'IFCCOLUMN', 'IFCSLAB', 'IFCFOOTING', 'IFCPILE'],
            'mep': ['IFCDUCT', 'IFCPIPE', 'IFCPUMP', 'IFCFAN', 'IFCVALVE'],
            'infrastructure': ['IFCBRIDGE', 'IFCROAD', 'IFCRAIL', 'IFCTUNNEL']
        }
        
        domain_scores = {domain: 0 for domain in entity_patterns.keys()}
        
        for spec in document.specifications:
            for facets in [spec.applicability or [], spec.requirements or []]:
                for facet in facets:
                    if facet.get('type') == 'entity':
                        entity_data = facet.get('data')
                        if entity_data and entity_data.name:
                            entity_name = entity_data.name.simple_value or ''
                            
                            for domain, patterns in entity_patterns.items():
                                for pattern in patterns:
                                    if pattern in entity_name.upper():
                                        domain_scores[domain] += 1
        
        # Return domain with highest score
        if max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        else:
            return 'general'
    
    def _assess_training_potential(self, document: IDSDocument) -> Dict[str, Any]:
        """Assess the training potential of this IDS document"""
        
        potential = {
            'quality': 'unknown',
            'usefulness': 0.0,
            'training_weight': 1.0,
            'recommended_use': []
        }
        
        # Assess quality based on completeness
        has_requirements = any(spec.requirements for spec in document.specifications)
        has_applicability = any(spec.applicability for spec in document.specifications)
        has_documentation = bool(document.info.description or document.info.purpose)
        
        quality_score = sum([has_requirements, has_applicability, has_documentation]) / 3
        
        if quality_score >= 0.8:
            potential['quality'] = 'high'
            potential['training_weight'] = 1.5
        elif quality_score >= 0.5:
            potential['quality'] = 'medium'
            potential['training_weight'] = 1.0
        else:
            potential['quality'] = 'low'
            potential['training_weight'] = 0.5
        
        # Calculate usefulness
        total_specs = len(document.specifications)
        complex_specs = sum(1 for spec in document.specifications 
                          if len(spec.requirements or []) > 2)
        
        potential['usefulness'] = min(1.0, (total_specs * 0.2) + (complex_specs * 0.3))
        
        # Recommend usage
        if potential['quality'] == 'high':
            potential['recommended_use'].append('validation_standard')
        if potential['usefulness'] > 0.7:
            potential['recommended_use'].append('training_data')
        if total_specs > 5:
            potential['recommended_use'].append('complexity_testing')
        
        return potential