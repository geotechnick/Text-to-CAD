"""
IDS Validator for validating IFC models against Information Delivery Specifications
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from .ids_parser import IDSParser, IDSDocument, FacetCardinality, DataType

@dataclass
class ValidationResult:
    """Result of IDS validation"""
    specification_name: str
    passed: bool
    errors: List[str]
    warnings: List[str]
    entity_count: int
    validated_entities: List[str]

@dataclass
class OverallValidationResult:
    """Overall validation result for multiple specifications"""
    ifc_file: str
    total_specifications: int
    passed_specifications: int
    failed_specifications: int
    overall_passed: bool
    results: List[ValidationResult]
    summary: Dict[str, Any]

class IDSValidator:
    """
    Validator for checking IFC models against IDS specifications
    """
    
    def __init__(self):
        self.parser = IDSParser()
        self.validation_cache = {}
        
        logging.info("IDSValidator initialized")
    
    def validate_ifc_against_ids(self, ifc_file_path: Path, ids_file_path: Path) -> OverallValidationResult:
        """Validate an IFC file against an IDS specification"""
        
        try:
            # Parse IDS document
            ids_document = self.parser.parse_ids_file(ids_file_path)
            
            # Read IFC content
            with open(ifc_file_path, 'r', encoding='utf-8') as f:
                ifc_content = f.read()
            
            # Validate each specification
            results = []
            for spec in ids_document.specifications:
                result = self._validate_specification(ifc_content, spec)
                results.append(result)
            
            # Calculate overall result
            passed_count = sum(1 for r in results if r.passed)
            failed_count = len(results) - passed_count
            overall_passed = failed_count == 0
            
            # Generate summary
            summary = self._generate_validation_summary(results, ifc_content)
            
            overall_result = OverallValidationResult(
                ifc_file=str(ifc_file_path),
                total_specifications=len(results),
                passed_specifications=passed_count,
                failed_specifications=failed_count,
                overall_passed=overall_passed,
                results=results,
                summary=summary
            )
            
            logging.info(f"Validated {ifc_file_path.name} against {ids_file_path.name}: "
                        f"{passed_count}/{len(results)} specifications passed")
            
            return overall_result
            
        except Exception as e:
            logging.error(f"Validation failed: {e}")
            # Return failed result
            return OverallValidationResult(
                ifc_file=str(ifc_file_path),
                total_specifications=0,
                passed_specifications=0,
                failed_specifications=1,
                overall_passed=False,
                results=[],
                summary={'error': str(e)}
            )
    
    def _validate_specification(self, ifc_content: str, specification) -> ValidationResult:
        """Validate IFC content against a single specification"""
        
        errors = []
        warnings = []
        validated_entities = []
        entity_count = 0
        
        try:
            # Find applicable entities based on applicability facets
            applicable_entities = self._find_applicable_entities(ifc_content, specification.applicability or [])
            entity_count = len(applicable_entities)
            
            if not applicable_entities:
                if specification.applicability:
                    warnings.append("No entities found matching applicability criteria")
                else:
                    warnings.append("No applicability criteria specified")
            
            # Validate requirements for each applicable entity
            for entity_line in applicable_entities:
                entity_id = self._extract_entity_id(entity_line)
                validated_entities.append(entity_id)
                
                # Validate requirements
                requirement_errors = self._validate_requirements(ifc_content, entity_line, specification.requirements or [])
                errors.extend([f"Entity {entity_id}: {err}" for err in requirement_errors])
            
            passed = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            passed = False
        
        return ValidationResult(
            specification_name=specification.name,
            passed=passed,
            errors=errors,
            warnings=warnings,
            entity_count=entity_count,
            validated_entities=validated_entities
        )
    
    def _find_applicable_entities(self, ifc_content: str, applicability_facets: List[Dict]) -> List[str]:
        """Find IFC entities that match applicability criteria"""
        
        applicable_entities = []
        
        if not applicability_facets:
            return applicable_entities
        
        # Get all entity lines from IFC content
        entity_lines = self._extract_entity_lines(ifc_content)
        
        for entity_line in entity_lines:
            if self._entity_matches_applicability(entity_line, applicability_facets):
                applicable_entities.append(entity_line)
        
        return applicable_entities
    
    def _entity_matches_applicability(self, entity_line: str, applicability_facets: List[Dict]) -> bool:
        """Check if an entity line matches all applicability criteria"""
        
        for facet in applicability_facets:
            facet_type = facet.get('type')
            facet_data = facet.get('data')
            
            if facet_type == 'entity':
                if not self._validate_entity_facet(entity_line, facet_data):
                    return False
            elif facet_type == 'property':
                if not self._validate_property_facet_applicability(entity_line, facet_data):
                    return False
            elif facet_type == 'classification':
                if not self._validate_classification_facet_applicability(entity_line, facet_data):
                    return False
            # Add other facet types as needed
        
        return True
    
    def _validate_requirements(self, ifc_content: str, entity_line: str, requirement_facets: List[Dict]) -> List[str]:
        """Validate requirements for a specific entity"""
        
        errors = []
        
        for facet in requirement_facets:
            facet_type = facet.get('type')
            facet_data = facet.get('data')
            
            if facet_type == 'property':
                property_errors = self._validate_property_requirement(ifc_content, entity_line, facet_data)
                errors.extend(property_errors)
            elif facet_type == 'classification':
                classification_errors = self._validate_classification_requirement(ifc_content, entity_line, facet_data)
                errors.extend(classification_errors)
            elif facet_type == 'material':
                material_errors = self._validate_material_requirement(ifc_content, entity_line, facet_data)
                errors.extend(material_errors)
            elif facet_type == 'attribute':
                attribute_errors = self._validate_attribute_requirement(entity_line, facet_data)
                errors.extend(attribute_errors)
            # Add other requirement types as needed
        
        return errors
    
    def _validate_entity_facet(self, entity_line: str, entity_facet) -> bool:
        """Validate entity facet against entity line"""
        
        if not entity_facet or not entity_facet.name:
            return True
        
        # Extract entity type from line
        entity_type = self._extract_entity_type(entity_line)
        
        # Check entity name
        if not self._validate_ids_value(entity_type, entity_facet.name):
            return False
        
        # Check predefined type if specified
        if entity_facet.predefined_type:
            predefined_type = self._extract_predefined_type(entity_line)
            if not self._validate_ids_value(predefined_type, entity_facet.predefined_type):
                return False
        
        return True
    
    def _validate_property_facet_applicability(self, entity_line: str, property_facet) -> bool:
        """Validate property facet for applicability (simplified)"""
        # For applicability, we might just check if the entity could have the property
        # This is a simplified implementation
        return True
    
    def _validate_classification_facet_applicability(self, entity_line: str, classification_facet) -> bool:
        """Validate classification facet for applicability (simplified)"""
        # For applicability, we might just check if the entity could have classifications
        # This is a simplified implementation
        return True
    
    def _validate_property_requirement(self, ifc_content: str, entity_line: str, property_facet) -> List[str]:
        """Validate property requirements for an entity"""
        
        errors = []
        
        if not property_facet:
            return errors
        
        entity_id = self._extract_entity_id(entity_line)
        
        # Find property sets for this entity
        property_sets = self._find_entity_property_sets(ifc_content, entity_id)
        
        # Check if required property set exists
        if property_facet.property_set:
            pset_name = property_facet.property_set.simple_value
            matching_psets = [ps for ps in property_sets if pset_name and pset_name in ps]
            
            if not matching_psets and property_facet.cardinality == FacetCardinality.REQUIRED:
                errors.append(f"Required property set '{pset_name}' not found")
                return errors
        
        # Check specific property if specified
        if property_facet.base_name:
            prop_name = property_facet.base_name.simple_value
            found_property = False
            
            for pset in property_sets:
                if self._property_exists_in_pset(ifc_content, pset, prop_name):
                    found_property = True
                    
                    # Validate property value if specified
                    if property_facet.value:
                        prop_value = self._get_property_value(ifc_content, pset, prop_name)
                        if not self._validate_ids_value(prop_value, property_facet.value):
                            errors.append(f"Property '{prop_name}' value '{prop_value}' does not match requirement")
                    break
            
            if not found_property and property_facet.cardinality == FacetCardinality.REQUIRED:
                errors.append(f"Required property '{prop_name}' not found")
        
        return errors
    
    def _validate_classification_requirement(self, ifc_content: str, entity_line: str, classification_facet) -> List[str]:
        """Validate classification requirements for an entity"""
        
        errors = []
        
        if not classification_facet:
            return errors
        
        entity_id = self._extract_entity_id(entity_line)
        
        # Find classifications for this entity
        classifications = self._find_entity_classifications(ifc_content, entity_id)
        
        # Check system requirement
        if classification_facet.system:
            system_name = classification_facet.system.simple_value
            matching_classifications = [c for c in classifications if system_name and system_name in c]
            
            if not matching_classifications and classification_facet.cardinality == FacetCardinality.REQUIRED:
                errors.append(f"Required classification system '{system_name}' not found")
                return errors
        
        # Check value requirement
        if classification_facet.value:
            value_name = classification_facet.value.simple_value
            found_value = False
            
            for classification in classifications:
                if value_name and value_name in classification:
                    found_value = True
                    break
            
            if not found_value and classification_facet.cardinality == FacetCardinality.REQUIRED:
                errors.append(f"Required classification value '{value_name}' not found")
        
        return errors
    
    def _validate_material_requirement(self, ifc_content: str, entity_line: str, material_facet) -> List[str]:
        """Validate material requirements for an entity"""
        
        errors = []
        
        if not material_facet or not material_facet.value:
            return errors
        
        entity_id = self._extract_entity_id(entity_line)
        
        # Find materials for this entity
        materials = self._find_entity_materials(ifc_content, entity_id)
        
        # Check material value
        material_name = material_facet.value.simple_value
        found_material = False
        
        for material in materials:
            if self._validate_ids_value(material, material_facet.value):
                found_material = True
                break
        
        if not found_material and material_facet.cardinality == FacetCardinality.REQUIRED:
            errors.append(f"Required material '{material_name}' not found")
        
        return errors
    
    def _validate_attribute_requirement(self, entity_line: str, attribute_facet) -> List[str]:
        """Validate attribute requirements for an entity"""
        
        errors = []
        
        if not attribute_facet or not attribute_facet.name:
            return errors
        
        attribute_name = attribute_facet.name.simple_value
        
        # Extract attribute value from entity line
        attribute_value = self._extract_attribute_value(entity_line, attribute_name)
        
        if attribute_value is None and attribute_facet.cardinality == FacetCardinality.REQUIRED:
            errors.append(f"Required attribute '{attribute_name}' not found")
        elif attribute_value is not None and attribute_facet.value:
            if not self._validate_ids_value(str(attribute_value), attribute_facet.value):
                errors.append(f"Attribute '{attribute_name}' value '{attribute_value}' does not match requirement")
        
        return errors
    
    def _validate_ids_value(self, actual_value: str, ids_value) -> bool:
        """Validate an actual value against an IDS value specification"""
        
        if not ids_value:
            return True
        
        if not actual_value:
            return False
        
        # Simple value check
        if ids_value.simple_value:
            return actual_value == ids_value.simple_value
        
        # Pattern check
        if ids_value.pattern:
            try:
                return bool(re.match(ids_value.pattern, actual_value))
            except re.error:
                return False
        
        # Enumeration check
        if ids_value.enumeration:
            return actual_value in ids_value.enumeration
        
        # Numeric range checks
        try:
            numeric_value = float(actual_value)
            
            if ids_value.min_inclusive is not None and numeric_value < ids_value.min_inclusive:
                return False
            if ids_value.max_inclusive is not None and numeric_value > ids_value.max_inclusive:
                return False
            if ids_value.min_exclusive is not None and numeric_value <= ids_value.min_exclusive:
                return False
            if ids_value.max_exclusive is not None and numeric_value >= ids_value.max_exclusive:
                return False
        except ValueError:
            pass  # Not a numeric value
        
        # Length checks
        if ids_value.length is not None and len(actual_value) != ids_value.length:
            return False
        if ids_value.min_length is not None and len(actual_value) < ids_value.min_length:
            return False
        if ids_value.max_length is not None and len(actual_value) > ids_value.max_length:
            return False
        
        return True
    
    # Helper methods for IFC parsing (simplified implementations)
    
    def _extract_entity_lines(self, ifc_content: str) -> List[str]:
        """Extract entity lines from IFC content"""
        lines = []
        for line in ifc_content.split('\n'):
            line = line.strip()
            if line.startswith('#') and '=' in line and 'IFC' in line.upper():
                lines.append(line)
        return lines
    
    def _extract_entity_id(self, entity_line: str) -> str:
        """Extract entity ID from entity line"""
        match = re.match(r'#(\d+)', entity_line)
        return match.group(1) if match else ''
    
    def _extract_entity_type(self, entity_line: str) -> str:
        """Extract entity type from entity line"""
        match = re.search(r'(IFC\w+)\(', entity_line)
        return match.group(1) if match else ''
    
    def _extract_predefined_type(self, entity_line: str) -> str:
        """Extract predefined type from entity line (simplified)"""
        # This is a simplified implementation
        # In reality, predefined type extraction depends on the specific IFC entity
        if 'PREDEFINEDTYPE' in entity_line.upper():
            match = re.search(r'\.(\w+)\.', entity_line)
            return match.group(1) if match else ''
        return ''
    
    def _extract_attribute_value(self, entity_line: str, attribute_name: str) -> Optional[str]:
        """Extract attribute value from entity line (simplified)"""
        # This is a very simplified implementation
        # In reality, attribute extraction requires proper IFC parsing
        return None
    
    def _find_entity_property_sets(self, ifc_content: str, entity_id: str) -> List[str]:
        """Find property sets associated with an entity"""
        property_sets = []
        
        # Look for IFCRELDEFINESBYPROPERTIES relationships
        for line in ifc_content.split('\n'):
            if 'IFCRELDEFINESBYPROPERTIES' in line and f'#{entity_id}' in line:
                # Extract property set reference (simplified)
                pset_match = re.search(r'#(\d+)', line.split('IFCRELDEFINESBYPROPERTIES')[1])
                if pset_match:
                    pset_id = pset_match.group(1)
                    property_sets.append(pset_id)
        
        return property_sets
    
    def _property_exists_in_pset(self, ifc_content: str, pset_id: str, prop_name: str) -> bool:
        """Check if a property exists in a property set"""
        # Find the property set line
        for line in ifc_content.split('\n'):
            if line.startswith(f'#{pset_id}=') and 'IFCPROPERTYSET' in line:
                # Check if property name exists in the property set (simplified)
                return prop_name.upper() in line.upper()
        return False
    
    def _get_property_value(self, ifc_content: str, pset_id: str, prop_name: str) -> Optional[str]:
        """Get the value of a property from a property set (simplified)"""
        # This would require more sophisticated IFC parsing
        return None
    
    def _find_entity_classifications(self, ifc_content: str, entity_id: str) -> List[str]:
        """Find classifications associated with an entity"""
        classifications = []
        
        # Look for classification relationships (simplified)
        for line in ifc_content.split('\n'):
            if 'IFCRELASSOCIATESCLASSIFICATION' in line and f'#{entity_id}' in line:
                classifications.append(line)
        
        return classifications
    
    def _find_entity_materials(self, ifc_content: str, entity_id: str) -> List[str]:
        """Find materials associated with an entity"""
        materials = []
        
        # Look for material relationships (simplified)
        for line in ifc_content.split('\n'):
            if 'IFCRELASSOCIATESMATERIAL' in line and f'#{entity_id}' in line:
                # Extract material name (simplified)
                if 'IFCMATERIAL' in line:
                    material_match = re.search(r"'([^']+)'", line)
                    if material_match:
                        materials.append(material_match.group(1))
        
        return materials
    
    def _generate_validation_summary(self, results: List[ValidationResult], ifc_content: str) -> Dict[str, Any]:
        """Generate summary of validation results"""
        
        total_entities = len(self._extract_entity_lines(ifc_content))
        validated_entities = sum(r.entity_count for r in results)
        total_errors = sum(len(r.errors) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)
        
        return {
            'total_entities_in_file': total_entities,
            'validated_entities': validated_entities,
            'validation_coverage': validated_entities / max(1, total_entities),
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'error_rate': total_errors / max(1, validated_entities),
            'specifications_summary': {
                'passed': [r.specification_name for r in results if r.passed],
                'failed': [r.specification_name for r in results if not r.passed]
            }
        }
    
    def generate_training_examples_from_ids(self, ids_document: IDSDocument) -> List[Dict[str, Any]]:
        """Generate training examples from IDS specifications"""
        
        training_examples = []
        
        for spec in ids_document.specifications:
            example = self._specification_to_training_example(spec)
            training_examples.append(example)
        
        return training_examples
    
    def _specification_to_training_example(self, specification) -> Dict[str, Any]:
        """Convert an IDS specification to a training example"""
        
        # Generate prompt from specification
        prompt = self._generate_prompt_from_specification(specification)
        
        # Extract requirements for training
        requirements = self._extract_training_requirements(specification)
        
        # Generate expected IFC structure
        expected_structure = self._generate_expected_ifc_structure(specification)
        
        return {
            'prompt': prompt,
            'specification_name': specification.name,
            'ifc_versions': specification.ifc_version,
            'requirements': requirements,
            'expected_structure': expected_structure,
            'training_weight': self._calculate_training_weight(specification),
            'complexity': self._assess_specification_complexity(specification)
        }
    
    def _generate_prompt_from_specification(self, specification) -> str:
        """Generate a natural language prompt from IDS specification"""
        
        prompt_parts = []
        
        # Start with specification name/description
        if specification.description:
            prompt_parts.append(f"Create a design that meets the following requirements: {specification.description}")
        else:
            prompt_parts.append(f"Design elements according to the '{specification.name}' specification")
        
        # Add entity requirements from applicability
        if specification.applicability:
            entity_requirements = []
            for facet in specification.applicability:
                if facet.get('type') == 'entity':
                    entity_data = facet.get('data')
                    if entity_data and entity_data.name:
                        entity_name = entity_data.name.simple_value or 'elements'
                        entity_name = entity_name.replace('IFC', '').lower()
                        if entity_data.predefined_type:
                            ptype = entity_data.predefined_type.simple_value or ''
                            entity_requirements.append(f"{entity_name} of type {ptype}")
                        else:
                            entity_requirements.append(entity_name)
            
            if entity_requirements:
                prompt_parts.append(f"Include {', '.join(entity_requirements)}")
        
        # Add property requirements
        if specification.requirements:
            property_requirements = []
            for facet in specification.requirements:
                if facet.get('type') == 'property':
                    prop_data = facet.get('data')
                    if prop_data and prop_data.base_name:
                        prop_name = prop_data.base_name.simple_value or 'property'
                        if prop_data.value and prop_data.value.simple_value:
                            property_requirements.append(f"{prop_name} = {prop_data.value.simple_value}")
                        else:
                            property_requirements.append(f"with {prop_name} specified")
            
            if property_requirements:
                prompt_parts.append(f"Ensure {', '.join(property_requirements)}")
        
        return '. '.join(prompt_parts) + '.'
    
    def _extract_training_requirements(self, specification) -> Dict[str, Any]:
        """Extract structured requirements for training"""
        
        requirements = {
            'entities': [],
            'properties': [],
            'classifications': [],
            'materials': [],
            'attributes': []
        }
        
        # Process all facets
        for facets in [specification.applicability or [], specification.requirements or []]:
            for facet in facets:
                facet_type = facet.get('type')
                facet_data = facet.get('data')
                
                if facet_type == 'entity' and facet_data:
                    requirements['entities'].append({
                        'name': facet_data.name.simple_value if facet_data.name else None,
                        'predefined_type': facet_data.predefined_type.simple_value if facet_data.predefined_type else None,
                        'cardinality': facet_data.cardinality.value
                    })
                elif facet_type == 'property' and facet_data:
                    requirements['properties'].append({
                        'property_set': facet_data.property_set.simple_value if facet_data.property_set else None,
                        'name': facet_data.base_name.simple_value if facet_data.base_name else None,
                        'value': facet_data.value.simple_value if facet_data.value else None,
                        'data_type': facet_data.data_type.value if facet_data.data_type else None,
                        'cardinality': facet_data.cardinality.value
                    })
                # Add other facet types as needed
        
        return requirements
    
    def _generate_expected_ifc_structure(self, specification) -> Dict[str, Any]:
        """Generate expected IFC structure for validation"""
        
        structure = {
            'required_entities': [],
            'required_properties': [],
            'spatial_requirements': [],
            'relationship_requirements': []
        }
        
        # Extract from specification facets
        if specification.applicability:
            for facet in specification.applicability:
                if facet.get('type') == 'entity':
                    entity_data = facet.get('data')
                    if entity_data and entity_data.name:
                        structure['required_entities'].append(entity_data.name.simple_value)
        
        return structure
    
    def _calculate_training_weight(self, specification) -> float:
        """Calculate training weight for specification"""
        
        weight = 1.0
        
        # Increase weight for complex specifications
        total_facets = len(specification.applicability or []) + len(specification.requirements or [])
        if total_facets > 5:
            weight *= 1.5
        
        # Increase weight for specifications with requirements
        if specification.requirements:
            weight *= 1.2
        
        return min(2.0, weight)
    
    def _assess_specification_complexity(self, specification) -> str:
        """Assess complexity level of specification"""
        
        total_facets = len(specification.applicability or []) + len(specification.requirements or [])
        
        if total_facets <= 2:
            return 'simple'
        elif total_facets <= 5:
            return 'moderate'
        else:
            return 'complex'