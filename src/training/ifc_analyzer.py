"""
IFC Analysis tools for training validation and quality assessment
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

class IFCAnalyzer:
    """
    Comprehensive IFC file analyzer for training validation
    """
    
    def __init__(self):
        self.element_counts = {}
        self.spatial_structure = {}
        self.properties = {}
        self.validation_results = {}
        
        logging.info("IFCAnalyzer initialized")
    
    def analyze_ifc_file(self, ifc_path: Path) -> Dict[str, Any]:
        """Comprehensive analysis of an IFC file"""
        
        try:
            with open(ifc_path, 'r', encoding='utf-8') as f:
                ifc_content = f.read()
        except Exception as e:
            logging.error(f"Failed to read IFC file {ifc_path}: {e}")
            return {}
        
        analysis = {
            'file_path': str(ifc_path),
            'file_size': ifc_path.stat().st_size,
            'element_analysis': self._analyze_elements(ifc_content),
            'spatial_analysis': self._analyze_spatial_structure(ifc_content),
            'property_analysis': self._analyze_properties(ifc_content),
            'geometry_analysis': self._analyze_geometry(ifc_content),
            'schema_validation': self._validate_schema(ifc_content),
            'engineering_classification': self._classify_engineering_domain(ifc_content, ifc_path),
            'quality_metrics': self._calculate_quality_metrics(ifc_content)
        }
        
        logging.info(f"Analyzed IFC file: {ifc_path.name} - {analysis['element_analysis']['total_elements']} elements")
        return analysis
    
    def _analyze_elements(self, ifc_content: str) -> Dict[str, Any]:
        """Analyze IFC elements in the file"""
        
        # Comprehensive element types for civil engineering
        element_types = {
            # Structural Elements
            'IFCWALL': 0, 'IFCBEAM': 0, 'IFCCOLUMN': 0, 'IFCSLAB': 0,
            'IFCFOOTING': 0, 'IFCPILE': 0, 'IFCREINFORCINGELEMENT': 0,
            
            # Building Elements
            'IFCDOOR': 0, 'IFCWINDOW': 0, 'IFCROOF': 0, 'IFCSTAIR': 0,
            'IFCRAMP': 0, 'IFCRAILING': 0,
            
            # Infrastructure Elements
            'IFCBRIDGE': 0, 'IFCROAD': 0, 'IFCRAIL': 0, 'IFCTUNNEL': 0,
            'IFCPAVEMENT': 0, 'IFCKERB': 0, 'IFCSIGN': 0,
            
            # MEP Elements
            'IFCDUCT': 0, 'IFCPIPE': 0, 'IFCPUMP': 0, 'IFCFAN': 0,
            'IFCBOILER': 0, 'IFCCHILLER': 0, 'IFCVALVE': 0,
            'IFCFLOWCONTROLLER': 0, 'IFCFLOWTERMINAL': 0,
            
            # Spaces and Zones
            'IFCSPACE': 0, 'IFCZONE': 0, 'IFCBUILDING': 0, 'IFCSITE': 0,
            'IFCBUILDINGSTOREY': 0,
            
            # Other Elements
            'IFCFURNISHINGELEMENT': 0, 'IFCDISTRIBUTIONELEMENT': 0,
            'IFCPROXY': 0, 'IFCBUILDINGELEMENTPROXY': 0
        }
        
        # Count elements
        for element_type in element_types:
            element_types[element_type] = ifc_content.count(element_type)
        
        # Calculate totals
        total_elements = sum(element_types.values())
        structural_elements = (element_types['IFCWALL'] + element_types['IFCBEAM'] + 
                             element_types['IFCCOLUMN'] + element_types['IFCSLAB'] +
                             element_types['IFCFOOTING'] + element_types['IFCPILE'])
        
        infrastructure_elements = (element_types['IFCBRIDGE'] + element_types['IFCROAD'] + 
                                 element_types['IFCRAIL'] + element_types['IFCTUNNEL'])
        
        mep_elements = (element_types['IFCDUCT'] + element_types['IFCPIPE'] + 
                       element_types['IFCPUMP'] + element_types['IFCFAN'])
        
        return {
            'element_counts': element_types,
            'total_elements': total_elements,
            'structural_elements': structural_elements,
            'infrastructure_elements': infrastructure_elements,
            'mep_elements': mep_elements,
            'element_distribution': {
                'structural_percentage': (structural_elements / max(1, total_elements)) * 100,
                'infrastructure_percentage': (infrastructure_elements / max(1, total_elements)) * 100,
                'mep_percentage': (mep_elements / max(1, total_elements)) * 100
            }
        }
    
    def _analyze_spatial_structure(self, ifc_content: str) -> Dict[str, Any]:
        """Analyze spatial hierarchy and relationships"""
        
        spatial_elements = {
            'projects': len(re.findall(r'IFCPROJECT\(', ifc_content)),
            'sites': len(re.findall(r'IFCSITE\(', ifc_content)),
            'buildings': len(re.findall(r'IFCBUILDING\(', ifc_content)),
            'storeys': len(re.findall(r'IFCBUILDINGSTOREY\(', ifc_content)),
            'spaces': len(re.findall(r'IFCSPACE\(', ifc_content)),
            'zones': len(re.findall(r'IFCZONE\(', ifc_content))
        }
        
        # Analyze spatial relationships
        rel_aggregates = len(re.findall(r'IFCRELAGGREGATES\(', ifc_content))
        rel_contains = len(re.findall(r'IFCRELCONTAINEDINSPATIALSTRUCTURE\(', ifc_content))
        
        return {
            'spatial_elements': spatial_elements,
            'relationships': {
                'aggregates': rel_aggregates,
                'spatial_containment': rel_contains
            },
            'hierarchy_depth': self._calculate_hierarchy_depth(spatial_elements),
            'spatial_complexity': self._calculate_spatial_complexity(spatial_elements, rel_aggregates + rel_contains)
        }
    
    def _analyze_properties(self, ifc_content: str) -> Dict[str, Any]:
        """Analyze property sets and quantities"""
        
        property_sets = len(re.findall(r'IFCPROPERTYSET\(', ifc_content))
        element_quantities = len(re.findall(r'IFCELEMENTQUANTITY\(', ifc_content))
        single_properties = len(re.findall(r'IFCPROPERTYSINGLEVALUE\(', ifc_content))
        
        # Common property set names for civil engineering
        common_psets = [
            'Pset_WallCommon', 'Pset_BeamCommon', 'Pset_ColumnCommon',
            'Pset_SlabCommon', 'Pset_FootingCommon', 'Pset_PileCommon',
            'Pset_BridgeCommon', 'Pset_RoadCommon', 'Pset_RailCommon'
        ]
        
        pset_coverage = {}
        for pset in common_psets:
            pset_coverage[pset] = ifc_content.count(pset)
        
        return {
            'property_sets': property_sets,
            'element_quantities': element_quantities,
            'single_properties': single_properties,
            'common_psets_coverage': pset_coverage,
            'property_density': single_properties / max(1, property_sets)
        }
    
    def _analyze_geometry(self, ifc_content: str) -> Dict[str, Any]:
        """Analyze geometric representations"""
        
        geometry_types = {
            'swept_solids': len(re.findall(r'IFCEXTRUDEDAREASOLID\(', ifc_content)),
            'brep_geometry': len(re.findall(r'IFCFACETEDBREP\(', ifc_content)),
            'mesh_geometry': len(re.findall(r'IFCTRIANGULATEDFACESET\(', ifc_content)),
            'curve_geometry': len(re.findall(r'IFCCOMPOSITECURVE\(', ifc_content)),
            'points': len(re.findall(r'IFCCARTESIANPOINT\(', ifc_content)),
            'directions': len(re.findall(r'IFCDIRECTION\(', ifc_content))
        }
        
        # Calculate geometry complexity
        total_geometry = sum(geometry_types.values())
        
        return {
            'geometry_types': geometry_types,
            'total_geometry_objects': total_geometry,
            'geometry_complexity': self._calculate_geometry_complexity(geometry_types),
            'coordinate_system_analysis': self._analyze_coordinate_systems(ifc_content)
        }
    
    def _validate_schema(self, ifc_content: str) -> Dict[str, Any]:
        """Basic schema validation"""
        
        # Check for IFC schema version
        schema_match = re.search(r'FILE_SCHEMA\s*\(\s*\(\s*[\'"]([^\'"]+)[\'"]', ifc_content)
        schema_version = schema_match.group(1) if schema_match else 'Unknown'
        
        # Basic validation checks
        validation_checks = {
            'has_header': 'ISO-10303-21' in ifc_content[:200],
            'has_schema': 'FILE_SCHEMA' in ifc_content,
            'has_project': 'IFCPROJECT(' in ifc_content,
            'proper_ending': 'END-ISO-10303-21' in ifc_content[-100:],
            'entity_consistency': self._check_entity_consistency(ifc_content)
        }
        
        validation_score = sum(validation_checks.values()) / len(validation_checks)
        
        return {
            'schema_version': schema_version,
            'validation_checks': validation_checks,
            'validation_score': validation_score,
            'schema_compliance': validation_score >= 0.8
        }
    
    def _classify_engineering_domain(self, ifc_content: str, file_path: Path) -> Dict[str, Any]:
        """Classify the engineering domain and project type"""
        
        # Analyze filename
        filename_lower = file_path.name.lower()
        
        # Domain indicators
        domain_scores = {
            'building_architecture': 0,
            'building_structural': 0,
            'building_mep': 0,
            'infrastructure_transport': 0,
            'infrastructure_utilities': 0,
            'civil_works': 0
        }
        
        # Filename analysis
        if 'architecture' in filename_lower:
            domain_scores['building_architecture'] += 3
        if 'structural' in filename_lower:
            domain_scores['building_structural'] += 3
        if 'hvac' in filename_lower or 'mep' in filename_lower:
            domain_scores['building_mep'] += 3
        if 'bridge' in filename_lower or 'road' in filename_lower or 'rail' in filename_lower:
            domain_scores['infrastructure_transport'] += 3
        if 'plumbing' in filename_lower or 'utilities' in filename_lower:
            domain_scores['infrastructure_utilities'] += 3
        
        # Content analysis
        wall_count = ifc_content.count('IFCWALL')
        beam_count = ifc_content.count('IFCBEAM')
        column_count = ifc_content.count('IFCCOLUMN')
        bridge_count = ifc_content.count('IFCBRIDGE')
        road_count = ifc_content.count('IFCROAD')
        duct_count = ifc_content.count('IFCDUCT')
        pipe_count = ifc_content.count('IFCPIPE')
        
        if wall_count > 5:
            domain_scores['building_architecture'] += 2
        if beam_count > 0 and column_count > 0:
            domain_scores['building_structural'] += 2
        if duct_count > 0 or pipe_count > 0:
            domain_scores['building_mep'] += 2
        if bridge_count > 0 or road_count > 0:
            domain_scores['infrastructure_transport'] += 2
        
        # Determine primary domain
        primary_domain = max(domain_scores, key=domain_scores.get)
        confidence = domain_scores[primary_domain] / max(1, sum(domain_scores.values()))
        
        return {
            'domain_scores': domain_scores,
            'primary_domain': primary_domain,
            'confidence': confidence,
            'complexity_level': self._assess_complexity_level(ifc_content)
        }
    
    def _calculate_quality_metrics(self, ifc_content: str) -> Dict[str, float]:
        """Calculate quality metrics for the IFC file"""
        
        # Completeness metrics
        total_entities = len(re.findall(r'#\d+\s*=', ifc_content))
        geometric_entities = (ifc_content.count('IFCEXTRUDEDAREASOLID') + 
                            ifc_content.count('IFCFACETEDBREP') +
                            ifc_content.count('IFCTRIANGULATEDFACESET'))
        
        completeness_score = min(1.0, geometric_entities / max(1, total_entities * 0.1))
        
        # Consistency metrics
        consistency_score = self._check_entity_consistency(ifc_content)
        
        # Richness metrics (properties and relationships)
        property_sets = ifc_content.count('IFCPROPERTYSET')
        relationships = (ifc_content.count('IFCRELAGGREGATES') + 
                        ifc_content.count('IFCRELCONTAINEDINSPATIALSTRUCTURE'))
        
        richness_score = min(1.0, (property_sets + relationships) / max(1, total_entities * 0.05))
        
        # Overall quality score
        quality_score = (completeness_score * 0.4 + consistency_score * 0.3 + richness_score * 0.3)
        
        return {
            'completeness_score': completeness_score,
            'consistency_score': consistency_score,
            'richness_score': richness_score,
            'overall_quality': quality_score,
            'entity_count': total_entities,
            'geometric_coverage': geometric_entities / max(1, total_entities)
        }
    
    def _calculate_hierarchy_depth(self, spatial_elements: Dict[str, int]) -> int:
        """Calculate spatial hierarchy depth"""
        depth = 0
        if spatial_elements['projects'] > 0:
            depth += 1
        if spatial_elements['sites'] > 0:
            depth += 1
        if spatial_elements['buildings'] > 0:
            depth += 1
        if spatial_elements['storeys'] > 0:
            depth += 1
        if spatial_elements['spaces'] > 0:
            depth += 1
        
        return depth
    
    def _calculate_spatial_complexity(self, spatial_elements: Dict[str, int], relationships: int) -> float:
        """Calculate spatial complexity score"""
        total_spatial = sum(spatial_elements.values())
        if total_spatial == 0:
            return 0.0
        
        # Complexity based on variety and relationships
        variety_score = len([x for x in spatial_elements.values() if x > 0]) / len(spatial_elements)
        relationship_score = min(1.0, relationships / max(1, total_spatial))
        
        return (variety_score + relationship_score) / 2
    
    def _calculate_geometry_complexity(self, geometry_types: Dict[str, int]) -> float:
        """Calculate geometry complexity score"""
        total_geometry = sum(geometry_types.values())
        if total_geometry == 0:
            return 0.0
        
        # Weight different geometry types by complexity
        complexity_weights = {
            'points': 0.1,
            'directions': 0.1,
            'curve_geometry': 0.3,
            'swept_solids': 0.5,
            'brep_geometry': 0.8,
            'mesh_geometry': 1.0
        }
        
        weighted_complexity = 0
        for geom_type, count in geometry_types.items():
            weight = complexity_weights.get(geom_type, 0.5)
            weighted_complexity += count * weight
        
        return min(1.0, weighted_complexity / max(1, total_geometry))
    
    def _analyze_coordinate_systems(self, ifc_content: str) -> Dict[str, Any]:
        """Analyze coordinate systems and units"""
        
        # Extract unit information
        unit_assignments = len(re.findall(r'IFCUNITASSIGNMENT\(', ifc_content))
        si_units = len(re.findall(r'IFCSIUNIT\(', ifc_content))
        
        # Look for common units
        has_metric = any(unit in ifc_content for unit in ['METRE', 'MILLIMETRE', 'KILOGRAM'])
        has_imperial = any(unit in ifc_content for unit in ['FOOT', 'INCH', 'POUND'])
        
        return {
            'unit_assignments': unit_assignments,
            'si_units': si_units,
            'uses_metric': has_metric,
            'uses_imperial': has_imperial,
            'coordinate_systems': len(re.findall(r'IFCAXIS2PLACEMENT3D\(', ifc_content))
        }
    
    def _check_entity_consistency(self, ifc_content: str) -> float:
        """Check basic entity consistency"""
        
        # Count entity references vs definitions
        entity_definitions = len(re.findall(r'#\d+\s*=', ifc_content))
        entity_references = len(re.findall(r'#\d+', ifc_content))
        
        if entity_definitions == 0:
            return 0.0
        
        # Basic consistency check
        consistency_ratio = min(1.0, entity_definitions / max(1, entity_references * 0.1))
        
        # Check for obvious errors
        has_unresolved_refs = '$' in ifc_content
        has_proper_structure = 'ENDSEC;' in ifc_content and 'DATA;' in ifc_content
        
        penalty = 0
        if has_unresolved_refs:
            penalty += 0.2
        if not has_proper_structure:
            penalty += 0.3
        
        return max(0.0, consistency_ratio - penalty)
    
    def _assess_complexity_level(self, ifc_content: str) -> str:
        """Assess overall complexity level"""
        
        total_entities = len(re.findall(r'#\d+\s*=', ifc_content))
        element_count = (ifc_content.count('IFCWALL') + ifc_content.count('IFCBEAM') + 
                        ifc_content.count('IFCCOLUMN') + ifc_content.count('IFCSLAB'))
        
        if total_entities < 100 and element_count < 10:
            return 'simple'
        elif total_entities < 1000 and element_count < 100:
            return 'moderate'
        elif total_entities < 10000 and element_count < 1000:
            return 'complex'
        else:
            return 'very_complex'
    
    def generate_training_metadata(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for training from IFC analysis"""
        
        metadata = {
            'domain': analysis['engineering_classification']['primary_domain'],
            'complexity': analysis['engineering_classification']['complexity_level'],
            'element_count': analysis['element_analysis']['total_elements'],
            'quality_score': analysis['quality_metrics']['overall_quality'],
            'spatial_hierarchy': analysis['spatial_analysis']['spatial_elements'],
            'recommended_training_weight': self._calculate_training_weight(analysis),
            'suitable_for_validation': analysis['quality_metrics']['overall_quality'] > 0.7,
            'training_tags': self._generate_training_tags(analysis)
        }
        
        return metadata
    
    def _calculate_training_weight(self, analysis: Dict[str, Any]) -> float:
        """Calculate recommended training weight for this example"""
        
        quality = analysis['quality_metrics']['overall_quality']
        complexity = analysis['engineering_classification']['complexity_level']
        
        # Base weight on quality
        weight = quality
        
        # Adjust for complexity
        complexity_multipliers = {
            'simple': 1.0,
            'moderate': 1.2,
            'complex': 1.5,
            'very_complex': 2.0
        }
        
        weight *= complexity_multipliers.get(complexity, 1.0)
        
        return min(2.0, weight)  # Cap at 2.0
    
    def _generate_training_tags(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate training tags based on analysis"""
        
        tags = []
        
        # Domain tags
        domain = analysis['engineering_classification']['primary_domain']
        tags.append(domain)
        
        # Complexity tags
        complexity = analysis['engineering_classification']['complexity_level']
        tags.append(f"complexity_{complexity}")
        
        # Element type tags
        elements = analysis['element_analysis']['element_counts']
        if elements['IFCWALL'] > 0:
            tags.append('walls')
        if elements['IFCBEAM'] > 0 or elements['IFCCOLUMN'] > 0:
            tags.append('structural_frame')
        if elements['IFCBRIDGE'] > 0:
            tags.append('bridge')
        if elements['IFCROAD'] > 0:
            tags.append('road')
        if elements['IFCDUCT'] > 0 or elements['IFCPIPE'] > 0:
            tags.append('mep_systems')
        
        # Quality tags
        quality = analysis['quality_metrics']['overall_quality']
        if quality > 0.8:
            tags.append('high_quality')
        elif quality > 0.6:
            tags.append('medium_quality')
        else:
            tags.append('low_quality')
        
        return tags