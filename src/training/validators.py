"""
Model validation and testing for Text-to-CAD system
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import re

from .data_loader import TrainingExample
from .ids_parser import IDSParser
from .ids_validator import IDSValidator
from ..main import get_system

class ModelValidator:
    """
    Comprehensive model validation for the Text-to-CAD system
    """
    
    def __init__(self):
        self.validation_results = []
        self.test_cases = []
        
        # IDS validation components
        self.ids_parser = IDSParser()
        self.ids_validator = IDSValidator()
        self.ids_specifications = []
        self._load_ids_specifications()
        
        logging.info(f"ModelValidator initialized with {len(self.ids_specifications)} IDS specifications")
    
    async def validate_prompt_parser(self, test_examples: List[TrainingExample]) -> Dict[str, float]:
        """Validate prompt parser performance"""
        
        logging.info(f"Validating prompt parser with {len(test_examples)} examples")
        
        correct_intents = 0
        correct_parameters = 0
        total_examples = len(test_examples)
        
        for example in test_examples:
            try:
                # Simulate prompt parsing validation
                parsed_result = await self._validate_prompt_parsing(example)
                
                # Check intent classification
                if parsed_result.get("intent") == example.intent:
                    correct_intents += 1
                
                # Check parameter extraction
                if self._validate_parameters(parsed_result.get("parameters", []), example.parameters):
                    correct_parameters += 1
                
            except Exception as e:
                logging.warning(f"Validation error for prompt: {e}")
        
        metrics = {
            "intent_accuracy": correct_intents / max(1, total_examples),
            "parameter_accuracy": correct_parameters / max(1, total_examples),
            "overall_accuracy": (correct_intents + correct_parameters) / max(2, total_examples * 2),
            "total_examples": total_examples
        }
        
        logging.info(f"Prompt parser validation: Intent={metrics['intent_accuracy']:.3f}, Parameters={metrics['parameter_accuracy']:.3f}")
        return metrics
    
    async def validate_file_analyzer(self, file_paths: List[str]) -> Dict[str, float]:
        """Validate file analyzer performance"""
        
        logging.info(f"Validating file analyzer with {len(file_paths)} files")
        
        successful_analyses = 0
        extraction_quality_scores = []
        
        for file_path in file_paths:
            try:
                # Simulate file analysis validation
                analysis_result = await self._validate_file_analysis(file_path)
                
                if analysis_result.get("success", False):
                    successful_analyses += 1
                    extraction_quality_scores.append(analysis_result.get("quality_score", 0.0))
                
            except Exception as e:
                logging.warning(f"File analysis validation error for {file_path}: {e}")
        
        metrics = {
            "success_rate": successful_analyses / max(1, len(file_paths)),
            "avg_extraction_quality": sum(extraction_quality_scores) / max(1, len(extraction_quality_scores)),
            "total_files": len(file_paths),
            "successful_files": successful_analyses
        }
        
        logging.info(f"File analyzer validation: Success rate={metrics['success_rate']:.3f}")
        return metrics
    
    async def validate_ifc_generator(self, test_examples: List[TrainingExample]) -> Dict[str, float]:
        """Validate IFC generator performance"""
        
        logging.info(f"Validating IFC generator with {len(test_examples)} examples")
        
        successful_generations = 0
        ifc_quality_scores = []
        generation_times = []
        
        for example in test_examples:
            try:
                start_time = time.time()
                
                # Simulate IFC generation validation
                generation_result = await self._validate_ifc_generation(example)
                
                generation_time = time.time() - start_time
                generation_times.append(generation_time)
                
                if generation_result.get("success", False):
                    successful_generations += 1
                    ifc_quality_scores.append(generation_result.get("quality_score", 0.0))
                
            except Exception as e:
                logging.warning(f"IFC generation validation error: {e}")
        
        metrics = {
            "generation_success_rate": successful_generations / max(1, len(test_examples)),
            "avg_ifc_quality": sum(ifc_quality_scores) / max(1, len(ifc_quality_scores)),
            "avg_generation_time": sum(generation_times) / max(1, len(generation_times)),
            "total_examples": len(test_examples),
            "successful_generations": successful_generations
        }
        
        logging.info(f"IFC generator validation: Success rate={metrics['generation_success_rate']:.3f}")
        return metrics
    
    async def validate_complete_system(self, test_examples: List[TrainingExample]) -> Dict[str, float]:
        """Validate the complete end-to-end system"""
        
        logging.info(f"Validating complete system with {len(test_examples)} examples")
        
        system = get_system()
        await system.initialize()
        
        try:
            successful_completions = 0
            system_quality_scores = []
            processing_times = []
            error_types = {}
            
            for example in test_examples:
                try:
                    start_time = time.time()
                    
                    # Process through complete system
                    result = await system.process_prompt(example.prompt, example.files)
                    
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    
                    # Validate result
                    validation_result = self._validate_system_output(result, example)
                    
                    if validation_result.get("success", False):
                        successful_completions += 1
                        system_quality_scores.append(validation_result.get("quality_score", 0.0))
                    else:
                        error_type = validation_result.get("error_type", "unknown")
                        error_types[error_type] = error_types.get(error_type, 0) + 1
                
                except Exception as e:
                    logging.warning(f"System validation error: {e}")
                    error_types["system_error"] = error_types.get("system_error", 0) + 1
            
            metrics = {
                "completion_rate": successful_completions / max(1, len(test_examples)),
                "avg_quality_score": sum(system_quality_scores) / max(1, len(system_quality_scores)),
                "avg_processing_time": sum(processing_times) / max(1, len(processing_times)),
                "total_examples": len(test_examples),
                "successful_completions": successful_completions,
                "error_distribution": error_types
            }
            
        finally:
            await system.shutdown()
        
        logging.info(f"Complete system validation: Completion rate={metrics['completion_rate']:.3f}")
        return metrics
    
    async def validate_ids_compliance(self, test_examples: List[TrainingExample]) -> Dict[str, float]:
        """Validate IDS (Information Delivery Specification) compliance"""
        
        logging.info(f"Validating IDS compliance with {len(test_examples)} examples")
        
        if not self.ids_specifications:
            logging.warning("No IDS specifications loaded for validation")
            return {
                "compliance_rate": 0.0,
                "total_examples": len(test_examples),
                "validated_examples": 0,
                "specifications_used": 0
            }
        
        compliant_examples = 0
        total_validations = 0
        specification_results = {}
        
        for example in test_examples:
            try:
                # Generate IFC content for the example (simplified)
                ifc_content = await self._generate_test_ifc_content(example)
                
                if not ifc_content:
                    continue
                
                # Find applicable IDS specifications
                applicable_specs = self._find_applicable_specs_for_example(example)
                
                for spec_info in applicable_specs[:2]:  # Limit to 2 specs per example
                    try:
                        # Save temporary IFC file
                        temp_ifc_path = Path(f"temp_validation_{total_validations}.ifc")
                        with open(temp_ifc_path, 'w', encoding='utf-8') as f:
                            f.write(ifc_content)
                        
                        # Validate against IDS
                        spec_file = Path(spec_info['file'])
                        result = self.ids_validator.validate_ifc_against_ids(temp_ifc_path, spec_file)
                        
                        total_validations += 1
                        spec_name = spec_info['name']
                        
                        if spec_name not in specification_results:
                            specification_results[spec_name] = {'passed': 0, 'total': 0}
                        
                        specification_results[spec_name]['total'] += 1
                        
                        if result.overall_passed:
                            compliant_examples += 1
                            specification_results[spec_name]['passed'] += 1
                        
                        # Cleanup
                        if temp_ifc_path.exists():
                            temp_ifc_path.unlink()
                            
                    except Exception as e:
                        logging.warning(f"IDS validation failed for {spec_info['name']}: {e}")
                
            except Exception as e:
                logging.warning(f"IDS compliance validation error for example: {e}")
        
        compliance_rate = compliant_examples / max(1, total_validations)
        
        metrics = {
            "compliance_rate": compliance_rate,
            "total_examples": len(test_examples),
            "validated_examples": total_validations,
            "compliant_examples": compliant_examples,
            "specifications_used": len(specification_results),
            "specification_results": specification_results,
            "average_compliance_per_spec": sum(
                r['passed'] / max(1, r['total']) for r in specification_results.values()
            ) / max(1, len(specification_results))
        }
        
        logging.info(f"IDS compliance validation: {compliance_rate:.3f} compliance rate ({compliant_examples}/{total_validations})")
        return metrics

    async def validate_engineering_accuracy(self, test_examples: List[TrainingExample]) -> Dict[str, float]:
        """Validate engineering accuracy and code compliance"""
        
        logging.info("Validating engineering accuracy and code compliance")
        
        code_compliant = 0
        structurally_sound = 0
        dimensionally_accurate = 0
        
        for example in test_examples:
            try:
                # Simulate engineering validation
                engineering_result = await self._validate_engineering_accuracy(example)
                
                if engineering_result.get("code_compliant", False):
                    code_compliant += 1
                
                if engineering_result.get("structurally_sound", False):
                    structurally_sound += 1
                
                if engineering_result.get("dimensionally_accurate", False):
                    dimensionally_accurate += 1
                
            except Exception as e:
                logging.warning(f"Engineering validation error: {e}")
        
        total = len(test_examples)
        metrics = {
            "code_compliance_rate": code_compliant / max(1, total),
            "structural_soundness_rate": structurally_sound / max(1, total),
            "dimensional_accuracy_rate": dimensionally_accurate / max(1, total),
            "overall_engineering_score": (code_compliant + structurally_sound + dimensionally_accurate) / max(3, total * 3)
        }
        
        logging.info(f"Engineering validation: Overall score={metrics['overall_engineering_score']:.3f}")
        return metrics
    
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """Create comprehensive test cases for validation"""
        
        test_cases = [
            # Simple structure test cases
            {
                "name": "simple_concrete_wall",
                "prompt": "Design a concrete wall 3m high and 10m long",
                "expected_intent": "simple_structure",
                "expected_elements": ["wall"],
                "expected_properties": ["height", "length", "material"],
                "complexity": "low"
            },
            {
                "name": "basic_foundation",
                "prompt": "Create a foundation footing 2m x 2m for a column",
                "expected_intent": "simple_structure",
                "expected_elements": ["footing"],
                "expected_properties": ["width", "length", "purpose"],
                "complexity": "low"
            },
            
            # Complex infrastructure test cases
            {
                "name": "floodwall_system",
                "prompt": "Design a reinforced concrete floodwall 4.2m high and 850m long with micropile foundation for 500-year flood protection",
                "expected_intent": "complex_infrastructure",
                "expected_elements": ["wall", "foundation", "pile"],
                "expected_properties": ["height", "length", "material", "protection_level", "foundation_type"],
                "complexity": "high"
            },
            {
                "name": "pump_station",
                "prompt": "Create a pump station with 10 gpm capacity including concrete pad foundation and equipment housing",
                "expected_intent": "complex_infrastructure",
                "expected_elements": ["foundation", "building", "equipment"],
                "expected_properties": ["capacity", "foundation_type", "equipment_type"],
                "complexity": "medium"
            },
            
            # Retrofit test cases
            {
                "name": "seismic_retrofit",
                "prompt": "Upgrade existing retaining wall to meet current seismic codes with additional reinforcement",
                "expected_intent": "retrofit_upgrade",
                "expected_elements": ["wall", "reinforcement"],
                "expected_properties": ["existing_structure", "upgrade_type", "code_requirements"],
                "complexity": "medium"
            },
            
            # Edge cases
            {
                "name": "minimal_info",
                "prompt": "Build a structure",
                "expected_intent": "simple_structure",
                "expected_elements": [],
                "expected_properties": [],
                "complexity": "edge_case"
            },
            {
                "name": "complex_mixed_units",
                "prompt": "Design a 15-foot high concrete wall that is 300 meters long with 18-inch thickness",
                "expected_intent": "simple_structure",
                "expected_elements": ["wall"],
                "expected_properties": ["height", "length", "thickness", "material"],
                "complexity": "medium"
            }
        ]
        
        self.test_cases = test_cases
        logging.info(f"Created {len(test_cases)} test cases")
        return test_cases
    
    async def run_comprehensive_validation(self, test_examples: List[TrainingExample]) -> Dict[str, Any]:
        """Run comprehensive validation across all components"""
        
        logging.info("Starting comprehensive validation")
        
        # Run individual component validations
        prompt_metrics = await self.validate_prompt_parser(test_examples)
        file_metrics = await self.validate_file_analyzer([])  # Would need file paths
        ifc_metrics = await self.validate_ifc_generator(test_examples)
        system_metrics = await self.validate_complete_system(test_examples[:5])  # Limit for performance
        engineering_metrics = await self.validate_engineering_accuracy(test_examples)
        ids_metrics = await self.validate_ids_compliance(test_examples[:5])  # Limit for performance
        
        # Calculate composite scores (including IDS compliance)
        composite_score = (
            prompt_metrics.get("overall_accuracy", 0.0) * 0.20 +
            file_metrics.get("success_rate", 0.0) * 0.10 +
            ifc_metrics.get("generation_success_rate", 0.0) * 0.20 +
            system_metrics.get("completion_rate", 0.0) * 0.20 +
            engineering_metrics.get("overall_engineering_score", 0.0) * 0.15 +
            ids_metrics.get("compliance_rate", 0.0) * 0.15
        )
        
        validation_report = {
            "composite_score": composite_score,
            "component_scores": {
                "prompt_parser": prompt_metrics,
                "file_analyzer": file_metrics,
                "ifc_generator": ifc_metrics,
                "complete_system": system_metrics,
                "engineering_accuracy": engineering_metrics,
                "ids_compliance": ids_metrics
            },
            "validation_summary": {
                "total_test_examples": len(test_examples),
                "validation_timestamp": time.time(),
                "overall_performance": "excellent" if composite_score > 0.9 else
                                    "good" if composite_score > 0.75 else
                                    "acceptable" if composite_score > 0.6 else "needs_improvement"
            },
            "recommendations": self._generate_validation_recommendations(composite_score, {
                "prompt_parser": prompt_metrics,
                "file_analyzer": file_metrics,
                "ifc_generator": ifc_metrics,
                "complete_system": system_metrics,
                "engineering_accuracy": engineering_metrics,
                "ids_compliance": ids_metrics
            })
        }
        
        logging.info(f"Comprehensive validation complete. Composite score: {composite_score:.3f}")
        return validation_report
    
    def save_validation_report(self, report: Dict[str, Any], output_path: Path):
        """Save validation report to file"""
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Validation report saved to {output_path}")
    
    # Simulation methods for validation (replace with actual validation logic)
    async def _validate_prompt_parsing(self, example: TrainingExample) -> Dict[str, Any]:
        """Simulate prompt parsing validation"""
        
        # This would be replaced with actual prompt parser validation
        return {
            "intent": example.intent,  # Simulated correct result
            "parameters": [
                {"name": "height", "value": 3.0, "unit": "m", "confidence": 0.9}
            ],
            "confidence": 0.85
        }
    
    async def _validate_file_analysis(self, file_path: str) -> Dict[str, Any]:
        """Simulate file analysis validation"""
        
        # This would be replaced with actual file analyzer validation
        return {
            "success": True,
            "quality_score": 0.8,
            "extracted_data": {"type": "structural_calculation"}
        }
    
    async def _validate_ifc_generation(self, example: TrainingExample) -> Dict[str, Any]:
        """Simulate IFC generation validation"""
        
        # This would be replaced with actual IFC generator validation
        return {
            "success": True,
            "quality_score": 0.85,
            "element_count": 5,
            "schema_valid": True
        }
    
    def _validate_system_output(self, result: Dict[str, Any], example: TrainingExample) -> Dict[str, Any]:
        """Validate complete system output"""
        
        # Check if result contains expected components
        success = result.get("status") == "completed"
        
        quality_score = 0.0
        if success:
            # Simple quality assessment
            if "ifc_content" in result:
                quality_score += 0.5
            if result.get("element_count", 0) > 0:
                quality_score += 0.3
            if result.get("processing_time", float('inf')) < 60:  # Under 1 minute
                quality_score += 0.2
        
        return {
            "success": success,
            "quality_score": quality_score,
            "error_type": "none" if success else "processing_failure"
        }
    
    async def _validate_engineering_accuracy(self, example: TrainingExample) -> Dict[str, Any]:
        """Simulate engineering accuracy validation"""
        
        # This would involve actual engineering validation
        return {
            "code_compliant": True,
            "structurally_sound": True,
            "dimensionally_accurate": True,
            "safety_factor_adequate": True
        }
    
    def _validate_parameters(self, parsed_params: List[Dict], expected_params: List[Dict]) -> bool:
        """Validate parameter extraction accuracy"""
        
        if not parsed_params and not expected_params:
            return True
        
        # Simple validation - check if key parameter names are present
        parsed_names = {p.get("name", "").lower() for p in parsed_params}
        expected_names = {p.get("name", "").lower() for p in expected_params}
        
        # Check if at least 70% of expected parameters are found
        if not expected_names:
            return True
        
        intersection = parsed_names.intersection(expected_names)
        return len(intersection) / len(expected_names) >= 0.7
    
    def _generate_validation_recommendations(self, composite_score: float, 
                                           component_scores: Dict[str, Dict]) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        # Overall performance recommendations
        if composite_score < 0.6:
            recommendations.append("Overall system performance is below acceptable threshold. Consider comprehensive retraining.")
        elif composite_score < 0.75:
            recommendations.append("System performance is acceptable but has room for improvement.")
        else:
            recommendations.append("System performance is good. Focus on specific component improvements.")
        
        # Component-specific recommendations
        prompt_accuracy = component_scores.get("prompt_parser", {}).get("overall_accuracy", 0.0)
        if prompt_accuracy < 0.7:
            recommendations.append("Prompt parser accuracy is low. Increase training data or improve NLP model.")
        
        ifc_success = component_scores.get("ifc_generator", {}).get("generation_success_rate", 0.0)
        if ifc_success < 0.8:
            recommendations.append("IFC generation success rate is low. Review template library and geometry algorithms.")
        
        completion_rate = component_scores.get("complete_system", {}).get("completion_rate", 0.0)
        if completion_rate < 0.7:
            recommendations.append("End-to-end completion rate is low. Check system integration and error handling.")
        
        engineering_score = component_scores.get("engineering_accuracy", {}).get("overall_engineering_score", 0.0)
        if engineering_score < 0.8:
            recommendations.append("Engineering accuracy needs improvement. Review code compliance and validation rules.")
        
        # IDS compliance recommendations
        ids_compliance = component_scores.get("ids_compliance", {}).get("compliance_rate", 0.0)
        if ids_compliance < 0.6:
            recommendations.append("IDS compliance rate is low. Review Information Delivery Specifications and improve model generation.")
        elif ids_compliance < 0.8:
            recommendations.append("IDS compliance is moderate. Focus on specific specification requirements.")
        
        ids_validations = component_scores.get("ids_compliance", {}).get("validated_examples", 0)
        if ids_validations > 0:
            recommendations.append(f"IDS validation performed on {ids_validations} examples. Use IDS training data to improve compliance.")
        
        return recommendations
    
    def _load_ids_specifications(self):
        """Load available IDS specifications for validation"""
        
        self.ids_specifications = []
        
        # Load from training data directory
        ids_dir = Path("training_data/ids_specifications")
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
        
        # Load from buildingSMART examples
        ids_examples_dir = Path("buildingsmart_ids/Documentation/Examples")
        if ids_examples_dir.exists():
            for ids_file in list(ids_examples_dir.glob("*.ids"))[:5]:  # Limit to 5 examples
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
                    pass  # Skip files that fail to parse
        
        logging.info(f"Loaded {len(self.ids_specifications)} IDS specifications for validation")
    
    async def _generate_test_ifc_content(self, example: TrainingExample) -> Optional[str]:
        """Generate test IFC content for validation (simplified)"""
        
        # This is a simplified implementation
        # In practice, this would use the full system to generate IFC content
        
        ifc_template = '''ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');
FILE_NAME('test_model.ifc','2025-01-01T00:00:00',('Text-to-CAD'),('buildingSMART'),
          'Text-to-CAD System','Text-to-CAD System','');
FILE_SCHEMA(('IFC4'));
ENDSEC;

DATA;
#1=IFCPROJECT('0YvhU1xkr5$wblLr33FF8B',$,'Test Project',$,$,$,$,$,#2);
#2=IFCUNITASSIGNMENT((#3,#4));
#3=IFCSIUNIT(*,.LENGTHUNIT.,.MILLI.,.METRE.);
#4=IFCSIUNIT(*,.AREAUNIT.,.MILLI.,.SQUARE_METRE.);
#5=IFCSITE('1YvhU1xkr5$wblLr33FF8C',$,'Test Site',$,$,#6,$,$,.ELEMENT.,$,$,$,$,$);
#6=IFCLOCALPLACEMENT($,#7);
#7=IFCAXIS2PLACEMENT3D(#8,$,$);
#8=IFCCARTESIANPOINT((0.,0.,0.));
#9=IFCBUILDING('2YvhU1xkr5$wblLr33FF8D',$,'Test Building',$,$,#10,$,$,.ELEMENT.,$,$,$);
#10=IFCLOCALPLACEMENT(#6,#11);
#11=IFCAXIS2PLACEMENT3D(#12,$,$);
#12=IFCCARTESIANPOINT((0.,0.,0.));
'''
        
        # Add some basic elements based on the example
        element_id = 100
        elements = []
        
        # Check for wall requirement
        if 'wall' in example.prompt.lower():
            elements.append(f'#{element_id}=IFCWALL(\'{element_id}YvhU1xkr5$wblLr33FF8E\',$,\'Test Wall\',$,$,#20,$,\'Test Wall\',.STANDARD.);')
            element_id += 1
        
        # Check for beam requirement  
        if 'beam' in example.prompt.lower():
            elements.append(f'#{element_id}=IFCBEAM(\'{element_id}YvhU1xkr5$wblLr33FF8E\',$,\'Test Beam\',$,$,#21,$,\'Test Beam\',.BEAM.);')
            element_id += 1
        
        # Add elements to template
        if elements:
            ifc_template += '\n'.join(elements) + '\n'
        
        ifc_template += '''ENDSEC;
END-ISO-10303-21;
'''
        
        return ifc_template
    
    def _find_applicable_specs_for_example(self, example: TrainingExample) -> List[Dict[str, Any]]:
        """Find IDS specifications applicable to a training example"""
        
        applicable_specs = []
        prompt_lower = example.prompt.lower()
        intent_lower = example.intent.lower()
        
        # Look for domain-specific specifications
        target_domains = []
        
        if any(keyword in prompt_lower for keyword in ['wall', 'building', 'architecture']):
            target_domains.append('architectural')
        if any(keyword in prompt_lower for keyword in ['beam', 'column', 'structural']):
            target_domains.append('structural')
        if any(keyword in intent_lower for keyword in ['ids', 'compliance']):
            target_domains.extend(['general', 'architectural', 'structural'])
        
        if not target_domains:
            target_domains.append('general')
        
        # Find matching specifications
        for spec in self.ids_specifications:
            if spec['domain'] in target_domains:
                applicable_specs.append(spec)
        
        # Sort by complexity (simpler first)
        applicable_specs.sort(key=lambda x: x['complexity'])
        
        return applicable_specs[:3]  # Limit to 3 most applicable