#!/usr/bin/env python3
"""
Test IDS integration with Text-to-CAD system
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.ids_parser import IDSParser
from src.training.ids_validator import IDSValidator

def main():
    print("=== IDS Integration Test ===\n")
    
    # Initialize parser and validator
    parser = IDSParser()
    validator = IDSValidator()
    
    # Find IDS example files
    ids_dir = Path("buildingsmart_ids/Documentation/Examples")
    if not ids_dir.exists():
        print("IDS examples directory not found. Make sure buildingSMART IDS repo is cloned.")
        return
    
    ids_files = list(ids_dir.glob("*.ids"))
    print(f"Found {len(ids_files)} IDS example files\n")
    
    # Test parsing
    parsed_documents = []
    
    for ids_file in ids_files[:3]:  # Test first 3 files
        print(f"Testing: {ids_file.name}")
        
        try:
            # Parse IDS file
            document = parser.parse_ids_file(ids_file)
            parsed_documents.append((ids_file.name, document))
            
            print(f"   Title: {document.info.title}")
            print(f"   Specifications: {len(document.specifications)}")
            
            # Generate summary
            summary = parser.generate_validation_summary(document)
            print(f"   Domain: {summary['domain_classification']}")
            print(f"   Complexity: {summary['complexity_score']:.3f}")
            
            # Validate IDS structure
            validation = parser.validate_ids_file(ids_file)
            print(f"   Valid: {validation['valid']}")
            if validation['errors']:
                print(f"   Errors: {len(validation['errors'])}")
            
            print("   Parse successful\n")
            
        except Exception as e:
            print(f"   Parse failed: {e}\n")
    
    # Test training example generation
    print("=== Training Example Generation ===")
    
    if parsed_documents:
        document_name, document = parsed_documents[0]
        print(f"Generating training examples from: {document_name}")
        
        training_examples = validator.generate_training_examples_from_ids(document)
        print(f"Generated {len(training_examples)} training examples\n")
        
        # Show first example
        if training_examples:
            example = training_examples[0]
            print("Sample Training Example:")
            print(f"   Prompt: {example['prompt']}")
            print(f"   Specification: {example['specification_name']}")
            print(f"   IFC Versions: {example['ifc_versions']}")
            print(f"   Complexity: {example['complexity']}")
            print(f"   Training Weight: {example['training_weight']:.2f}")
            
            # Show requirements
            req = example['requirements']
            if req['entities']:
                print(f"   Required Entities: {[e['name'] for e in req['entities'] if e['name']]}")
            if req['properties']:
                print(f"   Required Properties: {[p['name'] for p in req['properties'] if p['name']]}")
            print()
    
    # Test validation against IFC files
    print("=== IFC Validation Test ===")
    
    # Find sample IFC files
    ifc_dir = Path("training_data/ifc_models/reference")
    if ifc_dir.exists():
        ifc_files = list(ifc_dir.rglob("*.ifc"))
        
        if ifc_files and parsed_documents:
            ifc_file = ifc_files[0]
            ids_file = Path("buildingsmart_ids/Documentation/Examples") / (parsed_documents[0][0])
            
            print(f"Validating: {ifc_file.name}")
            print(f"Against IDS: {ids_file.name}")
            
            try:
                validation_result = validator.validate_ifc_against_ids(ifc_file, ids_file)
                
                print(f"   Overall Passed: {validation_result.overall_passed}")
                print(f"   Specifications: {validation_result.passed_specifications}/{validation_result.total_specifications} passed")
                print(f"   Validated Entities: {validation_result.summary.get('validated_entities', 0)}")
                print(f"   Total Errors: {validation_result.summary.get('total_errors', 0)}")
                print(f"   Coverage: {validation_result.summary.get('validation_coverage', 0):.1%}")
                
                # Show first few results
                for i, result in enumerate(validation_result.results[:2]):
                    print(f"   Spec {i+1} ({result.specification_name}): {'PASS' if result.passed else 'FAIL'}")
                    if result.errors:
                        print(f"     Errors: {len(result.errors)}")
                    if result.warnings:
                        print(f"     Warnings: {len(result.warnings)}")
                
                print("   Validation complete\n")
                
            except Exception as e:
                print(f"   Validation failed: {e}\n")
        else:
            print("No IFC files found for validation test\n")
    else:
        print("No IFC reference directory found for validation test\n")
    
    # Summary
    print("=== Integration Summary ===")
    print(f"IDS Parser: {'Working' if parsed_documents else 'Failed'}")
    print(f"Training Generation: {'Working' if 'training_examples' in locals() else 'Failed'}")
    print(f"IFC Validation: {'Working' if 'validation_result' in locals() else 'Failed'}")
    
    print("\nIDS integration test complete!")

if __name__ == "__main__":
    main()