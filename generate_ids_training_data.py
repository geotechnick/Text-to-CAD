#!/usr/bin/env python3
"""
Generate IDS-based training data for Text-to-CAD system
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.ids_parser import IDSParser
from src.training.ids_validator import IDSValidator

def main():
    print("=== Generating IDS-Based Training Data ===\n")
    
    # Initialize parser and validator
    parser = IDSParser()
    validator = IDSValidator()
    
    # Find all IDS files
    ids_files = []
    
    # Add buildingSMART examples
    ids_dir = Path("buildingsmart_ids/Documentation/Examples")
    if ids_dir.exists():
        ids_files.extend(list(ids_dir.glob("*.ids")))
    
    print(f"Found {len(ids_files)} IDS files to process\n")
    
    # Generate training examples from each IDS file
    all_training_examples = []
    ids_summaries = []
    
    for ids_file in ids_files:
        print(f"Processing: {ids_file.name}")
        
        try:
            # Parse IDS document
            document = parser.parse_ids_file(ids_file)
            
            # Generate validation summary
            summary = parser.generate_validation_summary(document)
            ids_summaries.append({
                'file': ids_file.name,
                'title': document.info.title,
                'specifications': len(document.specifications),
                'domain': summary['domain_classification'],
                'complexity': summary['complexity_score'],
                'ifc_versions': summary['ifc_versions'],
                'facet_analysis': summary['facet_analysis']
            })
            
            # Generate training examples
            training_examples = validator.generate_training_examples_from_ids(document)
            
            # Add metadata
            for example in training_examples:
                example['source_ids_file'] = ids_file.name
                example['source_domain'] = summary['domain_classification']
                example['ids_complexity'] = summary['complexity_score']
            
            all_training_examples.extend(training_examples)
            
            print(f"   Generated {len(training_examples)} training examples")
            print(f"   Domain: {summary['domain_classification']}")
            print(f"   Complexity: {summary['complexity_score']:.3f}")
            
        except Exception as e:
            print(f"   Failed to process: {e}")
        
        print()
    
    # Create training data directory structure
    training_dir = Path("training_data")
    training_dir.mkdir(exist_ok=True)
    (training_dir / "prompts").mkdir(exist_ok=True)
    (training_dir / "ids_specifications").mkdir(exist_ok=True)
    
    # Save IDS-based training prompts
    ids_prompts = {
        "prompts": []
    }
    
    for example in all_training_examples:
        # Convert to training format
        training_prompt = {
            "text": example['prompt'],
            "intent": "ids_compliance",
            "parameters": [
                {
                    "name": "ids_specification",
                    "value": example['specification_name'],
                    "confidence": 1.0
                },
                {
                    "name": "ifc_version",
                    "value": example['ifc_versions'][0] if example['ifc_versions'] else "IFC4",
                    "confidence": 1.0
                },
                {
                    "name": "complexity",
                    "value": example['complexity'],
                    "confidence": 1.0
                }
            ],
            "constraints": [
                {
                    "type": "ids_compliance",
                    "requirement": f"Must comply with {example['specification_name']} specification"
                }
            ],
            "metadata": {
                "source_ids_file": example['source_ids_file'],
                "domain": example['source_domain'],
                "training_weight": example['training_weight'],
                "expected_structure": example['expected_structure'],
                "requirements": example['requirements']
            }
        }
        
        # Add entity requirements as parameters
        for entity_req in example['requirements'].get('entities', []):
            if entity_req.get('name'):
                training_prompt['parameters'].append({
                    "name": "required_entity",
                    "value": entity_req['name'],
                    "confidence": 0.9
                })
        
        # Add property requirements as parameters
        for prop_req in example['requirements'].get('properties', []):
            if prop_req.get('name'):
                training_prompt['parameters'].append({
                    "name": "required_property",
                    "value": prop_req['name'],
                    "confidence": 0.8
                })
        
        ids_prompts["prompts"].append(training_prompt)
    
    # Save IDS training prompts
    ids_prompts_file = training_dir / "prompts" / "ids_compliance_prompts.json"
    with open(ids_prompts_file, 'w') as f:
        json.dump(ids_prompts, f, indent=2)
    
    print(f"Saved {len(ids_prompts['prompts'])} IDS-based training prompts to {ids_prompts_file}")
    
    # Generate domain-specific prompts
    domain_prompts = {}
    for example in all_training_examples:
        domain = example['source_domain']
        if domain not in domain_prompts:
            domain_prompts[domain] = []
        domain_prompts[domain].append(example)
    
    # Save domain-specific prompt files
    for domain, examples in domain_prompts.items():
        if len(examples) >= 3:  # Only save domains with enough examples
            domain_file = training_dir / "prompts" / f"ids_{domain}_prompts.json"
            
            domain_data = {
                "prompts": [
                    {
                        "text": ex['prompt'],
                        "intent": f"{domain}_with_ids_compliance",
                        "parameters": [
                            {"name": "domain", "value": domain, "confidence": 1.0},
                            {"name": "specification", "value": ex['specification_name'], "confidence": 1.0}
                        ],
                        "constraints": [
                            {"type": "ids_compliance", "requirement": f"Must meet {ex['specification_name']} requirements"}
                        ],
                        "metadata": ex
                    }
                    for ex in examples
                ]
            }
            
            with open(domain_file, 'w') as f:
                json.dump(domain_data, f, indent=2)
            
            print(f"Saved {len(examples)} {domain} domain prompts to {domain_file}")
    
    # Copy selected IDS files to training directory for validation
    ids_spec_dir = training_dir / "ids_specifications"
    
    # Copy a representative set of IDS files
    representative_files = []
    domains_covered = set()
    
    for summary in ids_summaries:
        domain = summary['domain']
        if domain not in domains_covered and summary['specifications'] > 0:
            representative_files.append(summary['file'])
            domains_covered.add(domain)
    
    # Copy the files
    for filename in representative_files[:5]:  # Limit to 5 representative files
        source_file = ids_dir / filename
        dest_file = ids_spec_dir / filename
        
        if source_file.exists():
            import shutil
            shutil.copy2(source_file, dest_file)
            print(f"Copied {filename} to training IDS specifications")
    
    # Generate summary report
    summary_report = {
        "generation_summary": {
            "total_ids_files_processed": len(ids_files),
            "total_training_examples": len(all_training_examples),
            "domains_covered": list(set(ex['source_domain'] for ex in all_training_examples)),
            "complexity_distribution": {
                "simple": len([ex for ex in all_training_examples if ex['complexity'] == 'simple']),
                "moderate": len([ex for ex in all_training_examples if ex['complexity'] == 'moderate']),
                "complex": len([ex for ex in all_training_examples if ex['complexity'] == 'complex'])
            },
            "ifc_versions_covered": list(set(v for ex in all_training_examples for v in ex['ifc_versions']))
        },
        "ids_files_summary": ids_summaries,
        "training_data_files": {
            "main_prompts": str(ids_prompts_file),
            "domain_specific": [str(f) for f in training_dir.glob("prompts/ids_*_prompts.json")],
            "ids_specifications": [str(f) for f in ids_spec_dir.glob("*.ids")]
        },
        "recommendations": [
            "Use IDS compliance prompts for training IFC generator validation",
            "Domain-specific prompts can be used for specialized training",
            "IDS specifications in training directory can be used for validation during training",
            "Training weight indicates relative importance for curriculum learning"
        ]
    }
    
    # Save summary report
    summary_file = training_dir / "ids_training_data_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"\nSummary report saved to {summary_file}")
    
    # Final summary
    print("\n=== IDS Training Data Generation Complete ===")
    print(f"Total Training Examples: {len(all_training_examples)}")
    print(f"Domains Covered: {', '.join(summary_report['generation_summary']['domains_covered'])}")
    print(f"Complexity Distribution:")
    for complexity, count in summary_report['generation_summary']['complexity_distribution'].items():
        print(f"  {complexity}: {count} examples")
    print(f"IFC Versions: {', '.join(summary_report['generation_summary']['ifc_versions_covered'])}")
    
    print(f"\nFiles Generated:")
    print(f"  Main prompts: {ids_prompts_file}")
    print(f"  Domain prompts: {len(domain_prompts)} files")
    print(f"  IDS specifications: {len(representative_files)} files")
    print(f"  Summary report: {summary_file}")
    
    print("\nIDS training data is ready for use in the Text-to-CAD training system!")

if __name__ == "__main__":
    main()