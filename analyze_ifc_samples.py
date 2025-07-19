#!/usr/bin/env python3
"""
Analyze buildingSMART sample IFC files for training validation
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.ifc_analyzer import IFCAnalyzer
from src.training.data_loader import DataLoader

def main():
    print("=== buildingSMART IFC Sample Analysis ===\n")
    
    # Initialize analyzer
    analyzer = IFCAnalyzer()
    data_loader = DataLoader("training_data")
    
    # Find all IFC reference files
    ifc_dir = Path("training_data/ifc_models/reference")
    if not ifc_dir.exists():
        print("No IFC reference files found. Run the training setup first.")
        return
    
    ifc_files = list(ifc_dir.rglob("*.ifc"))
    print(f"Found {len(ifc_files)} IFC reference files\n")
    
    analysis_results = []
    
    for ifc_file in ifc_files:
        print(f"Analyzing: {ifc_file.name}")
        
        try:
            # Comprehensive analysis
            analysis = analyzer.analyze_ifc_file(ifc_file)
            
            # Generate training metadata
            metadata = analyzer.generate_training_metadata(analysis)
            
            # Display key results
            print(f"   Domain: {analysis['engineering_classification']['primary_domain']}")
            print(f"   Complexity: {analysis['engineering_classification']['complexity_level']}")
            print(f"   Elements: {analysis['element_analysis']['total_elements']}")
            print(f"   Quality: {analysis['quality_metrics']['overall_quality']:.3f}")
            print(f"   Training Weight: {metadata['recommended_training_weight']:.2f}")
            print(f"   Tags: {', '.join(metadata['training_tags'])}")
            
            analysis_results.append({
                'file': ifc_file.name,
                'analysis': analysis,
                'metadata': metadata
            })
            print("   Analysis complete\n")
            
        except Exception as e:
            print(f"   Analysis failed: {e}\n")
    
    # Summary statistics
    print("=== Analysis Summary ===")
    
    domains = {}
    complexities = {}
    total_quality = 0
    
    for result in analysis_results:
        # Domain distribution
        domain = result['analysis']['engineering_classification']['primary_domain']
        domains[domain] = domains.get(domain, 0) + 1
        
        # Complexity distribution  
        complexity = result['analysis']['engineering_classification']['complexity_level']
        complexities[complexity] = complexities.get(complexity, 0) + 1
        
        # Average quality
        total_quality += result['analysis']['quality_metrics']['overall_quality']
    
    if analysis_results:
        avg_quality = total_quality / len(analysis_results)
        
        print(f"Analyzed {len(analysis_results)} files")
        print(f"Average Quality Score: {avg_quality:.3f}")
        print(f"Domain Distribution:")
        for domain, count in sorted(domains.items()):
            print(f"   - {domain}: {count} files")
        print(f"Complexity Distribution:")
        for complexity, count in sorted(complexities.items()):
            print(f"   - {complexity}: {count} files")
    
    # Test data loader integration
    print("\n=== Testing Data Loader Integration ===")
    
    # Load IFC examples through data loader
    ifc_examples = data_loader.load_ifc_data()
    print(f"Loaded {len(ifc_examples)} IFC examples through DataLoader")
    
    # Show examples with generated prompts
    examples_with_prompts = [ex for ex in ifc_examples if ex.source_prompt]
    print(f"{len(examples_with_prompts)} examples have auto-generated prompts:")
    
    for i, example in enumerate(examples_with_prompts[:3]):  # Show first 3
        print(f"\n{i+1}. Generated Prompt: \"{example.source_prompt}\"")
        print(f"   Elements: {example.element_count}")
        print(f"   Properties: {len(example.properties)}")
        print(f"   Spatial levels: {len([k for k, v in example.spatial_hierarchy.items() if v])}")
    
    # Save analysis results
    output_file = "ifc_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"\nAnalysis results saved to {output_file}")
    print("IFC analysis complete!")

if __name__ == "__main__":
    main()