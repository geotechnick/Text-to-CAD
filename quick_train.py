#!/usr/bin/env python3
"""
Quick training script for Text-to-CAD system - simplified version for testing
"""

import asyncio
import logging
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.data_loader import DataLoader
from src.training.trainer import SystemTrainer

# Configure simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

async def quick_train():
    """Quick training demonstration"""
    
    logging.info("=== Text-to-CAD Quick Training Demo ===")
    
    try:
        # Step 1: Create sample data
        logging.info("Step 1: Creating sample training data...")
        
        data_dir = Path("training_data")
        data_dir.mkdir(exist_ok=True)
        (data_dir / "prompts").mkdir(exist_ok=True)
        
        # Create simple training prompts
        sample_prompts = {
            "prompts": [
                {
                    "text": "Design a concrete wall 3m high and 10m long",
                    "intent": "simple_structure",
                    "parameters": [
                        {"name": "height", "value": 3.0, "unit": "m", "confidence": 0.95},
                        {"name": "length", "value": 10.0, "unit": "m", "confidence": 0.95},
                        {"name": "material", "value": "concrete", "confidence": 0.9}
                    ],
                    "constraints": []
                },
                {
                    "text": "Create a pump foundation for 10 gpm capacity",
                    "intent": "simple_structure",
                    "parameters": [
                        {"name": "capacity", "value": 10, "unit": "gpm", "confidence": 0.9},
                        {"name": "equipment_type", "value": "pump", "confidence": 1.0}
                    ],
                    "constraints": []
                },
                {
                    "text": "Build a retaining wall 5 feet high",
                    "intent": "simple_structure", 
                    "parameters": [
                        {"name": "height", "value": 5.0, "unit": "ft", "confidence": 0.9},
                        {"name": "structure_type", "value": "retaining_wall", "confidence": 1.0}
                    ],
                    "constraints": []
                }
            ]
        }
        
        # Save sample prompts
        with open(data_dir / "prompts" / "quick_train_prompts.json", 'w') as f:
            json.dump(sample_prompts, f, indent=2)
        
        logging.info(f"Created {len(sample_prompts['prompts'])} sample prompts")
        
        # Step 2: Load data
        logging.info("Step 2: Loading training data...")
        
        data_loader = DataLoader(str(data_dir))
        
        # Load prompt data
        prompt_examples = data_loader.load_prompt_data()
        logging.info(f"Loaded {len(prompt_examples)} prompt examples")
        
        # Generate some synthetic data
        synthetic_examples = data_loader.generate_synthetic_data(20)  # Small number for demo
        logging.info(f"Generated {len(synthetic_examples)} synthetic examples")
        
        # Combine data
        all_examples = prompt_examples + synthetic_examples
        
        # Split data
        data_splits = data_loader.split_data(test_size=0.2, val_size=0.2)
        logging.info(f"Data splits: Train={len(data_splits['train'])}, Val={len(data_splits['validation'])}, Test={len(data_splits['test'])}")
        
        # Step 3: Initialize trainer with simple config
        logging.info("Step 3: Initializing trainer...")
        
        # Create simple config file
        simple_config = {
            "training": {
                "epochs": 3,  # Very short for demo
                "batch_size": 2,
                "learning_rate": 0.001,
                "validation_split": 0.2,
                "output_dir": "quick_models",
                "save_checkpoints": False,
                "early_stopping": False,
                "log_interval": 1
            }
        }
        
        config_path = "quick_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(simple_config, f)
        
        trainer = SystemTrainer(config_path)
        
        # Step 4: Quick training demonstration
        logging.info("Step 4: Running quick training demo...")
        
        # Train just the prompt parser (fastest component)
        logging.info("Training Prompt Parser...")
        prompt_metrics = await trainer.train_prompt_parser(data_splits["train"])
        logging.info(f"Prompt Parser Results: {prompt_metrics}")
        
        # Test IFC generation on a small sample
        logging.info("Testing IFC Generator...")
        ifc_metrics = await trainer.train_ifc_generator(data_splits["train"][:2])  # Just 2 examples
        logging.info(f"IFC Generator Results: {ifc_metrics}")
        
        # Quick validation
        logging.info("Running validation...")
        validation_metrics = await trainer.validate_system(data_splits["validation"][:2])  # Just 2 examples
        logging.info(f"Validation Results: {validation_metrics}")
        
        # Step 5: Summary
        logging.info("=== Training Demo Complete ===")
        logging.info("Summary:")
        logging.info(f"  - Prompt Parser Accuracy: {prompt_metrics.get('accuracy', 0):.3f}")
        logging.info(f"  - IFC Generation Rate: {ifc_metrics.get('generation_rate', 0):.3f}")
        logging.info(f"  - Validation Score: {validation_metrics.get('composite_score', 0):.3f}")
        
        # Save results
        results = {
            "prompt_parser": prompt_metrics,
            "ifc_generator": ifc_metrics,
            "validation": validation_metrics,
            "data_stats": {
                "total_examples": len(all_examples),
                "train_examples": len(data_splits["train"]),
                "validation_examples": len(data_splits["validation"])
            }
        }
        
        with open("quick_train_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info("Results saved to quick_train_results.json")
        logging.info("Quick training demo completed successfully!")
        
        return True
        
    except Exception as e:
        logging.error(f"Quick training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_individual_components():
    """Test individual components separately"""
    
    logging.info("=== Testing Individual Components ===")
    
    try:
        # Test data loader
        logging.info("Testing DataLoader...")
        data_loader = DataLoader("training_data")
        
        # Generate a few synthetic examples
        synthetic_examples = data_loader.generate_synthetic_data(5)
        logging.info(f"Generated {len(synthetic_examples)} synthetic examples")
        
        # Print a sample
        if synthetic_examples:
            sample = synthetic_examples[0]
            logging.info(f"Sample prompt: {sample.prompt}")
            logging.info(f"Sample intent: {sample.intent}")
            logging.info(f"Sample parameters: {sample.parameters}")
        
        logging.info("DataLoader test completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick training for Text-to-CAD")
    parser.add_argument("--test-components", action="store_true", 
                       help="Test individual components only")
    
    args = parser.parse_args()
    
    # Fix Windows multiprocessing
    import multiprocessing as mp
    mp.freeze_support()
    
    if args.test_components:
        success = asyncio.run(test_individual_components())
    else:
        success = asyncio.run(quick_train())
    
    if success:
        logging.info("All tests completed successfully!")
        return 0
    else:
        logging.error("Tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())