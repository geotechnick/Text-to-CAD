#!/usr/bin/env python3
"""
Main training script for Text-to-CAD Multi-Agent System
"""

import asyncio
import argparse
import logging
import json
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.trainer import SystemTrainer, TrainingConfig, create_training_config
from src.training.data_loader import DataLoader
from src.training.metrics import TrainingMetrics
from src.training.validators import ModelValidator

def setup_logging(log_level: str = "INFO"):
    """Setup logging for training"""
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(description="Train Text-to-CAD Multi-Agent System")
    
    # Training modes
    parser.add_argument("--mode", type=str, default="full", 
                       choices=["full", "prompt_parser", "file_analyzer", "ifc_generator", "validate"],
                       help="Training mode")
    
    # Data options
    parser.add_argument("--data-dir", type=str, default="training_data",
                       help="Training data directory")
    parser.add_argument("--generate-synthetic", action="store_true", default=True,
                       help="Generate synthetic training data")
    parser.add_argument("--synthetic-count", type=int, default=1000,
                       help="Number of synthetic examples to generate")
    
    # Training configuration
    parser.add_argument("--config", type=str, help="Training configuration file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split ratio")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for models")
    parser.add_argument("--save-checkpoints", action="store_true", default=True,
                       help="Save training checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=5,
                       help="Checkpoint save interval (epochs)")
    
    # Performance options
    parser.add_argument("--early-stopping", action="store_true", default=True,
                       help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    # Validation options
    parser.add_argument("--validate-only", action="store_true",
                       help="Only run validation on existing model")
    parser.add_argument("--test-cases", action="store_true",
                       help="Run predefined test cases")
    
    return parser.parse_args()

async def prepare_training_data(args) -> dict:
    """Prepare training data"""
    
    logging.info("Preparing training data...")
    
    data_loader = DataLoader(args.data_dir)
    
    # Generate synthetic data if requested
    if args.generate_synthetic:
        logging.info(f"Generating {args.synthetic_count} synthetic examples...")
        synthetic_examples = data_loader.generate_synthetic_data(args.synthetic_count)
        logging.info(f"Generated {len(synthetic_examples)} synthetic examples")
    
    # Load all data
    prompt_examples = data_loader.load_prompt_data()
    file_examples = data_loader.load_file_data()
    ifc_examples = data_loader.load_ifc_data()
    
    # Print data statistics
    stats = data_loader.get_statistics()
    logging.info(f"Data statistics:\n{json.dumps(stats, indent=2)}")
    
    # Split data
    data_splits = data_loader.split_data(
        test_size=0.1,
        val_size=args.validation_split
    )
    
    return data_splits

async def create_sample_training_data():
    """Create sample training data for demonstration"""
    
    logging.info("Creating sample training data...")
    
    # Create directories
    data_dir = Path("training_data")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "prompts").mkdir(exist_ok=True)
    (data_dir / "files").mkdir(exist_ok=True)
    (data_dir / "ifc_models").mkdir(exist_ok=True)
    
    # Create sample prompts
    sample_prompts = {
        "prompts": [
            {
                "text": "Design a reinforced concrete wall 3m high and 10m long",
                "intent": "simple_structure",
                "parameters": [
                    {"name": "height", "value": 3.0, "unit": "m", "confidence": 0.95},
                    {"name": "length", "value": 10.0, "unit": "m", "confidence": 0.95},
                    {"name": "material", "value": "reinforced_concrete", "confidence": 0.9}
                ],
                "constraints": [
                    {"type": "structural", "requirement": "load_bearing"}
                ]
            },
            {
                "text": "Create a pump foundation for 10 gpm capacity",
                "intent": "simple_structure", 
                "parameters": [
                    {"name": "equipment_type", "value": "pump", "confidence": 1.0},
                    {"name": "capacity", "value": 10, "unit": "gpm", "confidence": 0.9},
                    {"name": "foundation_type", "value": "concrete_pad", "confidence": 0.8}
                ],
                "constraints": [
                    {"type": "equipment", "requirement": "vibration_isolation"}
                ]
            },
            {
                "text": "Design a floodwall 4.2m high for 500-year protection",
                "intent": "complex_infrastructure",
                "parameters": [
                    {"name": "height", "value": 4.2, "unit": "m", "confidence": 0.95},
                    {"name": "protection_level", "value": "500-year", "confidence": 0.9},
                    {"name": "structure_type", "value": "floodwall", "confidence": 1.0}
                ],
                "constraints": [
                    {"type": "flood_protection", "requirement": "design_flood_level"},
                    {"type": "safety", "requirement": "factor_of_safety_2.0"}
                ]
            }
        ]
    }
    
    # Save sample prompts
    with open(data_dir / "prompts" / "sample_prompts.json", 'w') as f:
        json.dump(sample_prompts, f, indent=2)
    
    logging.info("Sample training data created successfully")

async def train_prompt_parser_only(trainer: SystemTrainer, data_splits: dict):
    """Train only the prompt parser"""
    
    logging.info("Training Prompt Parser only...")
    
    train_data = data_splits["train"]
    validation_data = data_splits["validation"]
    
    # Train prompt parser
    metrics = await trainer.train_prompt_parser(train_data)
    
    # Validate
    validation_metrics = await trainer.validate_system(validation_data)
    
    logging.info(f"Prompt Parser training complete: {metrics}")
    return metrics

async def train_file_analyzer_only(trainer: SystemTrainer, data_splits: dict):
    """Train only the file analyzer"""
    
    logging.info("Training File Analyzer only...")
    
    # Get file examples
    file_examples = trainer.data_loader.file_examples
    
    if not file_examples:
        logging.warning("No file examples found for training")
        return {}
    
    # Train file analyzer
    metrics = await trainer.train_file_analyzer(file_examples)
    
    logging.info(f"File Analyzer training complete: {metrics}")
    return metrics

async def train_ifc_generator_only(trainer: SystemTrainer, data_splits: dict):
    """Train only the IFC generator"""
    
    logging.info("Training IFC Generator only...")
    
    train_data = data_splits["train"]
    validation_data = data_splits["validation"]
    
    # Train IFC generator
    metrics = await trainer.train_ifc_generator(train_data)
    
    # Validate
    validation_metrics = await trainer.validate_system(validation_data)
    
    logging.info(f"IFC Generator training complete: {metrics}")
    return metrics

async def validate_system_only(trainer: SystemTrainer, data_splits: dict):
    """Run validation only"""
    
    logging.info("Running system validation only...")
    
    validator = ModelValidator()
    
    # Create test cases
    test_cases = validator.create_test_cases()
    logging.info(f"Created {len(test_cases)} test cases")
    
    # Run comprehensive validation
    test_data = data_splits.get("test", data_splits.get("validation", []))
    
    if test_data:
        validation_report = await validator.run_comprehensive_validation(test_data)
        
        # Save validation report
        output_dir = Path(trainer.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        validator.save_validation_report(validation_report, output_dir / "validation_report.json")
        
        logging.info(f"Validation complete. Composite score: {validation_report['composite_score']:.3f}")
        return validation_report
    else:
        logging.warning("No test data available for validation")
        return {}

async def main():
    """Main training function"""
    
    args = parse_arguments()
    setup_logging(args.log_level)
    
    logging.info("Starting Text-to-CAD Multi-Agent Training")
    logging.info(f"Training mode: {args.mode}")
    
    try:
        # Create sample data if none exists
        if not Path(args.data_dir).exists() or not list(Path(args.data_dir).rglob("*.json")):
            await create_sample_training_data()
        
        # Create training configuration
        if args.config and Path(args.config).exists():
            config_path = args.config
        else:
            # Create default config
            config = create_training_config()
            
            # Override with command line arguments
            config["training"]["epochs"] = args.epochs
            config["training"]["batch_size"] = args.batch_size
            config["training"]["learning_rate"] = args.learning_rate
            config["training"]["validation_split"] = args.validation_split
            config["training"]["output_dir"] = args.output_dir
            config["training"]["save_checkpoints"] = args.save_checkpoints
            config["training"]["checkpoint_interval"] = args.checkpoint_interval
            config["training"]["early_stopping"] = args.early_stopping
            config["training"]["patience"] = args.patience
            
            # Save config
            config_path = "training_config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, indent=2)
            
            logging.info(f"Created training configuration: {config_path}")
        
        # Initialize trainer
        trainer = SystemTrainer(config_path)
        
        # Prepare data
        data_splits = await prepare_training_data(args)
        
        # Run training based on mode
        if args.mode == "full":
            await trainer.train_complete_system(data_splits)
            
        elif args.mode == "prompt_parser":
            await train_prompt_parser_only(trainer, data_splits)
            
        elif args.mode == "file_analyzer":
            await train_file_analyzer_only(trainer, data_splits)
            
        elif args.mode == "ifc_generator":
            await train_ifc_generator_only(trainer, data_splits)
            
        elif args.mode == "validate":
            await validate_system_only(trainer, data_splits)
        
        # Generate training plots if full training was completed
        if args.mode == "full":
            try:
                output_dir = Path(trainer.config.output_dir)
                trainer.metrics.plot_training_curves(output_dir / "plots")
                logging.info("Training plots generated")
            except Exception as e:
                logging.warning(f"Could not generate plots: {e}")
        
        logging.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    # Fix Windows multiprocessing
    import multiprocessing as mp
    mp.freeze_support()
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)