"""
Main trainer for the Text-to-CAD Multi-Agent System
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import yaml

from .data_loader import DataLoader, TrainingExample
from .metrics import TrainingMetrics
from .validators import ModelValidator
from ..main import get_system
from ..agents.prompt_parser_agent import get_prompt_parser_agent
from ..agents.file_analyzer_agent import get_file_analyzer_agent
from ..agents.ifc_generator_agent import get_ifc_generator_agent

@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 0.001
    validation_split: float = 0.2
    test_split: float = 0.1
    save_checkpoints: bool = True
    checkpoint_interval: int = 5
    early_stopping: bool = True
    patience: int = 3
    output_dir: str = "models"
    log_interval: int = 10

@dataclass
class AgentConfig:
    max_workers: int = 4
    cache_size: int = 100
    enable_augmentation: bool = True
    augmentation_factor: int = 2

class SystemTrainer:
    """
    Comprehensive trainer for the Text-to-CAD multi-agent system
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.data_loader = DataLoader()
        self.metrics = TrainingMetrics()
        self.validator = ModelValidator()
        
        # Create output directories
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_validation_score = 0.0
        self.patience_counter = 0
        
        logging.info(f"SystemTrainer initialized with config: {self.config}")
    
    def _load_config(self, config_path: Optional[str]) -> TrainingConfig:
        """Load training configuration"""
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Extract training config
            training_config = config_dict.get('training', {})
            return TrainingConfig(**training_config)
        else:
            return TrainingConfig()
    
    async def prepare_data(self, generate_synthetic: bool = True, synthetic_count: int = 1000):
        """Prepare training data"""
        
        logging.info("Preparing training data...")
        
        # Load existing data
        prompt_examples = self.data_loader.load_prompt_data()
        file_examples = self.data_loader.load_file_data()
        ifc_examples = self.data_loader.load_ifc_data()
        
        # Generate synthetic data if requested
        if generate_synthetic:
            synthetic_examples = self.data_loader.generate_synthetic_data(synthetic_count)
            prompt_examples.extend(synthetic_examples)
        
        # Augment data
        if len(prompt_examples) > 0:
            augmented_examples = self.data_loader.augment_data(
                prompt_examples, 
                self.config.batch_size // 2
            )
            prompt_examples = augmented_examples
        
        # Split data
        data_splits = self.data_loader.split_data(
            test_size=self.config.test_split,
            val_size=self.config.validation_split
        )
        
        # Save data splits
        for split_name, examples in data_splits.items():
            self.data_loader.save_training_data(examples, f"{split_name}_data.json")
        
        # Print statistics
        stats = self.data_loader.get_statistics()
        logging.info(f"Data statistics: {json.dumps(stats, indent=2)}")
        
        return data_splits
    
    async def train_prompt_parser(self, train_examples: List[TrainingExample]) -> Dict[str, float]:
        """Train the prompt parser agent"""
        
        logging.info("Training Prompt Parser...")
        
        parser = get_prompt_parser_agent()
        metrics = {}
        
        # Create batches
        batches = self.data_loader.create_batches(train_examples, self.config.batch_size)
        
        total_correct = 0
        total_examples = 0
        
        for batch_idx, batch in enumerate(batches):
            batch_correct = 0
            
            for example in batch:
                try:
                    # Simulate training (in real implementation, this would involve actual ML training)
                    # For now, we'll test the parser and collect metrics
                    
                    # The parser would process the prompt
                    parsed_result = await self._simulate_prompt_parsing(example.prompt)
                    
                    # Compare with expected results
                    intent_correct = parsed_result.get("intent") == example.intent
                    params_correct = self._compare_parameters(
                        parsed_result.get("parameters", []), 
                        example.parameters
                    )
                    
                    if intent_correct and params_correct:
                        batch_correct += 1
                    
                    total_examples += 1
                    
                except Exception as e:
                    logging.warning(f"Error processing example: {e}")
            
            total_correct += batch_correct
            
            if batch_idx % self.config.log_interval == 0:
                batch_accuracy = batch_correct / len(batch)
                logging.info(f"Batch {batch_idx}: Accuracy = {batch_accuracy:.3f}")
        
        # Calculate final metrics
        accuracy = total_correct / max(1, total_examples)
        metrics["accuracy"] = accuracy
        metrics["total_examples"] = total_examples
        
        logging.info(f"Prompt Parser training complete. Accuracy: {accuracy:.3f}")
        return metrics
    
    async def train_file_analyzer(self, file_examples: List[Any]) -> Dict[str, float]:
        """Train the file analyzer agent"""
        
        logging.info("Training File Analyzer...")
        
        analyzer = get_file_analyzer_agent()
        metrics = {}
        
        total_files = len(file_examples)
        successful_extractions = 0
        
        for file_example in file_examples:
            try:
                # Simulate file analysis training
                # In real implementation, this would involve ML training
                
                analysis_result = await self._simulate_file_analysis(file_example.file_path)
                
                if analysis_result.get("success", False):
                    successful_extractions += 1
                
            except Exception as e:
                logging.warning(f"Error analyzing file {file_example.file_path}: {e}")
        
        # Calculate metrics
        success_rate = successful_extractions / max(1, total_files)
        metrics["success_rate"] = success_rate
        metrics["total_files"] = total_files
        
        logging.info(f"File Analyzer training complete. Success rate: {success_rate:.3f}")
        return metrics
    
    async def train_ifc_generator(self, train_examples: List[TrainingExample]) -> Dict[str, float]:
        """Train the IFC generator agent"""
        
        logging.info("Training IFC Generator...")
        
        generator = get_ifc_generator_agent()
        metrics = {}
        
        total_generated = 0
        successful_generations = 0
        
        # Filter examples that can be used for IFC generation
        ifc_examples = [ex for ex in train_examples if ex.intent in ["simple_structure", "complex_infrastructure"]]
        
        for example in ifc_examples:
            try:
                # Simulate IFC generation training
                generation_result = await self._simulate_ifc_generation(example)
                
                if generation_result.get("success", False):
                    successful_generations += 1
                
                total_generated += 1
                
            except Exception as e:
                logging.warning(f"Error generating IFC for example: {e}")
        
        # Calculate metrics
        generation_rate = successful_generations / max(1, total_generated)
        metrics["generation_rate"] = generation_rate
        metrics["total_generated"] = total_generated
        
        logging.info(f"IFC Generator training complete. Generation rate: {generation_rate:.3f}")
        return metrics
    
    async def train_end_to_end(self, train_examples: List[TrainingExample]) -> Dict[str, float]:
        """Train the complete system end-to-end"""
        
        logging.info("Training End-to-End System...")
        
        system = get_system()
        await system.initialize()
        
        try:
            total_examples = len(train_examples)
            successful_completions = 0
            
            # Create batches for end-to-end training
            batches = self.data_loader.create_batches(train_examples, self.config.batch_size)
            
            for batch_idx, batch in enumerate(batches):
                batch_successes = 0
                
                for example in batch:
                    try:
                        # Process through complete system
                        result = await system.process_prompt(example.prompt, example.files)
                        
                        # Check if processing was successful
                        if result.get("status") == "completed":
                            batch_successes += 1
                            successful_completions += 1
                        
                    except Exception as e:
                        logging.warning(f"End-to-end processing failed: {e}")
                
                if batch_idx % self.config.log_interval == 0:
                    batch_success_rate = batch_successes / len(batch)
                    logging.info(f"End-to-end Batch {batch_idx}: Success rate = {batch_success_rate:.3f}")
            
            # Calculate final metrics
            completion_rate = successful_completions / max(1, total_examples)
            metrics = {
                "completion_rate": completion_rate,
                "total_examples": total_examples,
                "successful_completions": successful_completions
            }
            
            logging.info(f"End-to-end training complete. Completion rate: {completion_rate:.3f}")
            
        finally:
            await system.shutdown()
        
        return metrics
    
    async def validate_system(self, validation_examples: List[TrainingExample]) -> Dict[str, float]:
        """Validate the trained system"""
        
        logging.info("Validating system...")
        
        validation_metrics = await self.validator.validate_complete_system(validation_examples)
        
        # Calculate composite validation score
        composite_score = (
            validation_metrics.get("accuracy", 0.0) * 0.4 +
            validation_metrics.get("generation_rate", 0.0) * 0.3 +
            validation_metrics.get("completion_rate", 0.0) * 0.3
        )
        
        validation_metrics["composite_score"] = composite_score
        
        logging.info(f"Validation complete. Composite score: {composite_score:.3f}")
        return validation_metrics
    
    async def train_complete_system(self, data_splits: Dict[str, List[TrainingExample]]):
        """Train the complete multi-agent system"""
        
        train_data = data_splits["train"]
        validation_data = data_splits["validation"]
        
        logging.info(f"Starting training with {len(train_data)} training examples")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            logging.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            # Train individual agents
            prompt_metrics = await self.train_prompt_parser(train_data)
            file_metrics = await self.train_file_analyzer(self.data_loader.file_examples)
            ifc_metrics = await self.train_ifc_generator(train_data)
            
            # End-to-end training
            e2e_metrics = await self.train_end_to_end(train_data[:self.config.batch_size])  # Limit for demo
            
            # Validate
            validation_metrics = await self.validate_system(validation_data)
            
            # Record metrics
            epoch_metrics = {
                "epoch": epoch + 1,
                "prompt_parser": prompt_metrics,
                "file_analyzer": file_metrics,
                "ifc_generator": ifc_metrics,
                "end_to_end": e2e_metrics,
                "validation": validation_metrics,
                "epoch_time": time.time() - epoch_start_time
            }
            
            self.metrics.record_epoch(epoch_metrics)
            
            # Check for improvement
            current_score = validation_metrics.get("composite_score", 0.0)
            if current_score > self.best_validation_score:
                self.best_validation_score = current_score
                self.patience_counter = 0
                
                # Save best model
                await self._save_checkpoint(epoch, "best_model")
                logging.info(f"New best model saved with score: {current_score:.3f}")
            else:
                self.patience_counter += 1
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                await self._save_checkpoint(epoch, f"checkpoint_epoch_{epoch + 1}")
            
            # Early stopping
            if self.config.early_stopping and self.patience_counter >= self.config.patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            logging.info(f"Epoch {epoch + 1} complete in {epoch_metrics['epoch_time']:.2f}s")
        
        # Save final metrics
        self.metrics.save_training_history(self.output_dir / "training_history.json")
        
        logging.info("Training complete!")
    
    async def _save_checkpoint(self, epoch: int, checkpoint_name: str):
        """Save model checkpoint"""
        
        checkpoint_dir = self.output_dir / "checkpoints" / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save training state
        checkpoint_data = {
            "epoch": epoch,
            "best_validation_score": self.best_validation_score,
            "patience_counter": self.patience_counter,
            "config": self.config.__dict__
        }
        
        with open(checkpoint_dir / "training_state.json", 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logging.info(f"Checkpoint saved: {checkpoint_name}")
    
    # Simulation methods for training (replace with actual ML training)
    async def _simulate_prompt_parsing(self, prompt: str) -> Dict[str, Any]:
        """Simulate prompt parsing for training"""
        
        # This would be replaced with actual parser training/inference
        return {
            "intent": "simple_structure",  # Simulated result
            "parameters": [
                {"name": "height", "value": 3.0, "unit": "m", "confidence": 0.9}
            ],
            "confidence": 0.85
        }
    
    async def _simulate_file_analysis(self, file_path: str) -> Dict[str, Any]:
        """Simulate file analysis for training"""
        
        # This would be replaced with actual file analyzer training
        return {
            "success": True,
            "extracted_data": {"type": "structural_calculation"},
            "confidence": 0.8
        }
    
    async def _simulate_ifc_generation(self, example: TrainingExample) -> Dict[str, Any]:
        """Simulate IFC generation for training"""
        
        # This would be replaced with actual IFC generator training
        return {
            "success": True,
            "element_count": 5,
            "generation_time": 2.5
        }
    
    def _compare_parameters(self, parsed_params: List[Dict], expected_params: List[Dict]) -> bool:
        """Compare parsed parameters with expected parameters"""
        
        if len(parsed_params) != len(expected_params):
            return False
        
        # Simple comparison - in practice, this would be more sophisticated
        parsed_names = {p.get("name") for p in parsed_params}
        expected_names = {p.get("name") for p in expected_params}
        
        return parsed_names == expected_names

# Training configuration
def create_training_config() -> dict:
    """Create default training configuration"""
    
    return {
        "system": {
            "max_workers": 8,
            "cache_size": 100,
            "timeout": 300
        },
        "training": {
            "epochs": 20,
            "batch_size": 8,
            "learning_rate": 0.001,
            "validation_split": 0.2,
            "test_split": 0.1,
            "save_checkpoints": True,
            "checkpoint_interval": 5,
            "early_stopping": True,
            "patience": 5,
            "output_dir": "models",
            "log_interval": 10
        },
        "agents": {
            "prompt_parser": {
                "max_workers": 2,
                "enable_augmentation": True
            },
            "file_analyzer": {
                "max_workers": 4,
                "chunk_size": 1024
            },
            "ifc_generator": {
                "max_workers": 2,
                "template_cache_size": 100
            }
        },
        "data": {
            "generate_synthetic": True,
            "synthetic_count": 1000,
            "augmentation_factor": 3
        }
    }