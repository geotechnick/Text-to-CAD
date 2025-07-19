"""
Training metrics and evaluation for Text-to-CAD system
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class EpochMetrics:
    epoch: int
    timestamp: float
    prompt_parser_accuracy: float
    file_analyzer_success_rate: float
    ifc_generator_success_rate: float
    end_to_end_completion_rate: float
    validation_score: float
    training_time: float
    memory_usage: float
    processing_speed: float

class TrainingMetrics:
    """
    Comprehensive metrics collection and analysis for training
    """
    
    def __init__(self):
        self.epoch_history = []
        self.validation_history = []
        self.loss_history = []
        self.performance_history = []
        
        # Real-time metrics
        self.current_epoch_metrics = {}
        self.start_time = time.time()
        
        logging.info("TrainingMetrics initialized")
    
    def record_epoch(self, metrics: Dict[str, Any]):
        """Record metrics for an epoch"""
        
        epoch_data = {
            "epoch": metrics.get("epoch", 0),
            "timestamp": time.time(),
            "prompt_parser": metrics.get("prompt_parser", {}),
            "file_analyzer": metrics.get("file_analyzer", {}),
            "ifc_generator": metrics.get("ifc_generator", {}),
            "end_to_end": metrics.get("end_to_end", {}),
            "validation": metrics.get("validation", {}),
            "epoch_time": metrics.get("epoch_time", 0.0)
        }
        
        self.epoch_history.append(epoch_data)
        
        # Extract key metrics for trending
        key_metrics = {
            "epoch": epoch_data["epoch"],
            "accuracy": metrics.get("prompt_parser", {}).get("accuracy", 0.0),
            "file_success_rate": metrics.get("file_analyzer", {}).get("success_rate", 0.0),
            "ifc_generation_rate": metrics.get("ifc_generator", {}).get("generation_rate", 0.0),
            "completion_rate": metrics.get("end_to_end", {}).get("completion_rate", 0.0),
            "validation_score": metrics.get("validation", {}).get("composite_score", 0.0)
        }
        
        self.validation_history.append(key_metrics)
        
        logging.info(f"Recorded metrics for epoch {epoch_data['epoch']}")
    
    def record_batch_metrics(self, batch_idx: int, metrics: Dict[str, Any]):
        """Record metrics for a training batch"""
        
        batch_data = {
            "batch_idx": batch_idx,
            "timestamp": time.time(),
            "metrics": metrics
        }
        
        if "batch_history" not in self.current_epoch_metrics:
            self.current_epoch_metrics["batch_history"] = []
        
        self.current_epoch_metrics["batch_history"].append(batch_data)
    
    def calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends across epochs"""
        
        if len(self.validation_history) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Extract time series data
        epochs = [m["epoch"] for m in self.validation_history]
        accuracy = [m["accuracy"] for m in self.validation_history]
        validation_scores = [m["validation_score"] for m in self.validation_history]
        completion_rates = [m["completion_rate"] for m in self.validation_history]
        
        trends = {
            "total_epochs": len(epochs),
            "accuracy_trend": self._calculate_trend(accuracy),
            "validation_trend": self._calculate_trend(validation_scores),
            "completion_trend": self._calculate_trend(completion_rates),
            "best_epoch": {
                "epoch": epochs[np.argmax(validation_scores)],
                "validation_score": max(validation_scores),
                "accuracy": accuracy[np.argmax(validation_scores)]
            },
            "latest_metrics": self.validation_history[-1] if self.validation_history else {}
        }
        
        return trends
    
    def calculate_agent_performance(self) -> Dict[str, Dict[str, float]]:
        """Calculate individual agent performance metrics"""
        
        agent_metrics = {
            "prompt_parser": {
                "avg_accuracy": 0.0,
                "best_accuracy": 0.0,
                "improvement": 0.0
            },
            "file_analyzer": {
                "avg_success_rate": 0.0,
                "best_success_rate": 0.0,
                "improvement": 0.0
            },
            "ifc_generator": {
                "avg_generation_rate": 0.0,
                "best_generation_rate": 0.0,
                "improvement": 0.0
            },
            "system_overall": {
                "avg_completion_rate": 0.0,
                "best_completion_rate": 0.0,
                "improvement": 0.0
            }
        }
        
        if not self.validation_history:
            return agent_metrics
        
        # Calculate prompt parser metrics
        accuracies = [m["accuracy"] for m in self.validation_history]
        agent_metrics["prompt_parser"]["avg_accuracy"] = np.mean(accuracies)
        agent_metrics["prompt_parser"]["best_accuracy"] = max(accuracies)
        if len(accuracies) > 1:
            agent_metrics["prompt_parser"]["improvement"] = accuracies[-1] - accuracies[0]
        
        # Calculate file analyzer metrics
        file_rates = [m["file_success_rate"] for m in self.validation_history]
        agent_metrics["file_analyzer"]["avg_success_rate"] = np.mean(file_rates)
        agent_metrics["file_analyzer"]["best_success_rate"] = max(file_rates)
        if len(file_rates) > 1:
            agent_metrics["file_analyzer"]["improvement"] = file_rates[-1] - file_rates[0]
        
        # Calculate IFC generator metrics
        ifc_rates = [m["ifc_generation_rate"] for m in self.validation_history]
        agent_metrics["ifc_generator"]["avg_generation_rate"] = np.mean(ifc_rates)
        agent_metrics["ifc_generator"]["best_generation_rate"] = max(ifc_rates)
        if len(ifc_rates) > 1:
            agent_metrics["ifc_generator"]["improvement"] = ifc_rates[-1] - ifc_rates[0]
        
        # Calculate system overall metrics
        completion_rates = [m["completion_rate"] for m in self.validation_history]
        agent_metrics["system_overall"]["avg_completion_rate"] = np.mean(completion_rates)
        agent_metrics["system_overall"]["best_completion_rate"] = max(completion_rates)
        if len(completion_rates) > 1:
            agent_metrics["system_overall"]["improvement"] = completion_rates[-1] - completion_rates[0]
        
        return agent_metrics
    
    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        
        total_training_time = time.time() - self.start_time
        
        report = {
            "training_summary": {
                "total_epochs": len(self.epoch_history),
                "total_training_time": total_training_time,
                "avg_epoch_time": np.mean([e["epoch_time"] for e in self.epoch_history]) if self.epoch_history else 0,
                "training_start_time": self.start_time,
                "training_end_time": time.time()
            },
            "performance_trends": self.calculate_performance_trends(),
            "agent_performance": self.calculate_agent_performance(),
            "training_stability": self._analyze_training_stability(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def save_training_history(self, output_path: Path):
        """Save complete training history to file"""
        
        history_data = {
            "epoch_history": self.epoch_history,
            "validation_history": self.validation_history,
            "training_report": self.generate_training_report(),
            "metadata": {
                "total_epochs": len(self.epoch_history),
                "save_timestamp": time.time(),
                "training_duration": time.time() - self.start_time
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logging.info(f"Training history saved to {output_path}")
    
    def plot_training_curves(self, output_dir: Path, show_plots: bool = False):
        """Generate training curve plots"""
        
        if not self.validation_history:
            logging.warning("No validation history available for plotting")
            return
        
        output_dir.mkdir(exist_ok=True)
        
        # Extract data for plotting
        epochs = [m["epoch"] for m in self.validation_history]
        accuracy = [m["accuracy"] for m in self.validation_history]
        validation_scores = [m["validation_score"] for m in self.validation_history]
        completion_rates = [m["completion_rate"] for m in self.validation_history]
        file_rates = [m["file_success_rate"] for m in self.validation_history]
        ifc_rates = [m["ifc_generation_rate"] for m in self.validation_history]
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Text-to-CAD Training Progress', fontsize=16)
        
        # Plot 1: Prompt Parser Accuracy
        axes[0, 0].plot(epochs, accuracy, 'b-', linewidth=2, label='Accuracy')
        axes[0, 0].set_title('Prompt Parser Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: File Analyzer Success Rate
        axes[0, 1].plot(epochs, file_rates, 'g-', linewidth=2, label='Success Rate')
        axes[0, 1].set_title('File Analyzer Success Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: IFC Generator Success Rate
        axes[0, 2].plot(epochs, ifc_rates, 'r-', linewidth=2, label='Generation Rate')
        axes[0, 2].set_title('IFC Generator Success Rate')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Generation Rate')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
        
        # Plot 4: End-to-End Completion Rate
        axes[1, 0].plot(epochs, completion_rates, 'm-', linewidth=2, label='Completion Rate')
        axes[1, 0].set_title('End-to-End Completion Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Completion Rate')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 5: Validation Score
        axes[1, 1].plot(epochs, validation_scores, 'c-', linewidth=2, label='Validation Score')
        axes[1, 1].set_title('Overall Validation Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Score')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        # Plot 6: Combined Overview
        axes[1, 2].plot(epochs, accuracy, 'b-', linewidth=2, label='Parser Accuracy', alpha=0.7)
        axes[1, 2].plot(epochs, file_rates, 'g-', linewidth=2, label='File Success', alpha=0.7)
        axes[1, 2].plot(epochs, ifc_rates, 'r-', linewidth=2, label='IFC Generation', alpha=0.7)
        axes[1, 2].plot(epochs, completion_rates, 'm-', linewidth=2, label='E2E Completion', alpha=0.7)
        axes[1, 2].set_title('Combined Performance Overview')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Performance Metric')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        logging.info(f"Training curves saved to {plot_path}")
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """Calculate trend statistics for a metric"""
        
        if len(values) < 2:
            return {"trend": 0.0, "slope": 0.0, "r_squared": 0.0}
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate linear regression
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        
        # Calculate R-squared
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            "slope": float(slope),
            "r_squared": float(r_squared),
            "improvement": float(values[-1] - values[0]),
            "latest_value": float(values[-1])
        }
    
    def _analyze_training_stability(self) -> Dict[str, Any]:
        """Analyze training stability and convergence"""
        
        if len(self.validation_history) < 5:
            return {"status": "insufficient_data"}
        
        # Get recent validation scores
        recent_scores = [m["validation_score"] for m in self.validation_history[-5:]]
        
        # Calculate stability metrics
        mean_score = np.mean(recent_scores)
        std_score = np.std(recent_scores)
        coefficient_of_variation = std_score / mean_score if mean_score > 0 else float('inf')
        
        # Determine stability status
        if coefficient_of_variation < 0.05:
            stability_status = "stable"
        elif coefficient_of_variation < 0.15:
            stability_status = "moderately_stable"
        else:
            stability_status = "unstable"
        
        return {
            "status": stability_status,
            "coefficient_of_variation": float(coefficient_of_variation),
            "recent_mean": float(mean_score),
            "recent_std": float(std_score),
            "convergence_indicator": float(1.0 - coefficient_of_variation)
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate training recommendations based on metrics"""
        
        recommendations = []
        
        if not self.validation_history:
            return ["Insufficient training data for recommendations"]
        
        # Analyze trends
        trends = self.calculate_performance_trends()
        stability = self._analyze_training_stability()
        
        # Check for poor performance
        latest_metrics = self.validation_history[-1]
        if latest_metrics["accuracy"] < 0.7:
            recommendations.append("Prompt parser accuracy is low. Consider increasing training data or adjusting model parameters.")
        
        if latest_metrics["completion_rate"] < 0.6:
            recommendations.append("End-to-end completion rate is low. Review system integration and error handling.")
        
        # Check for instability
        if stability["status"] == "unstable":
            recommendations.append("Training appears unstable. Consider reducing learning rate or adding regularization.")
        
        # Check for convergence
        if len(self.validation_history) > 10:
            recent_improvement = trends["validation_trend"]["improvement"]
            if abs(recent_improvement) < 0.01:
                recommendations.append("Training may have converged. Consider early stopping or adjusting parameters.")
        
        # Check for overfitting
        if len(self.validation_history) > 5:
            validation_scores = [m["validation_score"] for m in self.validation_history]
            if max(validation_scores[:-3]) > max(validation_scores[-3:]):
                recommendations.append("Possible overfitting detected. Consider adding regularization or reducing model complexity.")
        
        if not recommendations:
            recommendations.append("Training appears to be progressing well. Continue with current configuration.")
        
        return recommendations
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time training metrics"""
        
        current_time = time.time()
        training_duration = current_time - self.start_time
        
        metrics = {
            "training_duration": training_duration,
            "epochs_completed": len(self.epoch_history),
            "current_epoch_data": self.current_epoch_metrics,
            "latest_performance": self.validation_history[-1] if self.validation_history else {},
            "training_speed": len(self.epoch_history) / (training_duration / 3600) if training_duration > 0 else 0  # epochs per hour
        }
        
        return metrics