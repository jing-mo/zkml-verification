import torch
from torch.utils.data import DataLoader
from typing import Dict, Any

from new_zkp_verify.train import ModelTrainer
from new_zkp_verify.verify import ModelVerifier
from new_zkp_verify.utils import get_data_loaders  # Update this import
from new_zkp_verify.config import (
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_BASELINE_MODELS, DEFAULT_CONFIDENCE
)

class VerificationExperiment:
    """Verification experiment runner"""
    
    def __init__(
        self,
        batch_size=DEFAULT_BATCH_SIZE,
        num_epochs=DEFAULT_EPOCHS,
        learning_rate=DEFAULT_LEARNING_RATE,
        num_baseline_models=DEFAULT_NUM_BASELINE_MODELS,
        confidence_level=DEFAULT_CONFIDENCE,
        sample_batches=10,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_baseline_models = num_baseline_models
        self.confidence_level = confidence_level
        self.sample_batches = sample_batches
        self.device = device
        
        # Add data loaders
        self.train_loader, self.test_loader = get_data_loaders(batch_size=batch_size)
        
        self.trainer = ModelTrainer(
            device=device,
            batch_size=batch_size,
            epochs=num_epochs,
            lr=learning_rate
        )
        self.verifier = ModelVerifier(device=device)

    def run_experiment(self, train_distilled=False) -> Dict[str, Any]:
        """Run the verification experiment
        
        Args:
            train_distilled: Whether to train distilled model
            
        Returns:
            Dict containing experiment results
        """
        print(f"\nStarting verification experiment with {self.num_baseline_models} baseline models")
        print(f"Batch size: {self.batch_size}, Epochs: {self.num_epochs}")
        
        # Train baseline models
        baseline_models = []
        baseline_stats = {}
        
        for i in range(self.num_baseline_models):
            print(f"\nTraining baseline model {i+1}/{self.num_baseline_models}")
            model, stats = self.trainer.train()
            baseline_models.append(model)
            baseline_stats[f"baseline_{i+1}"] = stats
        
        # Train target model (either standard or distilled)
        print("\nTraining target model")
        if train_distilled:
            target_model, target_stats = self.trainer.train_distilled(
                teacher_model=baseline_models[0]
            )
        else:
            target_model, target_stats = self.trainer.train()
        
        # Verify the model
        print("\nVerifying model")
        # Update the verify_black_box call to include test_loader
        verification_result = self.verifier.verify_black_box(
            target_model=target_model,
            baseline_models=baseline_models,
            test_loader=self.test_loader,  # Add this line
            confidence_level=self.confidence_level
        )
        
        # Compile results
        results = {
            "summary": {
                "passed_verification": verification_result["verification_passed"],
                "target_kl_mean": verification_result["target_kl_mean"],
                "baseline_avg_accuracy": verification_result["baseline_avg_accuracy"],
                "training_time": target_stats["training_time"],
                "verification_time": verification_result["verification_time"]
            },
            "target_model_stats": target_stats,
            "baseline_stats": baseline_stats,
            "verification_details": verification_result,
            "experiment_config": {
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "learning_rate": self.learning_rate,
                "num_baseline_models": self.num_baseline_models,
                "confidence_level": self.confidence_level,
                "device": self.device,
                "train_distilled": train_distilled
            }
        }
        
        return results