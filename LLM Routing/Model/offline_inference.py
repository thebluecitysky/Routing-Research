import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

class CustomModelTester:
    def __init__(self, model_path: str, tokenizer_path: str, device: str = "cuda"):
        """
        Initialize the custom model tester with APGR calculation capability.
        
        Args:
            model_path: Path to your trained model checkpoint
            tokenizer_path: Path to the tokenizer
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = self._load_model(model_path).to(self.device)
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.optimal_threshold = 0.5
        self.performance_metrics = {}  # Store performance metrics for APGR calculation
    
    def _load_model(self, model_path: str):
        """Load your custom model implementation"""
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        return model
    
    def _load_tokenizer(self, tokenizer_path: str):
        """Load the tokenizer"""
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tokenizer_path)
    
    def preprocess_input(self, input_text: str) -> Dict[str, torch.Tensor]:
        """Preprocess input text for your model"""
        inputs = self.tokenizer(
            input_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def predict_proba(self, input_text: str) -> np.ndarray:
        """Get raw prediction probabilities from your model"""
        with torch.no_grad():
            inputs = self.preprocess_input(input_text)
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        return probs
    
    def evaluate_on_dataset(self, dataset: List[Tuple[str, int]]) -> Dict[str, float]:
        """
        Evaluate model performance on a dataset and calculate metrics including APGR
        
        Args:
            dataset: List of (text, label) tuples
            
        Returns:
            Dictionary containing various performance metrics
        """
        all_probs = []
        all_labels = []
        
        for text, label in dataset:
            probs = self.predict_proba(text)
            all_probs.append(probs[0])
            all_labels.append(label)
            
        y_probs = np.array(all_probs)
        y_true = np.array(all_labels)
        
        # Calculate accuracy at different threshold levels
        threshold_range = np.linspace(0, 1, 11)
        accuracies = []
        strong_percentages = []
        
        for threshold in threshold_range:
            correct = 0
            strong_calls = 0
            for prob, true_label in zip(y_probs, y_true):
                pred_class = np.argmax(prob)
                if prob[pred_class] >= threshold:
                    strong_calls += 1
                    if pred_class == true_label:
                        correct += 1
                else:
                    # For weak model calls, we'd use a different model in practice
                    # Here we just count it as correct if the prediction matches
                    if pred_class == true_label:
                        correct += 1
            
            accuracy = correct / len(y_true)
            strong_percentage = strong_calls / len(y_true) * 100
            
            accuracies.append(accuracy)
            strong_percentages.append(strong_percentage)
        
        # Calculate metrics
        weak_accuracy = self._calculate_weak_model_accuracy(dataset)
        strong_accuracy = self._calculate_strong_model_accuracy(dataset)
        
        # Calculate AUC
        auc = np.trapz(accuracies, strong_percentages) / 100
        
        # Calculate APGR (Average Performance Gain Ratio)
        weak_auc = weak_accuracy  # Weak model would be flat line at its accuracy
        strong_auc = strong_accuracy  # Strong model would be flat line at its accuracy
        apgr = (auc - weak_auc) / (strong_auc - weak_auc) if (strong_auc - weak_auc) != 0 else 0
        
        # Store results
        results = {
            'thresholds': threshold_range,
            'accuracies': accuracies,
            'strong_percentages': strong_percentages,
            'weak_accuracy': weak_accuracy,
            'strong_accuracy': strong_accuracy,
            'AUC': auc,
            'APGR': apgr
        }
        
        self.performance_metrics = results
        return results
    
    def _calculate_weak_model_accuracy(self, dataset: List[Tuple[str, int]]) -> float:
        """
        Simulate weak model accuracy (replace with actual weak model evaluation if available)
        """
        # In practice, you would evaluate your weak model here
        # For demonstration, we'll use a fixed value or simple heuristic
        return 0.65  # Example value
    
    def _calculate_strong_model_accuracy(self, dataset: List[Tuple[str, int]]) -> float:
        """
        Simulate strong model accuracy (replace with actual strong model evaluation if available)
        """
        # In practice, you would evaluate your strong model here
        # For demonstration, we'll use a fixed value or simple heuristic
        return 0.85  # Example value
    
    def plot_performance_curve(self, save_path: str = None):
        """Plot the performance curve with APGR information"""
        if not self.performance_metrics:
            raise ValueError("No performance metrics available. Run evaluate_on_dataset first.")
            
        metrics = self.performance_metrics
        plt.figure(figsize=(8, 6))
        
        # Plot accuracy vs strong percentage
        plt.plot(
            metrics['strong_percentages'],
            metrics['accuracies'],
            label=f"Model (APGR={metrics['APGR']:.3f})",
            marker='o',
            linestyle='-'
        )
        
        # Plot weak and strong model baselines
        plt.axhline(
            y=metrics['weak_accuracy'],
            color='grey',
            linestyle='--',
            label=f'Weak Model ({metrics["weak_accuracy"]:.3f})'
        )
        plt.axhline(
            y=metrics['strong_accuracy'],
            color='red',
            linestyle='--',
            label=f'Strong Model ({metrics["strong_accuracy"]:.3f})'
        )
        
        plt.xlabel("Strong Model Calls (%)")
        plt.ylabel("Accuracy")
        plt.title("Model Performance with APGR Metric")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved performance plot to {save_path}")
        else:
            plt.show()
    
    def find_optimal_threshold(self, val_dataset, target_class_idx: int = 1):
        """Find optimal threshold with APGR considerations"""
        results = self.evaluate_on_dataset(val_dataset)
        
        # Find threshold that maximizes APGR
        best_idx = np.argmax([results['accuracies']])
        self.optimal_threshold = results['thresholds'][best_idx]
        
        print(f"Optimal threshold: {self.optimal_threshold:.4f}")
        print(f"APGR at optimal threshold: {results['APGR']:.4f}")
        
        return self.optimal_threshold
    
    def predict_with_threshold(self, input_text: str, threshold: float = None) -> Dict[str, Any]:
        """Make prediction using the specified threshold"""
        if threshold is None:
            threshold = self.optimal_threshold
        
        probs = self.predict_proba(input_text)[0]
        confidence = np.max(probs)
        predicted_class = np.argmax(probs)
        
        if probs[predicted_class] < threshold:
            predicted_class = -1  # Uncertain class
        
        return {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "probabilities": [float(p) for p in probs],
            "threshold_used": float(threshold),
            "is_strong_call": bool(probs[predicted_class] >= threshold)
        }

# Example usage
if __name__ == "__main__":
    # Initialize tester with your model paths
    tester = CustomModelTester(
        model_path="/data2/zhangtiant/DeepRouter/saved/best_model.pth",
        tokenizer_path="/data2/zhangtiant/DeepRouter/saved/"
    )
    
    # Example validation dataset (replace with your actual data)
    dataset = pd.read_csv('/data2/zhangtiant/DeepRouter/data/combined_data.csv')

    # 划分训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    graph_folder = '/data2/zhangtiant/DeepRouter/data/graph'
    from train import DualFeatureDataset
    from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AdamW
    )
    tokenizer = AutoTokenizer.from_pretrained("/data2/zhangtiant/DeepRouter/saved/", trust_remote_code=True)
    train_set = DualFeatureDataset(
        [dataset['prompt'][i] for i in train_dataset.indices],
        graph_folder,
        [dataset['score'][i] for i in train_dataset.indices],
        tokenizer,
        MAX_LENGTH=512,
        batch_size=100
    )
    val_set = DualFeatureDataset(
        [dataset['prompt'][i] for i in val_dataset.indices],
        graph_folder,
        [dataset['score'][i] for i in val_dataset.indices],
        tokenizer,
        MAX_LENGTH=512,
        batch_size=100
    )
    
    # Find optimal threshold (run this once with your validation data)
    optimal_threshold = tester.find_optimal_threshold(val_set, target_class_idx=1)

    # Evaluate on dataset and calculate APGR
    results = tester.evaluate_on_dataset(val_set)
    print("\nPerformance Metrics:")
    print(f"Weak Model Accuracy: {results['weak_accuracy']:.4f}")
    print(f"Strong Model Accuracy: {results['strong_accuracy']:.4f}")
    print(f"AUC: {results['AUC']:.4f}")
    print(f"APGR: {results['APGR']:.4f}")
    
    # Plot performance curve
    tester.plot_performance_curve("performance_plot.png")
    
    # Make predictions with threshold
    test_text = "This is a test input"
    prediction = tester.predict_with_threshold(test_text)

    print("\nPrediction with Threshold Adjustment:")
    print(f"Input: {test_text}")
    print(f"Predicted class: {prediction['predicted_class']}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print(f"Threshold used: {prediction['threshold_used']:.4f}")
    print(f"Is strong call: {prediction['is_strong_call']}")