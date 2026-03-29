"""
Evaluation metrics and utilities for healthcare models.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch


class HealthcareModelEvaluator:
    """Evaluate model performance on healthcare tasks."""
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba=None):
        """Calculate standard evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
        }
        
        if y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba, average='weighted')
        
        return metrics
    
    @staticmethod
    def get_confusion_matrix(y_true, y_pred):
        """Get confusion matrix."""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def calculate_sensitivity_specificity(y_true, y_pred):
        """Calculate sensitivity and specificity for binary classification."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return sensitivity, specificity


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(**batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    evaluator = HealthcareModelEvaluator()
    metrics = evaluator.calculate_metrics(all_labels, all_preds)
    return metrics


if __name__ == "__main__":
    evaluator = HealthcareModelEvaluator()
