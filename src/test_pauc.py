#!/usr/bin/env python3
"""
Script for testing PAUC calculation functionality
"""

import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def compute_pauc(y_true, y_scores, max_fpr=0.01):
    """
    Calculate pAUC (partial AUC) with max_fpr=0.01
    
    Parameters:
        - y_true: true labels (binary classification: 0=benign, 1=attack)
        - y_scores: prediction scores
        - max_fpr: maximum false positive rate, default 0.01 (1%)
    
    Returns:
        - pauc: partial AUC value
    """
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # Find the part where FPR <= max_fpr
    mask = fpr <= max_fpr
    fpr_partial = fpr[mask]
    tpr_partial = tpr[mask]
    
    # If max_fpr is not in fpr, interpolation is needed
    if max_fpr not in fpr_partial:
        # Add max_fpr point
        fpr_partial = np.append(fpr_partial, max_fpr)
        # Interpolate corresponding TPR
        tpr_at_max_fpr = np.interp(max_fpr, fpr, tpr)
        tpr_partial = np.append(tpr_partial, tpr_at_max_fpr)
    
    # Calculate pAUC (using trapezoidal rule)
    pauc = np.trapz(tpr_partial, fpr_partial)
    
    return pauc, fpr_partial, tpr_partial

def test_pauc():
    """Test PAUC calculation"""
    # Generate test data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate true labels: 80% benign (0), 20% attack (1)
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    
    # Generate prediction scores: lower scores for benign class, higher scores for attack class
    y_scores = np.random.normal(0, 1, n_samples)
    y_scores[y_true == 1] += 2  # Higher scores for attack class
    
    # Calculate pAUC
    pauc, fpr_partial, tpr_partial = compute_pauc(y_true, y_scores, max_fpr=0.01)
    
    print(f"pAUC (FPR ≤ 0.01): {pauc:.4f}")
    
    # Plot ROC curve and pAUC region
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    plt.figure(figsize=(10, 6))
    
    # Plot complete ROC curve
    plt.plot(fpr, tpr, 'b-', linewidth=2, label='Full ROC')
    
    # Plot pAUC region
    plt.fill_between(fpr_partial, 0, tpr_partial, alpha=0.3, color='red', 
                     label=f'pAUC region (FPR ≤ 0.01)')
    
    # Add max_fpr line
    plt.axvline(x=0.01, color='red', linestyle='--', alpha=0.7, 
                label='Max FPR = 0.01')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve with pAUC (FPR ≤ 0.01) = {pauc:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 0.1])  # Zoom in on FPR ≤ 0.1 region
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('test_pauc.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Test completed! Image saved as test_pauc.png")   

if __name__ == "__main__":
    test_pauc() 