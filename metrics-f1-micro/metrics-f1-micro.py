import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum(y_true == y_pred)
    
    total = len(y_true)
    
    precision = tp / total
    recall = tp / total
    
    if (precision + recall) == 0:
        return 0.0
        
    f1 = 2 * (precision * recall) / (precision + recall)
    return float(f1)