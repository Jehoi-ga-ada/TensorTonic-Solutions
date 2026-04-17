import numpy as np

def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_k = np.asarray(recommended[:k])
    relevant = np.asarray(relevant)

    mask = np.isin(top_k, relevant)
    hits = np.sum(mask)

    precision_at_k = hits / k
    recall_at_k = hits / len(relevant)
    
    return [precision_at_k, recall_at_k]