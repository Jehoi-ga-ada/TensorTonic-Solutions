import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    
    return np.sum(np.abs(np.asarray(x) - np.asarray(y)), axis=0, dtype=np.float64)