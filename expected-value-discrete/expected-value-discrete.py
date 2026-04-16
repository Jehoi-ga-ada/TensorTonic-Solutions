import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.asarray(x)
    p = np.asarray(p)

    ev = np.sum(x*p)
    
    if np.sum(p) <= 1-1e-7 or np.sum(p) >= 1+1e-7:
        raise ValueError("Probabilites must sum to 1")
        
    return ev 