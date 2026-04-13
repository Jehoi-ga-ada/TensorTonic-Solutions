import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    N = X.shape[0]
    w = np.zeros((X.shape[1], 1))
    b = 0.0
    y = y.reshape(-1, 1)
    params = [w, b]
    
    for _ in range(steps):
        out = X @ w + b
        p = _sigmoid(out)

        error = p - y
        
        dw = (X.T @ error) / N
        db = np.sum(error) / N

        w -= lr * dw
        b -= lr * db

    return w.flatten(), float(b)