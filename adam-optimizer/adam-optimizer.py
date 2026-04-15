import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    param = np.asarray(param, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    
    m = beta1 * m + (1-beta1) * grad
    v = beta2 * v + (1-beta2) * (grad**2)
    m_new = m/(1-beta1**t)
    v_new = v/(1-beta2**t)
    param_new = param - lr * m_new/(v_new**(1/2) + eps)

    return param_new, m, v