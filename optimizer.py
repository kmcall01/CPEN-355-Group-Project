import numpy as np


# ─────────────────────────────────────────────────────────────
# Gradient Descent
# ─────────────────────────────────────────────────────────────
def gradient_descent(f, grad_f, w0, lr, n_iters):
    w = w0.copy().astype(float)

    ws = [w.copy()]
    fs = [f(w)]

    for _ in range(n_iters):
        w = w - lr * grad_f(w)
        fval = f(w)

        ws.append(w.copy())
        fs.append(fval)

        if not np.isfinite(fval):
            break

    return np.array(ws), np.array(fs)


# ─────────────────────────────────────────────────────────────
# Adam Optimizer
# ─────────────────────────────────────────────────────────────
def adam(f, grad_f, w0, lr, n_iters,
         beta1=0.9, beta2=0.999, eps=1e-8):

    w = w0.copy().astype(float)

    m = np.zeros_like(w)
    v = np.zeros_like(w)

    ws = [w.copy()]
    fs = [f(w)]

    for t in range(1, n_iters + 1):
        g = grad_f(w)

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

        fval = f(w)

        ws.append(w.copy())
        fs.append(fval)

        if not np.isfinite(fval):
            break

    return np.array(ws), np.array(fs)


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────
def run_optimizer(f, grad_f, w0, learning_rates,
                  n_iters=100, method="gd"):

    results = {}

    for lr in learning_rates:
        if method == "adam":
            ws, fs = adam(f, grad_f, w0, lr, n_iters)
        else:
            ws, fs = gradient_descent(f, grad_f, w0, lr, n_iters)

        results[lr] = (ws, fs)

    return results