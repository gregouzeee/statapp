import numpy as np


def normalize_probs(probs, axis=-1, eps=1e-12):
    p = np.asarray(probs, dtype=float)
    s = p.sum(axis=axis, keepdims=True)
    k = p.shape[axis]
    out = np.full_like(p, 1.0 / k)
    np.divide(p, s, out=out, where=s > eps)
    return out


def entropy(probs, axis=-1, eps=1e-12, base=None):
    p = normalize_probs(probs, axis=axis, eps=eps)
    log_p = np.log(np.clip(p, eps, 1.0))
    h = -np.sum(p * log_p, axis=axis)
    if base is not None:
        h = h / np.log(base)
    return h


def normalized_entropy(probs, axis=-1, eps=1e-12):
    p = normalize_probs(probs, axis=axis, eps=eps)
    k = p.shape[axis]
    if k <= 1:
        return np.zeros(np.sum(p, axis=axis).shape, dtype=float)
    h = entropy(p, axis=axis, eps=eps)
    return h / np.log(k)


def confidence_from_entropy(probs, axis=-1, eps=1e-12):
    h_norm = normalized_entropy(probs, axis=axis, eps=eps)
    return np.clip(1.0 - h_norm, 0.0, 1.0)
