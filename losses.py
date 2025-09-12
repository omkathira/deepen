from deepen.core.tensor import Tensor

def mse(pred: Tensor, true: Tensor): return ((pred - true) ** 2).mean()

def mae(pred: Tensor, true: Tensor): return (pred - true).abs().mean()

def binary_cross_entropy(pred: Tensor, true: Tensor, epsilon=1e-7):
    p = pred.clip(epsilon, 1 - epsilon)
    loss = -(true * p.log() + (1 - true) * (1 - p).log())
    return loss.mean()

def cross_entropy(logits: Tensor, labels: Tensor):
    return 42