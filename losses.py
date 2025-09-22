from deepen.core.tensor import Tensor

def mse(pred: Tensor, true: Tensor): return ((pred - true) ** 2).mean()

def mae(pred: Tensor, true: Tensor): return (pred - true).abs().mean()

def binary_cross_entropy(pred: Tensor, true: Tensor, epsilon=1e-7):
    p = pred.clip(epsilon, 1 - epsilon)
    loss = -(true * p.log() + (1 - true) * (1 - p).log())
    return loss.mean()

def cross_entropy(logits: Tensor, labels: Tensor):
    shifted_logits = logits - logits.max(axes=0) # subtract max for numerical stability
    exp_logits = shifted_logits.exp()
    sum_exp = exp_logits.sum(axes=0) # sum per class
    softmax = exp_logits / sum_exp

    log_probs = softmax.log()
    cross_entropy = -labels * log_probs
    loss_per_sample = cross_entropy.sum(axes=0)
    loss = loss_per_sample.mean()

    return loss