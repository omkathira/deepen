from deepen.backend import active_backend as bx

_bx = bx() # backend singleton

# Dropout
class dropout:
    @staticmethod
    def forward(save, x, p):
        q = save.q = 1.0 - p
        u = _bx.random.uniform(size=x.shape, low=0.0, high=1.0, dtype=x.dtype)
        mask = save.mask = _bx.less(u, q).astype(x.dtype)
        output = _bx.divide(_bx.multiply(x, mask), q)
        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _bx.divide(_bx.multiply(output_grad, save.mask), save.q)
        return dx,