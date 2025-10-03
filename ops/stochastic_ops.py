from deepen.backend import active_backend as bx

_bx = bx() # backend singleton

# Dropout
class dropout:
    @staticmethod
    def forward(save, x, p):
        q = 1.0 - p
        u = _bx.random.uniform(size=x.shape, low=0.0, high=1.0, dtype=x.dtype)
        mask = _bx.less(u, q).astype(x.dtype)
        output = _bx.divide(_bx.multiply(x, mask), q)

        if save.active:
            save.q = q
            save.mask = mask

        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _bx.divide(_bx.multiply(output_grad, save.mask), save.q)
        return dx,

# Gaussian noise
class gaussian_noise:
    @staticmethod
    def forward(save, x, mean=0.0, std=0.1):
        noise = _bx.random.normal(size=x.shape, loc=mean, scale=std, dtype=x.dtype)
        output = _bx.add(x, noise)
        return output

    @staticmethod
    def backward(save, output_grad):
        dx = output_grad
        return dx,