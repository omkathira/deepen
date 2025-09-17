from deepen.backend import active_backend as bx

_bx = bx() # backend singleton

class Optimizer:
    def __init__(self, params):
        self.params = tuple(params)

    def zero_grad(self):
        for p in self.params:
            p._reset_grad()

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data = param.data - self.lr * param.grad

class SGD_with_momentum(Optimizer):
    def __init__(self, params):
        super.__init__(params)
    
    def step(self):
        pass

class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), epsilon=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.epsilon = epsilon
        self.t = 0
        self.m = [_bx.zeros_like(p.data) for p in self.params]
        self.v = [_bx.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is not None:
                g_t = param.grad

                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g_t # momentum updates
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g_t ** 2) # variance in momentum updates

                # bias correction
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                param.data -= self.lr * (m_hat / (_bx.sqrt(v_hat) + self.epsilon))