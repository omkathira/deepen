from deepen.backend import active_backend as bx

_bx = bx() # backend singleton

class Optimizer:
    def __init__(self, params):
        self.params = tuple(params)

    def zero_grad(self):
        for p in self.params:
            p._reset_grad()

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=None):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.v = [_bx.zeros_like(p.data) for p in self.params]
    
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                if self.momentum is None:
                    param.data -= self.lr * self.v[i]
                else:
                    self.v[i] = self.momentum * self.v[i] + param.grad
                    param.data -= self.lr * self.v[i]

class RMSprop(Optimizer):
    def __init__(self, params):
        super().__init__(params)
    
    def step(self):
        pass

class AdaGrad(Optimizer):
    def __init__(self, params):
        super().__init__(params)
    
    def step(self):
        pass

class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), epsilon=1e-8):
        super().__init__(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.epsilon = epsilon
        self.t = 0
        self.m = [_bx.zeros_like(p.data) for p in self.params]
        self.v = [_bx.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is not None:
                self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * param.grad # momentum updates
                self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (param.grad ** 2) # variance in momentum updates

                # bias correction
                m_hat = self.m[i] / (1 - self.b1 ** self.t)
                v_hat = self.v[i] / (1 - self.b2 ** self.t)

                param.data -= self.lr * (m_hat / (_bx.sqrt(v_hat) + self.epsilon))

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), epsilon=1e-8, weight_decay=0.01):
        super().__init__(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [_bx.zeros_like(p.data) for p in self.params]
        self.v = [_bx.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is not None:
                param.data -= self.lr * self.weight_decay * param.data # weight decay (L2 regularization)

                self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * param.grad
                self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (param.grad ** 2)

                # bias correction
                m_hat = self.m[i] / (1 - self.b1 ** self.t)
                v_hat = self.v[i] / (1 - self.b2 ** self.t)

                param.data -= self.lr * (m_hat / (_bx.sqrt(v_hat) + self.epsilon))