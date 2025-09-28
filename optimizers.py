from deepen.backend import active_backend as bx

_bx = bx() # backend singleton

class Optimizer:
    def __init__(self, params):
        self.params = tuple(params)

    def zero_grad(self):
        for p in self.params:
            p._reset_grad()

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=None, weight_decay=None):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.m = {} # stores running average of gradients

        for param in self.params:
            self.m[param] = _bx.zeros_like(param.data)
    
    def step(self):
        for param in self.params:
            if param.grad is not None:
                param_grad = _bx.copy(param.grad) # avoid modifying original gradient in-place

                if self.weight_decay is not None:
                    param_grad += self.weight_decay * param.data # coupled weight decay

                if self.momentum is None:
                    param.data -= self.lr * param_grad
                else:
                    self.m[param] = self.momentum * self.m[param] + param_grad
                    param.data -= self.lr * self.m[param]

class RMSprop(Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.99, weight_decay=None, epsilon=1e-8):
        super().__init__(params)
        self.lr = lr
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.v = {} # stores running average of squared gradients
        self.epsilon = epsilon

        for param in self.params:
            self.v[param] = _bx.zeros_like(param.data)
    
    def step(self):
        for param in self.params:
            if param.grad is not None:
                param_grad = _bx.copy(param.grad)

                if self.weight_decay is not None:
                    param_grad += self.weight_decay * param.data

                self.v[param] = self.alpha * self.v[param] + (1 - self.alpha) * (param_grad ** 2)

                param.data -= self.lr * (param_grad / (_bx.sqrt(self.v[param]) + self.epsilon))

class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=None, epsilon=1e-8):
        super().__init__(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}
        self.epsilon = epsilon

        for param in self.params:
            self.m[param] = _bx.zeros_like(param.data)
            self.v[param] = _bx.zeros_like(param.data)

    def step(self):
        self.t += 1
        for param in self.params:
            if param.grad is not None:
                param_grad = _bx.copy(param.grad)

                if self.weight_decay is not None:
                    param_grad += self.weight_decay * param.data

                self.m[param] = self.b1 * self.m[param] + (1 - self.b1) * param_grad # momentum updates
                self.v[param] = self.b2 * self.v[param] + (1 - self.b2) * (param_grad ** 2) # variance in momentum updates

                # bias correction
                m_hat = self.m[param] / (1 - self.b1 ** self.t)
                v_hat = self.v[param] / (1 - self.b2 ** self.t)

                param.data -= self.lr * (m_hat / (_bx.sqrt(v_hat) + self.epsilon))

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=None, epsilon=1e-8):
        super().__init__(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}
        self.epsilon = epsilon

        for param in self.params:
            self.m[param] = _bx.zeros_like(param.data)
            self.v[param] = _bx.zeros_like(param.data)

    def step(self):
        self.t += 1
        for param in self.params:
            if param.grad is not None:
                if self.weight_decay is not None:
                    param.data -= self.lr * self.weight_decay * param.data # decoupled weight decay
                
                param_grad = _bx.copy(param.grad)

                self.m[param] = self.b1 * self.m[param] + (1 - self.b1) * param_grad
                self.v[param] = self.b2 * self.v[param] + (1 - self.b2) * (param_grad ** 2)

                m_hat = self.m[param] / (1 - self.b1 ** self.t)
                v_hat = self.v[param] / (1 - self.b2 ** self.t)

                param.data -= self.lr * (m_hat / (_bx.sqrt(v_hat) + self.epsilon))