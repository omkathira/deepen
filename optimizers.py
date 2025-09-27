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
        self.m = {} # momentum

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
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), epsilon=1e-8, weight_decay=None):
        super().__init__(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {} # momentum
        self.v = {} # variance

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
                m_cor = self.m[param] / (1 - self.b1 ** self.t)
                v_cor = self.v[param] / (1 - self.b2 ** self.t)

                param.data -= self.lr * (m_cor / (_bx.sqrt(v_cor) + self.epsilon))

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), epsilon=1e-8, weight_decay=None):
        super().__init__(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}

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

                m_cor = self.m[param] / (1 - self.b1 ** self.t)
                v_cor = self.v[param] / (1 - self.b2 ** self.t)

                param.data -= self.lr * (m_cor / (_bx.sqrt(v_cor) + self.epsilon))