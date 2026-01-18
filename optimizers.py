from deepen.backend import active_backend as bx

_bx = bx() # backend singleton

class Optimizer:
    def __init__(self, params):
        params = list(params)
        self._params = {id(p): p for p in params}
        self._param_order = [id(p) for p in params]

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=None, weight_decay=None):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.m = {} # stores running average of gradients

        for p_id in self._param_order:
            self.m[p_id] = _bx.zeros_like(self._params[p_id].data)
    
    def step(self):
        for p_id in self._param_order:
            p = self._params[p_id]

            if p.grad is not None:
                # normally, we'd create a copy (or, call .detach()) to avoid modifying gradients in-place
                # but, our gradients here are not tensors - they're just backend arrays
                p_grad = p.grad

                if self.weight_decay is not None:
                    p_grad = p_grad + self.weight_decay * p.data # coupled weight decay

                if self.momentum is not None:
                    self.m[p_id] = self.momentum * self.m[p_id] + p_grad
                    p_grad = self.m[p_id]
                
                p.data -= self.lr * p_grad

class RMSprop(Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.99, weight_decay=None, epsilon=1e-8):
        super().__init__(params)
        self.lr = lr
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.v = {} # stores running average of squared gradients
        self.epsilon = epsilon

        for p_id in self._param_order:
            self.v[p_id] = _bx.zeros_like(self._params[p_id].data)
    
    def step(self):
        for p_id in self._param_order:
            p = self._params[p_id]

            if p.grad is not None:
                p_grad = p.grad

                if self.weight_decay is not None:
                    p_grad = p_grad + self.weight_decay * p.data

                self.v[p_id] = self.alpha * self.v[p_id] + (1 - self.alpha) * (p_grad ** 2)

                p.data -= self.lr * (p_grad / (_bx.sqrt(self.v[p_id]) + self.epsilon))

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

        for p_id in self._param_order:
            self.m[p_id] = _bx.zeros_like(self._params[p_id].data)
            self.v[p_id] = _bx.zeros_like(self._params[p_id].data)

    def step(self):
        self.t += 1
        for p_id in self._param_order:
            p = self._params[p_id]

            if p.grad is not None:
                p_grad = p.grad

                if self.weight_decay is not None:
                    p_grad = p_grad + self.weight_decay * p.data

                # biased momentum and variance updates
                self.m[p_id] = self.b1 * self.m[p_id] + (1 - self.b1) * p_grad
                self.v[p_id] = self.b2 * self.v[p_id] + (1 - self.b2) * (p_grad ** 2)

                # bias correction
                m_hat = self.m[p_id] / (1 - self.b1 ** self.t)
                v_hat = self.v[p_id] / (1 - self.b2 ** self.t)

                p.data -= self.lr * (m_hat / (_bx.sqrt(v_hat) + self.epsilon))

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

        for p_id in self._param_order:
            self.m[p_id] = _bx.zeros_like(self._params[p_id].data)
            self.v[p_id] = _bx.zeros_like(self._params[p_id].data)

    def step(self):
        self.t += 1
        for p_id in self._param_order:
            p = self._params[p_id]

            if p.grad is not None:
                if self.weight_decay is not None:
                    p.data -= self.lr * self.weight_decay * p.data # decoupled weight decay
                
                p_grad = p.grad

                self.m[p_id] = self.b1 * self.m[p_id] + (1 - self.b1) * p_grad
                self.v[p_id] = self.b2 * self.v[p_id] + (1 - self.b2) * (p_grad ** 2)

                m_hat = self.m[p_id] / (1 - self.b1 ** self.t)
                v_hat = self.v[p_id] / (1 - self.b2 ** self.t)

                p.data -= self.lr * (m_hat / (_bx.sqrt(v_hat) + self.epsilon))

class Muon(Optimizer):
    pass