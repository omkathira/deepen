from deepen.backend import active_backend as bx

_bx = bx() # backend singleton

# Squeeze
class squeeze:
    @staticmethod
    def forward(save, x, axes=None):
        if axes is None:
            output = _bx.squeeze(x)
        else:
            if isinstance(axes, int):
                ax = axes if axes >= 0 else axes + x.ndim # handle possible negative axis
                output = _bx.squeeze(x, axis=ax)
            else:
                norm_axes = tuple(ax if ax >= 0 else ax + x.ndim for ax in axes) # handle multiple possible negative axes
                output = _bx.squeeze(x, axis=norm_axes)

        if save.active:
            save.x_shape = x.shape

        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _bx.reshape(output_grad, save.x_shape)
        return dx,

# Unsqueeze
class unsqueeze:
    @staticmethod
    def forward(save, x, axes=None):
        if axes is None:
            raise ValueError("unsqueeze requires axes (int or iterable)")

        if isinstance(axes, int):
            ax = axes if axes >= 0 else axes + x.ndim + 1 # handle possible negative axis
            output = _bx.expand_dims(x, axis=ax)
        else:
            norm_axes = [ax if ax >= 0 else ax + x.ndim + 1 for ax in axes] # handle multiple possible negative axes
            output = x
            for ax in sorted(norm_axes, reverse=True):
                output = _bx.expand_dims(output, axis=ax)

        if save.active:
            save.x_shape = x.shape

        return output

    @staticmethod
    def backward(save, output_grad):
        dx = _bx.reshape(output_grad, save.x_shape)
        return dx,

# Transpose
class transpose:
    @staticmethod
    def forward(save, x, axes=None):
        output = _bx.transpose(x, axes)

        if save.active:
            save.axes = axes

        return output
    
    @staticmethod
    def backward(save, output_grad):
        # move back to native shape to accumulate gradients
        if save.axes is None:
            dx = _bx.transpose(output_grad)
        else:
            inverted_axes = [0] * len(save.axes)

            for i, ax in enumerate(save.axes):
                inverted_axes[ax] = i

            dx = _bx.transpose(output_grad, tuple(inverted_axes))

        return dx,

# Concatenate
class concatenate:
    @staticmethod
    def forward(save, x, y, axes=None):
        output = _bx.concatenate([x, y], axis=axes)

        if save.active:
            save.x_end = x.shape[axes]
            save.axes = axes

        return output

    @staticmethod
    def backward(save, output_grad):
        # dx = output_grad[..., :save.x_end, ...]
        # dy = output_grad[..., save.x_end:, ...]

        axis = 0 if save.axes is None else save.axes
        ndim = output_grad.ndim
        axis = axis if axis >= 0 else axis + ndim

        idx_x = [slice(None)] * ndim
        idx_y = [slice(None)] * ndim
        idx_x[axis] = slice(0, save.x_end)
        idx_y[axis] = slice(save.x_end, None)

        dx = output_grad[tuple(idx_x)]
        dy = output_grad[tuple(idx_y)]

        return dx, dy

# Reshape
class reshape:
    @staticmethod
    def forward(save, x, shape):
        output = _bx.reshape(x, shape)

        if save.active:
            save.x_shape = x.shape

        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = _bx.reshape(output_grad, save.x_shape)
        return dx,