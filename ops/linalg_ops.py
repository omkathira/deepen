from deepen.backend import active_backend as bx
from deepen.ops.utils import _reduce_grad # handles broadcasting for backward passes

_bx = bx() # backend singleton

# Inner product
class matmul:
    _save_data = ('x', 'x_shape', 'y', 'y_shape')

    @staticmethod
    def forward(save, x, y):
        output = _bx.matmul(x, y)

        if save.active:
            save.x, save.x_shape = x, x.shape
            save.y, save.y_shape = y, y.shape
        
        return output
    
    @staticmethod
    def backward(save, output_grad):
        dx = _reduce_grad(_bx.matmul(output_grad, save.y.T), save.x_shape)
        dy = _reduce_grad(_bx.matmul(save.x.T, output_grad), save.y_shape)
        return dx, dy

# Outer product
class outer:
    _save_data = ('x', 'x_shape', 'y', 'y_shape')

    @staticmethod
    def forward(save, x, y):
        output = _bx.outer(x, y)

        if save.active:
            save.x, save.x_shape = x, x.shape
            save.y, save.y_shape = y, y.shape
        
        return output
    
    @staticmethod
    def backward(save, output_grad):
        x_reshaped = _bx.reshape(save.x, (-1, 1))
        y_reshaped = _bx.reshape(save.y, (1, -1))
        dx = _reduce_grad(_bx.sum(_bx.multiply(output_grad, y_reshaped), axis=1), save.x_shape)
        dy = _reduce_grad(_bx.sum(_bx.multiply(output_grad, x_reshaped), axis=0), save.y_shape)
        return dx, dy

# Used for 2D convolutions
class _im2col:
    _save_data = ("x_shape", "k_h", "k_w", "H_out", "W_out", "stride", "padding")

    @staticmethod
    def forward(save, x, k_h, k_w, stride=1, padding=1):
        N, C, H, W = x.shape

        H_out = (H + 2 * padding - k_h) // stride + 1
        W_out = (W + 2 * padding - k_w) // stride + 1

        X_pad = _bx.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

        s_N, s_C, s_H, s_W = X_pad.strides

        patches = _bx.lib.stride_tricks.as_strided(
            X_pad,
            shape=(N, C, H_out, W_out, k_h, k_w),
            strides=(s_N, s_C, stride * s_H, stride * s_W, s_H, s_W)
        )

        patches = patches.transpose((1, 4, 5, 2, 3, 0)).reshape(C * k_h * k_w, H_out * W_out * N)

        if save.active:
            save.x_shape = x.shape
            save.k_h, save.k_w = k_h, k_w
            save.H_out, save.W_out = H_out, W_out
            save.stride = stride
            save.padding = padding

        return patches

    @staticmethod
    def backward(save, output_grad):
        N, C, H, W = save.x_shape

        output_grad = output_grad.reshape(C, save.k_h, save.k_w, save.H_out, save.W_out, N).transpose(5, 0, 3, 4, 1, 2)

        output_grad_pad = _bx.zeros((N, C, H + 2 * save.padding, W + 2 * save.padding))
        
        n_idx = _bx.arange(N)[:, None, None, None, None, None]
        
        c_idx = _bx.arange(C)[None, :, None, None, None, None]
        
        h_out_idx = _bx.arange(save.H_out)[None, None, :, None, None, None]
        w_out_idx = _bx.arange(save.W_out)[None, None, None, :, None, None]

        k_h_idx = _bx.arange(save.k_h)[None, None, None, None, :, None]
        k_w_idx = _bx.arange(save.k_w)[None, None, None, None, None, :]

        h_idx = h_out_idx * save.stride + k_h_idx
        w_idx = w_out_idx * save.stride + k_w_idx
        
        n_flat = _bx.broadcast_to(n_idx, output_grad.shape).ravel()
        c_flat = _bx.broadcast_to(c_idx, output_grad.shape).ravel()
        h_flat = _bx.broadcast_to(h_idx, output_grad.shape).ravel()
        w_flat = _bx.broadcast_to(w_idx, output_grad.shape).ravel()
        
        _bx.add.at(output_grad_pad, (n_flat, c_flat, h_flat, w_flat), output_grad.ravel())

        if save.padding > 0:
            return output_grad_pad[:, :, save.padding:-save.padding, save.padding:-save.padding],
        
        return output_grad_pad,

# Used for 2D upsampling (inverse of im2col)
class _col2im:
    _save_data = ("x_shape", "k_h", "k_w", "H_out", "W_out", "stride", "padding")

    @staticmethod
    def forward(save, x, output_shape, k_h, k_w, stride=1, padding=1):
        if save.active:
            save.x_shape = x.shape
            save.k_h, save.k_w = k_h, k_w
            save.stride = stride
            save.padding = padding
        
        N, C, H, W = output_shape

        H_out = (H + 2 * padding - k_h) // stride + 1
        W_out = (W + 2 * padding - k_w) // stride + 1

        if save.active:
            save.H_out, save.W_out = H_out, W_out

        output_grad = x.reshape(C, k_h, k_w, H_out, W_out, N).transpose(5, 0, 3, 4, 1, 2)

        output_grad_pad = _bx.zeros((N, C, H + 2 * padding, W + 2 * padding))
        
        n_idx = _bx.arange(N)[:, None, None, None, None, None]
        
        c_idx = _bx.arange(C)[None, :, None, None, None, None]
        
        h_out_idx = _bx.arange(H_out)[None, None, :, None, None, None]
        w_out_idx = _bx.arange(W_out)[None, None, None, :, None, None]

        k_h_idx = _bx.arange(k_h)[None, None, None, None, :, None]
        k_w_idx = _bx.arange(k_w)[None, None, None, None, None, :]

        h_idx = h_out_idx * stride + k_h_idx
        w_idx = w_out_idx * stride + k_w_idx
        
        n_flat = _bx.broadcast_to(n_idx, output_grad.shape).ravel()
        c_flat = _bx.broadcast_to(c_idx, output_grad.shape).ravel()
        h_flat = _bx.broadcast_to(h_idx, output_grad.shape).ravel()
        w_flat = _bx.broadcast_to(w_idx, output_grad.shape).ravel()
        
        _bx.add.at(output_grad_pad, (n_flat, c_flat, h_flat, w_flat), output_grad.ravel())

        if padding > 0:
            return output_grad_pad[:, :, padding:-padding, padding:-padding]
        
        return output_grad_pad

    @staticmethod
    def backward(save, output_grad):
        N, C, H, W = output_grad.shape

        X_pad = _bx.pad(output_grad, ((0, 0), (0, 0), (save.padding, save.padding), (save.padding, save.padding)))

        s_N, s_C, s_H, s_W = X_pad.strides

        patches = _bx.lib.stride_tricks.as_strided(
            X_pad,
            shape=(N, C, save.H_out, save.W_out, save.k_h, save.k_w),
            strides=(s_N, s_C, save.stride * s_H, save.stride * s_W, s_H, s_W)
        )

        patches = patches.transpose((1, 4, 5, 2, 3, 0)).reshape(C * save.k_h * save.k_w, save.H_out * save.W_out * N)

        return patches,

