from deepen.backend import active_backend as bx

_bx = bx() # backend singleton

# Gather
class gather:
    @staticmethod
    def forward(save, x, indices):
        output = x[indices]
        save.x_shape, save.indices = x.shape, indices
        return output
    
    @staticmethod
    def backward(save, output_grad):
        # Create zero gradient for source tensor
        dx = _bx.zeros(save.x_shape)
        
        # Flatten indices and output_grad for easier processing
        flat_indices = save.indices.reshape(-1)
        flat_grad = output_grad.reshape(-1, save.x_shape[1])
        
        # Scatter-add gradients back to source positions
        # This handles repeated indices by accumulating gradients
        for i, idx in enumerate(flat_indices):
            dx[idx] += flat_grad[i]
        
        return dx, None