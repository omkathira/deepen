from deepen.core.tensor import Tensor
from deepen.layers import Layer, Linear

class RNNCell(Layer):
    """
    Basic RNN Cell implementation using Linear layers.
    
    Computes: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
    
    Args:
        input_size (int): Size of input features
        hidden_size (int): Size of hidden state
    """
    
    def __init__(self, in_size, h_size):
        super().__init__()
        self.in_size = in_size
        self.h_size = h_size

        self.in_to_h = Linear(in_size, h_size) # input-to-hidden transform
        self.h_to_h = Linear(h_size, h_size, bias=False) # hidden-to-hidden transform
    
    def forward(self, x_t, h_prev): # computes: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
        in_cont = self.in_to_h.forward(x_t) # W_xh * x_t
        h_cont = self.h_to_h.forward(h_prev) # W_hh * h_{t-1}
        combined = in_cont + h_cont # W_hh * h_{t-1} + b + W_xh * x_t
        h_t = combined.tanh() # tanh(W_hh * h_{t-1} + b + W_xh * x_t)
        return h_t
    
    def init_h(self, batch_size):
        return Tensor.zeros((batch_size, self.h_size)) # create initial hidden state

class RNN(Layer):
    """
    RNN Layer that processes entire sequences using Linear-based RNN cells.
    
    Handles multiple timesteps and implements Backpropagation Through Time (BPTT).
    Uses Linear layers for cleaner parameter management and automatic registration.
    
    Args:
        input_size (int): Size of input features
        hidden_size (int): Size of hidden state  
        num_layers (int): Number of stacked RNN layers (default: 1)
    """
    
    def __init__(self, in_size, h_size, out_size, num_layers=1):
        super().__init__()
        self.in_size = in_size
        self.h_size = h_size
        self.num_layers = num_layers
        
        self.cells = [RNNCell(self.in_size, self.h_size) for _ in range(num_layers)] # create a stack of RNN cells
        
        # Optional: Output projection layer using Linear
        # Maps final hidden state to desired output size
        self.output = Linear(h_size, out_size)
    
    def forward(self, x_seq, h_0=None):
        """
        Forward pass through entire sequence using Linear-based RNN cells.
        
        Args:
            x_sequence (Tensor): Input sequence, shape (seq_len, batch_size, input_size)
            h_0 (Tensor, optional): Initial hidden state, shape (num_layers, batch_size, hidden_size)
            
        Returns:
            outputs (Tensor): Output sequence, shape (seq_len, batch_size, hidden_size)
            h_final (Tensor): Final hidden state, shape (num_layers, batch_size, hidden_size)
        """
        
        seq_len, batch_size, _ = x_seq.shape
        
        if h_0 is None: h_0 = Tensor.zeros((self.num_layers, batch_size, self.h_size), requires_grad=True) # initialize hidden state
        
        h_states = [[] for _ in range(self.num_layers)]
        output = []
        
        # Process sequence timestep by timestep
        # TODO: Loop through sequence length
        for t in range(seq_len):
            x_t = x_seq[t, :, :] # extract x_t from sequence at current timestep

            current_in = x_t
            
            for layer_idx in range(self.num_layers): # iterate through each RNNCell layer
                h_prev = h_0[layer_idx, :, :] # get the previous hidden state for the RNNCell layer
                
                if layer_idx == 0:
                    h_t = self.cells[layer_idx].forward(x_t, h_prev) # call RNNCell's forward method
                else:
                    h_t = self.cells[layer_idx].forward(current_in, h_prev)

                # Store hidden state for BPTT
                # TODO: Save h_t to hidden_states
                h_states[layer_idx].append(h_t)
                
                # Update hidden state for next timestep
                # TODO: Update h_0 with new hidden state
                h_0[layer_idx, :, :] = h_t

                current_in = h_t
            
            # Maps hidden state to desired output dimension
            # TODO: Project hidden state to output if needed
            out_proj = self.output.forward(h_t)
            
            # Store output for this timestep
            # TODO: Save output to outputs list
            output.append(out_proj)
        
        # Convert outputs to tensor for return
        # Stack outputs into tensor: (seq_len, batch_size, output_size)
        outputs_tensor = None
        
        return outputs_tensor, h_0
    
    def init_hidden(self, batch_size):
        """
        Initialize hidden states for all layers using Linear-based RNN cells.
        
        Args:
            batch_size (int): Number of samples in batch
            
        Returns:
            h_0 (Tensor): Initial hidden states, shape (num_layers, batch_size, hidden_size)
        """
        
        # Create initial hidden states for all layers
        # Each layer needs its own initial hidden state
        # TODO: Create initial hidden states for all layers
        h_0 = None
        
        return h_0