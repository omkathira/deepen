from deepen.layers import *

# Sequential model builder
class Sequential(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

        for i, layer in enumerate(layers):
            setattr(self, f'layer_{i}', layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Activation function wrapper (turns activation functions into usable layers)
class Activation(Layer):
    def __init__(self, activation_name, **kwargs):
        super().__init__()
        self.activation_func = getattr(Tensor, activation_name)
        self.kwargs = kwargs
    
    def forward(self, x):
        return self.activation_func(x, **self.kwargs)

# Factory for activation functions
def Sigmoid(): return Activation('sigmoid')
def Tanh(): return Activation('tanh')
def ReLU(): return Activation('relu')
def LeakyReLU(neg_slope=0.1): return Activation('leaky_relu', neg_slope=neg_slope)
def Swish(): return Activation('swish')

# Block for residuals
class ResidualBlock(Layer):
    pass

# Blocks for Recurrent Neural Networks (RNNs)
class LSTMCell(Layer):
    pass

class GRUCell(Layer):
    pass

# Blocks for Autoencoders (AEs), Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), etc

class EncoderBlock(Layer): # equivalent to a GeneratorBlock
    pass

class DecoderBlock(Layer): # equivalent to a DiscriminatorBlock
    pass

# Blocks for Transformers