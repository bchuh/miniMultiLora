"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor
import math
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
np.random.seed(10)
class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weight : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        ### BEGIN YOUR SOLUTION
        self.weight =  tensor_from_numpy(np.random.normal(0, 1, (num_embeddings, embedding_dim)),backend, requires_grad=True)
        self.weights = Parameter(self.weight)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        ### BEGIN YOUR SOLUTION
        x_onehot = one_hot(x, self.num_embeddings)
        x_onehot = x_onehot.view(x.shape[0]*x.shape[1], self.num_embeddings)
        output = x_onehot @ self.weights.value
        return output.view(x.shape[0], x.shape[1], self.embedding_dim)
        ### END YOUR SOLUTION

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.training = True
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)
        """
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = np.random.rand(*x.shape)

            mask = (mask > self.p_dropout).astype(int)
            mask = tensor_from_numpy( mask,x.backend, requires_grad=False)
            output = x * mask
            return output * (1-self.p_dropout)
        else:
            return x
        ### END YOUR SOLUTION

class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weight - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(1/in_size), 1/sqrt(1/in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(1/in_size), 1/sqrt(1/in_size)).
        """
        self.out_size = out_size
        ### BEGIN YOUR SOLUTION
        weight = tensor_from_numpy( np.random.uniform(-math.sqrt(1/in_size), math.sqrt(1/in_size), size=(in_size, out_size)),backend, requires_grad=True)
        self.weights = Parameter(weight)
        if bias:
            bias = tensor_from_numpy( np.random.uniform(-math.sqrt(1/in_size), math.sqrt(1/in_size), size=(out_size, )), backend=backend, requires_grad=True )
            self.bias = Parameter(bias)
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        #batch, in_size = x.shape
        orig_shape = None
        if len(x.shape) == 3:
            orig_shape = x.shape
            B, S, D = orig_shape
            x = x.view(B*S, D)

        ### BEGIN YOUR SOLUTION
        output = x @ self.weights.value
        if self.bias is not None:
            output += self.bias.value
        if orig_shape is not None:
            output = output.view(B, S, self.out_size)
        return output
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        weights = ones_tensor_from_numpy((self.dim, ), backend=backend)
        self.weights = Parameter(weights)
        bias = zeros_tensor_from_numpy((self.dim,), backend=backend)
        self.bias = Parameter(bias)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        ### BEGIN YOUR SOLUTION
        mean = x.mean(dim=1)
        std_0 = (x.var(dim=1) + self.eps)
        std = std_0**0.5
        normalized_x = (x - mean) / std
        output = self.weights.value * normalized_x + self.bias.value
        return output
        ### END YOUR SOLUTION