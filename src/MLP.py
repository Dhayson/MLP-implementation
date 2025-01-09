import numpy as np
from math import sqrt
from enum import Enum
from collections.abc import Callable
from src.activation_functions import ActivationFunction

class InitializationType(Enum):
    zero = 0
    gaussian = 1

class MLP:
    weight_tensor: list[np.ndarray]
    bias_tensor: list[np.ndarray]
    input_dim: int
    internal_layers_dims: list[int]
    exit_dim: int
    activations: list[ActivationFunction]
    layers_io: list[tuple[int,int]]
    depth: int
    
    
    def __init__(self, input_dim, internal_layers_dims, exit_dim, activations):
        self.input_dim = input_dim
        self.internal_layers_dims = internal_layers_dims
        self.exit_dim = exit_dim
        self.activations = activations
        
        self.layers_io = []
        self.layers_io.append((self.input_dim, self.internal_layers_dims[0]))
        for i in range(0,len(self.internal_layers_dims)-1):
            self.layers_io.append((self.internal_layers_dims[i], self.internal_layers_dims[i+1]))
        i = len(self.internal_layers_dims) - 1
        if i >= 0:
            self.layers_io.append((self.internal_layers_dims[i], self.exit_dim))
        else:
            self.layers_io.append((self.input_dim, self.exit_dim))
            
        self.depth = len(self.layers_io)
        assert self.depth == len(self.activations)
        
    
    def initialize(self, initialization_type: InitializationType = InitializationType.gaussian):
        """Inicializa um MLP, definindo seus pesos e bias iniciais
        """
        self.weight_tensor = []
        for input, output in self.layers_io:
            neuron_weight = np.zeros(shape=(input, output))
            if initialization_type == InitializationType.gaussian:
                neuron_weight = np.random.normal(loc = 0, scale=sqrt(2/(input+output)), size=(input, output))
            self.weight_tensor.append(neuron_weight)
        self.bias_tensor = []
        for _input, output in self.layers_io:
            self.bias_tensor.append(np.zeros(shape=(output)))
            
        assert self.depth == len(self.weight_tensor)
        assert self.depth == len(self.weight_tensor)
        
            
    def predict(self, input: np.ndarray) -> np.ndarray:
        """Predição simples

        Args:
            input (np.ndarray): Vetor de entrada

        Returns:
            np.ndarray: Vetor de saída do MLP
        """
        wx = self.weight_tensor[0].transpose()@input
        z: np.ndarray = wx + self.bias_tensor[0]
        a: np.ndarray =  np.vectorize(pyfunc=self.activations[0].f)(z)
        for L in range(1, self.depth):
            wx = self.weight_tensor[L].transpose()@a
            z = wx + self.bias_tensor[L]
            a = np.vectorize(pyfunc=self.activations[L].f)(z)
        return a
    
    def loss(self, prediction: np.ndarray, expected: np.ndarray) -> float:
        assert len(prediction) == len(expected)
        loss = 0.0
        for i in range(len(prediction)):
            l = prediction[i] - expected[i]
            loss += 0.5*l*l
        return loss
        