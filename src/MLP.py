import numpy as np
import pandas as pd
from math import sqrt
from enum import Enum
from collections.abc import Callable
from src.activation_functions import ActivationFunction
from src.loss_functions import LossFunction, MSE
from src.optimization import TrainOptimization

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
    loss: LossFunction
    train_optimization: TrainOptimization
    
    layers_io: list[tuple[int,int]]
    depth: int
    
    t: int = 0
    
    def __init__(
        self, 
        input_dim: int, 
        internal_layers_dims: list[int], 
        exit_dim: int, 
        activations: list[ActivationFunction], 
        loss: LossFunction = MSE(),
        train_optimization: TrainOptimization = None
    ):
        self.input_dim = input_dim
        self.internal_layers_dims = internal_layers_dims
        self.exit_dim = exit_dim
        self.activations = activations
        self.loss = loss
        self.train_optimization = train_optimization
        
        self.layers_io = []
        i = len(self.internal_layers_dims) - 1
        if i >= 0:
            self.layers_io.append((self.input_dim, self.internal_layers_dims[0]))
            for j in range(0,len(self.internal_layers_dims)-1):
                self.layers_io.append((self.internal_layers_dims[j], self.internal_layers_dims[j+1]))
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
            neuron_weight = np.zeros(shape=(input, output), dtype='float64')
            if initialization_type == InitializationType.gaussian:
                neuron_weight = np.random.normal(loc = 0, scale=sqrt(2/(input+output)), size=(input, output))
            self.weight_tensor.append(neuron_weight)
            
        self.bias_tensor = []
        for _input, output in self.layers_io:
            self.bias_tensor.append(np.zeros(shape=(output), dtype='float64'))
            
        if self.train_optimization != None:
            self.train_optimization.initialize(layers_io = self.layers_io)
            
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
    
    def forward_propagate(
        self, 
        input: np.ndarray, 
        expected: np.ndarray
    ) -> tuple[float, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Realiza o forward propagation, guardando os valores de ativação intermediários

        Args:
            input (np.ndarray): Vetor de entrada

        Returns:
            float: Resultado de aplicar o loss
            np.ndarray: Vetor de saída do MLP
            list[np.ndarray]: Valores de ativação das camadas, com o primeiro valor sendo a entrada
            list[np.ndarray]: Valores de ativação intermediário das camadas
        """
        activation_tensor = []
        activation_tensor.append(input)
        intermediate_tensor = []
        wx = self.weight_tensor[0].transpose()@input
        z: np.ndarray = wx + self.bias_tensor[0]
        intermediate_tensor.append(z)
        a: np.ndarray = np.vectorize(pyfunc=self.activations[0].f)(z)
        activation_tensor.append(a)
        for L in range(1, self.depth):
            wx = self.weight_tensor[L].transpose()@a
            z = wx + self.bias_tensor[L]
            intermediate_tensor.append(z)
            a = np.vectorize(pyfunc=self.activations[L].f)(z)
            activation_tensor.append(a)
        output = a
        loss = self.loss.loss(output, expected)
        return (loss, output, activation_tensor, intermediate_tensor)

    def backward_propagate(
        self, 
        output: np.ndarray, 
        expected: np.ndarray, 
        activation_tensor: list[np.ndarray], 
        intermediate_tensor: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Realiza o backward propagation, calculando o gradiente dos pesos e bias

        Args:
            output (np.ndarray): Output do MLP
            expected (np.ndarray): Valor esperado
            activation_tensor (list[np.ndarray]): Valores de ativação calculados no forward propagation, incluindo o input inicialmente
            intermediate_tensor (list[np.ndarray]): Valores intermediários calculados no forward propagation

        Returns:
            list[np.ndarray]: Gradiente dos pesos
            list[np.ndarray]: Gradiente dos bias
        """
        weight_gradient: list[np.ndarray] = []
        bias_gradient: list[np.ndarray] = []
        
        loss_der = self.loss.loss_der(output, expected)
        flag = "None"
        if type(loss_der) == str:
            flag = loss_der
        else:
            grad_activation = loss_der
        
        for L in reversed(range(self.depth)):
            if flag == "FlagEXP":
                grad_intermediate: np.ndarray = np.vectorize(pyfunc=self.activations[L].f_der_exp)(expected, None, output)
                flag = "None"
            else:
                grad_intermediate: np.ndarray = grad_activation*np.vectorize(pyfunc=self.activations[L].f_der)(intermediate_tensor[L], activation_tensor[L+1])
                
            grad_weight_L = []
            for i in range(len(grad_intermediate)):
                grad_int_i: float = grad_intermediate[i]
                grad_weight_L_i = grad_int_i*activation_tensor[L]
                grad_weight_L.append(grad_weight_L_i)
            weight_gradient.append(np.array(grad_weight_L, dtype='float64').transpose())
            
            grad_bias_L = grad_intermediate*1 # Bias não multiplica com ninguém
            bias_gradient.append(np.array(grad_bias_L, dtype='float64'))
            
            grad_activation_to_sum = []
            for i in range(len(grad_intermediate)):
                gats_i = grad_intermediate[i]*(self.weight_tensor[L].transpose()[i])
                grad_activation_to_sum.append(gats_i)
            grad_activation = sum(grad_activation_to_sum)
            
        weight_gradient.reverse()
        bias_gradient.reverse()
        for i in range(len(weight_gradient)):
            assert weight_gradient[i].shape == self.weight_tensor[i].shape
        for i in range(len(bias_gradient)):
            assert bias_gradient[i].shape == self.bias_tensor[i].shape
        return weight_gradient, bias_gradient
    
    def train(self, dataset: pd.DataFrame, expected: pd.DataFrame, learning_rate = 0.4, sample = "Batch", n=-1):
        assert len(dataset) == len(expected)
        if sample == "Batch":
            n = len(dataset)
        elif sample == "Stochastic":
            n = 1
        elif sample == "Minibatch":
            n = n
        assert n > 0
        
        # Coleta uma amostra de n elementos aleatórios
        dataset_sample = dataset.sample(n=n)
        expected_sample= expected.loc[dataset_sample.index]
        
        total_gradient_w = []
        for i in range(len(self.weight_tensor)):
            total_gradient_w.append(np.zeros_like(self.weight_tensor[i], dtype='float64'))
        total_gradient_b = []
        for i in range(len(self.bias_tensor)):
            total_gradient_b.append(np.zeros_like(self.bias_tensor[i], dtype='float64'))
        total_loss = 0.0
        for i in dataset_sample.index:
            loss, output, activate, intermediate = self.forward_propagate(dataset_sample.loc[i].to_numpy(), expected_sample.loc[i].to_numpy())
            grad_w, grad_b = self.backward_propagate(output, expected_sample.loc[i].to_numpy(), activate, intermediate)
            total_loss += loss
            for i in range(len(self.weight_tensor)):
                total_gradient_w[i] += grad_w[i]
            for i in range(len(self.bias_tensor)):
                total_gradient_b[i] += grad_b[i]
        
        total_loss /= n 
        for i in range(len(self.weight_tensor)):
            total_gradient_w[i] /= n 
        for i in range(len(self.bias_tensor)):
            total_gradient_b[i] /= n
        
        
        if self.train_optimization == None:
            factor_w = total_gradient_w
            factor_b = total_gradient_b
        else:
            factor_w, factor_b = self.train_optimization.apply(total_gradient_w, total_gradient_b, self.t)
            
        for i in range(len(self.weight_tensor)):
            self.weight_tensor[i] -= learning_rate*factor_w[i]
        for i in range(len(self.bias_tensor)):
            self.bias_tensor[i] -= learning_rate*factor_b[i]
        
        self.t += 1
        
        return total_loss
    
    def eval(self, dataset: pd.DataFrame, expected: pd.DataFrame, kind="None", tmax = None, tmin = None, do_print = False):
        assert len(dataset) == len(expected)
        n = len(dataset)
        
        train_loss = 0.0
        total_gradient_w = []
        for i in range(len(self.weight_tensor)):
            total_gradient_w.append(np.zeros_like(self.weight_tensor[i], dtype='float64'))
        total_gradient_b = []
        for i in range(len(self.bias_tensor)):
            total_gradient_b.append(np.zeros_like(self.bias_tensor[i], dtype='float64'))
        classification_pred = []
        classification_exp = []
        
        regression_pred = []
        regression_exp = []
        
        
        for i in dataset.index:
            output = self.predict(dataset.loc[i].to_numpy())
            if kind == "Classification":
                classification_pred.append((np.argmax(output), i))
                classification_exp.append(np.argmax(expected.loc[i].to_numpy()))
            if kind == "Regression":
                regression_pred.append((output*(tmax-tmin) + tmin, i))
                regression_exp.append(expected.loc[i].to_numpy()*(tmax-tmin) + tmin)
            train_loss += self.loss.loss(output, expected.loc[i])
        
        n_wrong = 0
        if kind == "Classification":
            for i in range(len(classification_pred)):
                if classification_pred[i][0] != classification_exp[i]:
                    n_wrong += 1
                    if do_print:
                        print(f"missed: {classification_pred[i]} expected {classification_exp[i]}")       
               
        rmse = 0
        mae = 0       
        if kind == "Regression":
            for i in range(len(regression_pred)):
                if do_print:
                    print(f"{regression_pred[i]}")
                    print("expected")
                    print(regression_exp[i])
                    print()
                l = regression_pred[i][0] - regression_exp[i]
                rmse += l*l
                mae += abs(l)
                
        
        train_loss /= n
        
        if kind == "Classification":
            accuracy = (n-n_wrong)/n
            return train_loss, accuracy
        elif kind == "Regression":
            rmse = sqrt(rmse.iloc[0]/n)
            mae = mae/n
            return train_loss, rmse, mae
        else:
            return train_loss, None

        
         
        
            
        
        
    
        