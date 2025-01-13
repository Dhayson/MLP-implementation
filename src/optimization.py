import math
import abc
import numpy as np

class TrainOptimization(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def initialize(self, **kwargs):
        pass
    
    @abc.abstractmethod
    def apply(self, grad_w: list[np.ndarray], grad_b: list[np.ndarray], t: int) -> tuple[float, float]:
        pass
    
class Adagrad(TrainOptimization):
    s_w_tensor: list[np.ndarray]
    s_b_tensor: list[np.ndarray]
    power: float
    delay_time: int
    
    def __init__(self, power = -0.5, delay_time = 100):
        super().__init__()
        self.power = power
        self.delay_time = delay_time
    
    def initialize(self, **kwargs):
        layers_io: list[tuple[int,int]] = kwargs.get('layers_io')
        self.s_w_tensor = []
        for input, output in layers_io:
            # Inicializa com esse valor para não somar toda vez, sendo matematicamente equivalente
            neuron_weight = np.zeros(shape=(input, output), dtype='float64')+0.000001
            self.s_w_tensor.append(neuron_weight)
            
        self.s_b_tensor = []
        for _input, output in layers_io:
            self.s_b_tensor.append(np.zeros(shape=(output), dtype='float64')+0.000001)
            
    def apply(self, grad_w: list[np.ndarray], grad_b: list[np.ndarray], t: int):
        # Gerar o quadrado do gradiente
        grad_w_2 = []
        for i in range(len(grad_w)):
            grad_w_2.append(np.power(grad_w[i], 2))
        grad_b_2 = []
        for i in range(len(grad_w)):
            grad_b_2.append(np.power(grad_b[i],2))
        
        # Atualizar variável de acumulação
        for i in range(len(grad_w)):
            self.s_w_tensor[i] += grad_w_2[i]
        for i in range(len(grad_b)):
            self.s_b_tensor[i] += grad_b_2[i]
        
        # Aplicar fator
        return_w = []
        return_b = []
        for i in range(len(grad_w)):
            factor_w = np.power(self.s_w_tensor[i], self.power)
            return_w.append(grad_w[i]*factor_w)
            factor_b = np.power(self.s_b_tensor[i], self.power)
            return_b.append(grad_b[i]*factor_b)
            # print(f"S weight {i} magnitude is {np.linalg.norm(self.s_w_tensor[i])}")
            # print(f"Factor w {i} magnitude is {np.linalg.norm(factor_w)}")
            # print(f"Return w {i} magnitude is {np.linalg.norm(return_w[i])}")
            
            
            # print(f"S bias {i} magnitude is {np.linalg.norm(self.s_b_tensor[i])}")
            # print(f"Factor b {i} magnitude is {np.linalg.norm(factor_b)}")
            # print(f"Return b {i} magnitude is {np.linalg.norm(return_b[i])}")
            # print()
        
        # Não aplica no começo, para evitar explosão do gradiente
        if t > self.delay_time:
            return return_w, return_b
        else:
            return grad_w, grad_b