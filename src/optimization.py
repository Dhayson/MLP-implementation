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
    do_print: tuple[bool, int]
    print_kind: str
    
    def __init__(self, power = 0.5, delay_time = 100, do_print = (False, 100), print_kind = "Full"):
        super().__init__()
        self.power = power
        self.delay_time = delay_time
        self.do_print = do_print
        self.print_kind = print_kind
    
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
            return_w.append(grad_w[i]/factor_w)
            factor_b = np.power(self.s_b_tensor[i], self.power)
            return_b.append(grad_b[i]/factor_b)
            if self.do_print[0] and t%self.do_print[1] == 0 and self.print_kind == "Full":
                print(f"Grad w   {i} is {grad_w[i]}")
                print(f"S weight {i} is {self.s_w_tensor[i]}")
                print(f"Factor w {i} is {factor_w}")
                print(f"Return w {i} is {return_w[i]}")
                print(f"Grad b   {i} is {grad_b[i]}")
                print(f"S bias   {i} is {self.s_b_tensor[i]}")
                print(f"Factor b {i} is {factor_b}")
                print(f"Return b {i} is {return_b[i]}")
                print()
                pass
                
        
        # Não aplica no começo, para evitar explosão do gradiente
        if t > self.delay_time:
            return return_w, return_b
        else:
            return grad_w, grad_b