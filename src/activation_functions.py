import math
import abc

class ActivationFunction(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def f(self, input: float) -> float:
        return
    
    @abc.abstractmethod
    def f_der(self, input: float, output: float) -> float:
        return
    
    @abc.abstractmethod
    def f_der_exp(self, expected: float, input: float, output: float) -> float:
        return

class ReLU(ActivationFunction):
    def f(self, input: float) -> float:
        if input > 0:
            return input
        else:
            return 0
        
    def f_der(self, input: float, output: float) -> float:
        if input > 0:
            return 1
        else:
            return 0

class Sigmoid(ActivationFunction):
    def f(self, input: float) -> float:
        exp_x = math.exp(input)
        return exp_x/(1 + exp_x)
        
    def f_der(self, input: float, output: float) -> float:
        return output*(1 - output)
    
class SigmoidBeforeCE(ActivationFunction):
    def f(self, input: float) -> float:
        exp_x = math.exp(input)
        return exp_x/(1 + exp_x)
        
    def f_der(self, input: float, output: float) -> float:
        raise Exception('Não deve ser chamado')
    
    def f_der_exp(self, expected: float, input, output: float):
        return output - expected

class Linear(ActivationFunction):
    def f(self, input: float) -> float:
        return input
    
    def f_der(self, input: float, output: float) -> float:
        return 1

class Tahn(ActivationFunction):
    def f(self, input: float) -> float:
        exp_2x = math.exp(-2*input)
        return (1-exp_2x)/(1+exp_2x)
    
    def f_der(self, input, output):
        return 1 - output*output
    
class LeakyReLU(ActivationFunction):
    alpha: int
    def __init__(self, alpha: int):
        super().__init__()
        self.alpha = alpha
        
    def f(self, input: float) -> float:
        if input > 0:
            return input
        else:
            return self.alpha*input
        
    def f_der(self, input: float, output: float) -> float:
        if input > 0:
            return 1
        else:
            return self.alpha