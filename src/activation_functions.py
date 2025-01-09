import math
import abc

class ActivationFunction(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def f(input: float) -> float:
        return
    
    @abc.abstractmethod
    def f_der(input: float, output: float) -> float:
        return

class ReLu(ActivationFunction):
    def f(input: float) -> float:
        if input > 0:
            return input
        else:
            return 0
        
    def f_der(input: float, output: float) -> float:
        if input > 0:
            return 1
        else:
            return 0

class Sigmoid(ActivationFunction):
    def f(input: float) -> float:
        exp_x = math.exp(input)
        return exp_x/(1 + exp_x)
        
    def f_der(input: float, output: float) -> float:
        return output*(1 - output)

class Linear(ActivationFunction):
    def f(input: float) -> float:
        return input
    
    def f_der(input: float, output: float) -> float:
        return 1