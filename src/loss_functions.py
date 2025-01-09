import math
import abc
import numpy as np

class LossFunction(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def loss(self, prediction: np.ndarray, expected: np.ndarray) -> float:
        return
    
    @abc.abstractmethod
    def loss_der(self, prediction: np.ndarray, expected: np.ndarray) -> np.ndarray:
        return
    
class MSE(LossFunction):
    def loss(self, prediction: np.ndarray, expected: np.ndarray) -> float:
        loss_val = 0.0
        for i in range(len(prediction)):
            l = prediction[i] - expected[i]
            loss_val += 0.5*l*l
        return loss_val
    
    def loss_der(self, prediction: np.ndarray, expected: np.ndarray) -> np.ndarray:
        return prediction - expected
    
class MAE(LossFunction):
    def loss(self, prediction: np.ndarray, expected: np.ndarray) -> float:
        loss_val = 0.0
        for i in range(len(prediction)):
            l = abs(prediction[i] - expected[i])
            loss_val += l
        return loss_val
    
    def loss_der(self, prediction: np.ndarray, expected: np.ndarray) -> np.ndarray:
        result = []
        for i in range(len(prediction)):
            if prediction[i] > expected[i]:
                result.append(1)
            elif prediction[i] < expected[i]:
                result.append(-1)
            else:
                result.append(0)
        return np.array(result)
    
    