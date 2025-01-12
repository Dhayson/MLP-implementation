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
    
class CrossEntropy(LossFunction):
    def loss(self, prediction, expected):
        sum = 0
        for p, e in zip(prediction, expected):
            p = p*0.99999 + 0.000005
            sum -= e*math.log(p)+(1-e)*math.log(1-p)
        return sum
    
    def loss_der(self, prediction: np.ndarray, expected: np.ndarray):
        prediction = prediction*0.99999 + 0.000005
        return -(expected/prediction - (1-expected)/(1-prediction))

class CrossEntropyAfterSigmoid(LossFunction):
    def loss(self, prediction, expected):
        sum = 0
        for p, e in zip(prediction, expected):
            p = p*0.99999 + 0.000005
            sum -= e*math.log(p)+(1-e)*math.log(1-p)
        return sum
    
    def loss_der(self, prediction: np.ndarray, expected: np.ndarray):
        return "FlagEXP"
    
class MQE(LossFunction):
    def loss(self, prediction, expected):
        loss_val = 0.0
        for i in range(len(prediction)):
            l = prediction[i] - expected[i]
            loss_val += 0.25*l*l*l*l
        return loss_val
    
    def loss_der(self, prediction, expected):
        pred_sq = prediction*prediction
        exp_sq = expected*expected
        return prediction*pred_sq - 3*pred_sq*expected + 3*prediction*exp_sq - expected*exp_sq
    
    