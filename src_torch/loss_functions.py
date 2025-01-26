import math
import abc
from torch import Tensor
import torch
import numpy as np

class LossFunction(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def loss(self, prediction: Tensor, expected: Tensor) -> Tensor:
        return
    
    @abc.abstractmethod
    def loss_der(self, prediction: Tensor, expected: Tensor) -> Tensor:
        return
    
class MSE(LossFunction):
    def loss(self, prediction: Tensor, expected: Tensor) -> Tensor:
        loss_val = 0.0
        for i in range(len(prediction)):
            l = prediction[i] - expected[i]
            loss_val += 0.5*l*l
        return loss_val
    
    def loss_der(self, prediction: Tensor, expected: Tensor) -> Tensor:
        return prediction - expected
    
class MAE(LossFunction):
    def loss(self, prediction: Tensor, expected: Tensor) -> Tensor:
        loss_val = 0.0
        for i in range(len(prediction)):
            l = abs(prediction[i] - expected[i])
            loss_val += l
        return loss_val
    
    def loss_der(self, prediction: Tensor, expected: Tensor) -> Tensor:
        result = []
        for i in range(len(prediction)):
            if prediction[i] > expected[i]:
                result.append(1)
            elif prediction[i] < expected[i]:
                result.append(-1)
            else:
                result.append(0)
        return torch.from_numpy(np.array(result))
    
class CrossEntropy(LossFunction):
    def loss(self, prediction, expected, device):
        sum = torch.zeros(1, device=device)
        for p, e in zip(prediction, expected):
            p = p*torch.tensor(0.99999, device=device) + torch.tensor(0.000005, device=device)
            cross = -(e*torch.log(p)+(torch.tensor(1, device=device)-e)*torch.log(torch.tensor(1, device=device)-p))
            sum += cross 
        return sum
    
    def loss_der(self, prediction: Tensor, expected: Tensor):
        prediction = prediction*torch.tensor(0.99999) + torch.tensor(0.000005)
        return -(expected/prediction - (1-expected)/(1-prediction))
    
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
    
    