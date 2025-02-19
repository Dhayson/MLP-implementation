import math
import abc
from torch import Tensor
import torch
import numpy as np

class LossFunction(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def loss(self, prediction: Tensor, expected: Tensor, device) -> Tensor:
        return
    
class MSE(LossFunction):
    def loss(self, prediction: Tensor, expected: Tensor, device) -> Tensor:
        loss_val = torch.zeros(1, device=device)
        for i in range(len(prediction)):
            l = prediction[i] - expected[i]
            loss_val += 0.5*l*l
        return loss_val
    
class CrossEntropy(LossFunction):
    def loss(self, prediction, expected, device):
        sum = torch.zeros(1, device=device)
        for p, e in zip(prediction, expected):
            p = p*torch.tensor(0.99999, device=device) + torch.tensor(0.000005, device=device)
            cross = -(e*torch.log(p)+(torch.tensor(1, device=device)-e)*torch.log(torch.tensor(1, device=device)-p))
            sum += cross 
        return sum
    
    