import math
from math import sqrt
import abc
import numpy as np

class WeightInitialization(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def gen_weights(self, input_dim: int, output_dim: int) -> np.ndarray:
        pass
    
class GaussianInitialization(WeightInitialization):
    def gen_weights(self, input_dim, output_dim):
        return np.random.normal(loc = 0, scale=sqrt(2/(input_dim+output_dim)), size=(input_dim, output_dim))