
from abc import ABC, abstractmethod
from .gaussian import Gaussian
# from .measurement import Measurement

class SystemEstimator(ABC):
    """ Interface for system estimators. """
    state_distribution: Gaussian

    @abstractmethod
    def initialise_state_distribution(self) -> Gaussian:
        """
        Initialises the initial state distribution for the system. 

        :return: Initial state distribution as a Gaussian object.
        :rtype: Gaussian
        """

    @abstractmethod
    def process_model(self, x):
        """
        TODO: Add docstring and type hint for x.
        """
    
    @abstractmethod
    def predict(self, time: float) -> Gaussian:
        """
        TODO:
        """