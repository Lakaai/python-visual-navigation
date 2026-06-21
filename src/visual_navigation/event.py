from abc import ABC, abstractmethod
from .system_estimator import SystemEstimator

class Event(ABC):
    """ Interface for events. """
    time: float
    verbosity: int = 1
    
    def process(self, system: SystemEstimator):
        """
        TODO: Add docstring 
        """

        # if verbosity > 0:
        #     print("Processing event: ")

        system.predict(self.time)

        self.update(system)

        @abstractmethod
        def get_process_string(self) -> str:
            """
            TODO: Add docstring 
            """



    # @abstractmethod
    # def update():
    #     """
    #     TODO: Add docstring 
    #     """