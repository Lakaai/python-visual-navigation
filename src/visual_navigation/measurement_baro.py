
from dataclasses import dataclass, field
from .measurement import Measurement
from .gaussian import Gaussian  
import numpy as np

@dataclass
class MeasurementBaro(Measurement):
    """
    TODO:
    """
    alt: np.ndarray
    
    R: np.ndarray = field(default_factory=lambda: np.array([[15.0]]))

    def get_process_string(self) -> str:
        """TODO:
        """
        return "Processing barometer measurement:"

    def log_likelihood(self, x):
        """
        TODO:
        """
        likelihood = self.predict_density(x)
        return likelihood.log_pdf(self.alt)

    def predict_density(self, x: np.ndarray):
        """
        TODO: 
        """
        alt = self.predict(x) # TODO: See if this should just be the measurement model from measurmeentmodel class
        return Gaussian(mean=alt, covariance=self.R)
    
    def predict(self, x: np.ndarray):
        """
        TODO:
        """
        # rBNnk = x[6:9]
        # Rnbk = Rotations.rpy2rot(x[9:12])
        # rCBb = CAMERA.translation_vector 
        # rCNnk = rBNnk + Rnbk.R @ rCBb
        # print("-x[8]:", -x[8])
        # print("np.array([-e3 @ rCNnk])", np.array([-e3 @ rCNnk]))
        # return np.array([-e3 @ rCNnk])
        return np.array([-x[8]])
    


    def as_vector(self):
        """TODO:
        Currently it is already a vector so just return the vector. 
        """
        return self.alt