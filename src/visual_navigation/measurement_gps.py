from dataclasses import dataclass, field
from .measurement import Measurement
import numpy as np

@dataclass
class MeasurementGPS(Measurement):
    """
    TODO:
    """
    lat: float
    lon: float
    alt: float

    noise_density: np.ndarray = field(default_factory=lambda: np.diag([1, 1, 1]))

    def log_likelihood(self, x):
        return    

    def as_vector(self):
        return np.array([self.lat, self.lon, self.alt])
