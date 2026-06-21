
from dataclasses import dataclass
from .measurement import Measurement

@dataclass
class MeasurementIMU(Measurement):
    """
    TODO: Placeholder.
    """

    gyro_x: float
    gyro_y: float
    gyro_z: float
    specific_force_x: float
    specific_force_y: float
    specific_force_z: float
