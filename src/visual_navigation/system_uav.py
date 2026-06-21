from .system_estimator import SystemEstimator
from .gaussian import Gaussian
import numpy as np 
from .rotations import Rotations
from .update_method import UpdateMethod
from typing import Callable
import logging

class SystemUAV(SystemEstimator):
    """
    
    The state vector is defined as: [nu(6); eta(6); eta_km1(6)], where:
        - nu: 6D velocity vector (translational and angular velocities)
        - eta: 6D pose vector (position and attitude)
        - eta_km1: 6D previous pose vector (position and attitude at the previous timestep)
    """
    def __init__(self):
        super().__init__()

        self.state_distribution = self.initialise_state_distribution()
        self.previous_time = None
        self.predict_method = UpdateMethod.UNSCENTED
        self.Qeta = np.diag([2, 2, 2, 0.02, 0.02, 0.02])

    def initialise_state_distribution(self) -> Gaussian:
        mu = np.zeros(18)

        h = 63.126999                               # Altitude (GPS) [m]
        ga = h - 7.0                                # Altitude (AGL) [m]
        
        etak = np.array([0.0, 0.0, -ga, -0.009, np.deg2rad(-5), 0])
        etakm1 = etak.copy()

        mu[0:3] = np.array([0.0, 0.0, 0.0])         # Initial translational velocity (m/s)
        mu[3:6] = np.array([0.0, 0.0, 0.0])         # Initial angular velocity (rad/s)
        mu[6:12] = etak
        mu[12:18] = etakm1
        
        P = np.eye(18)

        P[:3, :3] = np.eye(3) * 0.1                 # Initial translational velocity covariance (m^2/s^2)
        P[3:6, 3:6] = np.eye(3) * 0.01              # Initial angular velocity covariance (rad^2/s^2)
        
        P[6:9, 6:9] = np.eye(3) * 0.001             # Initial position covariance (m^2/s^2)
        P[9:12, 9:12] = np.eye(3) * 0.001           # Initial attitude covariance (rad^2/s^2)

        P[12:15, 12:15] = np.eye(3) * 0.001         # Initial previous position covariance (m^2/s^2)
        P[15:18, 15:18] = np.eye(3) * 0.001         # Initial previous attitude covariance (rad^2/s^2)

        return Gaussian.from_moment(mu, P)

    def process_model(self, x):
        """
        TODO:
        The state vector x = [nu(6), eta(6)].T
        """
        nu = x[0:6]
        eta = x[6:12]

        Jk = self.construct_euler_rate_matrix(eta)

        nudot = np.zeros(6)
        etadot = Jk @ nu

        # TODO SHOULD ETAD ACTUALLY BE APPENDED TO THE END? CHECK THIS STATE VEC 
        return np.concatenate((nudot, etadot, eta))

        
    def construct_euler_rate_matrix(self, eta):
        """
        TODO:
        """
        thetanb = eta[3:6]
        Rnb = Rotations.rpy2rot(thetanb)

        Jk = np.zeros((6, 6))

        phi = eta[3]
        theta = eta[4]

        Jk[0:3, 0:3] = Rnb.R

        Jk[3:6, 3:6] = [
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)],
        ]

        # print("Jk: ")
        # for row in Jk:
        #     print(f"| {' '.join(f'{val:6.2f}' for val in row)}   |")

        return Jk

    def rk4_step(self, func: Callable, x: np.ndarray, dt: float) -> np.ndarray:
        """
        TODO
        """
        k1 = func(x)
        k2 = func(x + k1 * dt / 2)
        k3 = func(x + k2 * dt / 2)
        k4 = func(x + k3 * dt)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

    def predict(self, current_time: float):
        """
        TODO
        """
        if not self.previous_time:
            logging.info("First timestep received, skipping prediction step.")
            self.previous_time = current_time

        dt = current_time - self.previous_time

        if dt <= 0:
            logging.warning(f"Non-positive time step: {dt}. Prediction step skipped.")
            return

        def func(x):
            return self.rk4_step(self.process_model, x, dt)

        if self.predict_method == UpdateMethod.UNSCENTED:
            predicted_state_distribution = self.state_distribution.unscented_transform(func)

        elif self.predict_method == UpdateMethod.AFFINE:
            predicted_state_distribution = self.state_distribution.affine_transform(func)

        else:
            raise ValueError(f"Invalid prediction method: {self.predict_method}")
        
        # TODO UNCOMMENT AND FIX 
        Qd = np.zeros((12, 12))
        Qd[6:12, 6:12] = self.Qeta * dt

            # # full_state_distribution_convariance = np.eye(18)
        # full_state_distribution_covariance[0:12, 0:12] = predicted_state_distribution.covariance

        # Add minor velocity noise (m^2/s^2) so the filter knows speed can change
        Qd = np.eye(18) * 0.01  

        # Inject it directly into the predicted covariance
        predicted_state_distribution.covariance += Qd

            # print("mean before rebuild", self.state_distribution.mean)

            # # Rebuild the full 18 state state_distribution and covaraince.
            # self.state_distribution.mean[0:12] = predicted_state_distribution.mean
            # self.state_distribution.covariance[0:12, 0:12] = predicted_state_distribution.covariance + Qd  # Inject noise

            # print("convariance after prediction and rebuild")
            # for row in self.state_distribution.covariance:
            #     print(f"| {' '.join(f'{val:6.2f}' for val in row)}   |")

            # print("mean after prediction and rebuild", self.state_distribution.mean)

        predicted_state_distribution.covariance = 0.5 * (predicted_state_distribution.covariance + predicted_state_distribution.covariance.T)  # Symmetrise to ensure pos def
        self.state_distribution = predicted_state_distribution
        
        self.previous_time = current_time

        return

    