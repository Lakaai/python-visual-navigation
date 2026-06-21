from dataclasses import dataclass
from .system_estimator import SystemEstimator

import numpy as np
from .event import Event
from .sensor_type import SensorType
from .update_method import UpdateMethod
from .measurement_models import MeasurementModels
from .gaussian import Gaussian
from scipy.optimize import minimize
from typing import Callable
from abc import abstractmethod

@dataclass
class Measurement(Event):
    """
    TODO:
    """

    time: float
    sensor: SensorType
    update_method: UpdateMethod

    def update(self, system: SystemEstimator):
        """
        TODO:
        """
        print("Performing measurement update: ", self.sensor)

        if self.update_method == UpdateMethod.BFGS:
            
            # Minimise the negative log posterior to find the MAP estimate

            # TODO: Cost joint state_distribution may be better as a method of measurement class since it is not specific to the flow bundle measurment. 
            def cost_function(x):
                return self.negative_log_posterior(x, system.state_distribution)

            initial_guess = system.state_distribution.mean
            if self.sensor == SensorType.CAMERA:
                if self.pk is not None:
                    # measurement.data.predict_state_distribution(self.state_distribution.mean)
                    # result = minimize(fun=cost_function, x0=initial_guess, args=(), method='BFGS', options={'gtol': 1e-3, 'disp': True, 'maxiter':100})
                    result = minimize(
                        fun=cost_function,
                        x0=initial_guess,
                        method="BFGS",
                        jac="3-point",
                        options={
                            "gtol": 1e-1,                  # temporarily relax this
                            "finite_diff_rel_step": 1e-4,  # more appropriate than tiny absolute eps
                            "disp": True,
                            "maxiter": 100,
                        },
                    )
                    print(result)

                    system.state_distribution = Gaussian(result.x, 0.5 * (result.hess_inv + result.hess_inv.T))

            elif self.sensor == SensorType.BARO:
                result = minimize(fun=cost_function, x0=initial_guess, args=(), method='BFGS', options={'gtol': 1e-3, 'disp': True, 'maxiter':100})
                # print(result)
                cov = result.hess_inv
                cov = 0.5 * (cov + cov.T)
                system.state_distribution = Gaussian(result.x, cov)

            return 


        n = len(self.as_vector())
        idx_y = np.arange(0, n)
        idx_x = np.arange(n, n + 18)

        measurement_model = self.generate_measurement_model()

        # Create joint state_distribution maybe?
        def augmented_predict_measurement(x):
            return self.augmented_predict_measurement(x, measurement_model)
        # augmented_predict_measurement = lambda x: self.augmented_predict_measurement(x, measurement_model)
        
        # Form the joint state_distribution p(xk, yk | y1:k-1) by propagating the prior p(xk | y1:k-1) through the transformation
        if self.update_method == UpdateMethod.UNSCENTED:
            # 1. Force perfect symmetry first to prevent rounding mismatches
            cov = system.state_distribution.covariance
            cov = 0.5 * (cov + cov.T)

            # 2. Create and add a small numerical safety jitter to the diagonal
            jitter = 1e-6 * np.eye(cov.shape[0])
            cov_with_jitter = cov + jitter

            # 3. Save the clean, stable covariance back to your system
            system.state_distribution.covariance = cov_with_jitter
            pyx = system.state_distribution.unscented_transform(augmented_predict_measurement)

        elif self.update_method == UpdateMethod.AFFINE:
            pyx = system.state_distribution.affine_transform(augmented_predict_measurement)

        elif self.update_method == UpdateMethod.BFGSTRUSTSQRT:
            
            # Create cost function with prototype V = cost_function(x, g)
            # TODO: Cost joint state_distribution may be better as a method of measurement class since it is not specific to the flow bundle measurment. 
            # def cost_function(x, g):
            #     return MeasurementFlowBundle.cost_joint_state_distribution(x, self.state_distribution, g)
            return
        else:
            raise ValueError(f"Invalid update method: {self.update_method}")
        # print(" pyx.covariance[np.ix_(idx_y, idx_y)]", pyx.covariance[np.ix_(idx_y, idx_y)])
        # print("self.noise_density", self.noise_density)
        pyx.covariance[np.ix_(idx_y, idx_y)] += self.noise_density

        # Symmetrise the covariance matrix to ensure it's positive definite
        pyx.covariance = 0.5 * (pyx.covariance + pyx.covariance.T)

        
        # Compute the conditional state_distribution p(xk | yk) by conditioning the joint state_distribution on the measurement yk
        system.state_distribution = pyx.conditional(idx_x, idx_y, self.as_vector())
        # self.print_matrix("covariance after cond:", state_distribution.covariance)
        # print("mean after cond:", state_distribution.mean)
        # state_distribution.covariance = (state_distribution.covariance + state_distribution.covariance.T) / 2
        # print("shape of cov:", state_distribution.covariance.shape)
        # print("Is symmetric:" , np.allclose(state_distribution.covariance, state_distribution.covariance.T))
        # print("IS pos def:", np.linalg.cholesky(state_distribution.covariance)) # Check pos def

        cov = system.state_distribution.covariance
        cov = 0.5 * (cov + cov.T)

        # 2. Create and add a small numerical safety jitter to the diagonal
        jitter = 1e-6 * np.eye(cov.shape[0])
        cov_with_jitter = cov + jitter

        # 3. Save the clean, stable covariance back to your system
        system.state_distribution.covariance = cov_with_jitter

        return
    
    @abstractmethod
    def log_likelihood(self, x: np.ndarray) -> float:
        """
        """

    
    def negative_log_posterior(self, x: np.ndarray, state_distribution: Gaussian) -> float:
        """
        TODO: 

        Defines the cost function for MAP estimator to minimise. The cost function is defined as the negative log posterior, that is the
        negative log of the measurement likelihood plus the negative log prior.

        V(xk) = -log p(yk | xk) - log p(xk; muk|k-1, Pk|k-1).

        For more information see slide 50 Gaussian Filtering I.

        :param x: The state at which to evaluate the negative log posterior, typically a hypothetical or candidate state estimate being searched over during optimisation.
        :type x: np.ndarray[float]
        :param distribution: TODO
        :type distribution: Gaussian
        :param measurement: TODO
        :type measurement: Measurement
        :param gradient: TODO
        :type gradient: bool
        :return: Negative log posterior
        :rtype: float
        """
        log_prior = state_distribution.log_pdf(x)
        log_likelihood = self.log_likelihood(x)
        return -(log_prior + log_likelihood)

    
    def generate_measurement_model(self) -> Callable:
        """
        Generate the appropriate measurement model function based on the sensor type of the measurement.

        :param measurement: The measurement for which to generate the measurement model.
        :type measurement: Measurement
        :return: The measurement model function corresponding to the sensor type of the measurement.
        :rtype: Callable
        """
        measurement_model = {
            SensorType.GPS: MeasurementModels.gps,
            SensorType.BARO: MeasurementModels.baro,
            SensorType.MAG: MeasurementModels.mag,
            SensorType.ACC: MeasurementModels.accelerometer,
            SensorType.GYRO: MeasurementModels.gyroscope,
        }.get(self.sensor)

        if measurement_model is None:
            raise KeyError(f"Invalid sensor type: {self.sensor}")

        return measurement_model

    def print_matrix(self, string, matrix):
        print(f"{string}")
        for row in matrix:
            print(f"| {' '.join(f'{val:6.3f}' for val in row)}   |")

    def augmented_predict_measurement(self, x, measurement_model):
        """
        TODO:
        """

        y = measurement_model(x)

        h_augmented = np.concatenate((y, x))

        return h_augmented
    

    
