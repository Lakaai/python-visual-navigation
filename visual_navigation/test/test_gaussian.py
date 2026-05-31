# import numpy as np
# from gaussian import Gaussian

# def test_conditional():

#     mu = np.array([1.0, 2.0, 3.0, 4.0])  
#     sigma = np.array([
#     [1.0, 0.2, 0.1, 0.0],
#     [0.2, 1.5, 0.3, 0.2],
#     [0.1, 0.3, 2.0, 0.5],
#     [0.0, 0.2, 0.5, 1.2]
#     ])       

#     idx_x = np.array([1, 2, 3])     
#     idx_y = np.array([0])           
#     y = np.array([0.8])     

#     distribution = Gaussian.from_moment(mu, sigma)

#     out = Gaussian.conditional(distribution, idx_x, idx_y, y, sqrt=False)

#     expected_mean = np.array([1.96, 2.98, 4.0])
#     expected_covariance = np.array([[
#         1.46, 0.27999999999999997, 0.2],
#         [0.27999999999999997, 1.99, 0.5],
#         [0.2, 0.5, 1.2]
#         ])

#     assert np.allclose(out.mean, expected_mean)
#     assert np.allclose(out.covariance, expected_covariance)

# def test_unscented_transform():

#     mu = np.array([1.0, 2.0, 3.0, 4.0])  
#     sigma = np.array([
#     [1.0, 0.2, 0.1, 0.0],
#     [0.2, 1.5, 0.3, 0.2],
#     [0.1, 0.3, 2.0, 0.5],
#     [0.0, 0.2, 0.5, 1.2]
#     ])       
         
#     distribution = Gaussian.from_moment(mu, sigma)
#     def nonlinear_func(x):
#         return x**2

#     out = Gaussian.unscented_transform(nonlinear_func, distribution, sqrt=False)

#     expected_mean = np.array([2.0, 5.5, 11.0, 17.2])
#     expected_covariance = np.array([
#     [9.0, 3.2600000000000002, 3.240000000000001, 1.199999999999999],
#     [3.2600000000000002, 34.782799999999995, 10.5152, 8.359999999999994],
#     [3.240000000000001, 10.5152, 91.00898615124791, 27.258344905235496],
#     [1.199999999999999, 8.359999999999994, 27.258344905235496, 82.80742110927032]
#     ])

#     assert np.allclose(out.mean, expected_mean)
#     assert np.allclose(out.covariance, expected_covariance)