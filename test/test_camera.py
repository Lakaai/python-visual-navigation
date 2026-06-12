import numpy as np
from camera import Camera
from rotations import RotationMatrix

# TODO: This should probably be a test for the camera calibration result rather than the camera class, perhaps it should live in the camera calibration application itself.
def test_vector_to_pixel():

    camera_matrix = np.array([[2273.16991, 0.0, 1375.02930], [0.0, 2276.38987, 723.107508], [0.0, 0.0, 1.0]], dtype=np.float32)
    distortion_coeffs = np.array([[0.0538893, -0.12069692, -0.00627124, 0.0045136, 0.15149127]], dtype=np.float32)
    translation_vector = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    rotation_matrix = RotationMatrix(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32))

    camera = Camera(matrix=camera_matrix,distortion_coeffs=distortion_coeffs, translation_vector= translation_vector, rotation_matrix=rotation_matrix)

    vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    pixel_coordinates = camera.vector_to_pixel(vector)

    expected_pixel_coordinates = np.array([1375.0293,  723.1075])

    assert np.allclose(pixel_coordinates, expected_pixel_coordinates)