"""
image coordinates (u, v) 
The (0,0) coordinate is the top-left corner

"""
from dataclasses import dataclass
import numpy as np
from .rotations import RotationMatrix
import cv2 

@dataclass
class Camera():
    """"
    
    """
    matrix: np.ndarray
    distortion_coeffs: np.ndarray
    translation_vector: np.ndarray
    rotation_matrix: RotationMatrix


    def undistort_features(self, features):
        """
        TODO: 
        """
        undistorted_features = cv2.undistortPoints(features, self.matrix, self.distortion_coeffs, P=self.matrix)
        return undistorted_features
    
    def is_vector_within_fov(self, rPCc: np.ndarray) -> bool:
        """
        rPCc : position of point in camera frame
        Assumes camera is looking along its z-axis and that the image plane is at z=1 in the camera frame

        Note if two vectors have unit length the dot product cos(theta) = A dot B/(norm(A)*norm(B)) simplifies toA dot B 
        so we can compute the angle between two vectors by normalising them and then taking the arccos of their dot product. 
        """
        uPCc = rPCc / np.linalg.norm(rPCc)

        principal_direction = np.array([0, 0, 1])

        # Compute horizontal angle
        horizontal_projection = np.array([uPCc[0], 0, uPCc[2]])
        horizontal_projection = horizontal_projection / np.linalg.norm(horizontal_projection)
        horizontal_angle = np.arccos(np.dot(horizontal_projection, principal_direction))

        # Compute vertical angle
        vertical_projection = np.array([0, uPCc[1], uPCc[2]])
        vertical_projection = vertical_projection / np.linalg.norm(vertical_projection)
        vertical_angle = np.arccos(np.dot(vertical_projection, principal_direction))

        # Convert angles to degrees
        horizontal_angle_deg = horizontal_angle * 180.0 / np.pi
        vertical_angle_deg = vertical_angle * 180.0 / np.pi

        # Project the vector onto the image plane
        pixel = self.vector_to_pixel(uPCc)

        # Check if the projected point is within the image bounds
        inside_image = (pixel[0] >= 0 and pixel[0] < self.matrix[0, 2] * 2 and
                        pixel[1] >= 0 and pixel[1] < self.matrix[1, 2] * 2)

        # Check if both angles are within their respective FOV limits
        inside_horizontal_fov = horizontal_angle_deg <= self.hFOV / 2.0
        inside_vertical_fov = vertical_angle_deg <= self.vFOV / 2.0

        return inside_image and inside_horizontal_fov and inside_vertical_fov
    
    def vector_to_pixel(self, rPCc):
        """
        TODO: 
        Compute the pixel location (rQOi) for the given unit vector (uPCc).

        Reference: https://www.geeksforgeeks.org/computer-vision/mapping-coordinates-from-3d-to-2d-using-opencv-python/
        """
        uPCc = rPCc / np.linalg.norm(rPCc)

        object_points = np.array([[uPCc[0], uPCc[1], uPCc[2]]], dtype=np.float32).reshape(1, 1, 3)
        
        # Since point is already in camera coordinates, rvec and tvec = zero
        rvec = np.zeros((3, 1), dtype=np.float32)   # No rotation
        tvec = np.zeros((3, 1), dtype=np.float32)   # No translation

        image_points, _ = cv2.projectPoints(
            object_points, 
            rvec, 
            tvec, 
            self.matrix, 
            self.distortion_coeffs
        )

        # image_points shape is (1, 1, 2) → extract the pixel
        pixel = image_points[0][0]   # Returns [u, v]
        return pixel

    def pixel_to_vector(self, rQOi, camera): 
        """ 
        TODO: 
        Compute unit vector (uPCc) for the given pixel location rQOi.
        """

        image_points = np.array([[rQOi[0], rQOi[1]]], dtype=np.float32)

        # Undistort and normalise the point
        normalised_points = cv2.undistortPoints(image_points, self.matrix, self.distortion_coeffs)

        # The normalised point is now [x, y, 1] in camera coordinates
        uPCc = np.array([normalised_points[0][0][0], normalised_points[0][0][1], 1.0])

        return uPCc / np.linalg.norm(uPCc)

