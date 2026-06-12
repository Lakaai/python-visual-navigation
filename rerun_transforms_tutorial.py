"""
TODO:

The script is a tutorial on how to visualise transforms (rotations and translations) using the rerun.
"""

import rerun as rr
import rerun.blueprint as rrb
import argparse
import numpy as np
import cv2
import json
from src.visual_navigation.rotations import RotationMatrix, Rotations
from scipy.spatial.transform import Rotation
from src.visual_navigation.camera import Camera

# def get_horizontal_basis(u: np.ndarray):
#     """
#     Given gravity vector in camera frame (gc), return two orthonormal
#     vectors u, v that lie in the plane perpendicular to gravity.
#     """
    
#     gc = np.asarray(gc)
#     gc = gc / np.linalg.norm(gc) # ensure unit vector

#     # Choose a reference vector not parallel to gc
#     if abs(np.dot(gc, [1, 0, 0])) < 0.9:
#         ref = np.array([1.0, 0.0, 0.0])

#     else:
#         ref = np.array([0.0, 1.0, 0.0])

#     u = np.cross(gc, ref)

#     u /= np.linalg.norm(u)

#     v = np.cross(gc, u) # v is now perpendicular to both gc and u

#     return u, v

def get_unit_orthonormal_vectors(n: np.ndarray):
    """
    Returns two unit length orthonormal vectors u, v perpendicular to n.
    """
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    
    # Choose the smallest axis as reference to maximise stability
    reference_index = np.argmin(np.abs(n))
    reference_vector = np.zeros(3)
    reference_vector[reference_index] = 1.0
    
    u = np.cross(n, reference_vector)
    u /= np.linalg.norm(u)
    
    # v is unit length and perpendicular to both n and u
    v = np.cross(n, u)

    return u, v

def draw_horizon(Rnb, Rbc, image, image_height, image_width, camera: Camera):
    """
    Rnb: world -> body
    Rbc: body -> camera
    rBNn: body position in world
    rCBb: camera position in body frame

    Note the roll pitch yaw states encode the rotation matrix Rnb, that is the rotation from world basis vectors into body basis vectors / frame 
    Rab is the matrix that rotates the vectors of the basis a into the vectors of the basis b

    The horizon (or true horizon) is the line in the image where the horizontal plane passing through the camera’s optical center intersects the image plane.

    The gravity vector in world frame is [0, 0, 1] (NED frame) and we can rotate this into the camera frame using Rcn = Rbc * Rnb and then applying Rcn.inv() to the gravity vector to get it in the camera frame. The horizon line will be perpendicular to the gravity vector in the camera frame. Therefore we can create a set of direction vectors in the camera frame that are perpendicular to the gravity vector and then project these onto the image plane using the camera matrix K to get the horizon points in the image for visualisation. We can also log these direction vectors as arrows in the world frame for visualisation in 3D.
    """                     
    K = camera.matrix
    
    horizon_pts = []
    camera_horizontal_vectors = []
    num_samples = 120
    # Rbc = Rbc.inv()
    # R = Rotation.from_euler("zxy", euler_angles, degrees=True)
    # Rnc = Rbc * Rnb
    Rnc = Rnb * Rbc # Check this order, we want to rotate the camera horizontal vectors from camera frame to world frame for visualisation in the world frame, that is we want to apply Rnc to the camera horizontal vectors to get them in the world frame for visualisation
    
    Rcn = Rnc.inv()

    # Define gravity vector in camera frame
    gc = Rcn.apply([0, 0, 1])

    # Get two orthonormal vectors in the horizontal plane of the camera frame
    u, v = get_unit_orthonormal_vectors(gc)

    # Create direction vectors pointing out of the camera optical axis
    thetas = np.linspace(0, 2 * np.pi, num_samples)

    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    direction_vectors = cos_thetas[:, np.newaxis] * u + sin_thetas[:, np.newaxis] * v

    # equivalent to K @ dir_c.T but faster
    p = direction_vectors @ K.T

    valid_depth = p[:, 2] > 1e-6

    p_img = p[valid_depth, :2] / p[valid_depth, 2:3]   # normalize

    pts = np.round(p_img).astype(np.int32)

    # Filter points inside image bounds
    in_bounds = (
        (pts[:, 0] >= 0) & (pts[:, 0] < image_width) &
        (pts[:, 1] >= 0) & (pts[:, 1] < image_height)
    )
    horizon_pts = pts[in_bounds]

    # for angle in angles:
    #     direction_vector = [np.sin(angle), 0, np.cos(angle)]   
    #     if direction_vector[2] > 0.02:  
    #         camera_horizontal_vectors.append(direction_vector)
    #         p = K @ direction_vector

    #         if p[2] > 1e-6:
    #             p_img = p[:2] / p[2]
    #             u = int(round(p_img[0]))
    #             v = int(round(p_img[1]))
    #             if 0 <= u < image_width and 0 <= v < image_height:
    #                 horizon_pts.append((u, v))

    # Draw on image
    if len(horizon_pts) > 5:
        pts = np.array(horizon_pts, dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=False, 
                     color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)

    rr.log(
        "world/inertial_frame/body_frame/camera_frame/horizon_from_camera",
        rr.Arrows3D(
            vectors=camera_horizontal_vectors,
            colors=[255, 100, 0],
            radii=0.02,
        )
    )

    return image

# def draw_horizon(Rnb, Rbc, image, image_height, image_width, camera: Camera):

#     K = camera.matrix
#     Rnc = Rbc * Rnb
#     Rcn = Rnc.inv()

#     # Compute the direction of gravity in the camera frame
#     gc = Rcn.apply([0,0,1])
#     print(gc)
#     gc /= np.linalg.norm(gc)

#     tmp = np.array([1.0, 0.0, 0.0])
#     if abs(np.dot(tmp, gc)) > 0.9:
#         tmp = np.array([0.0, 1.0, 0.0])

#     u = np.cross(gc, tmp)
#     u /= np.linalg.norm(u)

#     v = np.cross(gc, u)

#     horizon_pts = []
#     camera_horizontal_vectors = []

#     for theta in np.linspace(0, 2 * np.pi, 120):

#         dir_c = np.cos(theta) * u + np.sin(theta) * v

#         camera_horizontal_vectors.append(dir_c)

#         if dir_c[2] <= 0:
#             continue

#         p = K @ dir_c
#         p_img = p[:2] / p[2]

#         x = int(round(p_img[0]))
#         y = int(round(p_img[1]))

#         if 0 <= x < image_width and 0 <= y < image_height:
#             horizon_pts.append((x, y))

#     if len(horizon_pts) > 10:

#         pts = np.asarray(horizon_pts, dtype=np.float32)

#         vx, vy, x0, y0 = cv2.fitLine(
#             pts,
#             cv2.DIST_L2,
#             0,
#             0.01,
#             0.01,
#         )

#         vx = vx.item()
#         vy = vy.item()
#         x0 = x0.item()
#         y0 = y0.item()

#         y_left = int(y0 - x0 * vy / vx)
#         y_right = int(y0 + (image_width - x0) * vy / vx)

#         cv2.line(
#             image,
#             (0, y_left),
#             (image_width, y_right),
#             (0, 0, 255),
#             4,
#             cv2.LINE_AA,
#         )

#     rr.log(
#         "world/inertial_frame/body_frame/camera_frame/horizon_from_camera",
#         rr.Arrows3D(
#             vectors=camera_horizontal_vectors,
#             colors=[255, 100, 0],
#             radii=0.02,
#         ),
#     )

#     return image

def plot_inertial_frame(axis_scale=2):
    """ """
    rr.log(
        "world/inertial_frame/axes",
        rr.Arrows3D(
            vectors=[[axis_scale, 0, 0], [0, axis_scale, 0], [0, 0, axis_scale]],
            colors=[
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
            ],
        ),
    )

    rr.log(
        "world/inertial_frame/axis_labels",
        rr.Points3D(
            positions=[
                [axis_scale, 0, 0],
                [0, axis_scale, 0],
                [0, 0, axis_scale],
                [0, 0, 0],
            ],
            labels=["X", "Y", "Z", "Inertial"],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
        ),
    )


def plot_body_frame(rBNn: np.ndarray, Rnb: np.ndarray, axis_scale=2):
    """ """

    rr.log(
        "world/inertial_frame/body_frame",
        rr.Transform3D(
            translation=rBNn,
            rotation=rr.Quaternion(xyzw=Rnb.as_quat()),
        ),
    )

    rr.log(
        "world/inertial_frame/body_frame/axes",
        rr.Arrows3D(
            vectors=[[axis_scale, 0, 0], [0, axis_scale, 0], [0, 0, axis_scale]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )

    rr.log(
        "world/inertial_frame/body_frame/axis_labels",
        rr.Points3D(
            positions=[
                [axis_scale, 0, 0],
                [0, axis_scale, 0],
                [0, 0, axis_scale],
                [0, 0, 0],
            ],
            labels=["X", "Y", "Z", "Body"],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]],
        ),
    )


def plot_camera_frustum(width, height, image, rCBb, Rbc, camera: Camera):
    """
    rCBn : position of camera in navigation frame (NED)
    Rbc  : Body → Camera rotation
    Note when using rr.Transform3D the translations are relative to the parent's origin, for example the parent for the camera_frame is the body_frame therefore
    the translation is rCBb
    """

    rr.log(
        "world/inertial_frame/body_frame/camera_frame",
        rr.Transform3D(
            translation=rCBb,
            rotation=rr.Quaternion(xyzw=Rbc.as_quat()),
        ),
    )

    rr.log(
        "world/inertial_frame/body_frame/camera_frame/pinhole",
        rr.Pinhole(
            resolution=[width, height],
            image_from_camera=camera.matrix,
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=1.8,
        ),
    )

    rr.log(
        "world/inertial_frame/body_frame/camera_frame/pinhole/image",
        rr.Image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)),
    )

def construct_camera(path):
    """
    """
    with open(path, 'r') as f:
        camera_params = json.load(f)

    camera_matrix = np.array(camera_params['camera_matrix'], dtype=np.float32)
    distortion_coeffs = np.array(camera_params['dist_coeffs'], dtype=np.float32)
    translation_vector = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    rotation_matrix = RotationMatrix(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32))

    camera = Camera(matrix=camera_matrix,distortion_coeffs=distortion_coeffs, translation_vector= translation_vector, rotation_matrix=rotation_matrix)

    return camera


def visualise_transforms():
    """ 
    # The columns of a rotation matrix are the subscript basis vectors expressed in the superscript coordinates
    """

    camera = construct_camera(path="/home/luke/python-visual-navigation/camera.json")

    # Define the body roll pitch yaw euler angles
    rpy = np.array([0.5, 0, -35])

    # Intialise world frame as NED (Right-Handed Z-Down)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)

    # Define the body to camera rotation matrix
    Rbc = Rotation.from_matrix(
        np.array(
            [
                [0, 0, 1],  # b1 = c3
                [1, 0, 0],  # b2 = c1
                [0, 1, 0],  # b3 = c2
            ]
        )
    )

    # Define navigation to body rotation matrix
    Rnb = Rotation.from_euler("XYZ", rpy, degrees=True)
    print("Rnb:\n", Rnb.as_matrix())
    # Rnb = Rotations.rpy2rot(rpy, order='zyx')
    # print("Rnb:\n", Rnb)
    
    rBNn = np.array([3, 3, -3])
    rCBb = np.array([0.1, 0.1, -0.1])   # Apply some offset only for visualisation purposes

    cap = cv2.VideoCapture("/home/luke/MCHA4400/data/outdoor/flight.MOV")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    ret, frame = cap.read()

    plot_inertial_frame()

    plot_body_frame(rBNn=rBNn, Rnb=Rnb, axis_scale=2)

    annotated_image = draw_horizon(Rnb=Rnb, Rbc=Rbc,image=frame, image_height=height, image_width=width, camera=camera)

    rr.log("world/camera/image", rr.Image(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)))

    plot_camera_frustum(width=width, height=height, rCBb=rCBb, Rbc=Rbc, image=annotated_image, camera=camera)


def main():

    parser = argparse.ArgumentParser(description="TODO")
    rr.script_add_args(parser)
    args = parser.parse_args()

    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(origin="/world", name="World"),
        rrb.Spatial2DView(origin="world/camera", name="Camera"),
    )

    rr.script_setup(args, "transform_visualisation", default_blueprint=blueprint)

    visualise_transforms()

    rr.script_teardown(args)


if __name__ == "__main__":
    main()

