"""
TODO: 
"""
import sys
import json
import argparse
import numpy as np
from .camera import Camera
from .rotations import RotationMatrix
import rerun as rr
import rerun.blueprint as rrb
from .outdoor_optical_flow import run_outdoor_optical_flow

def construct_camera(path: str) -> Camera:
    """
    TODO:
    """
    with open(path, 'r') as f:
        camera_params = json.load(f)

    camera_matrix = np.array(camera_params['camera_matrix'], dtype=np.float32)
    distortion_coeffs = np.array(camera_params['dist_coeffs'], dtype=np.float32)
    translation_vector = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    rotation_matrix = RotationMatrix(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32))

    camera = Camera(matrix=camera_matrix,distortion_coeffs=distortion_coeffs, translation_vector= translation_vector, rotation_matrix=rotation_matrix)

    return camera

def main():
    parser = argparse.ArgumentParser(prog="Camera Calibration", description="TODO: description", epilog="TODO: text at the bottom of script help message")

    rr.script_add_args(parser)

    parser.add_argument('--video', help="Path to the video file")    
    parser.add_argument('--calibration-path', help="Path to the calibration data")
    parser.add_argument('-s', '--scenario', type=int, help="Visual navigation scenario to run, must be an integer between 1 and 6 inclusive")
    parser.add_argument('-v', '--visualise', action='store_true', help="Visualise the results using rerun")



    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(origin="/world", name="World"),
        rrb.Spatial2DView(origin="/world/camera", name="Camera"),
        # rrb.Spatial2DView(origin="/image", name="Frame", contents=["/image/**"]), # Can add this imge to camera frustum 
    )

    args = parser.parse_args()

    rr.init("Visual Navigation", spawn=args.visualise)

    rr.script_setup(args, "Visual Navigation", default_blueprint=blueprint)

    # Intialise world frame as NED (Right-Handed Z-Down)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)

    if not args.calibration_path:
        print("Error: Path to camera calibration JSON is required.")
        parser.print_help()
        sys.exit(1) 

    camera = construct_camera(args.calibration_path)

    match args.scenario:
        case 1:
            print("Running Scenario 1: ArUco SLAM Unique Tags!")
            run_aruco_slam(args.video, camera, use_rerun=not args.visualise)
            sys.exit(1)
            # Add Scenario 1 logic here
            
        case 2:
            print("Running Scenario 2: ArUco SLAM Identical Tags!")
            sys.exit(1)
            # Add Scenario 2 logic here
            
        case 3:
            print("Running Scenario 3: Indoor Point SLAM!")
            sys.exit(1)
            # Add Scenario 3 logic here
            
        case 4:
            print("Running Scenario 4: Outdoor Optical Flow!")
            run_outdoor_optical_flow(args.video, camera, use_rerun=not args.visualise)
            # Add Scenario 4 logic here
            
        case 5:
            print("Running Scenario 5: Indoor Visual Navigation!")
            sys.exit(1)
            # Add Scenario 5 logic here
            
        case 6:
            print("Running Scenario 6: Duck Apocalypse!")
            print("TODO")
            sys.exit(1)
            # Add Scenario 6 logic here
            
        case None:
            print("Error: Please specify a scenario using -s or --scenario.")
            parser.print_help()
            sys.exit(1)
            
        case _:
            print(f"Error: '{args.scenario}' is not a valid scenario. Must be between 1 and 6.")
            sys.exit(1)

    

if __name__ == "__main__":
    main()