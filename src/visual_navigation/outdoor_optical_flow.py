import cv2
import numpy as np

import navpy
from .srt_parser import parse_srt
from .system_uav import SystemUAV
from .rerun_helpers import RerunHelper
from .sensor_type import SensorType
from .measurement_flow_bundle import MeasurementFlowBundle
from .measurement_baro import MeasurementBaro
from .measurement_gps import MeasurementGPS
from .camera import Camera
import rerun as rr
from .update_method import UpdateMethod
import time

def setup_rerun():
    """
    """
    rerun_helper = RerunHelper()
    rerun_helper.update_inertial_frame(axis_scale=2)

    return rerun_helper


def run_outdoor_optical_flow(video_path, camera: Camera, use_rerun: bool = False):

    measurement_data = parse_srt("/home/luke/MCHA4400/data/outdoor/flight.SRT")

    if use_rerun:
        rerun_helper = setup_rerun()

    system = SystemUAV()

    t0 = measurement_data.timestamp[0]

    lat_ref = measurement_data.latitude[0]
    lon_ref = measurement_data.longitude[0]
    alt_ref = 7

    previous_frame = None
    previous_alt = 0
    rQOikm1 = None

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video resolution: {width}x{height}, Number of frames: {num_frames}")

    for i in range(len(measurement_data.timestamp)):
        
        ret, frame = cap.read()

        if not ret:
            break  # No more frames therefore end of video

        alt = measurement_data.altitude[i]
        lat = measurement_data.latitude[i]
        lon = measurement_data.longitude[i]
        t = measurement_data.timestamp[i] - t0

        print(f"Time: {t}, Frame: {i}, Altitude: {alt}, Latitude: {lat}, Longitude: {lon}")

        measurement_gps = MeasurementGPS(
            time=t,
            sensor=SensorType.GPS,
            update_method=UpdateMethod.UNSCENTED,
            lat=lat,
            lon=lon,
            alt=alt
        )
        measurement_gps.process(system)

        # measurement_flow_bundle = MeasurementFlowBundle(
        #     time=t, 
        #     sensor=SensorType.CAMERA,
        #     update_method=UpdateMethod.BFGS, 
        #     imgk_raw=frame,
        #     imgkm1_raw=previous_frame,
        #     rQOikm1=rQOikm1,
        #     camera=camera
        # )
        
        # # Predict system forward and update with the optical flow measurement.
        # measurement_flow_bundle.process(system)

        # 4. Update the previous frame and previous flow measurement for the next iteration.
        # previous_frame = frame.copy()
        # rQOikm1 = measurement_flow_bundle.rQOik.copy()

        if alt != previous_alt:
            previous_alt = alt
            print(f"Altitude changed from {previous_alt} to {alt} on frame {i}, processing barometer measurement.")
        #     measurement_baro = MeasurementBaro(
        #         time=t, 
        #         sensor=SensorType.BARO, 
        #         update_method=UpdateMethod.BFGS,
        #         alt=np.array([alt])
        #         )
            
        #     measurement_baro.process(system)
        #     # system.update(measurement_baro, UpdateMethod.UNSCENTED)

        else:
            print("Altitude is the same for this timestep!")

        cv2.putText(frame, f"Frame: {i}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Visualise the points
        # if measurement_flow_bundle.data.pk is not None:
            
            # for pt in measurement_flow_bundle.data.pk.T:
            #     # pt is [x, y, z] or [x, y, 1]
            #     x, y = pt[0], pt[1]
        
            #     # Draw using integer coordinates
            #     cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        if use_rerun:

            # Convert GPS coordinates to ECEF
            ned = navpy.lla2ned(lat, lon, alt, lat_ref, lon_ref, alt_ref)
            print(f"North: {ned[0]}, East: {ned[1]}, Down: {ned[2]}")

            # frame = draw_horizon(frame, width, height, system, camera)
            # cv2.imshow("Good Points", frame)
            # cv2.waitKey(2000)
            
            # rr.set_time(timeline="Timeline", duration=t)

            # trajectory_enu.append(np.array([ned[1], ned[0], ned[2]]))

            # Log the camera extrinsics as a transform, this will move the camera frustrum to the correct position in the world
            # rr.log("world/camera_frame", rr.Transform3D(translation=[ned[0], ned[1], ned[2]],  rotation=rr.Quaternion(xyzw=(0, 0, 0, 1))))

            # Log a visible point at the camera origin
            # rr.log("world/camera_frame/origin", rr.Points3D([ned[0], ned[1], ned[2]], colors=np.array([[255, 0, 0]]), radii=0.05))
            # rr.log("world/gps_trajectory", rr.LineStrips3D([np.array(trajectory_enu)], colors=[[0, 255, 0]]))
            
            rerun_helper.update_body_frame(system=system, axis_scale=2)
            rerun_helper.update_gps_position(ned=ned)
            rerun_helper.update_camera_frustum(width, height, frame, camera=camera)
            frame = rerun_helper.draw_horizon(euler_angles=np.array(system.state_distribution.mean[9:12]), image=frame, image_height=height, image_width=width, camera=camera)
            # cv2.imshow("Good Points", frame)
            # cv2.waitKey(2000)

            rr.log("world/camera/image", rr.Image(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))
            
            # estimated_trajectory_enu.append(np.array([system.density.mean[6], system.density.mean[7], -system.density.mean[8]]))

            # Add state estimate to the rerun timeline.
            # rr.log("world/estimated_trajectory", rr.LineStrips3D([np.array(estimated_trajectory_enu)], colors=[[0, 0, 255]]))

            # rr.log("world/camera_frame/pinhole", rr.Pinhole(image_from_camera=CAMERA.matrix, resolution=(width, height), camera_xyz=rr.ViewCoordinates.RDF))
        
            # _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

            # rr.log("/image/frame", rr.EncodedImage())
            time.sleep(2)
        
        # rr.log("world/chessboard_corners", rr.Points3D(grid_points, colors=np.array([[0, 255, 0]]), radii=0.0015))
        
        # gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # # keypoints, descriptors = orb.detectAndCompute(gray_image, None)
        # corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)
        # # annotated_frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
        # # cv2.imshow("Features", annotated_frame)
        # corners = np.int32(corners)

    cap.release()
    cv2.destroyAllWindows()
