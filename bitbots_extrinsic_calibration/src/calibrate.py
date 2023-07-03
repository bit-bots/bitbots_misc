import os
import pickle
from rosbags.rosbag2 import Reader
import tf2_ros
from rclpy.time import Duration, Time
from rclpy.serialization import deserialize_message
import rosbag2_py
from yoeo import models, detect
from typing import TYPE_CHECKING
from typing import Any
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge
import cv2
from soccer_ipm import soccer_ipm
from ipm_library.ipm import IPM
from soccer_ipm.utils import create_horizontal_plane
import numpy as np
from sensor_msgs_py.point_cloud2 import create_cloud
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
import tf2_ros as tf2
from ipm_library.exceptions import CameraInfoNotSetException
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
import time
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.affines import compose, decompose
from transforms3d.quaternions import mat2quat, quat2mat
import math
import copy
import optuna
import image_geometry



def get_rosbag_options(path, storage_id, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(
        uri=path, storage_id=storage_id)

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options

reader = rosbag2_py.SequentialReader()
bag_path = '/home/jasper/bitbots/rosbag2_2023_04_27-19_18_41'
storage_options, converter_options = get_rosbag_options(bag_path, "mcap")
reader.open(storage_options, converter_options)

topic_types = reader.get_all_topics_and_types()

# Create a map for quicker lookup
type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

# Set filter for topic of string type
storage_filter = rosbag2_py.StorageFilter(topics=['/tf_static', '/tf', '/camera/image_unrect', '/camera/camera_info'])
reader.set_filter(storage_filter)



transforms = []
transforms_static = []
images = []
camera_infos = []

i = 0
skip = 10

while reader.has_next():
    (topic, data, t) = reader.read_next()
    msg_type = get_message(type_map[topic])
    msg = deserialize_message(data, msg_type)
    if topic == '/tf_static':
        transforms_static.extend(msg.transforms)
    elif topic == '/tf':
        transforms.extend(msg.transforms)
    i += 1  
    if i % skip != 0:
        pass
        #continue

    if topic == '/camera/image_unrect':
        images.append(msg)
    elif topic == '/camera/camera_info':
        camera_infos.append(msg)
    



cv_bridge = CvBridge()
model = models.load_model(
  "/home/jasper/gits/YOEO/yoeo.cfg",
  "/home/jasper/gits/YOEO/yoeo.pth",)
    
def reproject_image(ipm : IPM, img, header) -> PointCloud2:

    # Get params
    scale = 1
    output_frame = "base_footprint"
    # Get field plane
    field = create_horizontal_plane()
    if field is None:
        return

    # Convert subsampled image
    image = cv2.resize(
        img,
        (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    point_idx_tuple = np.where(image == 1)

    # Restructure index tuple to a array
    point_idx_array = np.empty((point_idx_tuple[0].shape[0], 2))
    point_idx_array[:, 0] = point_idx_tuple[1] / scale
    point_idx_array[:, 1] = point_idx_tuple[0] / scale

    # Map points
    try:
        points_on_plane = ipm.map_points(
                    field,
                    point_idx_array,
                    header.stamp,
                    plane_frame_id=output_frame,
                    output_frame_id=output_frame)[1]
    except CameraInfoNotSetException:
        print(
            'Inverse perspective mapping should be performed, '
            'but no camera info was recived yet!')
        return None, None
    except (tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException) as e:
        print(
            'Inverse perspective mapping should be performed, '
            f'but no transform was found: {e}')
        return None, None

    # Define fields of the point cloud
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]

    # Build pointcloud
    pc = create_cloud(
        Header(
            stamp=header.stamp,
            frame_id=output_frame
        ),
        fields,
        points_on_plane)
    return pc, points_on_plane

rclpy.init()
node = rclpy.create_node('calibrate')
pub = node.create_publisher(PointCloud2, '/calibration_points', 10)

#check that file exists
if os.path.isfile("segmentations.pkl") and os.path.isfile("cv_images.pkl"):
    print("loading pickles")
    segmentations = pickle.load(open("segmentations.pkl", "rb"))
    cv_images = pickle.load(open("cv_images.pkl", "rb"))
else:
    segmentations = []
    cv_images = []
    for i, ros_img in enumerate(images):
        img = cv_bridge.imgmsg_to_cv2(ros_img, desired_encoding="passthrough")
        _, segmentation = detect.detect_image(model, img)
        segmentations.append(segmentation)
        cv_images.append(img)
        if i % 10 == 0:
            print(f"{i} / {len(images)}")
    pickle.dump(segmentations, open("segmentations.pkl", "wb"))
    pickle.dump(cv_images, open("cv_images.pkl", "wb"))

linepng = cv2.imread("/home/jasper/colcon_ws/src/bitbots_meta/bitbots_misc/bitbots_extrinsic_calibration/src/lines.png", cv2.IMREAD_GRAYSCALE)

def xy_robot_to_img(x,y, robot_x, robot_y, robot_yaw):
    corner_to_center = compose([(4.5+0.7), (3+0.7), 0], euler2mat(math.radians(180),0,0), [1,1,1])
    center_to_robot = compose([robot_x, robot_y, 0], euler2mat(0,0,robot_yaw), [1,1,1])
    p_robot = np.array([[x, y, 0,1]]).T
    p_corner = corner_to_center @ center_to_robot @ p_robot
    p_img = (p_corner[0][0] / 0.01, p_corner[1][0] / 0.01)
    return p_img

def objective(t : optuna.Trial, viz=False):
    ## camera projection parameters
    fov_x_estimate = 1336
    fov_y_estimate = 1334
    image_center_x_estimate = 2048/2
    image_center_y_estimate = 1536/2
    #fov_x = t.suggest_float("fov_x", fov_x_estimate-100, fov_x_estimate+100)
    fov_x = fov_y_estimate
    #fov_y = t.suggest_float("fov_y", fov_y_estimate-100, fov_y_estimate+100)
    fov_y = fov_y_estimate
    #image_center_offset_x = t.suggest_int("offset_x", image_center_x_estimate-100, image_center_x_estimate+100)
    image_center_offset_x = image_center_x_estimate
    #image_center_offset_y = t.suggest_int("offset_y", image_center_y_estimate-100, image_center_y_estimate+100)
    image_center_offset_y = image_center_y_estimate

    # camera distortion parameters
    k1_estimate = -0.285950
    k2_estimate = 0.081479
    k3_estimate = -0.000093
    k4_estimate = -0.000320
    k5_estimate = 0.0

    #k1 = t.suggest_float("k1", k1_estimate-0.001, k1_estimate+0.001)
    k1 = k1_estimate
    #k2 = t.suggest_float("k2", k2_estimate-0.001, k2_estimate+0.001)
    k2 = k2_estimate
    #k3 = t.suggest_float("k3", k3_estimate-0.001, k3_estimate+0.001)
    k3 = k3_estimate
    #k4 = t.suggest_float("k4", k4_estimate-0.001, k4_estimate+0.001)
    k4 = k4_estimate
    #k5 = t.suggest_float("k5", k5_estimate-0.001, k5_estimate+0.001)
    k5 = k5_estimate


    # robot extrinsics
    robot_x_estimate = -0.5
    robot_y_estimate = -3.25
    robot_x = t.suggest_float("robot_x", robot_x_estimate-0.5, robot_x_estimate+0.5)
    robot_y = t.suggest_float("robot_y", robot_y_estimate-0.5, robot_y_estimate+0.5)
    robot_yaw = t.suggest_float("robot_yaw", math.radians(70), math.radians(110))
    imu_pitch_offset = t.suggest_float("imu_pitch_offset", -0.2, 0.2)
    imu_roll_offset = t.suggest_float("imu_roll_offset", -0.2, 0.2)
    head_pitch_offset = t.suggest_float("head_pitch_offset", -0.2, 0.2)
    head_roll_offset = t.suggest_float("head_roll_offset", -0.2, 0.2)

    tf_buffer = tf2_ros.tf2_ros.Buffer(cache_time=Duration(seconds=10000))
    # TODO modify transforms accoring to offsets of trial
    for static_tf in transforms_static:
        tf_buffer.set_transform_static(static_tf, 'default_authority')
    
    for tf in transforms:
        if tf.header.frame_id == 'camera' and tf.child_frame_id == 'camera_optical_frame':
            tf_copy = copy.deepcopy(tf)
            rotation_quat =  [tf_copy.transform.rotation.w, tf_copy.transform.rotation.x, tf_copy.transform.rotation.y, tf_copy.transform.rotation.z]
            rotation_mat = quat2mat(rotation_quat)
            roll, pitch, yaw = mat2euler(rotation_mat, axes='sxyz')
            roll += head_pitch_offset
            pitch += head_roll_offset
            rotation_mat = euler2mat(roll, pitch, yaw, axes='sxyz')
            rotation_quat = mat2quat(rotation_mat)
            tf_copy.transform.rotation.x = rotation_quat[1]
            tf_copy.transform.rotation.y = rotation_quat[2]
            tf_copy.transform.rotation.z = rotation_quat[3]
            tf_copy.transform.rotation.w = rotation_quat[0]
            tf_buffer.set_transform(tf_copy, 'default_authority')
        elif tf.header.frame_id == 'base_link' and tf.child_frame_id == 'base_footprint':
            tf_copy = copy.deepcopy(tf)
            rotation_quat =  np.array([tf_copy.transform.rotation.w, tf_copy.transform.rotation.x, tf_copy.transform.rotation.y, tf_copy.transform.rotation.z])
            rotation_mat = quat2mat(rotation_quat)
            translation = np.array([tf_copy.transform.translation.x, tf_copy.transform.translation.y, tf_copy.transform.translation.z])
            tf_affine = compose(translation, rotation_mat, np.ones(3))
            offset_affine = compose(np.zeros(3), euler2mat(imu_roll_offset, imu_pitch_offset, 0), np.ones(3))
            tf_composed = offset_affine @ tf_affine
            t, rotmat, _, _ = decompose(tf_composed)
            quat = mat2quat(rotmat)
            tf_copy.transform.translation.x = t[0]
            tf_copy.transform.translation.y = t[1]
            tf_copy.transform.translation.z = t[2]
            tf_copy.transform.rotation.x = quat[1]
            tf_copy.transform.rotation.y = quat[2]
            tf_copy.transform.rotation.z = quat[3]
            tf_copy.transform.rotation.w = quat[0]
            tf_buffer.set_transform(tf_copy, 'default_authority')
        else:
            tf_buffer.set_transform(tf, 'default_authority')

    ipm = IPM(tf_buffer, None)
    total_error = 0
    c = CameraInfo()
    c.header = camera_infos[0].header
    c.width = camera_infos[0].width
    c.height = camera_infos[0].height
    c.distortion_model = 'plumb_bob'
    c.d = [k1, k2, k3, k4, k5]
    c.k = [fov_x, 0, image_center_offset_x, 0, fov_y, image_center_offset_y, 0, 0, 1]
    c.r = np.identity(3)
    c.p = [fov_x, 0, image_center_offset_x, 0, 0, fov_y, image_center_offset_y, 0, 0, 0, 1, 0]
    c.binning_x = camera_infos[0].binning_x
    c.binning_y = camera_infos[0].binning_y
    cam_model = image_geometry.PinholeCameraModel()
    cam_model.fromCameraInfo(c)
    cam_model.width = c.width//c.binning_x # slightly hacky
    cam_model.height = c.height//c.binning_y # slightly hacky
    ipm.set_camera_info(c)
    for ros_img , cv_img, segmentation in zip(images, cv_images, segmentations):
        class_id = 1
        
        segmentation_mask = np.zeros_like(segmentation[0], dtype=np.float32)
        segmentation_mask[segmentation[0] == class_id] = 1.0
        #mask_resized = cv2.resize(segmentation_mask, cv_img.shape[:2])
        rect_segmentation = np.zeros_like(segmentation_mask)
        cam_model.rectifyImage(segmentation_mask, rect_segmentation)
        if viz and False:
            cv2.imshow("segmentation_rect", rect_segmentation)
            cv2.imshow("segmentation", segmentation_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        pc, points_on_plane = reproject_image(ipm, rect_segmentation, ros_img.header)
        
        error = 0
        if points_on_plane is not None:
            if viz:
                viz_img = linepng.copy()
            for point in points_on_plane:
                x,y = xy_robot_to_img(point[0], point[1], robot_x, robot_y, robot_yaw)
                # check x and y are in image
                if np.isnan(x) or np.isnan(y) or x < 0 or x > linepng.shape[1] or y < 0 or y > linepng.shape[0]:
                    error += 1
                    continue
                else:
                    # draw pixel at x y coordinate white
                    error += linepng[int(y), int(x)] / 255
                    if viz:
                        viz_img[int(y), int(x)] = 255
            if viz:
                cv2.imshow("viz", viz_img)
                cv2.waitKey(25)
        total_error += error
    cv2.destroyAllWindows()
    return total_error
                    
        #if pc:
        #    print("we did it bois")
        #    pub.publish(pc)
        #    time.sleep(0)
        

#e_2 = trial([-0.5, -3.25, math.radians(90), 0, 0, -0.5, 0])
# print("e-2:", e_2)
# e_1 = trial([-0.5, -3.25, math.radians(90), 0, 0, -0.1, 0])
# print("e-1:", e_1)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print(study.best_params)

objective(study.best_trial, viz=True)

#current_best = {'robot_x': -0.2497426051059045, 'robot_y': -3.7317841870621034, 'robot_yaw': 1.5570264819927397, 'head_pitch_offset': -0.1940921501909509, 'head_roll_offset': -0.035970377641765736}

#objective(optuna.trial.FixedTrial(current_best), viz=True)