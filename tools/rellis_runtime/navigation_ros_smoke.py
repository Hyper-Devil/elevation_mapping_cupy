#!/usr/bin/python3
"""Drive the actual ROS navigation launch with real RELLIS fixtures and model worker.

Run inside an isolated validation container with the model worker already ready.
This validates interfaces and geometry, not prediction quality on RELLIS.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xmlrpc.client

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--uam2te', default='/uam2te')
    parser.add_argument('--worker-port', type=int, default=18765)
    parser.add_argument('--ros-port', type=int, default=11661)
    args = parser.parse_args()
    root, output = Path(args.fixtures), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(ROS_MASTER_URI=f'http://127.0.0.1:{args.ros_port}', ROS_IP='127.0.0.1',
                      ROS_HOSTNAME='localhost', ROS_LOG_DIR=str(output/'ros_logs'))
    children, handles = [], []
    def start(command, name):
        handle = open(output/name, 'w')
        handles.append(handle)
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        return child
    try:
        start(['roscore', '-p', str(args.ros_port)], 'master.log')
        deadline = time.monotonic()+20
        while time.monotonic() < deadline:
            try:
                if xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getPid('/navigation_test')[0] == 1:
                    break
            except Exception:
                time.sleep(.2)
        else:
            raise RuntimeError('Isolated master failed to start')
        launch = start(['roslaunch', args.uam2te+'/scripts/ros_workspace/src/ros_interface/launch/navigation.launch',
                        'reference_speed:=0.8', f'worker_url:=http://127.0.0.1:{args.worker_port}',
                        'map_frame:=navtest/map', 'base_frame:=navtest/base_link',
                        'pointcloud_topic:=/navtest/points', 'image_topic:=/navtest/image',
                        'camera_info_topic:=/navtest/camera_info', 'map_length:=30',
                        'origin_x:=-15', 'origin_y:=-15', 'save_path:='+str(output/'statistics.npz')], 'launch.log')
        import rospy
        import tf2_ros
        from tf.transformations import quaternion_from_matrix
        from geometry_msgs.msg import TransformStamped
        from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
        from grid_map_msgs.msg import GridMap
        from nav_msgs.msg import OccupancyGrid
        from cv_bridge import CvBridge
        from std_srvs.srv import Trigger
        rospy.init_node('navigation_smoke_driver')
        received, scores = [], []
        subscribers = [rospy.Subscriber('/similarity_costmap', OccupancyGrid, lambda m: received.append(m), queue_size=2),
                       rospy.Subscriber('/uam2te/conservative_score', Image, lambda m: scores.append(m), queue_size=2)]
        points = rospy.Publisher('/navtest/points', PointCloud2, queue_size=1)
        images = rospy.Publisher('/navtest/image', Image, queue_size=1)
        infos = rospy.Publisher('/navtest/camera_info', CameraInfo, queue_size=1, latch=True)
        meta = json.loads((root/'metadata.json').read_text())
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        transforms = []
        for child, matrix in [('base_link', np.eye(4)), ('lidar', np.array(meta['T_map_lidar'])),
                              ('camera_optical', np.array(meta['camera']['T_map_camera']))]:
            if child == 'base_link': matrix[2, 3] = .4
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id, transform.child_frame_id = 'navtest/map', 'navtest/'+child
            transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = matrix[:3, 3]
            quaternion = quaternion_from_matrix(matrix)
            transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = quaternion
            transforms.append(transform)
        broadcaster.sendTransform(transforms)
        deadline = time.monotonic()+60
        while points.get_num_connections() == 0 or images.get_num_connections() == 0:
            assert time.monotonic() < deadline, 'Navigation subscribers did not start'
            assert launch.poll() is None, 'Navigation launch exited'
            time.sleep(.2)
        for path in sorted(root.glob('scan_*.npy')):
            cloud = np.load(path).astype('<f4')
            msg = PointCloud2()
            msg.header.stamp, msg.header.frame_id = rospy.Time.now(), 'navtest/lidar'
            msg.height, msg.width, msg.is_dense = 1, len(cloud), False
            msg.fields = [PointField(name, i*4, PointField.FLOAT32, 1) for i, name in enumerate(['x','y','z','intensity'])]
            msg.point_step, msg.row_step, msg.data = 16, 16*len(cloud), cloud.tobytes()
            points.publish(msg)
            time.sleep(.5)
        bridge = CvBridge()
        rgb = np.load(root/'rgb.npy')
        stamps = []
        for _ in range(5):
            msg = bridge.cv2_to_imgmsg(rgb, 'rgb8')
            msg.header.stamp, msg.header.frame_id = rospy.Time.now(), 'navtest/camera_optical'
            stamps.append(msg.header.stamp.to_nsec())
            info = CameraInfo()
            info.header, info.width, info.height = msg.header, msg.width, msg.height
            info.K, info.D = meta['camera']['K'], meta['camera']['D']
            info.distortion_model = meta['camera']['distortion_model']
            infos.publish(info)
            time.sleep(.1)
            images.publish(msg)
            time.sleep(1.5)
        deadline = time.monotonic()+30
        while not received and time.monotonic() < deadline:
            time.sleep(.2)
        assert scores, 'No model output on ROS'
        assert received, 'No global risk map'
        prediction = bridge.imgmsg_to_cv2(scores[-1], '32FC1')
        assert prediction.shape == rgb.shape[:2] and np.isfinite(prediction).all()
        assert scores[-1].header.frame_id == 'navtest/camera_optical'
        assert scores[-1].header.stamp.to_nsec() in stamps
        cost = np.asarray(received[-1].data)
        assert (cost >= 0).sum() > 30, 'No projected model observations in global map'
        assert cost.min() >= -1 and cost.max() <= 100
        rospy.wait_for_service('/navigation_risk_costmap/save', timeout=10)
        saved = rospy.ServiceProxy('/navigation_risk_costmap/save', Trigger)()
        assert saved.success, saved.message
        with np.load(output/'statistics.npz') as data:
            assert data['count'].max() >= 1
            assert np.isfinite(data['variance']).all()
            report = dict(passed=True, observed_global_cells=int(data['valid'].sum()),
                          max_observations=float(data['count'].max()), model_score_min=float(prediction.min()),
                          model_score_max=float(prediction.max()), reference_speed=float(data['reference_speed']),
                          cvar_alpha=float(data['cvar_alpha']), ema_gamma=float(data['ema_gamma']),
                          scope='real checkpoint + real RELLIS RGB/XYZI + static calibration -> actual ROS1/GPU map; not model-quality or moving-robot validation')
        (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    finally:
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        for handle in handles: handle.close()


if __name__ == '__main__':
    main()
