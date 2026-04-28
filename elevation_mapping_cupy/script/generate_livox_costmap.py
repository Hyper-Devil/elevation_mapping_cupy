#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 Livox 原始点云过滤地面后转换为 Local Costmap (OccupancyGrid)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Topic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  订阅: ~cloud_topic  (sensor_msgs/PointCloud2，默认 /livox/points)
  发布: livox_costmap (nav_msgs/OccupancyGrid)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Local Costmap 特性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - 每帧刷新，不做历史积累，动态障碍不留痕迹
  - 滚动窗口随机器人移动，以 robot_frame 当前位置为中心
  - 发布在 costmap_frame（默认 odom），与 Nav2 local_costmap.global_frame 对齐

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
地面过滤与代价映射
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  点在 odom 系中高度 z 满足以下条件时保留为障碍：
    robot_z + min_obstacle_height ≤ z ≤ robot_z + max_obstacle_height

  z < robot_z + min_obstacle_height  : 地面，忽略
  z > robot_z + max_obstacle_height  : 过高（树冠/天花板），忽略
  同一格内障碍点数 ≥ min_points_per_cell : cost 100（障碍）

  空间可见性（以传感器当前位置为圆心）：
    距离 ≤ sensor_max_range 且无障碍点 : cost 0（自由）
    距离 > sensor_max_range            : cost -1（未知）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROS 参数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ~cloud_topic          (默认 /livox/points)
  ~costmap_frame        (默认 odom)
  ~robot_frame          (默认 base_link)
  ~local_size           (默认 20.0 m，地图边长)
  ~resolution           (默认 0.10 m/格)
  ~min_obstacle_height  (默认 0.2 m，相对 robot_frame z)
  ~max_obstacle_height  (默认 2.0 m，相对 robot_frame z)
  ~sensor_max_range     (默认 15.0 m，超出此距离标为未知)
  ~min_points_per_cell  (默认 2，噪声过滤)
  ~min_range            (默认 0.25 m，传感器盲区，原点附近的点忽略)
"""

import math

import numpy as np
import rospy
import tf2_ros
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2


def _quat_to_rot(q):
    """四元数 → 3×3 旋转矩阵"""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _pc2_to_xyz(msg):
    """
    从 PointCloud2 提取 (N,3) float32 xyz 数组。
    从消息 header 读取各字段的实际字节偏移，支持任意 point_step（包括 Livox 的 18 字节布局）。
    """
    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in ('x', 'y', 'z')):
        return np.empty((0, 3), dtype=np.float32)

    n = msg.width * msg.height
    step = msg.point_step
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, step)  # (N, step)

    xyz = np.empty((n, 3), dtype=np.float32)
    for i, name in enumerate(('x', 'y', 'z')):
        off = fields[name].offset
        # 取每个点中偏移 off 处的 4 字节，解释为 float32
        xyz[:, i] = raw[:, off:off + 4].copy().view(np.float32).ravel()

    valid = np.isfinite(xyz).all(axis=1)
    return xyz[valid]


class LivoxToLocalCostmap:
    def __init__(self):
        rospy.init_node('generate_livox_costmap', anonymous=True)

        self.cloud_topic = rospy.get_param('~cloud_topic', '/livox/points')
        self.costmap_frame = rospy.get_param('~costmap_frame', 'odom')
        self.robot_frame = rospy.get_param('~robot_frame', 'base_link')
        self.local_size = rospy.get_param('~local_size', 20.0)
        self.resolution = rospy.get_param('~resolution', 0.10)
        self.min_obstacle_height = rospy.get_param('~min_obstacle_height', 0.2)
        self.max_obstacle_height = rospy.get_param('~max_obstacle_height', 2.0)
        self.sensor_max_range = rospy.get_param('~sensor_max_range', 15.0)
        self.min_points_per_cell = rospy.get_param('~min_points_per_cell', 2)
        self.min_range = rospy.get_param('~min_range', 0.25)

        self.n_cells = int(round(self.local_size / self.resolution))

        # 预计算格子中心相对于地图原点的偏移（用于自由空间掩码）
        idx = np.arange(self.n_cells)
        cx, cy = np.meshgrid(idx, idx, indexing='xy')  # cx=col, cy=row
        # 格子中心在地图坐标系中的偏移（以地图左下角为原点）
        self._cell_dx = (cx + 0.5) * self.resolution  # shape (n, n)
        self._cell_dy = (cy + 0.5) * self.resolution

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.costmap_pub = rospy.Publisher('livox_costmap', OccupancyGrid, queue_size=1)
        self.cloud_sub = rospy.Subscriber(
            self.cloud_topic, PointCloud2, self.cloud_callback, queue_size=1)

        rospy.loginfo("Livox to Local Costmap initialized")
        rospy.loginfo(f"  Cloud topic     : {self.cloud_topic}")
        rospy.loginfo(f"  Costmap frame   : {self.costmap_frame}")
        rospy.loginfo(f"  Robot frame     : {self.robot_frame}")
        rospy.loginfo(f"  Local size      : {self.local_size} m  ({self.n_cells}×{self.n_cells} cells)")
        rospy.loginfo(f"  Resolution      : {self.resolution} m")
        rospy.loginfo(f"  Obstacle height : [{self.min_obstacle_height}, {self.max_obstacle_height}] m (rel. robot z)")
        rospy.loginfo(f"  Sensor range    : {self.sensor_max_range} m")
        rospy.loginfo(f"  Min range       : {self.min_range} m (blind zone)")

    def cloud_callback(self, msg):
        try:
            stamp = msg.header.stamp
            cloud_frame = msg.header.frame_id

            # ── 查询 TF ────────────────────────────────────────────────
            try:
                tf_cloud = self.tf_buffer.lookup_transform(
                    self.costmap_frame, cloud_frame, stamp, rospy.Duration(0.1))
                tf_robot = self.tf_buffer.lookup_transform(
                    self.costmap_frame, self.robot_frame, stamp, rospy.Duration(0.1))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn_throttle(5.0, f"TF lookup failed: {e}")
                return

            # 机器人在 costmap_frame 中的位置
            robot_x = tf_robot.transform.translation.x
            robot_y = tf_robot.transform.translation.y
            robot_z = tf_robot.transform.translation.z

            # 传感器在 costmap_frame 中的位置（用于自由空间圆心）
            sensor_x = tf_cloud.transform.translation.x
            sensor_y = tf_cloud.transform.translation.y

            # ── 读取并变换点云 ──────────────────────────────────────────
            pts = _pc2_to_xyz(msg)
            if pts.shape[0] == 0:
                return

            # 传感器坐标系中的盲区过滤：xy 任意一轴超出阈值才保留（方框判断）
            if self.min_range > 0:
                r = self.min_range
                pts = pts[(np.abs(pts[:, 0]) >= r) | (np.abs(pts[:, 1]) >= r)]
            if pts.shape[0] == 0:
                return

            t = tf_cloud.transform.translation
            R = _quat_to_rot(tf_cloud.transform.rotation)
            pts_odom = (R @ pts.T).T + np.array([t.x, t.y, t.z])

            # ── 高度过滤（相对于 robot_frame z）──────────────────────────
            z_rel = pts_odom[:, 2] - robot_z
            obs_mask = (z_rel >= self.min_obstacle_height) & (z_rel <= self.max_obstacle_height)
            obs_pts = pts_odom[obs_mask]

            # ── 初始化地图 ─────────────────────────────────────────────
            # 地图以机器人为中心，左下角为原点
            origin_x = robot_x - self.local_size / 2.0
            origin_y = robot_y - self.local_size / 2.0

            grid = np.full((self.n_cells, self.n_cells), -1, dtype=np.int8)

            # 传感器有效范围内的格子初始化为自由（0）
            # _cell_dx/dy 是以地图左下角为原点的格子中心坐标
            world_cx = origin_x + self._cell_dx  # 格子中心世界 x
            world_cy = origin_y + self._cell_dy  # 格子中心世界 y
            dist_to_sensor = np.sqrt(
                (world_cx - sensor_x) ** 2 + (world_cy - sensor_y) ** 2)
            grid[dist_to_sensor <= self.sensor_max_range] = 0

            # ── 标记障碍格子 ────────────────────────────────────────────
            if obs_pts.shape[0] > 0:
                col = np.floor((obs_pts[:, 0] - origin_x) / self.resolution).astype(int)
                row = np.floor((obs_pts[:, 1] - origin_y) / self.resolution).astype(int)
                valid = (col >= 0) & (col < self.n_cells) & (row >= 0) & (row < self.n_cells)
                col, row = col[valid], row[valid]

                if self.min_points_per_cell > 1:
                    counts = np.zeros((self.n_cells, self.n_cells), dtype=np.int32)
                    np.add.at(counts, (row, col), 1)
                    grid[counts >= self.min_points_per_cell] = 100
                else:
                    grid[row, col] = 100

            # ── 发布 ───────────────────────────────────────────────────
            # grid[row, col]，row=0 对应 min y（南端），与 OccupancyGrid 约定一致
            self._publish(stamp, origin_x, origin_y, grid)

        except Exception as e:
            rospy.logerr(f"Error in livox_to_localmap: {e}")

    def _publish(self, stamp, origin_x, origin_y, grid):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.costmap_frame
        msg.info.resolution = self.resolution
        msg.info.width = self.n_cells
        msg.info.height = self.n_cells
        msg.info.origin.position.x = origin_x
        msg.info.origin.position.y = origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().tolist()
        self.costmap_pub.publish(msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = LivoxToLocalCostmap()
        node.run()
    except rospy.ROSInterruptException:
        pass
