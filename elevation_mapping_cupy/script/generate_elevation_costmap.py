#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 GridMap 的 inpaint（填充后的 elevation）层转换为 Local Costmap (OccupancyGrid)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Topic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  订阅: /elevation_mapping/filtered_elevation_map  (grid_map_msgs/GridMap)
  发布: elevation_costmap                          (nav_msgs/OccupancyGrid)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Local Costmap 特性
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - 每帧刷新，不做历史积累（区别于 global similarity_costmap）
  - 地图范围与当前 GridMap 一致，随机器人滚动
  - 发布在 odom 坐标系，与 Nav2 local_costmap.global_frame 对齐
  - GridMap 中心（map 系）经 TF 变换后转换到 odom 系作为 costmap 原点

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Elevation → Cost 映射规则（基于坡度/梯度）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - 以 numpy.gradient 计算 inpaint 梯度幅值（单位: m/m，即 tan(slope)）
  - NaN（inpaint 仍无法填充的区域）           : -1  未知
  - slope < flat_threshold   (默认 0.15)       :  0  可通行
  - flat_threshold ≤ slope < 0.25              : 30
  - 0.25          ≤ slope < obstacle_threshold : 60
  - slope ≥ obstacle_threshold (默认 0.40)     : 100  障碍

  阈值均可通过 ROS 参数配置:
    ~flat_threshold      (默认 0.15)
    ~obstacle_threshold  (默认 0.40)
    ~costmap_frame       (默认 odom)
    ~gridmap_frame       (默认 map，即 GridMap 的源坐标系)
"""

import math

import numpy as np
import rospy
import tf2_ros
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import OccupancyGrid


class ElevationToCostmap:
    def __init__(self):
        rospy.init_node('elevation_to_costmap', anonymous=True)

        self.elevation_layer = rospy.get_param('~elevation_layer', 'inpaint')
        self.costmap_frame = rospy.get_param('~costmap_frame', 'odom')
        self.gridmap_frame = rospy.get_param('~gridmap_frame', 'map')
        self.flat_threshold = rospy.get_param('~flat_threshold', 0.15)
        self.obstacle_threshold = rospy.get_param('~obstacle_threshold', 0.40)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.costmap_pub = rospy.Publisher('elevation_costmap', OccupancyGrid, queue_size=1)
        self.gridmap_sub = rospy.Subscriber(
            '/elevation_mapping/filtered_elevation_map',
            GridMap,
            self.gridmap_callback,
            queue_size=1,
        )

        rospy.loginfo("Elevation to Local Costmap converter initialized")
        rospy.loginfo(f"  Elevation layer : {self.elevation_layer}")
        rospy.loginfo(f"  Costmap frame   : {self.costmap_frame}")
        rospy.loginfo(f"  GridMap frame   : {self.gridmap_frame}")
        rospy.loginfo(f"  Flat threshold  : {self.flat_threshold} m/m")
        rospy.loginfo(f"  Obstacle thresh : {self.obstacle_threshold} m/m")

    def gridmap_callback(self, msg):
        try:
            if self.elevation_layer not in msg.layers:
                rospy.logwarn_throttle(5.0, f"Layer '{self.elevation_layer}' not found in GridMap")
                return

            layer_index = msg.layers.index(self.elevation_layer)

            gm_resolution = msg.info.resolution
            gm_pos_x = msg.info.pose.position.x  # GridMap 中心，map 系
            gm_pos_y = msg.info.pose.position.y

            # GridMap 数据布局：dim[0]=列(y方向), dim[1]=行(x方向)
            gm_cols = msg.data[layer_index].layout.dim[0].size  # y方向格数
            gm_rows = msg.data[layer_index].layout.dim[1].size  # x方向格数

            elev = np.array(msg.data[layer_index].data, dtype=np.float32)
            elev = elev.reshape((gm_cols, gm_rows))

            # 计算梯度幅值（单位 m/m）
            # axis-0 对应世界 y 方向，axis-1 对应世界 x 方向
            grad_y, grad_x = np.gradient(elev, gm_resolution)
            slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
            slope[np.isnan(elev)] = np.nan

            cost_grid = self.slope_to_cost(slope)

            # GridMap i=0 对应最大 y（北端），OccupancyGrid row=0 对应最小 y（南端）→ 翻转 i 轴
            # GridMap j=0 对应最大 x（东侧），OccupancyGrid col=0 对应最小 x（西侧）→ 翻转 j 轴
            cost_oc = cost_grid[::-1, ::-1]

            # 将 GridMap 中心从 gridmap_frame(map) 变换到 costmap_frame(odom)
            origin_x, origin_y = self.transform_origin(
                gm_pos_x, gm_pos_y, gm_rows, gm_cols, gm_resolution,
                msg.info.header.stamp,
            )
            if origin_x is None:
                return

            self.publish_costmap(
                stamp=msg.info.header.stamp,
                resolution=gm_resolution,
                width=gm_rows,
                height=gm_cols,
                origin_x=origin_x,
                origin_y=origin_y,
                data=cost_oc,
            )

        except Exception as e:
            rospy.logerr(f"Error processing elevation GridMap: {e}")

    def transform_origin(self, gm_pos_x, gm_pos_y, gm_rows, gm_cols, resolution, stamp):
        """将 GridMap 中心从 gridmap_frame 变换到 costmap_frame，返回 costmap 左下角原点。"""
        if self.costmap_frame == self.gridmap_frame:
            # 无需变换
            return (
                gm_pos_x - (gm_rows / 2.0) * resolution,
                gm_pos_y - (gm_cols / 2.0) * resolution,
            )

        try:
            tf = self.tf_buffer.lookup_transform(
                self.costmap_frame,
                self.gridmap_frame,
                stamp,
                rospy.Duration(0.1),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0, f"TF {self.gridmap_frame}->{self.costmap_frame} failed: {e}")
            return None, None

        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y ** 2 + q.z ** 2))

        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        center_x = cos_y * gm_pos_x - sin_y * gm_pos_y + tx
        center_y = sin_y * gm_pos_x + cos_y * gm_pos_y + ty

        return (
            center_x - (gm_rows / 2.0) * resolution,
            center_y - (gm_cols / 2.0) * resolution,
        )

    def slope_to_cost(self, slope):
        cost = np.full(slope.shape, -1, dtype=np.int8)
        known = ~np.isnan(slope)
        s = slope[known]
        c = np.empty(s.shape, dtype=np.int8)
        c[s < self.flat_threshold] = 0
        c[(s >= self.flat_threshold) & (s < 0.25)] = 30
        c[(s >= 0.25) & (s < self.obstacle_threshold)] = 60
        c[s >= self.obstacle_threshold] = 100
        cost[known] = c
        return cost

    def publish_costmap(self, stamp, resolution, width, height, origin_x, origin_y, data):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.costmap_frame
        msg.info.resolution = resolution
        msg.info.width = width
        msg.info.height = height
        msg.info.origin.position.x = origin_x
        msg.info.origin.position.y = origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = data.flatten().tolist()
        self.costmap_pub.publish(msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = ElevationToCostmap()
        node.run()
    except rospy.ROSInterruptException:
        pass
