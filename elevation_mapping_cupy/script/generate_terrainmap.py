#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为点云的强度通道赋予 Costmap 信息

订阅:
  - /accumulated_map_points (sensor_msgs/PointCloud2)
  - /similarity_costmap (nav_msgs/OccupancyGrid)

发布:
  - /terrain_map (sensor_msgs/PointCloud2)

强度赋值规则:
  - costmap = -1 (未知): intensity = vehicleHeight (1.5)
  - costmap = 100 (障碍): intensity = vehicleHeight (1.5)
  - costmap = 30 (中间): intensity = 0.3
  - costmap = 0 (自由): intensity = 0
"""

import rospy
import numpy as np
import message_filters
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header


class TerrainMapGenerator:
    def __init__(self):
        rospy.init_node('terrain_map_generator', anonymous=True)

        # 参数配置
        self.vehicle_height = rospy.get_param('~vehicle_height', 1.5)
        self.voxel_size = rospy.get_param('~voxel_size', 0.25)
        self.time_slop = rospy.get_param('~time_slop', 0.5)  # 时间同步容差

        # Costmap 缓存
        self.costmap = None
        self.costmap_info = None

        # 发布者
        self.terrain_pub = rospy.Publisher('/terrain_map', PointCloud2, queue_size=1)

        # 使用 message_filters 进行近似时间同步
        self.pc_sub = message_filters.Subscriber('/accumulated_map_points', PointCloud2)
        self.costmap_sub = message_filters.Subscriber('/similarity_costmap', OccupancyGrid)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.pc_sub, self.costmap_sub],
            queue_size=10,
            slop=self.time_slop
        )
        self.sync.registerCallback(self.sync_callback)

        rospy.loginfo("Terrain map generator initialized")
        rospy.loginfo(f"  Vehicle height: {self.vehicle_height}")
        rospy.loginfo(f"  Voxel size: {self.voxel_size}")
        rospy.loginfo(f"  Time slop: {self.time_slop}")

    def sync_callback(self, pc_msg, costmap_msg):
        """同步回调：处理点云和 costmap"""
        try:
            # 更新 costmap 缓存
            self.costmap_info = costmap_msg.info
            self.costmap = np.array(costmap_msg.data, dtype=np.int8).reshape(
                (costmap_msg.info.height, costmap_msg.info.width)
            )

            # 处理点云
            self.process_pointcloud(pc_msg)

        except Exception as e:
            rospy.logerr(f"Error in sync_callback: {e}")

    def process_pointcloud(self, pc_msg):
        """处理点云：裁剪、降采样、赋值强度"""
        if self.costmap is None:
            rospy.logwarn_throttle(5.0, "Costmap not available yet")
            return

        # 读取点云
        points_list = list(pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True))
        if len(points_list) == 0:
            return

        points = np.array(points_list, dtype=np.float32)

        # 获取 costmap 边界
        origin_x = self.costmap_info.origin.position.x
        origin_y = self.costmap_info.origin.position.y
        map_width = self.costmap_info.width * self.costmap_info.resolution
        map_height = self.costmap_info.height * self.costmap_info.resolution

        # 裁剪点云到 costmap 范围
        mask = (
            (points[:, 0] >= origin_x) & (points[:, 0] < origin_x + map_width) &
            (points[:, 1] >= origin_y) & (points[:, 1] < origin_y + map_height)
        )
        points = points[mask]

        if len(points) == 0:
            return

        # Voxel 降采样
        points = self.voxel_downsample(points)

        if len(points) == 0:
            return

        # 为每个点赋值强度
        intensities = self.assign_intensity(points)

        # 发布点云
        self.publish_terrain_map(pc_msg.header, points, intensities)

    def voxel_downsample(self, points):
        """Voxel 降采样"""
        # 计算每个点的 voxel 索引
        voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)

        # 使用字典去重，保留每个 voxel 的第一个点
        voxel_dict = {}
        for i, idx in enumerate(voxel_indices):
            key = tuple(idx)
            if key not in voxel_dict:
                voxel_dict[key] = points[i]

        return np.array(list(voxel_dict.values()), dtype=np.float32)

    def assign_intensity(self, points):
        """根据 costmap 为点云赋值强度"""
        intensities = np.zeros(len(points), dtype=np.float32)

        origin_x = self.costmap_info.origin.position.x
        origin_y = self.costmap_info.origin.position.y
        resolution = self.costmap_info.resolution

        for i, point in enumerate(points):
            # 世界坐标转 costmap 索引
            cm_x = int((point[0] - origin_x) / resolution)
            cm_y = int((point[1] - origin_y) / resolution)

            # 检查是否在 costmap 范围内
            if 0 <= cm_x < self.costmap_info.width and 0 <= cm_y < self.costmap_info.height:
                cost = self.costmap[cm_y, cm_x]
                intensities[i] = self.cost_to_intensity(cost)
            else:
                # 超出范围，视为未知
                intensities[i] = self.vehicle_height

        return intensities

    def cost_to_intensity(self, cost):
        """
        将 costmap 代价值转换为强度值
        
        参数:
            cost: costmap 值 (-1, 0, 30, 100)
            
        返回:
            intensity: 强度值
        """
        if cost == -1:  # 未知
            return self.vehicle_height
        elif cost >= 100:  # 障碍
            return self.vehicle_height
        elif cost >= 30:  # 中间值
            return 0.3
        else:  # 自由 (cost == 0)
            return 0.0

    def publish_terrain_map(self, header, points, intensities):
        """发布带强度的点云"""
        # 创建点云数据 (x, y, z, intensity)
        cloud_data = []
        for i in range(len(points)):
            cloud_data.append([
                float(points[i, 0]),
                float(points[i, 1]),
                float(points[i, 2]),
                float(intensities[i])
            ])

        # 定义点云字段
        fields = [
            pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='intensity', offset=12, datatype=pc2.PointField.FLOAT32, count=1),
        ]

        # 创建 PointCloud2 消息
        terrain_msg = pc2.create_cloud(header, fields, cloud_data)
        self.terrain_pub.publish(terrain_msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = TerrainMapGenerator()
        node.run()
    except rospy.ROSInterruptException:
        pass
