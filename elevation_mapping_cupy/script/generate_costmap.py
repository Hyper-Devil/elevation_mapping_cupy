#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 GridMap 的 similarity_cvar 层转换为 Costmap (OccupancyGrid)

订阅: filtered_elevation_map (grid_map_msgs/GridMap)
发布: similarity_costmap (nav_msgs/OccupancyGrid)

值映射规则:
- similarity < 0.01 或 NaN: 不更新（保持原值）
- similarity < 0.3: 100 (障碍)
- similarity 0.3~0.5: 70
- similarity 0.5~0.7: 50
- similarity 0.7~0.9: 30
- similarity >= 0.9: 0 (自由)
"""

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from grid_map_msgs.msg import GridMap


class GridMapToCostmap:
    def __init__(self):
        rospy.init_node('gridmap_to_costmap', anonymous=True)

        # 参数配置
        self.costmap_frame = rospy.get_param('~costmap_frame', 'map')
        self.map_size = rospy.get_param('~map_size', 50.0)  # 米
        self.resolution = rospy.get_param('~resolution', 0.5)  # 米/格
        self.similarity_layer = rospy.get_param('~similarity_layer', 'similarity')
        
        # 阈值参数
        self.unknown_threshold = rospy.get_param('~unknown_threshold', 0.01)
        self.obstacle_threshold = rospy.get_param('~obstacle_threshold', 0.3)
        self.free_threshold = rospy.get_param('~free_threshold', 0.9)

        # 计算地图尺寸
        self.map_width = int(self.map_size / self.resolution)   # 100格
        self.map_height = int(self.map_size / self.resolution)  # 100格
        self.origin_x = -self.map_size / 2.0  # -25m
        self.origin_y = -self.map_size / 2.0  # -25m

        # 初始化 Costmap 数据 (-1 表示未知)
        self.costmap_data = np.full((self.map_height, self.map_width), -1, dtype=np.int8)

        # 发布者
        self.costmap_pub = rospy.Publisher('similarity_costmap', OccupancyGrid, queue_size=1)

        # 订阅者
        self.gridmap_sub = rospy.Subscriber('/elevation_mapping/filtered_elevation_map', GridMap, self.gridmap_callback, queue_size=1)

        rospy.loginfo("GridMap to Costmap converter initialized")
        rospy.loginfo(f"  Costmap size: {self.map_size}m x {self.map_size}m")
        rospy.loginfo(f"  Resolution: {self.resolution}m")
        rospy.loginfo(f"  Grid size: {self.map_width} x {self.map_height}")
        rospy.loginfo(f"  Origin: ({self.origin_x}, {self.origin_y})")

    def gridmap_callback(self, msg):
        """处理 GridMap 消息并更新 Costmap"""
        try:
            # 查找 similarity 层的索引
            if self.similarity_layer not in msg.layers:
                rospy.logwarn_throttle(5.0, f"Layer '{self.similarity_layer}' not found in GridMap")
                return

            layer_index = msg.layers.index(self.similarity_layer)
            
            # 获取 GridMap 参数
            gm_resolution = msg.info.resolution
            gm_length_x = msg.info.length_x
            gm_length_y = msg.info.length_y
            gm_pos_x = msg.info.pose.position.x
            gm_pos_y = msg.info.pose.position.y
            
            # GridMap 数据布局
            gm_cols = msg.data[layer_index].layout.dim[0].size  # 列数
            gm_rows = msg.data[layer_index].layout.dim[1].size  # 行数
            
            # 提取 similarity 数据并重塑为2D数组
            similarity_data = np.array(msg.data[layer_index].data, dtype=np.float32)
            similarity_data = similarity_data.reshape((gm_cols, gm_rows))

            # 遍历 GridMap 的每个格子
            for gm_i in range(gm_cols):
                for gm_j in range(gm_rows):
                    # GridMap 索引转世界坐标
                    # GridMap 中心在 (gm_pos_x, gm_pos_y)
                    # gm_i 对应 y 方向，gm_j 对应 x 方向
                    world_x = gm_pos_x + (gm_rows / 2.0 - gm_j - 0.5) * gm_resolution
                    world_y = gm_pos_y + (gm_cols / 2.0 - gm_i - 0.5) * gm_resolution

                    # 世界坐标转 Costmap 索引
                    cm_i = int((world_x - self.origin_x) / self.resolution)
                    cm_j = int((world_y - self.origin_y) / self.resolution)

                    # 检查是否在 Costmap 范围内
                    if 0 <= cm_i < self.map_width and 0 <= cm_j < self.map_height:
                        similarity = similarity_data[gm_i, gm_j]
                        cost = self.similarity_to_cost(similarity)
                        # 只有当 cost 不为 None 时才更新
                        if cost is not None:
                            self.costmap_data[cm_j, cm_i] = cost

            # 发布 Costmap
            self.publish_costmap(msg.info.header.stamp)

        except Exception as e:
            rospy.logerr(f"Error processing GridMap: {e}")

    def similarity_to_cost(self, similarity):
        """
        将 similarity 值转换为 costmap 代价值
        
        参数:
            similarity: 0~1 的相似度值
            
        返回:
            cost: None(不更新), 0~100(自由~障碍)
        """
        # 检查 NaN 或接近 0 的值 -> 不更新
        if np.isnan(similarity) or similarity < self.unknown_threshold:
            return None
        
        # 低于 obstacle_threshold -> 障碍
        if similarity < self.obstacle_threshold:
            return 100

        # 0.3~0.5 -> 70
        if similarity < 0.5:
            return 70

        # 0.5~0.7 -> 50
        if similarity < 0.7:
            return 50

        # 0.7~0.9 -> 30
        if similarity < self.free_threshold:
            return 30

        # 高于 free_threshold -> 自由
        if similarity >= self.free_threshold:
            return 0

        return 0

    def publish_costmap(self, stamp):
        """发布 OccupancyGrid 消息"""
        costmap_msg = OccupancyGrid()
        
        # Header
        costmap_msg.header.stamp = stamp
        costmap_msg.header.frame_id = self.costmap_frame
        
        # MapMetaData
        costmap_msg.info.resolution = self.resolution
        costmap_msg.info.width = self.map_width
        costmap_msg.info.height = self.map_height
        costmap_msg.info.origin.position.x = self.origin_x
        costmap_msg.info.origin.position.y = self.origin_y
        costmap_msg.info.origin.position.z = 0.0
        costmap_msg.info.origin.orientation.w = 1.0
        
        # Data (行优先存储)
        costmap_msg.data = self.costmap_data.flatten().tolist()
        
        self.costmap_pub.publish(costmap_msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = GridMapToCostmap()
        node.run()
    except rospy.ROSInterruptException:
        pass
