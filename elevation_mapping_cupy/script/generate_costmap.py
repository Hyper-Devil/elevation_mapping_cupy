#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 GridMap 的 similarity_cvar 层转换为 Costmap (OccupancyGrid)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Topic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  订阅: /elevation_mapping/filtered_elevation_map  (grid_map_msgs/GridMap)
  发布: similarity_costmap                         (nav_msgs/OccupancyGrid)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Costmap 地图设置
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - 尺寸: map_size × map_size（默认 50m × 50m），固定静态地图
  - 分辨率: 默认 0.5m/格，可通过 ROS 参数 ~resolution 配置
  - 原点: (-map_size/2, -map_size/2)，机器人初始位于地图中心
  - 初始值: -1（未知），已观测区域只增不减，不因机器人离开而清除
    （数据融合由 GridMap 自身维护，本节点仅做格式转换）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GridMap 数据布局（grid_map_msgs/Float32MultiArray）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - 数据以 Eigen 列优先存储，reshape 后形状为 (dim[0].size, dim[1].size)
  - dim[0]: label="column"，对应 y 方向，大小 = y_cells
  - dim[1]: label="row"，   对应 x 方向，大小 = x_cells
  - 索引 [i, j] = y方向第 i 格、x方向第 j 格

  坐标转换公式（实测验证方向正确）:
    world_y = pos_y + (y_cells/2 - i - 0.5) * resolution
    world_x = pos_x + (x_cells/2 - j - 0.5) * resolution

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
similarity → cost 映射规则（分档阈值）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NaN 或 < unknown_threshold(0.01) : -1  不更新，保持原值
  [unknown_threshold, 0.3)          :  40
  [0.3, 0.5)                        :  20
  [0.5, free_threshold(0.9))        :  10
  >= free_threshold(0.9)            :   0 自由

  阈值均可通过 ROS 参数配置:
    ~unknown_threshold  (默认 0.01)
    ~obstacle_threshold (默认 0.3)
    ~free_threshold     (默认 0.9)
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
        self.similarity_layer = rospy.get_param('~similarity_layer', 'similarity_cvar')
        
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
            gm_pos_x = msg.info.pose.position.x
            gm_pos_y = msg.info.pose.position.y
            
            # GridMap 数据布局
            gm_cols = msg.data[layer_index].layout.dim[0].size  # 列数
            gm_rows = msg.data[layer_index].layout.dim[1].size  # 行数
            
            # 提取 similarity 数据并重塑为2D数组
            similarity_data = np.array(msg.data[layer_index].data, dtype=np.float32)
            similarity_data = similarity_data.reshape((gm_cols, gm_rows))

            # 向量化计算：GridMap 索引 → 世界坐标 → Costmap 索引
            # dim[0](gm_cols) 对应 y 方向，dim[1](gm_rows) 对应 x 方向
            i_idx = np.arange(gm_cols)  # y方向索引
            j_idx = np.arange(gm_rows)  # x方向索引

            world_y = gm_pos_y + (gm_cols / 2.0 - i_idx - 0.5) * gm_resolution  # (gm_cols,)
            world_x = gm_pos_x + (gm_rows / 2.0 - j_idx - 0.5) * gm_resolution  # (gm_rows,)

            cm_rows = np.floor((world_y - self.origin_y) / self.resolution).astype(int)  # (gm_cols,)
            cm_cols = np.floor((world_x - self.origin_x) / self.resolution).astype(int)  # (gm_rows,)

            # 过滤超出 Costmap 范围的索引
            valid_row = (cm_rows >= 0) & (cm_rows < self.map_height)
            valid_col = (cm_cols >= 0) & (cm_cols < self.map_width)

            gi_valid = np.where(valid_row)[0]
            gj_valid = np.where(valid_col)[0]

            if gi_valid.size == 0 or gj_valid.size == 0:
                return

            # 提取有效子区域并批量计算 cost
            sub_sim = similarity_data[np.ix_(gi_valid, gj_valid)]      # (ny, nx)
            cost_grid = self.similarity_to_cost_vectorized(sub_sim)     # (ny, nx)，-1=不更新

            # 构建 costmap 行列索引网格并批量写入
            CM_ROWS, CM_COLS = np.meshgrid(cm_rows[gi_valid], cm_cols[gj_valid], indexing='ij')
            update_mask = cost_grid >= 0
            self.costmap_data[CM_ROWS[update_mask], CM_COLS[update_mask]] = cost_grid[update_mask]

            # 发布 Costmap
            self.publish_costmap(msg.info.header.stamp)

        except Exception as e:
            rospy.logerr(f"Error processing GridMap: {e}")

    def similarity_to_cost_vectorized(self, similarity):
        """
        向量化版本：将 similarity 数组批量转换为 costmap 代价值

        参数:
            similarity: np.ndarray，值域 0~1
        返回:
            cost: np.ndarray (int8)，-1=不更新，0~100(自由~障碍)
        """
        cost = np.full(similarity.shape, -1, dtype=np.int8)

        # NaN 或低于 unknown_threshold -> 不更新
        known = ~np.isnan(similarity) & (similarity >= self.unknown_threshold)
        s = similarity[known]

        c = np.empty(s.shape, dtype=np.int8)
        c[s < self.obstacle_threshold] = 100
        c[(s >= self.obstacle_threshold) & (s < 0.3)] = 40
        c[(s >= 0.3) & (s < 0.5)] = 20
        c[(s >= 0.5) & (s < self.free_threshold)] = 10
        c[s >= self.free_threshold] = 0

        cost[known] = c
        return cost

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
