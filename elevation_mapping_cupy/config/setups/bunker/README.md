# Bunker Robot Elevation Mapping Configuration

## 📁 文件结构

```
elevation_mapping_cupy/
├── launch/
│   ├── bunker.launch                    # 通用启动文件
│   └── bit-bunker.launch                # BIT-Bunker 启动文件（含 Costmap 节点）
├── script/
│   ├── generate_costmap.py              # similarity_cvar → /similarity_costmap（全局地图）
│   ├── generate_elevation_costmap.py    # inpaint elevation → /elevation_costmap（坡度局部地图）
│   └── generate_livox_costmap.py        # /livox/points → /livox_costmap（点云局部地图，可替代上者）
└── config/setups/bunker/
    ├── bunker_parameters.yaml           # 核心参数配置（分辨率 0.1m，地图 15×15m）
    ├── bunker_sensor.yaml               # 传感器和发布器配置
    ├── bit-bunker_sensor.yaml           # BIT-Bunker 传感器配置
    ├── bunker_plugin.yaml               # 插件配置
    └── bit-bunker_plugin.yaml           # BIT-Bunker 插件配置
```

## 🚀 使用方法

### 1. 启动elevation mapping（BIT-Bunker）
```bash
roslaunch elevation_mapping_cupy bit-bunker.launch
```

### 2. 需要修改的话题名称

在 `bunker_sensor.yaml` 中，请根据实际情况修改以下话题：

```yaml
subscribers:
  lidar_points:
    topic_name: '/cloud_registered'                    # ⚠️ FAST-LIO点云话题
  
  rgb_camera:
    topic_name: '/camera/image_rect_color'             # ⚠️ RGB相机话题
    camera_info_topic_name: '/camera/camera_info'      # ⚠️ 相机内参话题
  
  similarity_feature:
    topic_name: '/feature_extractor/similarity_map'    # ⚠️ 你的特征图话题
    camera_info_topic_name: '/camera/camera_info'      # ⚠️ 相机内参（应与RGB相同）
    channel_info_topic_name: '/feature_extractor/channel_info'  # ⚠️ 通道信息话题
```

### 3. 需要修改的坐标系

在 `bunker_parameters.yaml` 中，请根据FAST-LIO的配置修改：

```yaml
map_frame: 'camera_init'      # ⚠️ FAST-LIO全局坐标系（可能是 'camera_init' 或 'map'）
base_frame: 'body'            # ⚠️ 机器人本体坐标系（可能是 'body' 或 'base_link'）
```

## 📡 需要发布的话题

你的特征提取节点需要发布以下话题：

### 1. 特征图 (必需)
```
话题: /feature_extractor/similarity_map
类型: sensor_msgs/Image
编码: "passthrough"
数据: float32, shape (H, W, 1)
```

### 2. 相机内参 (必需)
```
话题: /camera/camera_info
类型: sensor_msgs/CameraInfo
内容: K矩阵, D畸变系数, 图像尺寸
```

### 3. 通道信息 (必需)
```
话题: /feature_extractor/channel_info
类型: elevation_map_msgs/ChannelInfo
内容: channels = ['similarity']
```

## 🎨 特征图示例代码

```python
#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, CameraInfo
from elevation_map_msgs.msg import ChannelInfo
from cv_bridge import CvBridge
import numpy as np

class SimilarityFeaturePublisher:
    def __init__(self):
        rospy.init_node('similarity_feature_publisher')
        self.bridge = CvBridge()
        
        # 发布器
        self.feature_pub = rospy.Publisher(
            '/feature_extractor/similarity_map', Image, queue_size=2)
        self.channel_pub = rospy.Publisher(
            '/feature_extractor/channel_info', ChannelInfo, queue_size=2)
        
        # 假设相机内参由其他节点发布
        # 如果没有，也需要发布相机内参
        
        # 订阅RGB图像
        rospy.Subscriber('/camera/image_rect_color', Image, self.callback)
        
    def callback(self, rgb_msg):
        # 1. 转换图像
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        
        # 2. 计算similarity特征 (你的算法)
        similarity_map = self.compute_similarity(rgb_image)
        # shape: (H, W) 单通道
        
        # 3. 转换为3D数组
        similarity_map = similarity_map[:, :, np.newaxis].astype(np.float32)
        
        # 4. 发布特征图
        feature_msg = self.bridge.cv2_to_imgmsg(similarity_map, "passthrough")
        feature_msg.header = rgb_msg.header
        self.feature_pub.publish(feature_msg)
        
        # 5. 发布通道信息
        channel_info = ChannelInfo()
        channel_info.header = rgb_msg.header
        channel_info.channels = ['similarity']  # 通道名称
        self.channel_pub.publish(channel_info)
    
    def compute_similarity(self, image):
        # 你的特征提取算法
        # 返回 (H, W) 的 float32 数组，值在 0-1 之间
        pass

if __name__ == '__main__':
    node = SimilarityFeaturePublisher()
    rospy.spin()
```

## 📊 输出话题

启动后，elevation_mapping 及 Costmap 节点发布以下话题：

```
/elevation_mapping/elevation_map_raw              # 原始地图 (5 Hz)
    layers: elevation, traversability, variance, rgb,
            similarity, similarity_var, similarity_cvar, max_categories

/elevation_mapping/filtered_elevation_map         # 滤波地图 (3 Hz)
    layers: inpaint, smooth, min_filter, elevation, traversability,
            similarity, similarity_var, similarity_cvar, max_categories

/similarity_costmap                               # 全局代价地图 (nav_msgs/OccupancyGrid)
    frame_id: map，200×200m，分辨率 0.1m，持续累积
    由 generate_costmap.py 从 similarity_cvar 层转换

/elevation_costmap                                # 坡度局部代价地图 (nav_msgs/OccupancyGrid)
    frame_id: odom，15×15m（跟随 GridMap 窗口），分辨率 0.1m，每帧刷新
    由 generate_elevation_costmap.py 从 inpaint 层坡度转换
    坐标轴：i 轴（y）和 j 轴（x）均已翻转，与 OccupancyGrid 约定对齐

/livox_costmap                                    # 点云局部代价地图 (nav_msgs/OccupancyGrid)
    frame_id: odom，20×20m（以机器人为中心滚动），分辨率 0.1m，每帧刷新
    由 generate_livox_costmap.py 从 /livox/points 过滤地面后转换
    过滤规则：25cm 盲区 | 高度 0.2m~2.0m（相对 base_link z）| 每格 ≥2 个点才标障碍
    不累积历史，动态障碍不留痕迹
```

## ⚙️ 关键参数说明

### 地图参数
- `resolution: 0.1` - 10cm 分辨率
- `map_length: 15` - 15m × 15m 地图尺寸（即 150×150 格）

### slope_filter 参数（bit-bunker_plugin_heruistic.yaml）
- `contact_length: 0.56` - 履带接地长度 (m)
- `track_width: 0.15` - 单侧履带宽 (m)
- `total_width: 0.778` - 车体总宽 (m)，履带中心间距 = 0.778 - 0.15 = 0.628 m
- `critical_value: 1.047` - 坡度达到 60°（π/3 rad）时可通行性降为 0

坡度计算在 **0.6 × 0.6 m 的足印范围**内取均值高程后再求梯度，对应接地长度（560 mm）和履带中心间距（628 mm），使坡度估计反映整车实际经历的俯仰/横滚，而非单点噪声。

### 性能参数
- `update_pose_fps: 20.0` - 位姿更新频率，匹配FAST-LIO
- `map_acquire_fps: 10.0` - 地图获取频率
- `publish fps: 5.0/2.0/3.0` - 不同发布器的频率

### 传感器参数
- `min_valid_distance: 0.5` - 过滤0.5m内的点（避免机器人本体）
- `max_height_range: 2.0` - 过滤2m以上的点（避免天花板）

## 🔧 调试技巧

### 1. 检查话题是否发布
```bash
rostopic list | grep -E 'cloud_registered|similarity_map|camera_info'
```

### 2. 检查TF树
```bash
rosrun tf view_frames
# 确保 camera_init -> body -> camera_optical_frame 的变换存在
```

### 3. 检查elevation mapping状态
```bash
rostopic echo /elevation_mapping/elevation_map_raw/layers
# 应该看到: elevation, traversability, variance, rgb, similarity
```

### 4. RViz可视化
添加GridMap显示插件：
- Topic: /elevation_mapping/elevation_map_raw
- 可选择显示的层: elevation, traversability, similarity

## ⚠️ 注意事项

1. **特征图尺寸**不需要在配置中指定，会从消息自动获取
2. **相机内参必须正确**，否则图像-地图投影会失败
3. **坐标系必须对齐**，确保TF树完整
4. **特征值范围**建议归一化到 [0, 1]
5. **时间戳同步**，相机和特征图的时间戳应尽量接近

## 📝 下一步

1. ✅ 修改话题名称（bunker_sensor.yaml）
2. ✅ 修改坐标系名称（bunker_parameters.yaml）
3. ✅ 实现特征提取节点
4. ✅ 测试launch文件
5. ✅ 在RViz中查看结果

## 🐛 常见问题

**Q: 特征图没有融合到地图中？**
A: 检查 channel_info 的 channels 是否为 ['similarity']，与配置文件一致

**Q: 图像投影位置不对？**
A: 检查相机内参K矩阵和畸变参数D，确保相机已校正

**Q: TF错误？**
A: 确保 camera_init, body, camera_optical_frame 的TF都在发布

**Q: 地图更新慢？**
A: 调整 update_pose_fps 和 map_acquire_fps，确保不超过硬件能力
