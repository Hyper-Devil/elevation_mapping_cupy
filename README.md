# Elevation Mapping cupy

![python tests](https://github.com/leggedrobotics/elevation_mapping_cupy/actions/workflows/python-tests.yml/badge.svg)

[Documentation](https://leggedrobotics.github.io/elevation_mapping_cupy/)

## Overview

The Elevaton Mapping CuPy software package represents an advancement in robotic navigation and locomotion.
Integrating with the Robot Operating System (ROS) and utilizing GPU acceleration, this framework enhances point cloud registration and ray casting,
crucial for efficient and accurate robotic movement, particularly in legged robots.
![screenshot](docs/media/main_repo.png)
![screenshot](docs/media/main_mem.png)
![gif](docs/media/convex_approximation.gif)

## Key Features

- **Height Drift Compensation**: Tackles state estimation drifts that can create mapping artifacts, ensuring more accurate terrain representation.

- **Visibility Cleanup and Artifact Removal**: Raycasting methods and an exclusion zone feature are designed to remove virtual artifacts and correctly interpret overhanging obstacles, preventing misidentification as walls.

- **Learning-based Traversability Filter**: Assesses terrain traversability using local geometry, improving path planning and navigation.

- **Versatile Locomotion Tools**: Incorporates smoothing filters and plane segmentation, optimizing movement across various terrains.

- **Multi-Modal Elevation Map (MEM) Framework**: Allows seamless integration of diverse data like geometry, semantics, and RGB information, enhancing multi-modal robotic perception.

- **GPU-Enhanced Efficiency**: Facilitates rapid processing of large data structures, crucial for real-time applications.

## Overview

![Overview of multi-modal elevation map structure](docs/media/overview.png)

Overview of our multi-modal elevation map structure. The framework takes multi-modal images (purple) and multi-modal (blue) point clouds as
input. This data is input into the elevation map by first associating the data to the cells and then fused with different fusion algorithms into the various
layers of the map. Finally the map can be post-processed with various custom plugins to generate new layers (e.g. traversability) or process layer for
external components (e.g. line detection).

## Citing

If you use the Elevation Mapping CuPy, please cite the following paper:
Elevation Mapping for Locomotion and Navigation using GPU

[Elevation Mapping for Locomotion and Navigation using GPU](https://arxiv.org/abs/2204.12876)

Takahiro Miki, Lorenz Wellhausen, Ruben Grandia, Fabian Jenelten, Timon Homberger, Marco Hutter  

```bibtex
@inproceedings{miki2022elevation,
  title={Elevation mapping for locomotion and navigation using gpu},
  author={Miki, Takahiro and Wellhausen, Lorenz and Grandia, Ruben and Jenelten, Fabian and Homberger, Timon and Hutter, Marco},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={2273--2280},
  year={2022},
  organization={IEEE}
}
```

[MEM: Multi-Modal Elevation Mapping for Robotics and Learning](https://arxiv.org/abs/2309.16818v1)

Gian Erni, Jonas Frey, Takahiro Miki, Matias Mattamala, Marco Hutter

```bibtex
@inproceedings{erni2023mem,
  title={MEM: Multi-Modal Elevation Mapping for Robotics and Learning},
  author={Erni, Gian and Frey, Jonas and Miki, Takahiro and Mattamala, Matias and Hutter, Marco},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={11011--11018},
  year={2023},
  organization={IEEE}
}
```

## Quick instructions to run

### Installation

First, clone to your catkin_ws

```zsh
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone https://github.com/leggedrobotics/elevation_mapping_cupy.git
```

Then install dependencies.
You can also use docker which already install all dependencies.
When you run the script it should pull the image.

```zsh
cd docker
./run.sh
```

You can also build locally by running `build.sh`, but in this case change `IMAGE_NAME` in `run.sh` to `elevation_mapping_cupy:latest`.

For more information, check [Document](https://leggedrobotics.github.io/elevation_mapping_cupy/getting_started/installation.html)

### Build package

Inside docker container.

```zsh
cd $HOME/catkin_ws
catkin build elevation_mapping_cupy
catkin build convex_plane_decomposition_ros  # If you want to use plane segmentation
catkin build semantic_sensor  # If you want to use semantic sensors
```

### Run turtlebot example

![Elevation map examples](docs/media/turtlebot.png)

```bash
export TURTLEBOT3_MODEL=waffle
roslaunch elevation_mapping_cupy turtlesim_simple_example.launch
```

For fusing semantics into the map such as rgb from a multi modal pointcloud:

```bash
export TURTLEBOT3_MODEL=waffle
roslaunch elevation_mapping_cupy turtlesim_semantic_pointcloud_example.launch
```

For fusing semantics into the map such as rgb semantics or features from an image:

```bash
export TURTLEBOT3_MODEL=waffle
roslaunch elevation_mapping_cupy turtlesim_semantic_image_example.launch
```

For plane segmentation:

```bash
catkin build convex_plane_decomposition_ros
export TURTLEBOT3_MODEL=waffle
roslaunch elevation_mapping_cupy turtlesim_plane_decomposition_example.launch
```

To control the robot with a keyboard, a new terminal window needs to be opened.
Then run

```bash
export TURTLEBOT3_MODEL=waffle
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

Velocity inputs can be sent to the robot by pressing the keys `a`, `w`, `d`, `x`. To stop the robot completely, press `s`.

---

## 本地定制内容目录

下文为本仓库相对上游 `leggedrobotics/elevation_mapping_cupy` 增量内容的索引：

- [本地定制修改记录（Bunker 语义分割接入）](#本地定制修改记录bunker-语义分割接入) — 单通道 index 图自动展开 one-hot、RViz 语义层显示约定、Bunker sensor 配置
- [针对 Conda Python 3.10 (Numpy/PyTorch 高版本) 的编译与部署指南](#针对-conda-python-310-numpypytorch-高版本-的编译与部署指南) — empy 版本、GLIBCXX、CMake Python 重定向、`cp.bool8` → `cp.bool_`
- [几何可通行性插件（Geometric Traversability Plugins）](#几何可通行性插件geometric-traversability-plugins)
  - [SlopeFilter（坡度）](#slopefilter坡度) — 足印均值 + sigmoid 响应
  - [RoughnessFilter（粗糙度）](#roughnessfilter粗糙度) — 多尺度圆形均值差 + sigmoid
  - [StepFilter（台阶）](#stepfilter台阶) — 两次滑窗 max−min + 障碍格数敏感度
- [语义类别属性插件（SemanticClassLayer）](#语义类别属性插件semanticclasslayer) — argmax + SemCost/SemSensitivity 查表
- [层值约定](#层值约定越大代表通行代价越低还是越高) — 可通行性"越大越好"，`sem_cost` 例外
- [可通行性融合插件（HeuristicTraversability）](#可通行性融合插件heuristictraversability) — 几何门 × 语义门 + 输出平滑
- [SmoothFilter 增强](#smoothfilter-增强可配置核大小与迭代次数) — 可配置 `size`/`num_passes`
- [Bunker Heuristic Pipeline 集成调试记录（2026-05-13）](#bunker-heuristic-pipeline-集成调试记录2026-05-13) — 端到端联调坑点汇总

---

## 本地定制修改记录（Bunker 语义分割接入）

### 单通道 index 图自动展开为 one-hot（`elevation_mapping_ros.cpp`）

**背景**：上游语义分割节点（san_app）发布 H×W uint8 的类别索引图（3MB），而不是 H×W×N float32 的 one-hot 概率图（264MB），以减少消息大小和发布延迟。

**修改位置**：`src/elevation_mapping_ros.cpp`，`inputImage()` 函数中，`cv::split` 之后、通道数校验之前。python文件修改无效。

**逻辑**：
- 若图像为单通道 **整数类型**（`CV_8U/CV_8S/CV_16U/CV_16S/CV_32S`）且配置了多个语义通道：
  - 将像素值视为类别索引，展开为 N 个二值通道（属于该类=1，否则=0）
- 若图像为单通道 **浮点类型**（`CV_32F` 等）：不做展开，作为单通道概率图直接传入
- 若图像为多通道：走原有路径，不受影响

这样做到了对下游代码零侵入：多通道 float32 图像的原有行为完全不变。

**重新编译**：
```bash
source /opt/ros/noetic/setup.bash
catkin_make --pkg elevation_mapping_cupy
```

### rviz 可视化语义层（重要）

`semantic_filter` 插件输出的 `max_categories` 层存储的是 **RGB 打包成 float32 的颜色值**，不是强度值。

在 rviz 的 GridMap 显示中必须设置：
- `Color Layer`: `max_categories`
- `Color Transformer`: **`ColorLayer`**（不能用 `IntensityLayer`，否则显示为灰度梯度）

### Bunker 传感器配置

语义分割接入配置文件：`config/setups/bunker/bunker_sensor.yaml`

关键字段：
```yaml
semantic_seg:
  topic_name: '/hikrobot_camera_L/semantic_image'   # 接收单通道 uint8 index 图
  channel_info_topic_name: '/hikrobot_camera_L/semantic_info'  # 动态获取通道名列表
  data_type: image
  channels: ['sem_6s5f2w', 'sem_road', ...]         # 22 个语义类别
```

`max_categories` 已加入两个发布话题的 `layers` 列表：
- `/elevation_mapping/elevation_map_raw`（5Hz）
- `/elevation_mapping/filtered_elevation_map`（3Hz）
---

## 针对 Conda Python 3.10 (Numpy/PyTorch 高版本) 的编译与部署指南

在使用 Ubuntu 20.04 (ROS Noetic) 并结合高版本深度学习框架 (通过 Conda 管理 Python 3.10) 时，直接编译本工程会遇到一系列系统环境、路径、以及新旧库的兼容性问题。在此记录成功编译并运行的方法以及踩坑记录。

### 1. 编译前环境准备与依赖修复

由于 Conda 环境自带独立的 Python 版本，ROS的默认编译系统找不到相关的 Python 包。需要在启动 `catkin_make` 前，在 Conda 环境中安装 ROS 构建所需模块。

**踩坑 1：缺少 `empy` 或 `empy` 版本过高导致报错 `PY_em`**
*   **现象**: `cmake` 时报错 `Unable to find either executable 'empy' or Python module 'em'...`
*   **原因**: Python 3.10 下如果直接 `pip install empy`，默认会安装 4.x 版本的 `empy`，而 ROS Noetic 的 CMake 脚本还在使用已被 4.x 移除的旧版 `em` 模块语法。
*   **解决**: 强制安装 `<4` 的版本：
    ```bash
    conda activate <your_env>
    pip install catkin_pkg rospkg defusedxml netifaces
    pip install "empy<4"
    ```

### 2. 绕开 GLIBCXX (libstdc++) 版本冲突

**踩坑 2：Conda 标准库与系统 GCC 不兼容**
*   **现象**: 运行 `catkin_make` 最后进行链接(Link)生成 `.so` 和可执行文件时，提示 `undefined reference to std::condition_variable::wait(...)@GLIBCXX_3.4.30` 等类似错误。
*   **原因**: Conda 中安装的 PyTorch 或其他三方库是基于更高版本 C++ 标准库 (`libstdc++.so.6`) 编译的，而 Ubuntu 20.04 系统默认的 GCC 9 工具链支持的最高版本不足。
*   **错误尝试**: 如果试图在 Conda 里通过 `conda install gcc_linux-64` 引入 Conda 的编译器来编译 ROS 环境，又会导致链接时找不到外部的 ROS 库、系统 OpenCV 甚至无法定位系统 `libpthread`。
*   **正确解决**: 保持使用系统的默认 `cc/c++`，但升级 Ubuntu 系统的 `libstdc++6` 动态库：
    ```bash
    sudo apt-get update && sudo apt-get install software-properties-common -y
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
    sudo apt-get update && sudo apt-get install libstdc++6 -y
    ```
    *注意：若之前为了解决此问题在 Conda 中安装过 `sysroot_linux-64`、`gcc_linux-64` 等，请先用 `conda remove` 将其卸载并清理 C/C++ 环境变量 (`unset CC CXX LDFLAGS CFLAGS CXXFLAGS`)，防止它们污染系统的链接路径。*

### 3. 执行特殊的 CMake 编译指令

环境清理完毕后，由于 ROS 默认绑定系统的 python3，需要通过传入 `-D` 参数强制重定向到你的 Conda Python，再进行编译。

```bash
# 激活 ROS 环境
source /opt/ros/noetic/setup.bash

# 清理旧的编译缓存（重要）
rm -rf build devel

# 使用指定的 Conda Python 路径进行增量或者独立编译
catkin_make -DCATKIN_WHITELIST_PACKAGES="elevation_map_msgs;elevation_mapping_cupy" \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/opt/conda/envs/<your_env>/bin/python \
            -DPYTHON_INCLUDE_DIR=/opt/conda/envs/<your_env>/include/python3.10 \
            -DPYTHON_LIBRARY=/opt/conda/envs/<your_env>/lib/libpython3.10.so
```

### 4. 运行时 numpy 高版本 API 废弃的修复

**踩坑 3：`numpy.bool8` AttributeError 导致节点在启动瞬间闪退崩溃**
*   **现象**: `roslaunch` 后，RViz 正常跳出（如果有屏幕），但 `elevation_mapping_node` 崩溃报错：`AttributeError: module 'numpy' has no attribute 'bool8'`。
*   **原因**: Python 3.10 环境中使用的 `numpy >= 1.24` 已经彻底移除了 `np.bool8`，而之前编写的 CuPy 脚本 (`cp.ones(..., cp.bool8)`) 还在使用这个旧名称。
*   **解决**: 需要手动修改代码。搜索工作空间下所有 `cp.bool8` 的使用，并替换成 `cp.bool_`：
    *   文件位置：`elevation_mapping_cupy/script/elevation_mapping_cupy/semantic_map.py` (在 `__init__` 和 `clear` 附近)。
    *   将 `cp.bool8` 修改为安全的 `cp.bool_`。
修改完后，重新 `roslaunch` 即可完美建立基于高版本 Python / CuPy / PyTorch 的地形建图环境。

---

## 几何可通行性插件（Geometric Traversability Plugins）

基于 [ETH Zurich traversability_estimation](https://github.com/leggedrobotics/traversability_estimation) 的 C++ 实现，将三种经典几何可通行性评估算法移植为 CuPy GPU 加速插件。

**文件位置**：`elevation_mapping_cupy/script/elevation_mapping_cupy/plugins/`

| 文件 | 算法 | 输出层值域 |
|---|---|---|
| `slope_filter.py` | 坡度可通行性（足印均值 + sigmoid） | [0, 1] |
| `roughness_filter.py` | 多尺度粗糙度可通行性（sigmoid） | [0, 1] |
| `step_filter.py` | 台阶高度可通行性（两次滑窗） | [0, 1] |

三个插件均输出 **[0, 1] 可通行性分数**（1 = 完全可通行，0 = 不可通行），可直接通过 `plugin_config.yaml` 配置启用，无需修改任何 C++ 代码。

> **注意**：三个插件的 `input_layer` 建议指定 `inpaint` 或 `smooth` 层（即由 `Inpainting`/`SmoothFilter` 输出的稠密层），避免高程层中的空洞（NaN）干扰梯度和滤波计算。`SmoothFilter` 的 `input_layer_name` 也应改用 `inpaint` 而不是 `min_filter`，否则 `uniform_filter` 会把残留 NaN 向邻域扩散。

---

### SlopeFilter（坡度）

**原理**：先在车体足印范围内做均值平滑（等价于在足印内拟合平面），再用法向量–垂直方向夹角计算坡度，最后用 sigmoid 映射为可通行性。

```
足印尺度       footprint_cells = max(round(contact_length/res), round(track_separation/res))
足印内均值     h_fp = uniform_filter(h, size=footprint_cells)
物理梯度       gx, gy = cp.gradient(h_fp) / resolution
法向量 z 分量   n_z = 1 / sqrt(1 + gx² + gy²)
坡度角         slope = arccos(n_z)  ∈ [0, π/2]
可通行性        traversability = 1 / (1 + exp(steepness × (slope − midpoint)))
```

`track_separation = total_width − track_width`（履带中心间距）。pitch 方向取接地长度、roll 方向取履带中心间距，二者取最大值作为方形核边长保证各向同性。

相比线性公式（`1 − slope/critical_value`），sigmoid 在 `midpoint` 以下响应弱（缓坡几乎不扣分），在 `midpoint` 以上快速下降。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `input_layer` | str | `"elevation_inpainted"` | 输入高程层名称 |
| `resolution` | float | `0.1` | 地图分辨率 (m/cell)，须与 `core_param.yaml` 一致 |
| `contact_length` | float | `0.56` | 履带接地长度 (m)，pitch 估计尺度 |
| `track_width` | float | `0.15` | 单侧履带宽度 (m) |
| `total_width` | float | `0.778` | 车体总宽 (m) |
| `midpoint` | float | `0.698`（40°） | 坡度达到此值（弧度）时 traversability = 0.5 |
| `steepness` | float | `8.4` | sigmoid 过渡陡峭程度（1/rad），可由 `trav(60°)≈0.05` 反算 |

**plugin_config.yaml 配置示例**（Bunker 履带车，midpoint=20°、`trav(40°)≈0.05`）：
```yaml
traversability_slope:
  type: slope_filter
  enable: True
  fill_nan: False
  is_height_layer: False
  layer_name: "traversability_slope"
  extra_params:
    input_layer:    "smooth"       # smooth 先去除测量噪声，足印均值再做车体尺度平均
    resolution:     0.1
    contact_length: 0.56
    track_width:    0.15
    total_width:    0.778
    midpoint:       0.349          # ≈ 20°
    steepness:      8.4            # ln(19)/(0.698 − 0.349)
```

---

### RoughnessFilter（粗糙度）

**原理**：对高程层分别做小尺度和大尺度圆形均值滤波，两者之差反映局部地形相对宏观趋势面的起伏程度（多尺度粗糙度），sigmoid 映射为可通行性。

```
smooth_low  = circular_mean(elevation, radius=low_radius)   # 保留局部起伏
smooth_high = circular_mean(elevation, radius=high_radius)  # 代表整体趋势
roughness   = |smooth_low - smooth_high|
可通行性     traversability = 1 / (1 + exp(steepness × (roughness − midpoint)))
```

圆形卷积核在初始化时预计算（对应 C++ `CircleIterator`），运行时无额外开销。相比线性公式，sigmoid 对 LiDAR 噪声/草地微起伏更宽容。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `input_layer` | str | `"elevation_inpainted"` | 输入高程层名称 |
| `resolution` | float | `0.1` | 地图分辨率 (m/cell) |
| `low_radius` | float | `0.1` | 小尺度均值滤波半径 (m) |
| `high_radius` | float | `0.56` | 大尺度均值滤波半径 (m)，建议对应车体接地长度 |
| `midpoint` | float | `0.12` | roughness 达到此值时 traversability = 0.5 (m) |
| `steepness` | float | `30.0` | sigmoid 过渡陡峭程度 (1/m) |

**plugin_config.yaml 配置示例**：
```yaml
traversability_roughness:
  type: roughness_filter
  enable: True
  fill_nan: False
  is_height_layer: False
  layer_name: "traversability_roughness"
  extra_params:
    input_layer: "smooth"
    resolution:  0.1
    low_radius:  0.1            # ≈ track_width/2
    high_radius: 0.56           # = contact_length
    midpoint:    0.18           # 18cm 高差时 trav=0.5
    steepness:   25.0           # ln(19)/(0.30 − 0.18)
```

---

### StepFilter（台阶）

**原理**：两次滑窗聚合，检测局部高差并以障碍格数做敏感度控制，避免单格噪声引起大范围误判。

```
# 第一次（first_window_radius）
step_height = max(elevation) - min(elevation)   # 窗口内局部高差

# 第二次（second_window_radius）
step_max = max(step_height)                     # 窗口内最大台阶高度
n_cells  = count(step_height > critical_value)  # 超阈值格数
step     = min(step_max, n_cells / n_cell_critical × step_max)
可通行性  = max(1 - step / critical_value, 0)
```

> **注意**：此滤波器对坡度也敏感——斜坡在滑窗内产生较大的 max-min 高差，因此陡坡同样会输出较低分数。这与 C++ 原版行为一致，与 `SlopeFilter` 配合使用时属于正常的双重约束。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `input_layer` | str | `"elevation_inpainted"` | 输入高程层名称 |
| `resolution` | float | `0.04` | 地图分辨率 (m/cell) |
| `first_window_radius` | float | `0.08` | 第一次滑窗半径 (m)，用于计算局部高差 |
| `second_window_radius` | float | `0.08` | 第二次滑窗半径 (m)，用于聚合台阶信息 |
| `critical_value` | float | `0.2` | 台阶高度阈值 (m)，超出则可通行性为 0 |
| `n_cell_critical` | int | `4` | 第二次窗口内允许的最大障碍格数，超出则不再衰减 |

**plugin_config.yaml 配置示例**：
```yaml
traversability_step:
  type: step_filter
  enable: True
  fill_nan: False
  is_height_layer: False
  layer_name: "traversability_step"
  extra_params:
    input_layer: "inpaint"
    resolution: 0.04
    first_window_radius: 0.3
    second_window_radius: 0.3
    critical_value: 0.2       # 台阶超过 20cm 时可通行性为 0
    n_cell_critical: 4
```

---

### 完整流水线示例（参考 `bit-bunker_plugin_heruistic.yaml`）

```yaml
publishers:
  your_topic:
    layers: ['elevation', 'traversability_slope',
             'traversability_roughness', 'traversability_step']
```

完整可运行配置参见 `elevation_mapping_cupy/config/setups/bunker/bit-bunker_plugin_heruistic.yaml`。

---

## 语义类别属性插件（SemanticClassLayer）

**文件位置**：`elevation_mapping_cupy/script/elevation_mapping_cupy/plugins/semantic_class_layer.py`

### 设计思路

语义分割网络为每格输出多个类别的概率。本插件只取置信度最高的类别（argmax），**置信度本身不再参与后续计算**，获胜类别决定该格的语义。

对获胜类别，从配置中查表得到两个标量属性：

- **SemCost**：通行代价（0~1），越高越难通行。反映该语义类别本身固有的通行难度，与地形几何无关。例：road/dirt=0.1（易），grass=0.2，bush/water/stone=0.5，障碍物类=1.0

- **SemSensitivity**：几何属性对该类别可通行性的影响程度（0~1）。越高表示坡度、台阶、粗糙度等几何特征对该类别越重要。例：road/dirt=1.0（几何完全有效），grass=0.5，bush/water/stone=0.2，障碍物类=0.0（几何无意义）

两层输出可在后续可通行性合成插件中结合几何分量使用。

### 输出层

每次调用返回一层，通过 `output_mode` 控制输出哪一层：

| `output_mode` | 输出层内容 | 值域 |
|---|---|---|
| `"cost"` | 每格获胜类别的 SemCost | [0, 1] |
| `"sensitivity"` | 每格获胜类别的 SemSensitivity | [0, 1] |

由于插件管理器每个条目只返回一层，需在 `plugin_config.yaml` 中**写两个条目**（指定不同 `output_mode`）以同时生成两个层，两次 argmax 计算开销可忽略。

### 边界情况处理

| 场景 | 行为 |
|---|---|
| 某类别置信度最高 | argmax 获胜，查表输出对应值 |
| 获胜类别不在配置列表中（如 `sem_building` 未配置） | 所有已配置层在该格概率均为 0，退回 `default_cost` / `default_sensitivity` |
| 格子尚未被语义网络覆盖（全零） | 同上，退回默认值 |

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `layers` | list[str] | `[]` | 语义层名称列表，顺序须与 SemCost/SemSensitivity 一一对应 |
| `SemCost` | list[float] | `[]` | 每个类别的通行代价，长度须与 `layers` 一致 |
| `SemSensitivity` | list[float] | `[]` | 每个类别的几何敏感度，长度须与 `layers` 一致 |
| `output_mode` | str | `"cost"` | `"cost"` 或 `"sensitivity"` |
| `default_cost` | float | `0.5` | 未观测格或获胜类别不在列表时的 SemCost 填充值 |
| `default_sensitivity` | float | `1.0` | 未观测格或获胜类别不在列表时的 SemSensitivity 填充值 |


## 层值约定：越大代表通行代价越低还是越高？

本项目所有**可通行性类**层统一采用以下约定：

> **值越大 = 越可通行 = 通行代价越低**

| 层名 | 来源 | 值域 | 约定验证 |
|---|---|---|---|
| `traversability`（内置） | 神经网络 `exp(-out)` | (0, 1] | `safe_thresh: 0.7`——低于此值为不安全格；`traversability_inlier: 0.9`——高于此值才用于漂移补偿 |
| `traversability_slope` | SlopeFilter | [0, 1] | 平地=1.0；坡度=`midpoint` 时=0.5；远超 `midpoint` 时→0 |
| `traversability_roughness` | RoughnessFilter | [0, 1] | 光滑=1.0；粗糙度=`midpoint` 时=0.5；远超时→0 |
| `traversability_step` | StepFilter | [0, 1] | 无台阶=1.0，台阶高=`critical_value` 时=0.0 |
| `traversability_heuristic` | HeuristicTraversability | [0, 1] | 综合几何+语义后两次均值平滑，1=完全可通行 |
| `similarity` | 语义网络输出 | [0, 1] | `similarity 越高越安全`（见 cvar_layer.py 注释） |
| `similarity_cvar` | CvarLayer | (~−∞, 1] | `generate_costmap.py`：≥0.9→cost=0（自由），<0.3→cost=100（障碍） |

### 例外：`sem_cost` 与 `sem_sensitivity`

`SemanticClassLayer` 输出的两个层**不是可通行性分数**，约定与上表相反：

| 层名 | 约定 | 说明 |
|---|---|---|
| `sem_cost` | **越大 = 越难通行** | 语义通行代价，road=0.1（易），building=0.9（难） |
| `sem_sensitivity` | 权重，非可通行性分数 | 控制几何特征对该类别的影响程度，不直接参与大小比较 |

**使用注意**：将 `sem_cost` 与其他可通行性分量合成时，需先取反：

```python
traversability_semantic = 1.0 - sem_cost   # 转换为"越大越可通行"约定
```

在 `plugin_config.yaml` 中可利用 `MaxLayerFilter` 的 `reverse` 参数完成此转换：

```yaml
traversability_semantic:
  type: max_layer_filter
  layer_name: "traversability_semantic"
  extra_params:
    layers:     ['sem_cost']
    reverse:    [True]          # 1 - sem_cost → 越大越可通行
    min_or_max: "max"
```

---

## 可通行性融合插件（HeuristicTraversability）

**文件位置**：`elevation_mapping_cupy/script/elevation_mapping_cupy/plugins/heuristic_traversability.py`

### 公式

```
geo       = u_rough × traversability_roughness + v_slope × traversability_slope
geo_gated = 1 − sem_sensitivity × (1 − geo)              # 几何门，按信任度加权
trav      = clip(geo_gated × (1 − sem_cost), 0, 1)       # 几何门 AND 语义门
HeuristicTraversability = uniform_filter(trav, 3) ×2     # 两次 3×3 均值平滑，消除语义投影棋盘格噪声
```

- 结果 `cp.clip` 到 [0, 1] 防止浮点溢出
- 输出连续做两次 `size=3` 均值滤波，平滑掉语义投影的棋盘格颗粒
- 任一缺失层都按"最乐观/中性"值替代（见 [缺失层的 Fallback 策略](#缺失层的-fallback-策略)）

### 设计思路

`sem_sensitivity` 表示**对几何读数的信任度**，不是几何贡献的乘数惩罚：

- `sens=1.0`（道路）：`geo_gated = geo`，几何完全决定
- `sens=0.5`（草地）：`geo_gated = 0.5×geo + 0.5`，半信半疑——草地几何完美时不扣分（=1），几何差时只扣半分
- `sens=0.0`（行人）：`geo_gated = 1.0`，无视几何，全凭 `sem_cost` 决定

`sem_cost` 作为乘法门（AND 逻辑）：

- `cost=0.0`：完全放行，`trav = geo_gated`
- `cost=1.0`：硬障碍，`trav = 0`（任何几何都救不回来）

这样两个语义项各司其职：`sensitivity` 调节"对几何的信任度"，`cost` 调节"该类别本身的可通行性"。

### 各项含义

`traversability_roughness` 由 **RoughnessFilter** 输出（sigmoid 响应，0.6 m × 0.6 m 足印尺度）

`traversability_slope` 由 **SlopeFilter** 输出（车体足印拟合平面法向量）

`sem_sensitivity` / `sem_cost` 由 **SemanticClassLayer** 输出（按类别查表）

### 归一化约束

`u_rough + v_slope = 1.0`（内层权重）。违反时插件启动会警告，`cp.clip` 兜底。

外层不再有 `alpha_sen`/`beta_cost`：由 `(1−sens×(1−geo)) × (1−cost)` 自然保证输出 ∈ [0, 1]，无需归一化。

### 典型结果（默认权重 u=0.4, v=0.6）

| 场景 | roughness | slope | sensitivity | cost | geo_gated | 输出 |
|---|---|---|---|---|---|---|
| 道路+平地 | 1.0 | 1.0 | 1.0 | 0.1 | 1.00 | **0.90** |
| 草地+平地 | 1.0 | 1.0 | 0.5 | 0.2 | 1.00 | **0.80** |
| 草地+陡坡 | 0.2 | 0.2 | 0.5 | 0.2 | 0.60 | **0.48** |
| 道路+陡坡 | 0.2 | 0.2 | 1.0 | 0.1 | 0.20 | **0.18** |
| 行人      | 任意 | 任意 | 0.0 | 1.0 | 1.00 | **0.00** |
| 灌木+平地 | 1.0 | 1.0 | 0.2 | 0.5 | 1.00 | **0.50** |

### 缺失层的 Fallback 策略

| 层缺失 | 处理方式 | 原因 |
|---|---|---|
| `traversability_roughness` / `traversability_slope` | 替换为全 1（最乐观） | 避免几何层缺失时对所有格子施加不公平惩罚 |
| `sem_sensitivity` | 替换为全 1 | 等效于"完全信任几何" |
| `sem_cost` | 替换为全 0 | 等效于"无语义惩罚" |

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `roughness_layer` | str | `"traversability_roughness"` | 粗糙度可通行性层名 |
| `slope_layer` | str | `"traversability_slope"` | 坡度可通行性层名 |
| `sem_sensitivity_layer` | str | `"sem_sensitivity"` | 语义几何信任度层名 |
| `sem_cost_layer` | str | `"sem_cost"` | 语义通行代价层名 |
| `u_rough` | float | `0.4` | 粗糙度内层权重 |
| `v_slope` | float | `0.6` | 坡度内层权重（坡度对轮式/履带机器人更关键） |

### 上游依赖与执行顺序

`plugin_config.yaml` 中本插件必须排在以下所有插件**之后**：

```
min_filter / inpainting / smooth_filter   (高程预处理，提供稠密无 NaN 高程)
        ├──► slope_filter                 (input_layer: smooth)
        └──► roughness_filter             (input_layer: smooth)
semantic_class_layer (output_mode: cost)        ──► sem_cost
semantic_class_layer (output_mode: sensitivity) ──► sem_sensitivity
                                ↓
                   heuristic_traversability
```

### plugin_config.yaml 配置示例

```yaml
heuristic_traversability:
  type: heuristic_traversability
  enable: True
  fill_nan: False
  is_height_layer: False
  layer_name: "traversability_heuristic"
  extra_params:
    roughness_layer:       "traversability_roughness"
    slope_layer:           "traversability_slope"
    sem_sensitivity_layer: "sem_sensitivity"
    sem_cost_layer:        "sem_cost"
    u_rough:   0.4               # 粗糙度内层权重（与 v_slope 之和应为 1）
    v_slope:   0.6               # 坡度内层权重
```

---

## SmoothFilter 增强（可配置核大小与迭代次数）

**文件位置**：`elevation_mapping_cupy/script/elevation_mapping_cupy/plugins/smooth_filter.py`

`SmoothFilter` 在原实现（写死两次 `size=3` `uniform_filter`）基础上，将核大小与迭代次数提取为可配置参数，便于按地图分辨率与噪声特征调参。等效平滑宽度 = `num_passes × (size − 1) + 1` 格。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `input_layer_name` | str | `"elevation"` | 输入高程层名（**推荐 `inpaint`**，避免 NaN 扩散污染） |
| `size` | int | `3` | 每次 `uniform_filter` 的核大小（格数，建议奇数） |
| `num_passes` | int | `2` | 迭代次数 |

**重要约束**：`uniform_filter` 不处理 NaN，会向邻域扩散。若 `input_layer_name` 选用包含未填充 NaN 的层（如 `min_filter` 输出），输出会被 NaN 污染。Bunker pipeline 中由 `inpainting` 喂入 `smooth_filter`。

---

## Bunker Heuristic Pipeline 集成调试记录（2026-05-13）

将几何 + 语义融合的可通行性流水线接入 bunker（`heuristic:=true` 模式）过程中发现并修复了多个底层问题，按时间顺序记录如下，便于回溯和复用经验。

### 启动流程：`bit-bunker.launch heuristic:=true`

依赖话题：

| 来源 | 话题 | 用途 |
|---|---|---|
| FAST-LIO-SAM | `/accumulated_map_points` (`camera_init` frame) | LiDAR 点云输入 |
| 语义分割节点 | `/hikrobot_camera_L/semantic_image` (8UC1 索引图) | 单通道语义类别索引 |
| 语义分割节点 | `/hikrobot_camera_L/semantic_info` (ChannelInfo) | 动态通道名列表 |
| calibrator | `/hikrobot_camera_L/camera_info_raw` (1536×2048, frame_id=cam_left) | **新增**：原始分辨率 camera_info |
| TF | `cam_left ← body ← odom ← map` | 由 fast_lio_sam launch 静态 TF 提供 |

---

### 调试发现的问题与修复

#### 问题 1：YAML 文件含中文注释导致启动崩溃

**现象**：节点启动后立刻抛 `UnicodeDecodeError: 'ascii' codec can't decode byte 0xe6`。

**根因**：`plugin_manager.py` 用 `open(file_path, "r")` 读 YAML，未指定编码。系统 locale 不是 UTF-8 时，含中文注释的 YAML 直接报错。

**修复**：`elevation_mapping_cupy/script/elevation_mapping_cupy/plugins/plugin_manager.py` 第 139 行 `open(..., "r")` → `open(..., "r", encoding="utf-8")`。

---

#### 问题 2：`semantic_filter` 的正则匹配到了新插件输出层

**现象**：`heuristic:=true` 启动后，`max_categories` 层在 RViz 中显示几乎全黑（packed RGB ≈ 0），而非预期的颜色编码。

**根因**：`semantic_filter` 用 `classes: ['^sem_.*$']` 匹配所有 `sem_` 开头的层。新加的 `sem_cost`（默认 0.5）和 `sem_sensitivity`（默认 1.0）也被误匹配为语义类别。此时 num_classes = 2，`step = 255/2 = 127`，`sem_sensitivity` 默认值 1.0 永远胜出 → `color_encoding[127]` = RGB(0, 32, 0) ≈ 黑色。

**修复**：`bit-bunker_plugin_heruistic.yaml` 中将 `classes: ['^sem_.*$']` 改为 `classes: ['^sem_(?!cost|sensitivity).*$']`（负向预查，排除两个新插件层）。

---

#### 问题 3：`slope_filter` 输入用 `smooth` 导致全 NaN

**现象**：`traversability_slope` 层数据全为 NaN，但 `traversability_roughness` 正常。

**根因**：`smooth_filter` 用 `cupyx.scipy.ndimage.uniform_filter`，**不处理 NaN，会向邻居扩散**。原本 smooth 的输入是 `min_filter`，其输出仍包含未填充的 NaN 区域，传给 `uniform_filter` 后整张图被污染为 NaN。`slope_filter` 用 `cp.gradient(smooth)` → 全 NaN。

**修复**：

- `slope_filter` 的 `input_layer` 从 `smooth` 改为 `inpaint`（OpenCV inpainting 输出，保证无 NaN）
- `smooth_filter` 的 `input_layer_name` 从 `min_filter` 改为 `inpaint`（让 smooth 层也有完整覆盖，仅作可视化）

---

#### 问题 4：相机话题与 frame_id 双重不匹配

**现象**：所有语义通道（`sem_*`）值为 0，`sem_cost` 全 0.5、`sem_sensitivity` 全 1.0（都是 default 值）。日志中相机 image 看似订阅成功，但融合从未发生。

**根因（两层）**：

**4a. 话题名陈旧**：sensor 配置使用 `/hikrobot_camera/camera_left/camera_info`，而当前系统发布的是 `/hikrobot_camera_L/image_rect/camera_info`。

**4b. frame_id 不在 TF 树**：原 `camera_info` 的 `frame_id: "hikrobot_camera"`，但 TF 树里没有这个 frame（仅有 `cam_left`、`camera_left`）。即使话题改对，elevation_mapping 也无法把 `hikrobot_camera → map` 投影到地图。

**4c. 已有的 `image_rect/camera_info`（frame_id=cam_left）分辨率是 960×720（裁剪+下采样后），但 `semantic_image` 是 1536×2048（原始分辨率）**。直接使用会让相机内参与图像分辨率不匹配，投影误差约 2 倍。

**修复**（多文件协同）：

- 在 `calibrator` 项目（`camera_rect_ros_mono.cpp`）中新增 `raw_info_pub_` 和 `prepareRawCameraInfoMsg()`，发布**原始分辨率的 camera_info**（1536×2048，frame_id=cam_left，含原始畸变系数 D），话题为 `/hikrobot_camera_L/camera_info_raw`。
- `bit-bunker_sensor_heruistic.yaml`（以及 `bit-bunker_sensor.yaml`）的所有 `camera_info_topic_name` 改为新话题 `/hikrobot_camera_L/camera_info_raw`。
- `rgb_camera.topic_name` 从 `/hikrobot_camera/camera_left/image_raw` 改为 `/hikrobot_camera_L/image_raw`。

---

#### 问题 5：`plumb_bob` 畸变模型被错误清零

**现象**：上一步修复后，语义可以投影了，但仍有大量栅格无语义数据。

**根因**：`elevation_mapping.py` 中处理 `distortion_model` 的代码：

```python
elif distortion_model == "plumb_bob":
    # Not implemented yet.
    D *= 0   # ← BUG
```

`plumb_bob` 与 `radtan` 是**同一个数学模型**（OpenCV 5 参畸变，`D = [k1, k2, p1, p2, k3]`），但代码当作未实现处理，把畸变系数全清零。新发布的 `camera_info_raw` 的 `distortion_model` 字段正是 `plumb_bob`，于是 kernel 把含畸变的原始图当作矫正图处理，边缘像素投影错位。

**修复**：把 `plumb_bob` 分支改为 `pass`（与 `radtan` 等价处理）。

---

#### 问题 6（**根因**）：原始 `elevation` 层稀疏导致投影丢失

**现象**：即使上述都修复后，`max_categories` 仍然呈"棋盘式"——相邻栅格交替"有正确语义"和"默认 sem_6s5f2w"。提高 `tolerance_z_collision` (0.10 → 0.50) 无效；将 kernel 的 `is_valid` 检查从严格 `!= 1` 改为 `< 0.5` 再改为 `isnan` 都只有轻微改善。

**根因（兜兜转转最终定位）**：

原始 `elevation` 层（`elevation_map[0]`）本身就是**稀疏的**：

- LiDAR 输入是 `/accumulated_map_points`，被 FAST-LIO-SAM 用 **0.3m voxel 下采样**（性能优化，README 中有历史记录）
- elevation map 分辨率是 **0.1m**
- 每个 0.3m voxel 覆盖 ~9 个 elevation 格子，但只有 1 个点 → **~89% 的栅格 elevation 是 NaN**
- 旋转 LiDAR 本身在车辆周围地面就只有几根线束，单帧地面也是条纹状

kernel 用 `map[cell_idx]`（即原始 elevation）算 3D 位置 `p3`，NaN 高程的栅格无法投影 → 棋盘式缺失。

**修复**（系统性方案）：

1. **重新启用 `min_filter`** 插件（之前因"未被任何 publisher 使用"禁用过），用作快速 GPU 填充：`dilation_size=1, iteration_n=10` → 最大向外扩张 1m，足够补 LiDAR 条纹间隙
2. **修改 image kernel** `custom_image_kernels.py::image_to_map_correspondence_kernel`：
   - 新增输入参数 `raw U elevation_filled`
   - 投影计算 `float p3 = elevation_filled[i] + center[2]`
   - Bresenham 起点 `float z0 = elevation_filled[i]`
   - NaN 检查 `if (isnan(elevation_filled[i])) return`
   - 中间格的占用检查仍用原始 `map[idx]`（因为只有真正观测到的格子才会作为遮挡来源）
3. **修改 Python 端** `elevation_mapping.py`：图像到达时先调用 `plugin_manager.update_with_name("min_filter", ...)` 刷新填充层，再传入 kernel
4. **回退兼容**：如果 min_filter 插件未启用，自动回退到原始 `self.elevation_map[0]`

---

### 配置文件最终改动

| 文件 | 改动 |
|---|---|
| `config/setups/bunker/bunker_parameters.yaml` | `cleanup_step: 0.1 → 0.02`（更稳定，validity 衰减更慢）；`max_height_range: 2.0 → 1.0`（Livox 安装高度约 1m，将接受上限控制在地面以上 ~2m，过滤低垂树枝） |
| `config/setups/bunker/bit-bunker_plugin.yaml` | `smooth_filter` 新增 `size`/`num_passes` 配置 |
| `config/setups/bunker/bit-bunker_sensor.yaml` | LiDAR 改 `/accumulated_map_points`；所有相机 `camera_info_topic_name` 改 `/hikrobot_camera_L/camera_info_raw`；`rgb_camera.topic_name` 改 `/hikrobot_camera_L/image_raw` |
| `rviz/bunker.rviz` | RGB image topic、点云 topic 同步更新；默认 GridMap 显示层改为 `traversability_heuristic` |
| `config/setups/bunker/bit-bunker_sensor_heruistic.yaml`（新建） | LiDAR 改 `/accumulated_map_points`；所有相机 `camera_info_topic_name` 改 `/hikrobot_camera_L/camera_info_raw`；`rgb_camera.topic_name` 改 `/hikrobot_camera_L/image_raw` |
| `config/setups/bunker/bit-bunker_plugin_heruistic.yaml`（新建） | `min_filter` 启用（iter=10）；`smooth_filter` 移到 `inpainting` 之后并改用 `inpaint` 输入；`slope_filter`/`roughness_filter` 用 `inpaint`；`semantic_filter` 正则排除 `sem_cost`/`sem_sensitivity` |
| `launch/bit-bunker.launch` | 新增 `heuristic` 参数切换两套 sensor 配置；generate_costmap 的 `similarity_layer` 在 heuristic 模式下改为 `traversability_heuristic`，阈值调整 |

### 代码改动

| 文件 | 改动 |
|---|---|
| `script/elevation_mapping_cupy/plugins/plugin_manager.py` | YAML 读取改 UTF-8 |
| `script/elevation_mapping_cupy/elevation_mapping.py` | `plumb_bob` 不再清零 D；图像融合前刷新 min_filter；调用 kernel 时多传一个 `elevation_filled` |
| `script/elevation_mapping_cupy/kernels/custom_image_kernels.py` | kernel 增 `elevation_filled` 输入；投影/Bresenham 起点/NaN 检查均改用之；`tolerance_z_collision` 默认值 0.10 → 0.50 |
| `script/elevation_mapping_cupy/plugins/semantic_filter.py` | （此次未改）维持 packed RGB 颜色编码 |

calibrator 项目（`/home/whd/catkin_slam/src/calibrator/`）：

| 文件 | 改动 |
|---|---|
| `src/camera_rect_ros_mono.cpp` | 新增 `raw_info_pub_` 与 `prepareRawCameraInfoMsg()`，每帧同时发布原始分辨率 camera_info |
| `launch/camera_rect_ros_mono.launch` | 新增 `raw_info_topic` 参数（默认 `/hikrobot_camera_L/camera_info_raw`） |

### 调参与运行经验

- LiDAR 输入选 `/accumulated_map_points`（保留累积建图能力），voxel 0.3m 的稀疏性靠 min_filter 在 GPU 上快速填补，比切回 `/cloud_registered`（条纹状）更稳定
- `min_filter` 用最小值填充偏保守（不会把高的"想象成"低的），适合避障；视觉效果想更平滑可改用 `inpaint`，但 OpenCV inpainting 需要 GPU↔CPU 传输，~30ms/帧，比 GPU min_filter（<1ms）慢一个数量级
- 调试棋盘问题花了较多时间，关键经验：**当观察到规则空间模式而非时间跳变时，应优先检查数据稀疏性，而非融合算法或反应速度**
