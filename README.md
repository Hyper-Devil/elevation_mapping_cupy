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