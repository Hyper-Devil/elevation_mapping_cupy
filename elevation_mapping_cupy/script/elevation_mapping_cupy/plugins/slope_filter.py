#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
"""
SlopeFilter —— 基于履带底盘足印的坡度可通行性插件

参考: traversability_estimation/traversability_estimation_filters/src/SlopeFilter.cpp
      (ETH Zurich, Autonomous Systems Lab)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原理
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
在计算坡度前，先对高程层在车体足印范围内做均值平滑：

  h_fp = uniform_filter(h, size=footprint_cells)

其中 footprint_cells = max(
    round(contact_length / resolution),   # pitch 方向：接地长度
    round(track_separation / resolution), # roll  方向：履带中心间距
)

均值平滑等价于在足印内拟合平面法向量，使坡度估计反映整车
实际经历的前后俯仰（pitch）和左右横滚（roll），而非单格噪声。

对平滑后的高程计算梯度，并用 sigmoid 非线性映射到可通行性：

  n_z = 1 / sqrt(1 + (∂h_fp/∂x)² + (∂h_fp/∂y)²)
  slope = arccos(n_z)   ∈ [0, π/2]
  traversability = 1 / (1 + exp(steepness × (slope − midpoint)))

sigmoid 在 midpoint 以下响应弱（缓坡几乎不扣分），以上快速下降，
比线性公式（1 - slope/critical_value）对缓坡更宽容。

由于地图方向与车体朝向无关，footprint_cells 取 pitch/roll 两方向
的最大值，以方形核保证各向同性。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
参数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  input_layer    : 输入高程层（建议 smooth，无 NaN）
  resolution     : 地图分辨率 (m/cell)，需与 bunker_parameters.yaml 一致
  contact_length : 履带接地长度 (m)，决定 pitch 估计尺度
  track_width    : 单侧履带宽度 (m)，用于计算履带中心间距
  total_width    : 车体总宽 (m)
  midpoint       : 坡度达到此值（弧度）时 traversability = 0.5，默认 40°
  steepness      : sigmoid 过渡陡峭程度 (1/rad)，由 trav(60°)≈0.05 反算：ln(19)/(π/3−midpoint)
"""
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import math
from typing import List

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class SlopeFilter(PluginBase):
    def __init__(
        self,
        cell_n: int = 100,
        input_layer: str = "elevation_inpainted",
        resolution: float = 0.1,
        contact_length: float = 0.56,
        track_width: float = 0.15,
        total_width: float = 0.778,
        midpoint: float = math.radians(40),
        steepness: float = 8.4,
        **kwargs,
    ):
        super().__init__()
        self.input_layer = input_layer
        self.resolution = float(resolution)
        self.midpoint = float(midpoint)
        self.steepness = float(steepness)

        # 履带中心间距 = 车体总宽 - 单侧履带宽（左右各减半个履带宽，合计减一个）
        track_separation = float(total_width) - float(track_width)

        # pitch（接地长度）和 roll（轮距）方向的足印格数，取较大值保证各向同性
        pitch_cells = max(1, round(float(contact_length) / self.resolution))
        roll_cells  = max(1, round(track_separation       / self.resolution))
        self.footprint_cells = max(pitch_cells, roll_cells)

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map: cp.ndarray,
        semantic_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        h = self.get_layer_data(
            elevation_map, layer_names,
            plugin_layers, plugin_layer_names,
            semantic_map, semantic_layer_names,
            self.input_layer,
        )
        if h is None:
            return cp.zeros_like(elevation_map[0], dtype=cp.float32)

        h = h.astype(cp.float64)

        # 在车体足印范围内做均值平滑，使梯度反映整车尺度的倾斜而非单格噪声
        h = ndimage.uniform_filter(h, size=self.footprint_cells)

        # cp.gradient 返回 [∂h/∂row, ∂h/∂col]，单位 m/cell；除以 resolution 转为无量纲
        grad = cp.gradient(h)
        gy = grad[0] / self.resolution
        gx = grad[1] / self.resolution

        # 地表法向量 z 分量
        normal_z = 1.0 / cp.sqrt(1.0 + gx ** 2 + gy ** 2)

        # 坡度角（弧度）
        slope = cp.arccos(cp.clip(normal_z, -1.0, 1.0))

        # sigmoid：midpoint 以下响应弱，以上快速压低
        traversability = 1.0 / (1.0 + cp.exp(self.steepness * (slope - self.midpoint)))
        return traversability.astype(cp.float32)
