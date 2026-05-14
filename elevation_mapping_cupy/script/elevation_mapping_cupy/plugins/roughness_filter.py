#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
"""
RoughnessFilter —— 基于多尺度高程差的粗糙度可通行性插件（sigmoid 响应）

参考: traversability_estimation/traversability_estimation_filters/src/RoughnessFilter.cpp
      (ETH Zurich, Autonomous Systems Lab)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原理
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
粗糙度定义为不同平滑尺度之间的高程差（多尺度分析）：

  smooth_low  = circular_mean(elevation, radius=low_radius)   [小尺度，保留局部起伏]
  smooth_high = circular_mean(elevation, radius=high_radius)  [大尺度，代表整体趋势]
  roughness   = |smooth_low - smooth_high|

可通行性采用 sigmoid 非线性响应：

  traversability = 1 / (1 + exp(steepness × (roughness − midpoint)))

相比原线性公式（1 - roughness/critical_value），sigmoid 在 midpoint 以下
响应极弱（平坦草地几乎不扣分），在 midpoint 以上响应快速增强，
避免对微小高度差（LiDAR 噪声、草地微起伏）过度惩罚。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
参数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  input_layer : 输入高程层（建议 smooth，已去除测量噪声）
  resolution  : 地图分辨率 (m/cell)
  low_radius  : 小尺度均值滤波半径 (m)
  high_radius : 大尺度均值滤波半径 (m)，建议对应车体接地长度
  midpoint    : roughness 达到此值时 traversability = 0.5 (m)
  steepness   : sigmoid 过渡陡峭程度 (1/m)，越大则过渡越锐利
"""
import cupy as cp
import numpy as np
import cupyx.scipy.ndimage as ndimage
from typing import List

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class RoughnessFilter(PluginBase):
    def __init__(
        self,
        cell_n: int = 100,
        input_layer: str = "elevation_inpainted",
        resolution: float = 0.1,
        low_radius: float = 0.1,
        high_radius: float = 0.56,
        midpoint: float = 0.12,
        steepness: float = 30.0,
        **kwargs,
    ):
        super().__init__()
        self.input_layer = input_layer
        self.midpoint = float(midpoint)
        self.steepness = float(steepness)

        # Pre-build circular kernels (mirrors C++ CircleIterator behaviour)
        self._kernel_low = self._make_circular_kernel(low_radius, resolution)
        self._kernel_high = self._make_circular_kernel(high_radius, resolution)

    @staticmethod
    def _make_circular_kernel(radius: float, resolution: float) -> cp.ndarray:
        """Build a normalised circular mean kernel of given physical radius."""
        r = max(1, int(round(radius / resolution)))
        y, x = np.mgrid[-r : r + 1, -r : r + 1]
        mask = (x ** 2 + y ** 2) <= r ** 2
        k = mask.astype(np.float32)
        k /= k.sum()
        return cp.asarray(k)

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

        h = h.astype(cp.float32)

        # Two-scale circular mean filtering
        smooth_low = ndimage.convolve(h, self._kernel_low)
        smooth_high = ndimage.convolve(h, self._kernel_high)

        roughness = cp.abs(smooth_low - smooth_high)
        # sigmoid: 在 midpoint 以下响应弱，以上快速压低
        traversability = 1.0 / (1.0 + cp.exp(self.steepness * (roughness - self.midpoint)))
        return traversability.astype(cp.float32)
