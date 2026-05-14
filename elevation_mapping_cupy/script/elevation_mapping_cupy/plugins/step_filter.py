#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
"""
StepFilter —— 基于台阶高度的可通行性插件

参考: traversability_estimation/traversability_estimation_filters/src/StepFilter.cpp
      (ETH Zurich, Autonomous Systems Lab)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
两次滑窗算法
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
第一次（first_window_radius）：
  对每格在 first_window_radius 圆形窗口内计算：
    step_height = max(elevation) - min(elevation)   [局部高差]

第二次（second_window_radius）：
  对每格在 second_window_radius 窗口内聚合：
    step_max = max(step_height)                      [最大台阶高度]
    n_cells  = 窗口内 step_height > critical_value 的格数
    step     = min(step_max, n_cells / n_cell_critical × step_max)
    traversability = max(1 - step / critical_value, 0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
两次滑窗的物理意义
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  第一次：捕捉局部高差（台阶、石块等障碍）
  第二次：在更大范围内聚合，n_cell_critical 控制敏感度——
          只有障碍格数达到阈值时才判为不可通行，
          避免单格噪声导致大范围误报。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
参数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  input_layer          : 输入高程层（建议 inpaint 后）
  resolution           : 地图分辨率 (m/cell)
  first_window_radius  : 第一次滑窗半径 (m)，默认 0.08
  second_window_radius : 第二次滑窗半径 (m)，默认 0.08
  critical_value       : 台阶高度阈值 (m)，超出则可通行性为 0，默认 0.2
  n_cell_critical      : 第二次窗口内超过 critical_value 的格数上限，默认 4
"""
import cupy as cp
import cupyx.scipy.ndimage as ndimage
from typing import List

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class StepFilter(PluginBase):
    def __init__(
        self,
        cell_n: int = 100,
        input_layer: str = "elevation_inpainted",
        resolution: float = 0.04,
        first_window_radius: float = 0.08,
        second_window_radius: float = 0.08,
        critical_value: float = 0.2,
        n_cell_critical: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.input_layer = input_layer
        self.resolution = float(resolution)
        self.critical_value = float(critical_value)
        self.n_cell_critical = max(1, int(n_cell_critical))

        self._size1 = self._radius_to_size(first_window_radius, resolution)
        self._size2 = self._radius_to_size(second_window_radius, resolution)

    @staticmethod
    def _radius_to_size(radius: float, resolution: float) -> int:
        """Convert a physical radius (m) to an odd square-kernel side length."""
        return 2 * max(1, int(round(radius / resolution))) + 1

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

        # Pass 1: step_height = local max - local min within first window
        max_h = ndimage.maximum_filter(h, size=self._size1)
        min_h = ndimage.minimum_filter(h, size=self._size1)
        step_height = max_h - min_h

        # Pass 2: aggregate within second window
        step_max = ndimage.maximum_filter(step_height, size=self._size2)

        # uniform_filter returns the local mean; multiply by kernel area to
        # approximate the count of cells where step_height > critical_value.
        is_bad = (step_height > self.critical_value).astype(cp.float32)
        n_cells = ndimage.uniform_filter(is_bad, size=self._size2) * (self._size2 ** 2)

        # Attenuate step_max by the fraction of bad cells vs. tolerance
        step = cp.minimum(step_max, (n_cells / self.n_cell_critical) * step_max)

        traversability = cp.clip(1.0 - step / self.critical_value, 0.0, 1.0)
        return traversability.astype(cp.float32)
