#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
"""
HeuristicTraversability —— 几何 × 语义融合可通行性插件

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
公式
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  geo       = u_rough × traversability_roughness + v_slope × traversability_slope
  geo_gated = 1 − sem_sensitivity × (1 − geo)              # 几何门（按信任度加权）
  HeuristicTraversability = geo_gated × (1 − sem_cost)     # 几何门 AND 语义门

  输出前对结果做两次 size=3 均值平滑，消除语义投影棋盘格噪声。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
设计思路
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sem_sensitivity 表示"对几何读数的信任度"，而不是几何贡献的乘数：
  sens=1.0（道路）：geo_gated = geo，几何完全决定
  sens=0.5（草地）：geo_gated = 0.5×geo + 0.5，半信半疑，几何差时只扣半分
  sens=0.0（行人）：geo_gated = 1.0，无视几何，全凭 sem_cost 决定

sem_cost 作为乘法门（AND 逻辑）：
  cost=1.0：trav=0，任何几何都救不回来（硬障碍）
  cost=0.0：trav=geo_gated，完全由几何决定

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
各项含义
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  geo 几何项：
    traversability_roughness   : 粗糙度可通行性，由 RoughnessFilter 输出
    traversability_slope       : 坡度可通行性，由 SlopeFilter 输出
    u_rough / v_slope          : 粗糙度与坡度的内层权重，建议 u_rough + v_slope = 1

  语义项：
    sem_sensitivity            : 几何读数的信任度（0=忽略几何，1=完全依赖）
    sem_cost                   : 该类别的固有通行代价（0=完全允许，1=硬障碍）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
典型结果（u_rough=0.4, v_slope=0.6）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  道路+平地（sens=1.0, cost=0.1, geo=1.0）  → 1.0  × 0.9 = 0.90
  草地+平地（sens=0.5, cost=0.2, geo=1.0）  → 1.0  × 0.8 = 0.80
  草地+陡坡（sens=0.5, cost=0.2, geo=0.2）  → 0.60 × 0.8 = 0.48
  道路+陡坡（sens=1.0, cost=0.1, geo=0.2）  → 0.20 × 0.9 = 0.18
  行人      （sens=0.0, cost=1.0）          → 1.0  × 0.0 = 0.00
  灌木+平地（sens=0.2, cost=0.5, geo=1.0）  → 1.0  × 0.5 = 0.50
  未观测格（mask_unobserved=True）          → NaN

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
上游依赖
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  smooth_filter      → 供 slope_filter 使用（input_layer: smooth）
  inpainting         → 供 roughness_filter 使用（input_layer: inpaint）
  slope_filter       → 输出 traversability_slope
  roughness_filter   → 输出 traversability_roughness
  semantic_class_layer (cost)        → 输出 sem_cost
  semantic_class_layer (sensitivity) → 输出 sem_sensitivity

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
参数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  roughness_layer      : traversability_roughness 层名
  slope_layer          : traversability_slope 层名
  sem_sensitivity_layer: sem_sensitivity 层名
  sem_cost_layer       : sem_cost 层名
  u_rough              : 粗糙度内层权重（建议与 v_slope 之和为 1）
  v_slope              : 坡度内层权重（建议与 u_rough 之和为 1）
"""
import cupy as cp
import cupyx.scipy.ndimage as ndimage
from typing import List

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class HeuristicTraversability(PluginBase):
    def __init__(
        self,
        cell_n: int = 100,
        roughness_layer: str = "traversability_roughness",
        slope_layer: str = "traversability_slope",
        sem_sensitivity_layer: str = "sem_sensitivity",
        sem_cost_layer: str = "sem_cost",
        u_rough: float = 0.4,
        v_slope: float = 0.6,
        **kwargs,
    ):
        super().__init__()
        self.roughness_layer       = roughness_layer
        self.slope_layer           = slope_layer
        self.sem_sensitivity_layer = sem_sensitivity_layer
        self.sem_cost_layer        = sem_cost_layer
        self.u_rough    = float(u_rough)
        self.v_slope    = float(v_slope)

        if abs(self.u_rough + self.v_slope - 1.0) > 1e-3:
            print(
                f"[HeuristicTraversability] Warning: u_rough({u_rough}) + v_slope({v_slope}) "
                f"= {u_rough + v_slope:.3f} ≠ 1.0, output may exceed [0, 1]"
            )

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
        # ── Retrieve upstream layers ──────────────────────────────────────────
        roughness = self.get_layer_data(
            elevation_map, layer_names, plugin_layers, plugin_layer_names,
            semantic_map, semantic_layer_names, self.roughness_layer,
        )
        slope = self.get_layer_data(
            elevation_map, layer_names, plugin_layers, plugin_layer_names,
            semantic_map, semantic_layer_names, self.slope_layer,
        )
        sensitivity = self.get_layer_data(
            elevation_map, layer_names, plugin_layers, plugin_layer_names,
            semantic_map, semantic_layer_names, self.sem_sensitivity_layer,
        )
        cost = self.get_layer_data(
            elevation_map, layer_names, plugin_layers, plugin_layer_names,
            semantic_map, semantic_layer_names, self.sem_cost_layer,
        )

        # ── Fallbacks for missing layers ──────────────────────────────────────
        # If a geometric layer is missing, treat as fully traversable (1.0)
        # so the missing component does not unfairly penalise traversability.
        # If a semantic layer is missing, use neutral values.
        h, w = elevation_map[0].shape
        if roughness   is None: roughness   = cp.ones((h, w),  dtype=cp.float32)
        if slope       is None: slope       = cp.ones((h, w),  dtype=cp.float32)
        if sensitivity is None: sensitivity = cp.ones((h, w),  dtype=cp.float32)
        if cost        is None: cost        = cp.zeros((h, w), dtype=cp.float32)

        # ── Formula ───────────────────────────────────────────────────────────
        # geo:        geometric traversability (linear blend of roughness + slope)
        # geo_gated:  sensitivity-weighted gate. sens=1 → geo, sens=0 → 1 (geometry ignored).
        # result:     geometric gate AND semantic gate (multiplicative)
        geo = (self.u_rough * roughness + self.v_slope * slope).astype(cp.float32)
        sensitivity = sensitivity.astype(cp.float32)
        cost = cost.astype(cp.float32)
        geo_gated = 1.0 - sensitivity * (1.0 - geo)
        result = cp.clip(geo_gated * (1.0 - cost), 0.0, 1.0)

        # ── Output smoothing ──────────────────────────────────────────────────
        result = ndimage.uniform_filter(result, size=3)
        result = ndimage.uniform_filter(result, size=3)

        return result.astype(cp.float32)
