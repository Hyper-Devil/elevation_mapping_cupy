#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
'''
CvarLayer —— 基于 CVaR 的风险感知可通行性插件

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
背景
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
可通行性分析结果（来自相机语义分割，映射到 GridMap）天然存在不确定性：
同一格子被反复观测时，similarity 服从某分布。
仅用均值 μ 导航会忽略不确定性，使机器人可能冒险进入"均值尚可但方差很大"的区域。
CVaR 通过对方差施加惩罚来量化并控制这一风险。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入层 / 输出层
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  输入 mean_layer     (默认 "similarity")     : 可通行性均值 μ，值域 0~1，越高越安全
  输入 variance_layer (默认 "similarity_var") : 可通行性方差 σ²
  输出 similarity_cvar : 经 CVaR 调整后的保守可通行性分数，供 generate_costmap 转为 OccupancyGrid

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
数学原理
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
假设每格 similarity 服从正态分布 X ~ N(μ, σ²)。

下尾条件期望（最差 (1-α) 情况下的期望值）：

  CVaR_α = E[ X | X ≤ q_α ]
          = μ - σ · φ(z_α) / (1-α)

  其中：
    α          = cvar_alpha，置信水平
    z_α        = Φ⁻¹(α)，标准正态分位数
    φ(z)       = (1/√2π) · exp(-z²/2)，标准正态 PDF
    cvar_scale = φ(z_α) / (1-α)（初始化时预计算，始终为正）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
直觉解释
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CVaR_α = μ - cvar_scale · σ

  - μ 高、σ 小（确定且安全）→ CVaR 接近 μ，cost 低，可通行
  - μ 高、σ 大（均值好但不确定）→ CVaR 被压低，cost 升高，机器人保守绕行
  - μ 低（明确障碍）→ CVaR 更低，cost 最高，强烈规避

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
参数 cvar_alpha 与风险偏好
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  α → 1.0  : cvar_scale 增大，对方差惩罚越重，越保守（风险厌恶）
  α → 0.0  : cvar_scale → 0，退化为纯均值，忽略方差（风险中性）
  典型取值  : 0.9（惩罚最差 10%）/ 0.95（惩罚最差 5%）/ 0.99（惩罚最差 1%）

数值示例（α=0.9，cvar_scale ≈ 1.755）：
  σ=0.05（高确定性）: CVaR ≈ μ - 0.09  （轻微惩罚）
  σ=0.20（低确定性）: CVaR ≈ μ - 0.35  （显著惩罚，cost 明显升高）
'''
import cupy as cp
import math
from statistics import NormalDist
from typing import List

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class CvarLayer(PluginBase):
    def __init__(
        self,
        mean_layer_name: str = "similarity",
        variance_layer_name: str = "similarity_var",
        cvar_alpha: float = 0.9,
        **kwargs,
    ):
        super().__init__()
        self.mean_layer_name = mean_layer_name
        self.variance_layer_name = variance_layer_name

        alpha = float(cvar_alpha)
        alpha = min(max(alpha, 1e-6), 1.0 - 1e-6)
        z = NormalDist().inv_cdf(alpha)
        phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        self.cvar_scale = phi / (1.0 - alpha)

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
        mean_map = self.get_layer_data(
            elevation_map,
            layer_names,
            plugin_layers,
            plugin_layer_names,
            semantic_map,
            semantic_layer_names,
            self.mean_layer_name,
        )
        variance_map = self.get_layer_data(
            elevation_map,
            layer_names,
            plugin_layers,
            plugin_layer_names,
            semantic_map,
            semantic_layer_names,
            self.variance_layer_name,
        )

        if mean_map is None or variance_map is None:
            return cp.zeros_like(elevation_map[0], dtype=cp.float32)

        variance_map = cp.maximum(variance_map, 0.0)
        sigma_map = cp.sqrt(variance_map)
        # 下尾 CVaR：悲观估计，对不确定区域施加惩罚
        # similarity 越高越安全，故取下尾（减去方差项）
        # α 越大 → cvar_scale 越大 → 对方差惩罚越重 → 越保守
        cvar_map = mean_map - sigma_map * self.cvar_scale
        return cvar_map.astype(cp.float32)
