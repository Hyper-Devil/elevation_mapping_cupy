#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
"""
SemanticClassLayer —— 语义类别属性查表插件

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
设计思路
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
语义分割网络为每格输出多个类别的概率。本插件只取置信度最高的类别
（argmax），置信度本身不再参与后续计算。

对获胜类别，从配置中查表得到两个标量属性：

  SemCost        : 通行代价（0~1），越高越难通行。
                   例：road=0.1（易），grass=0.3，building=0.9（难）

  SemSensitivity : 几何属性对该类别可通行性的影响程度（0~1）。
                   越高表示该类别的几何特征（坡度、台阶等）越重要。
                   例：dirt=1.0（几何完全有效），grass=0.5（几何部分有效）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输出
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
每次调用返回一层，通过 output_mode 控制：
  "cost"        → SemCost 层，值域 [0, 1]
  "sensitivity" → SemSensitivity 层，值域 [0, 1]

需在 plugin_config.yaml 中写两个条目（分别指定不同 output_mode）
以同时生成两个层，两次 argmax 计算开销可忽略。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
未观测格（all-zero probability）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
若某格所有指定语义层概率均为 0（尚未被语义网络覆盖），
则使用 default_cost / default_sensitivity 填充。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
参数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  layers             : 语义层名称列表，顺序与 SemCost/SemSensitivity 对应
  SemCost            : 每个类别的通行代价，长度须与 layers 一致
  SemSensitivity     : 每个类别的几何敏感度，长度须与 layers 一致
  output_mode        : "cost" 或 "sensitivity"
  default_cost       : 未观测格的 SemCost 默认值，默认 0.5
  default_sensitivity: 未观测格的 SemSensitivity 默认值，默认 1.0
"""
import cupy as cp
from typing import List

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class SemanticClassLayer(PluginBase):
    def __init__(
        self,
        cell_n: int = 100,
        layers: list = [],
        SemCost: list = [],
        SemSensitivity: list = [],
        output_mode: str = "cost",
        default_cost: float = 0.5,
        default_sensitivity: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        # Only the list matching output_mode is required; the other may be omitted.
        if output_mode == "cost":
            if len(SemCost) != len(layers):
                raise ValueError(
                    f"SemanticClassLayer(cost): layers({len(layers)}) and SemCost({len(SemCost)}) must have the same length"
                )
        else:
            if len(SemSensitivity) != len(layers):
                raise ValueError(
                    f"SemanticClassLayer(sensitivity): layers({len(layers)}) and SemSensitivity({len(SemSensitivity)}) must have the same length"
                )
        self.layers = layers
        self.sem_cost = cp.asarray(SemCost, dtype=cp.float32) if SemCost else cp.zeros(len(layers), dtype=cp.float32)
        self.sem_sensitivity = cp.asarray(SemSensitivity, dtype=cp.float32) if SemSensitivity else cp.zeros(len(layers), dtype=cp.float32)
        self.output_mode = output_mode
        self.default_cost = float(default_cost)
        self.default_sensitivity = float(default_sensitivity)

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
        # Collect available semantic probability layers
        probs = []
        valid_class_indices = []
        for i, name in enumerate(self.layers):
            layer = self.get_layer_data(
                elevation_map, layer_names,
                plugin_layers, plugin_layer_names,
                semantic_map, semantic_layer_names,
                name,
            )
            if layer is not None:
                probs.append(layer.astype(cp.float32))
                valid_class_indices.append(i)

        default_val = self.default_cost if self.output_mode == "cost" else self.default_sensitivity

        if not probs:
            return cp.full_like(elevation_map[0], default_val, dtype=cp.float32)

        # Stack to [N_classes, H, W], then argmax along class axis
        stacked = cp.stack(probs, axis=0)          # [N, H, W]
        local_idx = cp.argmax(stacked, axis=0)     # [H, W], range [0, N-1]

        # Map local indices back to original class table indices
        # (handles the case where some layers were missing)
        idx_map = cp.asarray(valid_class_indices, dtype=cp.int32)
        class_idx = idx_map[local_idx]             # [H, W], range within full class list

        # Look up the requested attribute for each cell's winning class
        if self.output_mode == "cost":
            result = self.sem_cost[class_idx]
        else:
            result = self.sem_sensitivity[class_idx]

        # Fill unobserved cells (all probabilities == 0) with the default value
        max_prob = cp.max(stacked, axis=0)
        result = cp.where(max_prob < 1e-6, default_val, result)

        return result.astype(cp.float32)
