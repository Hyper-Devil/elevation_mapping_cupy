#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import math
from statistics import NormalDist
from typing import List

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class CvarLayer(PluginBase):
    def __init__(
        self,
        cell_n: int = 100,
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
        cvar_map = mean_map + sigma_map * self.cvar_scale
        return cvar_map.astype(cp.float32)
