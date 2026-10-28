"""Paper EMA at world cells; retain statistics when the terrain window moves."""
import math
import cupy as cp
from .fusion_manager import FusionBase
from elevation_mapping_cupy.risk_mapping import ema_update


class ImageEMAVariance(FusionBase):
    def __init__(self, params):
        self.name = 'image_ema_variance'
        self.cell_n = params.cell_n
        self.resolution = params.resolution
        self.gamma = float(params.image_ema_gamma)
        if not math.isfinite(self.gamma) or not 0 < self.gamma < 1:
            raise ValueError('image_ema_gamma must be in (0,1)')
        self.size = int(round(params.semantic_global_map_length/self.resolution))
        if self.size <= 0:
            raise ValueError('semantic_global_map_length must be positive')
        self.origin_x = float(params.semantic_global_origin_x)
        self.origin_y = float(params.semantic_global_origin_y)
        self.states = {}
        self.center = None

    def set_map_center(self, center):
        self.center = cp.asarray(center)

    def clear(self):
        self.states.clear()

    def __call__(self, sem_map_idx, image, j, uv_correspondence, valid_correspondence,
                 image_height, image_width, semantic_map, new_map, sem_map_var_idx,
                 sem_map_aux_idxs=None):
        if self.center is None:
            raise ValueError('World-aligned EMA requires the terrain map center')
        key = int(sem_map_idx)
        if key not in self.states:
            self.states[key] = cp.zeros((3, self.size, self.size), dtype=cp.float32)
        state = self.states[key]
        offset = (cp.arange(self.cell_n)-self.cell_n//2)*self.resolution
        gx = cp.floor((self.center[0]+offset-self.origin_x)/self.resolution+1e-4).astype(cp.int64)
        gy = cp.floor((self.center[1]+offset-self.origin_y)/self.resolution+1e-4).astype(cp.int64)
        inside = (gx[:, None] >= 0) & (gx[:, None] < self.size) & (gy[None, :] >= 0) & (gy[None, :] < self.size)
        safe_x, safe_y = cp.clip(gx, 0, self.size-1), cp.clip(gy, 0, self.size-1)
        # Restore all local statistics from world storage, including cells revisited
        # after rolling out. No update occurs without a new valid image sample.
        local = state[:, safe_x[:, None], safe_y[None, :]].copy()
        local[:, ~inside] = 0
        u = uv_correspondence[0].astype(cp.int64)
        v = uv_correspondence[1].astype(cp.int64)
        # Invalid correspondences may contain arbitrary UV; never index them.
        u = cp.clip(u, 0, image.shape[2]-1)
        v = cp.clip(v, 0, image.shape[1]-1)
        sample = image[j, v, u]
        valid = inside & valid_correspondence & cp.isfinite(sample) & (sample >= 0) & (sample <= 1)
        x, y = cp.nonzero(valid)
        m, var, count = ema_update(local[0, x, y], local[1, x, y], local[2, x, y], sample[x, y], self.gamma)
        local[0, x, y], local[1, x, y], local[2, x, y] = m, var, count
        state[:, gx[x], gy[y]] = local[:, x, y]
        for target, value in zip((key, int(sem_map_var_idx), int(sem_map_aux_idxs[0])), local):
            semantic_map[target] = value
            new_map[target] = value
