#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import string

from .fusion_manager import FusionBase


def exponential_variance_correspondences_to_map_kernel(width, height, alpha):
    exponential_variance_correspondences_to_map_kernel = cp.ElementwiseKernel(
        in_params="raw U sem_map, raw U map_idx_mean, raw U map_idx_var, raw U map_idx_count, raw U image_mono, raw U uv_correspondence, raw B valid_correspondence, raw U image_height, raw U image_width",
        out_params="raw U new_sem_map",
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            """
        ).substitute(width=width, height=height),
        operation=string.Template(
            """
            int cell_idx = get_map_idx(i, 0);
            int mean_idx = get_map_idx(i, map_idx_mean);
            int var_idx = get_map_idx(i, map_idx_var);

            if (valid_correspondence[cell_idx]) {
                int cell_idx_2 = get_map_idx(i, 1);
                int idx = int(uv_correspondence[cell_idx]) + int(uv_correspondence[cell_idx_2]) * image_width;

                int count_idx = get_map_idx(i, map_idx_count);

                U x = image_mono[idx];
                U m_prev = sem_map[mean_idx];
                U v_prev = sem_map[var_idx];
                U n_prev = sem_map[count_idx];

                U n_new = n_prev + 1.0;
                U m_new = x;
                U v_new = 0.0;

                if (n_prev > 0.5) {
                    U delta = x - m_prev;
                    m_new = m_prev + delta / n_new;
                    U delta2 = x - m_new;
                    U m2_prev = (n_prev > 1.5) ? v_prev * (n_prev - 1.0) : 0.0;
                    U m2_new = m2_prev + delta * delta2;
                    v_new = (n_new > 1.5) ? m2_new / (n_new - 1.0) : 0.0;
                }

                new_sem_map[mean_idx] = m_new;
                new_sem_map[var_idx] = v_new > 0 ? v_new : 0;
                new_sem_map[count_idx] = n_new;
            } else {
                int count_idx = get_map_idx(i, map_idx_count);
                new_sem_map[mean_idx] = sem_map[mean_idx];
                new_sem_map[var_idx] = sem_map[var_idx];
                new_sem_map[count_idx] = sem_map[count_idx];
            }
            """
        ).substitute(alpha=alpha),
        name="exponential_variance_correspondences_to_map_kernel",
    )
    return exponential_variance_correspondences_to_map_kernel


class ImageExponentialVariance(FusionBase):
    def __init__(self, params, *args, **kwargs):
        self.name = "image_exponential_variance"
        self.cell_n = params.cell_n
        self.resolution = params.resolution

        self.exponential_variance_correspondences_to_map_kernel = exponential_variance_correspondences_to_map_kernel(
            width=self.cell_n, height=self.cell_n, alpha=0.7,
        )

    def __call__(
        self,
        sem_map_idx,
        image,
        j,
        uv_correspondence,
        valid_correspondence,
        image_height,
        image_width,
        semantic_map,
        new_map,
        sem_map_var_idx,
        sem_map_aux_idxs=None,
    ):
        sem_map_count_idx = sem_map_aux_idxs[0]
        self.exponential_variance_correspondences_to_map_kernel(
            semantic_map,
            cp.uint64(sem_map_idx),
            cp.uint64(sem_map_var_idx),
            cp.uint64(sem_map_count_idx),
            image[j],
            uv_correspondence,
            valid_correspondence,
            image_height,
            image_width,
            new_map,
            size=int(self.cell_n * self.cell_n),
        )
        semantic_map[sem_map_idx] = new_map[sem_map_idx]
        semantic_map[sem_map_var_idx] = new_map[sem_map_var_idx]
        semantic_map[sem_map_count_idx] = new_map[sem_map_count_idx]
