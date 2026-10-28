"""ROS-independent risk mathematics and GridMap layout conversion."""
import math
from statistics import NormalDist
import numpy as np


def cvar_coefficient(alpha):
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError('cvar_alpha must be finite and in [0,1)')
    if alpha == 0:
        return 0.0
    z = NormalDist().inv_cdf(alpha)
    return math.exp(-0.5*z*z)/math.sqrt(2*math.pi)/(1-alpha)


def ema_update(mean, variance, count, observation, gamma):
    """Paper recurrence: residual against the UPDATED mean, not sample variance.

    Caller supplies only valid observations. Works with NumPy and CuPy arrays.
    """
    if not math.isfinite(gamma) or not 0 < gamma < 1:
        raise ValueError('ema_gamma must be in (0,1)')
    first = count == 0
    updated_mean = (1-gamma)*mean + gamma*observation
    updated_mean = updated_mean*(~first) + observation*first
    updated_variance = ((1-gamma)*variance + gamma*(observation-updated_mean)**2)*(~first)
    return updated_mean, updated_variance, count+1


def decode_grid_layer(msg, layer):
    """Return [y-index,x-index], both descending in world coordinates.

    GridMap uses column-major Eigen matrices [x-index,y-index]. Account for
    circular-buffer starts and MultiArray data_offset before assigning positions.
    """
    field = msg.data[msg.layers.index(layer)]
    dims = field.layout.dim
    if len(dims) != 2 or not dims[0].label.startswith('column') or not dims[1].label.startswith('row'):
        raise ValueError('Expected standard column-major GridMap layout')
    ny, nx = dims[0].size, dims[1].size
    offset = field.layout.data_offset
    values = np.asarray(field.data, dtype=np.float32)[offset:offset+nx*ny]
    if len(values) != nx*ny:
        raise ValueError('Incomplete GridMap layer')
    values = values.reshape(ny, nx)
    return np.roll(values, (-msg.inner_start_index, -msg.outer_start_index), axis=(0, 1))


def score_to_cost(score):
    """Finite zero/negative scores are high risk, never unknown."""
    result = np.full(score.shape, -1, dtype=np.int8)
    valid = np.isfinite(score)
    result[valid] = np.rint(100*(1-np.clip(score[valid], 0, 1))).astype(np.int8)
    return result
