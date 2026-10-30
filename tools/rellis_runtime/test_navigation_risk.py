"""GPU tests for paper EMA, world persistence, CVaR and map serialization."""
from types import SimpleNamespace as NS
import cupy as cp
import numpy as np
import pytest
from elevation_mapping_cupy.parameter import Parameter
from elevation_mapping_cupy.semantic_map import SemanticMap
from elevation_mapping_cupy.plugins.cvar_layer import CvarLayer
from elevation_mapping_cupy.risk_mapping import cvar_coefficient, decode_grid_layer, ema_update, score_to_cost


def test_paper_variance_uses_updated_mean():
    m, v, n = np.zeros(1), np.zeros(1), np.zeros(1)
    m, v, n = ema_update(m, v, n, np.array([.2]), .5)
    np.testing.assert_allclose([m[0], v[0], n[0]], [.2, 0, 1])
    m, v, n = ema_update(m, v, n, np.array([.8]), .5)
    np.testing.assert_allclose([m[0], v[0], n[0]], [.5, .045, 2])


def test_world_ema_survives_window_shift_and_ignores_nan(tmp_path):
    p = Parameter(resolution=1., map_length=4., image_ema_gamma=.5,
                  semantic_global_map_length=20., semantic_global_origin_x=-10., semantic_global_origin_y=-10.,
                  image_channel_fusions={'similarity': 'ema_variance'},image_ema_trace_dir=str(tmp_path))
    p.update()
    mapping = SemanticMap(p)
    mapping.initialize_fusion()
    uv = cp.zeros((2, p.cell_n, p.cell_n), dtype=cp.float32)
    visible = cp.zeros((p.cell_n, p.cell_n), dtype=cp.bool_)
    visible[p.cell_n//2, p.cell_n//2] = True
    def observe(value, center):
        mapping.update_layers_image(cp.full((1, 1, 1), value, dtype=cp.float32), ['similarity'],
                                    uv, visible, cp.float32(1), cp.float32(1), cp.array(center))
    observe(.2, [0, 0, 0])
    observe(.9, [6, 0, 0])  # Original world cell leaves the local terrain window.
    observe(.8, [0, 0, 0])
    at = (slice(None), p.cell_n//2, p.cell_n//2)
    np.testing.assert_allclose(cp.asnumpy(mapping.semantic_map[at]), [.5, .045, 2], atol=1e-6)
    observe(float('nan'), [0, 0, 0])
    np.testing.assert_allclose(cp.asnumpy(mapping.semantic_map[at]), [.5, .045, 2], atol=1e-6)
    mapping.clear()
    observe(.7, [0, 0, 0])
    np.testing.assert_allclose(cp.asnumpy(mapping.semantic_map[at]), [.7, 0, 1], atol=1e-6)
    files=sorted(tmp_path.glob('frame_*.npz'))
    assert len(files)==5
    with np.load(files[2]) as trace:
        np.testing.assert_allclose(trace['observation'],[.8])
        np.testing.assert_allclose(trace['posterior'][:,0],[.5,.045,2],atol=1e-6)
        assert trace['generation']==0
    with np.load(files[3]) as trace:assert len(trace['observation'])==0
    with np.load(files[4]) as trace:assert trace['generation']==1


def test_cvar_masks_unknown_preserves_high_cost_and_alpha_zero():
    values = cp.array([[[.2, .8, .1]], [[.04, .01, 0]], [[2, 2, 0]]], dtype=cp.float32)
    names = ['similarity', 'similarity_var', 'similarity_count']
    def run(alpha):
        plugin = CvarLayer(cvar_alpha=alpha, count_layer_name='similarity_count')
        return cp.asnumpy(plugin(cp.zeros((1, 1, 3)), ['elevation'], cp.empty((0, 1, 3)), [], values, names))
    risk_neutral, risk_averse = run(0), run(.95)
    np.testing.assert_allclose(risk_neutral[0, :2], [.2, .8])
    assert np.isnan(risk_neutral[0, 2])
    assert risk_averse[0, 0] == 0
    assert (risk_averse[0, :2] <= risk_neutral[0, :2]).all()
    np.testing.assert_array_equal(score_to_cost(risk_averse)[0, [0, 2]], [100, -1])
    for invalid in [-1, 1, float('nan')]:
        with pytest.raises(ValueError): cvar_coefficient(invalid)


def test_gridmap_column_major_offset_and_circular_start():
    original = np.arange(6, dtype=np.float32).reshape(2, 3)
    storage = np.roll(original, (1, 2), axis=(0, 1))
    field = NS(layout=NS(dim=[NS(label='column_index', size=2), NS(label='row_index', size=3)], data_offset=1),
               data=[999]+storage.ravel().tolist())
    msg = NS(layers=['x'], data=[field], inner_start_index=1, outer_start_index=2)
    np.testing.assert_array_equal(decode_grid_layer(msg, 'x'), original)


def test_global_export_recomputes_alpha_and_overwrites_free_with_high_risk(monkeypatch):
    import importlib.util
    from pathlib import Path
    import rospy
    path = Path(__file__).resolve().parents[2]/'elevation_mapping_cupy/script/generate_risk_costmap.py'
    spec = importlib.util.spec_from_file_location('navigation_export_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    node = module.RiskCostmap.__new__(module.RiskCostmap)
    node.frame, node.resolution, node.size = 'map', .1, 2
    node.origin_x = node.origin_y = 0
    node.alpha = 0
    node.cost_encoding = 'continuous'
    node.cost_thresholds = (.3,.5,.7,.9)
    node.state = np.zeros((3, 2, 2), dtype=np.float32)
    node.state[:, 0, 0] = [1., .04, 1]
    output = []
    node.pub = NS(publish=output.append)
    settings = {'alpha': 0.}
    monkeypatch.setattr(rospy, 'get_param', lambda *args: settings['alpha'])
    node.publish(rospy.Time(1))
    assert output[-1].data == [0, -1, -1, -1]
    settings['alpha'] = .95
    node.publish(rospy.Time(2))
    assert output[-1].data[0] > 0
    node.state[0, 0, 0] = 0
    node.publish(rospy.Time(3))
    assert output[-1].data == [100, -1, -1, -1]
    # Re-publishing statistics is not an additional observation.
    assert node.state[2, 0, 0] == 1
    node.cost_encoding = 'five_level'
    node.state[:, 0, 0] = [.8, 0., 1]
    node.publish(rospy.Time(4))
    assert output[-1].data == [30, -1, -1, -1]
    node.cost_thresholds = (.2,.35,.5,.65)
    node.publish(rospy.Time(5))
    assert output[-1].data == [0, -1, -1, -1]


def test_five_level_cost_thresholds_and_unknowns():
    values = np.array([-.1, 0., .299, .3, .499, .5, .699, .7, .899, .9, 1., 1.1,
                       np.nan, np.inf, -np.inf], dtype=np.float32)
    np.testing.assert_array_equal(score_to_cost(values, 'five_level'),
                                  [100, 100, 100, 70, 70, 50, 50, 30, 30, 0, 0, 0, -1, -1, -1])
    np.testing.assert_array_equal(score_to_cost(np.array([0., .25, 1., np.nan])), [100, 75, 0, -1])
    with pytest.raises(ValueError): score_to_cost(values, 'unknown_encoding')
    np.testing.assert_array_equal(score_to_cost(np.array([.1,.2,.35,.5,.65,.8,np.nan],dtype=np.float32),
                                  'five_level',(.2,.35,.5,.65)), [100,70,50,30,0,0,-1])
    for invalid in [(.2,.3,.4),(.2,.1,.5,.9),(.2,.2,.5,.9),(.2,.3,.5,np.nan),(-.1,.3,.5,.9)]:
        with pytest.raises(ValueError):score_to_cost(values,'five_level',invalid)
