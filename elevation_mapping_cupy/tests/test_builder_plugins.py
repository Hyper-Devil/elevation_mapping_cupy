"""Behavioral checks for the custom dataset plugin contract on CUDA."""
import cupy as cp
import numpy as np
import pytest
from elevation_mapping_cupy.plugins.step_filter import StepFilter
from elevation_mapping_cupy.plugins.heuristic_traversability import HeuristicTraversability
from elevation_mapping_cupy.plugins.semantic_class_layer import SemanticClassLayer
from elevation_mapping_cupy.plugins.slope_filter import SlopeFilter


def invoke(plugin, h, extra=None, names=None, semantic=None, classes=None):
    e=cp.stack([h,cp.zeros_like(h),cp.ones_like(h)])
    return plugin(e,['elevation','variance','is_valid'],
                  cp.empty((0,*h.shape)) if extra is None else extra,names or [],
                  cp.empty((0,*h.shape)) if semantic is None else semantic,classes or [])


def test_step_detects_obstacle_but_not_flat_ground():
    h=cp.zeros((70,70),dtype=cp.float32)
    plugin=StepFilter(input_layer='elevation',resolution=.1,first_window_radius=.15,
                      second_window_radius=.3,critical_value=.25,n_cell_critical=4)
    assert float(invoke(plugin,h).min())==1.
    h[:,35:]=.5
    out=invoke(plugin,h)
    assert float(out[10:-10,33:37].max())<.05
    assert float(out[10:-10,10:20].min())>.99


@pytest.mark.parametrize('sensitivity,expected',[(1.,0.),(.5,.48),(.2,.672)])
def test_step_gating_preserves_soft_terrain(sensitivity,expected):
    h=cp.zeros((40,40),dtype=cp.float32)
    layers=cp.stack([cp.full_like(h,.2),cp.full_like(h,.2),cp.zeros_like(h),
                     cp.full_like(h,sensitivity),cp.full_like(h,.2)])
    out=invoke(HeuristicTraversability(step_layer='step'),h,layers,
               ['traversability_roughness','traversability_slope','step','sem_sensitivity','sem_cost'])
    np.testing.assert_allclose(cp.asnumpy(out),expected,atol=1e-6)


def test_semantics_export_integer_ids_and_unknown():
    h=cp.zeros((20,20),dtype=cp.float32); classes=['sem_unknown','sem_asphalt','sem_grass']
    s=cp.zeros((3,20,20),dtype=cp.float32);s[1,:,:10]=.8;s[2,:,10:15]=1
    out=invoke(SemanticClassLayer(layers=classes,output_mode='index',default_index=0),h,
               semantic=s,classes=classes)
    expected=np.zeros((20,20));expected[:,:10]=1;expected[:,10:15]=2
    np.testing.assert_array_equal(cp.asnumpy(out),expected)


def test_extra_slope_smoothing_suppresses_narrow_ripple():
    axis=cp.arange(100,dtype=cp.float32)*.1;x,y=cp.meshgrid(axis,axis)
    h=.2*cp.sin(x*2*np.pi/.5)
    raw=invoke(SlopeFilter(input_layer='elevation',resolution=.1,midpoint=.349,num_passes=0),h)
    smooth=invoke(SlopeFilter(input_layer='elevation',resolution=.1,midpoint=.349,size=5,num_passes=2),h)
    assert float(smooth[20:-20,20:-20].mean())>float(raw[20:-20,20:-20].mean())+.05
