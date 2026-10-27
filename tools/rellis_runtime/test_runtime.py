"""Runtime acceptance checks; execute in the dedicated container environment."""
import json
import os
from pathlib import Path
import numpy as np
import cupy as cp
import torch
import pytest
from cupyx.scipy.ndimage import uniform_filter
from elevation_mapping_cupy.parameter import Parameter
from elevation_mapping_cupy.elevation_mapping import ElevationMap
from elevation_mapping_cupy.plugins.slope_filter import SlopeFilter
from elevation_mapping_cupy.plugins.roughness_filter import RoughnessFilter
from elevation_mapping_cupy.plugins.semantic_class_layer import SemanticClassLayer
from elevation_mapping_cupy.plugins.heuristic_traversability import HeuristicTraversability

PKG=Path(__file__).resolve().parents[2]/'elevation_mapping_cupy'
VAL=Path(os.environ['EMCP_VALIDATION_OUTPUT'])
FIXTURES=Path(os.environ['EMCP_FIXTURES'])

def test_blackwell_cuda_and_torch():
    x=cp.arange(4096,dtype=cp.float32).reshape(64,64)
    np.testing.assert_allclose(cp.asnumpy(x.sum()),np.arange(4096,dtype=np.float32).sum())
    assert bool(cp.isfinite(uniform_filter(x,size=5)).all())
    conv=torch.nn.Conv2d(1,2,3,bias=False).cuda()
    conv.weight.data.fill_(1/9)
    result=conv(torch.as_tensor(cp.ones((1,1,32,32),dtype=cp.float32),device='cuda'))
    torch.cuda.synchronize()
    np.testing.assert_allclose(cp.asnumpy(cp.asarray(result.detach())),1,atol=1e-6)
    info=dict(cupy=cp.__version__,torch=torch.__version__,device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),torch_cuda=torch.version.cuda)
    (VAL/'gpu_runtime.json').write_text(json.dumps(info,indent=2))


def plugin_call(plugin,height,semantic=None,names=None,upstream=None,upstream_names=None):
    elevation=cp.stack([height,cp.zeros_like(height),cp.ones_like(height)])
    return plugin(elevation,['elevation','variance','is_valid'],cp.empty((0,*height.shape)) if upstream is None else upstream,upstream_names or [],cp.empty((0,*height.shape)) if semantic is None else semantic,names or [])


def test_geometry_responds_to_slope_and_roughness():
    axis=cp.arange(100,dtype=cp.float32)*0.1
    x,y=cp.meshgrid(axis,axis)
    flat=cp.zeros_like(x)
    slope=SlopeFilter(input_layer='elevation',resolution=0.1,midpoint=np.deg2rad(20),steepness=8.4)
    flat_score=plugin_call(slope,flat)[15:-15,15:-15].mean().item()
    steep_score=plugin_call(slope,x*np.tan(np.deg2rad(35)))[15:-15,15:-15].mean().item()
    assert flat_score>0.9 and steep_score<0.2
    rough=RoughnessFilter(input_layer='elevation',resolution=0.1,low_radius=0.1,high_radius=0.56,midpoint=0.18,steepness=25)
    flat_rough=plugin_call(rough,flat)[15:-15,15:-15].mean().item()
    rough_score=plugin_call(rough,0.5*cp.sin(2*np.pi*x/0.7))[15:-15,15:-15].mean().item()
    assert flat_rough>0.9 and rough_score<flat_rough-0.2
    (VAL/'geometry_response.json').write_text(json.dumps(dict(flat_slope=flat_score,slope_35deg=steep_score,flat_roughness=flat_rough,rough_surface=rough_score),indent=2))


@pytest.mark.parametrize('kind,expected',[('road',0.2),('grass',0.48),('obstacle',0.0),('unobserved',0.1)])
def test_semantic_fusion(kind,expected):
    height=cp.zeros((40,40),dtype=cp.float32)
    semantic=cp.zeros((3,40,40),dtype=cp.float32)
    names=['sem_road','sem_grass','sem_obstacle']
    if kind!='unobserved': semantic[['road','grass','obstacle'].index(kind)]=1
    cost=plugin_call(SemanticClassLayer(layers=names,SemCost=[0,0.2,1],output_mode='cost'),height,semantic,names)
    sens=plugin_call(SemanticClassLayer(layers=names,SemSensitivity=[1,0.5,1],output_mode='sensitivity'),height,semantic,names)
    upstream=cp.stack([cp.full_like(height,0.2),cp.full_like(height,0.2),sens,cost])
    result=plugin_call(HeuristicTraversability(),height,upstream=upstream,upstream_names=['traversability_roughness','traversability_slope','sem_sensitivity','sem_cost'])
    np.testing.assert_allclose(cp.asnumpy(result),expected,atol=1e-5)


def test_real_rellis_pointcloud_to_gpu_map():
    p=Parameter(resolution=0.1,map_length=15,use_chainer=False,
        weight_file=str(PKG/'config/core/weights.dat'),
        plugin_config_file=str(PKG/'config/setups/bunker/bit-bunker_plugin_heruistic.yaml'),
        subscriber_cfg={'ouster':{'data_type':'pointcloud','channels':['intensity']}},
        pointcloud_channel_fusions={'default':'average'},
        enable_drift_compensation=False,enable_overlap_clearance=False)
    p.update()
    m=ElevationMap(p)
    m.move_to(np.array([0,0,0.4]),np.eye(3))
    metadata=json.loads((FIXTURES/'metadata.json').read_text())
    T=np.array(metadata['T_map_lidar'])
    for path in sorted(FIXTURES.glob('scan_*.npy')):
        m.input_pointcloud(np.load(path),['x','y','z','intensity'],T[:3,:3],T[:3,3],0.,0.)
    names=['elevation','variance','traversability']+m.plugin_manager.layer_names
    output={}
    summary={}
    for name in names:
        arr=np.full((p.cell_n-2,p.cell_n-2),np.nan,dtype=np.float32)
        m.get_map_with_name_ref(name,arr)
        output[name]=arr
        finite=arr[np.isfinite(arr)]
        summary[name]=dict(finite_cells=int(len(finite)),min=float(finite.min()) if len(finite) else None,max=float(finite.max()) if len(finite) else None)
    cp.cuda.Stream.null.synchronize()
    assert summary['elevation']['finite_cells']>500
    for name in ['traversability_slope','traversability_roughness','traversability_heuristic']:
        assert summary[name]['finite_cells']>500
        assert -1e-6<=summary[name]['min']<=summary[name]['max']<=1+1e-6
    np.savez_compressed(VAL/'core_gridmap.npz',**output)
    (VAL/'core_gridmap_summary.json').write_text(json.dumps(summary,indent=2))
