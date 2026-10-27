#!/usr/bin/python3
"""ROS Noetic system-Python driver; production C++ node embeds Python 3.10."""
import json,time,os
from pathlib import Path
import numpy as np
import rospy
import tf2_ros
from tf.transformations import quaternion_from_matrix
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2,PointField,Image,CameraInfo
from grid_map_msgs.msg import GridMap
VAL=Path(os.environ['EMCP_VALIDATION_OUTPUT'])
FIXTURES=Path(os.environ['EMCP_FIXTURES'])
meta=json.loads((FIXTURES/'metadata.json').read_text())
rospy.init_node('emcupy_smoke_driver')
received=[]
def callback(msg):
    received.append(msg)
    if len(received)>20: del received[:-20]
sub=rospy.Subscriber('/emcupy_smoke/elevation_mapping/elevation_map_raw',GridMap,callback,queue_size=4)
pub=rospy.Publisher('/emcupy_smoke/points',PointCloud2,queue_size=1)
image_pub=rospy.Publisher('/emcupy_smoke/image',Image,queue_size=1)
info_pub=rospy.Publisher('/emcupy_smoke/camera_info',CameraInfo,queue_size=1)
broadcaster=tf2_ros.StaticTransformBroadcaster()
transforms=[]
for child,T in [('base_link',np.diag([1.,1.,1.,1.])),('lidar',np.array(meta['T_map_lidar'])),('camera_optical',np.array(meta['camera']['T_map_camera']))]:
    if child=='base_link': T[2,3]=0.4
    tf=TransformStamped(); tf.header.stamp=rospy.Time.now(); tf.header.frame_id='emcupy_smoke/map'; tf.child_frame_id='emcupy_smoke/'+child
    tf.transform.translation.x,tf.transform.translation.y,tf.transform.translation.z=T[:3,3]
    q=quaternion_from_matrix(T)
    tf.transform.rotation.x,tf.transform.rotation.y,tf.transform.rotation.z,tf.transform.rotation.w=q
    transforms.append(tf)
broadcaster.sendTransform(transforms)
deadline=time.monotonic()+120
while not pub.get_num_connections():
    assert time.monotonic()<deadline,'Pointcloud subscriber did not start'
    time.sleep(0.2)
time.sleep(1.)
scans=sorted(FIXTURES.glob('scan_*.npy'))
for path in scans:
    a=np.load(path).astype('<f4')
    msg=PointCloud2();msg.header.stamp=rospy.Time.now();msg.header.frame_id='emcupy_smoke/lidar'
    msg.height=1;msg.width=len(a);msg.is_dense=False;msg.is_bigendian=False
    msg.fields=[PointField(name,i*4,PointField.FLOAT32,1) for i,name in enumerate(['x','y','z','intensity'])]
    msg.point_step=16;msg.row_step=16*len(a);msg.data=a.tobytes()
    pub.publish(msg)
    time.sleep(1.)
# Exercise the C++ cv_bridge -> embedded Python 3.10 image interface.
cam=meta['camera']; rgb=np.load(FIXTURES/'rgb.npy')
for _ in range(5):
    im=Image();im.header.stamp=rospy.Time.now();im.header.frame_id='emcupy_smoke/camera_optical'
    im.height,im.width=rgb.shape[:2];im.encoding='rgb8';im.step=im.width*3;im.data=rgb.tobytes()
    info=CameraInfo();info.header=im.header;info.width=im.width;info.height=im.height;info.K=cam['K'];info.D=cam['D'];info.distortion_model=cam['distortion_model']
    info_pub.publish(info);image_pub.publish(im)
    time.sleep(1.)
deadline=time.monotonic()+30
while time.monotonic()<deadline:
    if received and 'rgb' in received[-1].layers:
        packed=np.asarray(received[-1].data[received[-1].layers.index('rgb')].data,dtype=np.float32)
        if ((packed.view(np.uint32)&0xFFFFFF)!=0).sum()>30: break
    time.sleep(0.2)
assert received,'No GridMap published'
m=received[-1]
assert m.info.header.frame_id=='emcupy_smoke/map',m.info.header.frame_id
assert abs(m.info.resolution-0.1)<1e-8
layers={n:np.asarray(d.data,dtype=np.float32) for n,d in zip(m.layers,m.data)}
summary={}
for name,a in layers.items():
    finite=a[np.isfinite(a)]
    summary[name]=dict(finite_cells=int(len(finite)),min=float(finite.min()) if len(finite) else None,max=float(finite.max()) if len(finite) else None)
assert summary['elevation']['finite_cells']>500,summary
for name in ['traversability_slope','traversability_roughness','traversability_heuristic']:
    assert summary[name]['finite_cells']>500
    assert -1e-6<=summary[name]['min']<=summary[name]['max']<=1+1e-6
colored_cells=int(((layers['rgb'].view(np.uint32)&0xFFFFFF)!=0).sum())
assert colored_cells>30,colored_cells
np.savez_compressed(VAL/'ros_gridmap.npz',**layers)
report=dict(passed=True,source_sequence='00002',sent_frames=len(scans),rgb_colored_cells=colored_cells,received_maps=len(received),frame=m.info.header.frame_id,resolution=m.info.resolution,map_length=[m.info.length_x,m.info.length_y],layers=summary,scope='stationary real XYZI + real RGB/CameraInfo -> actual GPU emcupy C++ ROS node -> GridMap; semantics use configured unknown defaults, not RELLIS semantic labels')
(VAL/'ros_smoke_result.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
