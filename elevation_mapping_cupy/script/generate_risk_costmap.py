#!/usr/bin/python3
"""Export paper EMA/CVaR statistics to a persistent global costmap."""
import os
import json
import threading
import time
import numpy as np
import rospy
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import String
from elevation_mapping_cupy.risk_mapping import cvar_coefficient, decode_grid_layer, score_to_cost, validate_cost_thresholds


class RiskCostmap:
    def __init__(self):
        self.frame = rospy.get_param('~map_frame', 'map')
        self.resolution = float(rospy.get_param('~resolution', 0.1))
        length = float(rospy.get_param('~map_length', 200.0))
        if self.resolution <= 0 or length <= 0:
            raise ValueError('Map dimensions must be positive')
        self.size = int(round(length/self.resolution))
        self.origin_x = float(rospy.get_param('~origin_x', -length/2))
        self.origin_y = float(rospy.get_param('~origin_y', -length/2))
        self.alpha = float(rospy.get_param('~cvar_alpha', 0.9))
        self.coefficient = cvar_coefficient(self.alpha)
        self.gamma = float(rospy.get_param('~ema_gamma', 0.2))
        self.reference_speed = float(rospy.get_param('~reference_speed'))
        self.cost_encoding = rospy.get_param('~cost_encoding', 'continuous')
        if self.cost_encoding not in ('continuous', 'five_level'):
            raise ValueError('cost_encoding must be continuous or five_level')
        self.cost_thresholds=validate_cost_thresholds(rospy.get_param('~cost_thresholds',[.3,.5,.7,.9]))
        self.save_path = rospy.get_param('~save_path', '')
        self.lock = threading.RLock()
        self.metadata = None
        self.last_publish = 0.
        self.publish_interval = 1./float(rospy.get_param('~publish_rate', 1.0))
        self.state = np.zeros((3, self.size, self.size), dtype=np.float32)
        self.pub = rospy.Publisher('/similarity_costmap', OccupancyGrid, queue_size=1, latch=True)
        self.service = rospy.Service('~save', Trigger, self.save)
        rospy.Subscriber('/uam2te/navigation_metadata', String, self.on_metadata, queue_size=1)
        rospy.Subscriber(rospy.get_param('~grid_topic'), GridMap, self.callback, queue_size=1)
        rospy.on_shutdown(self.save)

    def on_metadata(self, msg):
        metadata = json.loads(msg.data)
        if abs(metadata['reference_speed']-self.reference_speed) > 1e-6 or (self.metadata is not None and metadata != self.metadata):
            rospy.signal_shutdown('Prediction identity changed; restart the complete mapping launch')
            return
        self.metadata = metadata

    def callback(self, msg):
        try:
            if not all(layer in msg.layers for layer in ('similarity', 'similarity_var', 'similarity_count')):
                rospy.loginfo_throttle(5, 'Waiting for the first projected model observation')
                return
            if msg.info.header.frame_id != self.frame:
                raise ValueError('GridMap and output map frame differ; supply a consistent TF/map configuration')
            if not np.isclose(msg.info.resolution, self.resolution):
                raise ValueError('Global statistics and terrain map must use the same resolution')
            q = msg.info.pose.orientation
            if abs(q.x)+abs(q.y)+abs(q.z) > 1e-6 or abs(abs(q.w)-1) > 1e-6:
                raise ValueError('Expected world-aligned terrain GridMap')
            mean = decode_grid_layer(msg, 'similarity')
            variance = decode_grid_layer(msg, 'similarity_var')
            count = decode_grid_layer(msg, 'similarity_count')
            ny, nx = mean.shape
            x = msg.info.pose.position.x+(nx/2-np.arange(nx)-0.5)*self.resolution
            y = msg.info.pose.position.y+(ny/2-np.arange(ny)-0.5)*self.resolution
            gx = np.floor((x-self.origin_x)/self.resolution).astype(int)
            gy = np.floor((y-self.origin_y)/self.resolution).astype(int)
            xx, yy = np.meshgrid(gx, gy)
            valid = ((xx >= 0) & (xx < self.size) & (yy >= 0) & (yy < self.size)
                     & (count > 0) & np.isfinite(mean) & np.isfinite(variance))
            # Copy already-fused statistics, NEVER integrate periodically published
            # GridMap messages a second time. Full world history lives in EMA fusion.
            with self.lock:
                self.state[:, yy[valid], xx[valid]] = np.stack((mean, variance, count))[:, valid]
                now = time.monotonic()
                if now-self.last_publish >= self.publish_interval:
                    self.publish(msg.info.header.stamp)
                    self.last_publish = now
        except (ValueError, KeyError) as exc:
            rospy.logerr_throttle(5, 'Risk costmap: %s', exc)

    def publish(self, stamp):
        alpha = float(rospy.get_param('~cvar_alpha', self.alpha))
        coefficient = cvar_coefficient(alpha)
        self.alpha, self.coefficient = alpha, coefficient
        mean, variance, count = self.state
        conservative = mean-coefficient*np.sqrt(np.maximum(variance, 0))
        conservative = np.where(count > 0, conservative, np.nan)
        result = score_to_cost(conservative, self.cost_encoding, self.cost_thresholds)
        msg = OccupancyGrid()
        msg.header.frame_id, msg.header.stamp = self.frame, stamp
        msg.info.resolution = self.resolution
        msg.info.width = msg.info.height = self.size
        msg.info.origin.position.x, msg.info.origin.position.y = self.origin_x, self.origin_y
        msg.info.origin.orientation.w = 1
        msg.data = result.ravel().tolist()
        self.pub.publish(msg)

    def save(self, request=None):
        if not self.save_path:
            return TriggerResponse(success=False, message='Set ~save_path to an absolute .npz path')
        try:
            path = os.path.abspath(os.path.expanduser(self.save_path))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with self.lock, open(path+'.tmp', 'wb') as handle:
                np.savez_compressed(handle, score_mean=self.state[0], cost_mean=1-self.state[0],
                                    variance=self.state[1], count=self.state[2],
                                    valid=self.state[2] > 0, resolution=self.resolution,
                                    origin=[self.origin_x, self.origin_y], map_frame=self.frame,
                                    cvar_alpha=self.alpha, ema_gamma=self.gamma,
                                    cost_encoding=self.cost_encoding,
                                    cost_thresholds=self.cost_thresholds,
                                    reference_speed=self.reference_speed,
                                    support_min=float((self.metadata or {}).get('mapping_support_min', 0.)),
                                    prediction_metadata=json.dumps(self.metadata, sort_keys=True),
                                    schema='paper_ema_cvar_v1')
            os.replace(path+'.tmp', path)
            return TriggerResponse(success=True, message=path)
        except Exception as exc:
            return TriggerResponse(success=False, message=str(exc))


if __name__ == '__main__':
    rospy.init_node('risk_costmap')
    RiskCostmap()
    rospy.spin()
