#!/usr/bin/env python3
"""Extract real sensor fixtures from a foundation sequence; never generate labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def prepare(source, output):
    cal = json.loads((source/'calibration/calibration.json').read_text())
    scans = sorted((source/'lidar_raw').glob('*.npz'))[:10]
    if len(scans) != 10:
        raise ValueError('Need at least ten raw Ouster scans')
    with (source/'images.csv').open() as stream:
        image_row = next(csv.DictReader(stream))
    image_path = source/image_row['image_path']
    rgb = cv2.imread(str(image_path))
    if rgb is None:
        raise ValueError('Cannot decode the first source image')
    output.mkdir(parents=True, exist_ok=False)
    # Isolated stationary transport smoke only; this is not a production pose.
    map_base = np.eye(4); map_base[2, 3] = .4
    map_body = map_base @ np.linalg.inv(np.asarray(cal['T_body_base']))
    metadata = dict(sequence=source.name, purpose='stationary GPU/ROS transport smoke; approximate poses',
                    T_map_lidar=(map_body @ np.asarray(cal['T_body_lidar'])).tolist(),
                    base_height_m=.4, scans=[])
    for i, path in enumerate(scans):
        with np.load(path) as raw:
            points = np.column_stack([raw[k].reshape(-1) for k in ['x', 'y', 'z', 'intensity']]).astype('<f4')
        np.save(output/f'scan_{i:02d}.npy', points)
        metadata['scans'].append(dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), point_count=len(points)))
    np.save(output/'rgb.npy', cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    metadata['camera'] = {k: cal[k] for k in ['K', 'D', 'width', 'height', 'distortion_model']}
    metadata['camera'].update(source=str(image_path), sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                              T_map_camera=(map_body @ np.asarray(cal['T_body_camera'])).tolist())
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foundation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.foundation, args.output)
