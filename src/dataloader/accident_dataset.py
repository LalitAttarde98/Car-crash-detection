import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from collections import defaultdict
import cv2
from numba import jit

class AccidentDataset(Dataset):
    def __init__(self, 
                train_dir=None, test_list=None, test_dir=None,
                train_dtype=[
                ('rgb_path', '<U200'), ('annotations_path', '<U200'), ('type', '<U9'), ('accident_time', '<f8'),
                ('accident_frame', '<i8'), ('center_x', '<f8'), ('center_y', '<f8'), ('x1', '<f8'), ('y1', '<f8'),
                ('x2', '<f8'), ('y2', '<f8'), ('map', '<U8'), ('weather', '<U6'), ('camera_position', '<i8'),
                ('no_frames', '<i8'), ('duration', '<f8'), ('height', '<i8'), ('width', '<i8'),
                ('annotations_start_offset', '<i8')
                ],
                test_dtype = [
                ('rgb_path', '<U200'), ('region', '<U200'), ('scene_layout', '<U9'), ('weather', '<f8'),
                ('day_time', '<i8'), ('quality', '<f8'), ('no_frames', '<f8'), ('duration', '<f8'), ('height', '<f8'),
                ('width', '<f8'), 
                ],
                class_map={'head-on': 0, 'rear-end': 1, 'sideswipe': 2, 'single': 3, 't-bone': 4, 'Normal': 5},
                train_classes=None, infer_CLASSES=None,
                mode = 'train', window_size=30, stride=10,
                 ):
        self.mode = mode
        if self.mode == 'train':
            labels_csv_path = os.path.join(train_dir, 'labels.csv')
            self.labels_csv = np.genfromtxt(labels_csv_path, delimiter=',', dtype=train_dtype, names=True, encoding='utf-8')
            self.labels_csv = self.labels_csv[self.labels_csv['type'] == 't-bone'][:60]

            self.cls_id_to_name = train_classes
            self.valid_class_names = set(train_classes.values())

            self.labels_csv['rgb_path'] = np.char.add(os.path.join(train_dir, ''), self.labels_csv['rgb_path'])
            self.labels_csv['annotations_path'] = np.char.rstrip(self.labels_csv['annotations_path'], '.gz')
            lables_files = np.char.lstrip(self.labels_csv['annotations_path'], 'video_annotations')
            full_paths = np.char.add(train_dir, self.labels_csv['annotations_path'])
            self.labels_csv['annotations_path'] = np.char.add(full_paths, lables_files)

            self.window_size=window_size
            self.stride=stride
            self.collision_limit = 5
            self.samples = self._gather_samples(self.valid_class_names)

        elif self.mode == 'test' or self.mode == 'optimize':
            labels_csv_path = os.path.join(train_dir, 'labels.csv')
            self.test_csv = np.genfromtxt(labels_csv_path, delimiter=',', dtype=train_dtype, names=True, encoding='utf-8')
            self.test_csv = np.concatenate([self.test_csv[(self.test_csv['type'] == 't-bone') & (self.test_csv['weather'] == 'clear')][:10],
                                           self.test_csv[(self.test_csv['type'] == 'single') & (self.test_csv['weather'] == 'clear')][:10],
                                           self.test_csv[(self.test_csv['type'] == 'rear-end') & (self.test_csv['weather'] == 'clear')][:10],
                                           self.test_csv[(self.test_csv['type'] == 'head-on') & (self.test_csv['weather'] == 'clear')][:10],
                                           self.test_csv[(self.test_csv['type'] == 'sideswipe') & (self.test_csv['weather'] == 'clear')][:10]],
                                            axis=0)
            self.test_csv['rgb_path'] = np.char.add(os.path.join(train_dir, ''), self.test_csv['rgb_path'])

        elif self.mode == 'inference':
            self.test_csv = np.genfromtxt(test_list, delimiter=',', dtype=test_dtype, names=True, encoding='utf-8')
            #self.test_csv = self.test_csv[self.test_csv['rgb_path'] == 'videos/fzWY0vLAXzI_00.mp4']   #[(self.test_csv['quality'] == 'Good')]
            self.test_csv = self.test_csv[self.test_csv['quality'] == 'Very_Poor'][-30:-10]
            self.test_csv['rgb_path'] = np.char.add(os.path.join(test_dir, ''), self.test_csv['rgb_path'])


    def _gather_samples(self, target_classes):
        samples = []
        for row in self.labels_csv:
            annotation_path = row['annotations_path']

            with open(annotation_path, 'r') as file:
                data = json.load(file)

            offset = row['annotations_start_offset']
            video_width, video_height = row['width'], row['height']
            fps = row['no_frames'] / row['duration'] if row['duration'] > 0 else 30.0

            raw_tracks = self._extract_raw_tracks(data, offset, target_classes)
            feature_tracks = self._compute_velocities(raw_tracks, fps, video_width, video_height)
            
            collision_frames, anomalous_track_ids = self._get_collision_data(data, offset)

            max_frame = 0
            if feature_tracks:
                max_frame = int(max(np.max(track_data[:, 0]) for track_data in feature_tracks.values()))

            start_frame = 0
            while start_frame < max_frame:
                end_frame = start_frame + self.window_size
                window_tracks, num_tracks = self._get_window_tracks(feature_tracks, start_frame, end_frame)
                
                if num_tracks > 0:
                    is_accident, anomalous_ids = self._get_window_labels(collision_frames, anomalous_track_ids, start_frame, end_frame)
                    samples.append({
                        'tracks': window_tracks,
                        'anomalous_track_ids': anomalous_ids,
                        'is_accident': 1.0 if is_accident else 0.0,
                        'type': row['type'] if is_accident else 'Normal',
                        'video_id': row['rgb_path'],
                    })
                
                start_frame += 5 if is_accident else self.stride

        return samples

    def _extract_raw_tracks(self, data, offset, target_classes):
        raw_tracks = defaultdict(list)

        for frame_data in data.get('base', []):
            frame_number = frame_data.get('iteration', 0) - offset
            for obj in frame_data.get('objects', []):
                class_name = self.cls_id_to_name.get(obj.get('tag'))
                if class_name and class_name in target_classes:
                    raw_tracks[obj['id']].append((frame_number, obj['2d_bbox']))
        return raw_tracks

    def _compute_velocities(self, raw_tracks, fps, width, height):

        feature_tracks = {}
        wh_array = np.array([width, height, width, height], dtype=np.float32)

        for track_id, frames_and_boxes in raw_tracks.items():
            frames = np.array([x[0] for x in frames_and_boxes], dtype=np.float32)
            bboxes = np.array([x[1] for x in frames_and_boxes], dtype=np.float32).reshape(-1, 4)

            sort_idx = np.argsort(frames)
            frames = frames[sort_idx]
            bboxes = bboxes[sort_idx]

            bboxes /= wh_array
            xmin, ymin, xmax, ymax = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            w = xmax - xmin
            h = ymax - ymin
            xc = xmin + w / 2.0
            yc = ymin + h / 2.0

            feats = np.column_stack((xc, yc, w, h)) 
            velocities = np.zeros_like(feats) 
            if len(frames) > 1:
                frame_diffs = np.diff(frames)[:, None] 
                feat_diffs = np.diff(feats, axis=0)
                
                valid = (frame_diffs > 0).flatten()
                
                if np.any(valid):
                    # v = dx * (fps / dt) pixels/s
                    velocities[1:][valid] = feat_diffs[valid] * (fps / frame_diffs[valid])

            velocities = velocities +  np.random.randn(*velocities.shape) * 1e-2 # Regularization noise
            feats = feats +  np.random.randn(*feats.shape) * 1e-2 # Regularization noise
            track_data = np.column_stack((frames, feats, velocities))
            feature_tracks[track_id] = track_data

        return feature_tracks

    def _get_collision_data(self, data, offset):
        collision_frames = []
        anomalous_track_ids = set()
        
        for i, collision in enumerate(data.get('collision', [])):
            anomalous_track_ids.update(collision.get('ids', []))
            collision_frames.append(collision.get('iteration', 0) - offset)
            if i >= self.collision_limit: break
            
        return collision_frames, anomalous_track_ids

    @staticmethod
    def _get_window_tracks(feature_tracks, start_frame, end_frame):
  
        window_tracks = {}
        for track_id, track_data in feature_tracks.items():
            frames = track_data[:, 0]
            mask = (frames >= start_frame) & (frames < end_frame)
            
            if np.any(mask):
                valid_data = track_data[mask]
                window_tracks[track_id] = [
                    (row[0] - start_frame, row[1:]) for row in valid_data
                ]
        
        return window_tracks, len(window_tracks)

    @staticmethod
    def _get_window_labels(collision_frames, anomalous_ids, start_frame, end_frame):
        is_accident = any(start_frame <= cf < end_frame for cf in collision_frames)
        return (is_accident, list(anomalous_ids) if is_accident else [])

    def __len__(self):
        if self.mode == 'train':
            return len(self.samples)
        return len(self.test_csv)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.samples[idx]

        video_path = self.test_csv[idx]['rgb_path']
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        video_tensor = torch.from_numpy(np.array(frames, dtype=np.uint8)).permute(0, 3, 1, 2)

        return {
            'video': video_tensor,
            'total_frames': len(frames),
            'total_duration': self.test_csv[idx]['duration'],
            'rgb_path': self.test_csv[idx]['rgb_path'],
            'height': self.test_csv[idx]['height'],
            'width': self.test_csv[idx]['width'],
            'accident_frame': self.test_csv[idx]['accident_frame'] if 'accident_frame' in self.test_csv.dtype.names else -1,
            'quality': self.test_csv[idx]['quality'] if 'quality' in self.test_csv.dtype.names else 'Good'
        }