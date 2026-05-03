import numpy as np
data_cfg = {
    'train_dir': '/sim_dataset/', # To be replaced by a user-defined path
    'test_list': '/test_metadata.csv',
    'test_dir': '/',
    'train_dtype': [
        ('rgb_path', '<U200'), ('annotations_path', '<U200'), ('type', '<U9'), ('accident_time', '<f8'),
        ('accident_frame', '<i8'), ('center_x', '<f8'), ('center_y', '<f8'), ('x1', '<f8'), ('y1', '<f8'),
        ('x2', '<f8'), ('y2', '<f8'), ('map', '<U8'), ('weather', '<U6'), ('camera_position', '<i8'),
        ('no_frames', '<i8'), ('duration', '<f8'), ('height', '<i8'), ('width', '<i8'),
        ('annotations_start_offset', '<i8')
    ],
    'test_dtype': [
        ('path', '<U200'), ('region', '<U200'), ('scene_layout', '<U9'), ('weather', '<U6'),
        ('day_time', '<U8'), ('quality', '<U10'), ('no_frames', '<i8'), ('duration', '<f8'), ('height', '<i8'),
        ('width', '<i8'), 
    ],
    'class_map': {'head-on': 0, 'rear-end': 1, 'sideswipe': 2, 'single': 3, 't-bone': 4, 'Normal': 5},
    'train_classes': {12: 'Pedestrian', 13: 'Rider', 14: 'Car', 15: 'Truck', 16: 'Bus', 17: 'Train',
                      18: 'Motorcycle', 19: 'Bicycle', 21: 'Dynamic', 29: 'Van'},
    'infer_CLASSES': {2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 7: "train", 8: "truck"},
    'mode': 'inference', # train / test / inference
    'window_size': 30,
    'stride': 10,
}

model_cfg = {
    'feature_dim':8,  # [xc, yc, w, h, vx, vy, vw, vh]
    'hidden_dim':32,
    'num_classes':len(data_cfg['class_map']),
    'num_heads':2, 
    'num_layers':1,
}


train_cfg = {
    'epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'batch_size': 16,
    'num_workers': 8,
    'loss_weights': {
        'anomaly': 0.5,       
        'type': 1.0,            
        'location': 0.5,     
    },
    'ckpt_path':'spatio_temporal_transformer_50.pth',
    'checkpoint_dir': './checkpoints',
}

predict_cfg = {
    'ckpt_path': 'spatio_temporal_epoch_50.pth',
    'use_ml_predictor': False,
    'track_thresh':0.4, 
    'track_buffer':data_cfg['window_size'], 
    'match_thresh':0.8,
    'fuse_score':False,
    'detection_thresh':0.4,
    # For good quality
    'accl_thres': -2.0, # accleration threshold (pixel/s^2)
    'angle_thres':np.pi / 6, #angle threshold (radians) 
    'min_history_len':8, # Minimum history length for tracking objects
    'accel_window':5, # Number of past frames to detect accleration spikes
    'closing_speed_thres':5.0, # Closing speed between two objects
    'angle_step':5,
    'proximity_ratio':0.3, 
    'flow_energy_thresh':2.0, # Flow energy spike threshold
    'flow_percentile':90, 
    'flow_scale':0.5, # Resize frame for optical flow calculations
    'flow_spike_thres': 6.0, # Standard deviation for optical flow anamoly
    'alignment':0.90,
    'anomaly_threshold': 1.5,
    'score_decay':0.90, # Exponential decay factor for anomaly scores over time

    # Optical Flow Analyser & Pipeline Parameters
    'flow_baseline_secs': 4.0,         # Long adaptive baseline length in seconds
    'flow_sustain_frames': 3,          # Required consecutive frames for spike trigger
    'flow_concentration_ratio': 2.5,   # Minimum spatial concentration ratio
    'flow_min_time_s': 2.0,            # Minimum time gate to skip early encoding artifacts
    'flow_energy_veto': 0.5,          # Absolute energy floor to qualify as valid optical flow
    'physics_confirm_frames': 2,       # Confirmation frames required if flow doesn't corroborate physics

    # For poor quality
    'poor_quality': {
        'min_history_len':5, 
        'flow_energy_thresh':1.0,
        'flow_spike_thres': 4.0,
        'anomaly_threshold': 1.4,
    }
}
