import cv2, optuna
import numpy as np
from PIL import Image
from tqdm import tqdm

from bytetrack.byte_tracker import BYTETracker
from models.crash_detector import CrashDetector

def evaluate_model(video_data, tracker, crash_detector, valid_classes, 
                   flow_energy_thresh, flow_percentile, flow_scale):

    height = video_data['height']
    width = video_data['width']
    fps = video_data['fps']
    
    tracker.reset()
    detected_accidents = []
    
    for frame_id in range(len(video_data['detections_per_frame'])):
        detections = video_data['detections_per_frame'][frame_id]
        mag_small = video_data['flow_mag_per_frame'][frame_id]
        
        mask = np.isin(detections.class_id, list(valid_classes.keys()))
        detections_for_tracker = np.empty((0, 5))
        if mask.sum() > 0:
            scores = detections.confidence.reshape(-1, 1)
            detections_for_tracker = np.hstack((detections.xyxy, scores))[mask]

        online_targets = tracker.update(detections_for_tracker, (height, width), (height, width), fps)
        
        if online_targets and mag_small is not None:
            flow_h, flow_w = mag_small.shape
            
            flow_energies = {}
            for t in online_targets:
                if t.track_id is None: continue
                box_energy = 0.0 
                x1, y1, w, h = t.tlwh
                if not any(v is None for v in [x1, y1, w, h]):
                    x1_s, y1_s = max(0, int(x1 * flow_scale)), max(0, int(y1 * flow_scale))
                    x2_s, y2_s = min(flow_w, int((x1 + w) * flow_scale)), min(flow_h, int((y1 + h) * flow_scale))
                    
                    roi_mag = mag_small[y1_s:y2_s, x1_s:x2_s]
                    box_energy = np.percentile(roi_mag, flow_percentile) if roi_mag.size > 0 else 0.0
                    box_energy = (box_energy*100.0) / (np.hypot(w, h) + 1e-6)
                
                t.curr_flow_energy = box_energy 
                flow_energies[t.track_id] = box_energy
            
            anomalous_tracks = tracker.anomaly_detector.update(tracker, fps, flow_energies)

            for t in online_targets:
                if t.track_id in anomalous_tracks:
                    t.is_anomaly = True
                    t.anomaly_score = anomalous_tracks[t.track_id]
                    if t.curr_flow_energy < flow_energy_thresh: 
                        t.is_anomaly = False
                else:
                    t.is_anomaly = False

            accidents = crash_detector.update(online_targets, frame_id, fps)
            detected_accidents.extend(accidents)

    return detected_accidents

def precompute_data(dataloader, model, detection_thresh, flow_scale):

    precomputed_data = []
    
    for data in tqdm(dataloader):
        video_frames = data['video'][0]
        height = data['height'][0].item()
        width = data['width'][0].item()
        
        total_frames = data['total_frames'][0].item()
        total_duration = data['total_duration'][0].item()
        fps = (total_frames / total_duration) if total_duration > 0.0 else 30.0
        
        gt_accident_frame = data['accident_frame'][0].item()
        
        flow_w, flow_h = int(width * flow_scale), int(height * flow_scale)
        prev_gray_small = None
        
        detections_per_frame = []
        flow_mag_per_frame = []
        
        for frame_chw in video_frames:
            frame = frame_chw.permute(1, 2, 0).numpy() 
            image_pil = Image.fromarray(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray_small = cv2.resize(gray, (flow_w, flow_h))
            
            detections = model.predict(image_pil, threshold=detection_thresh)
            detections_per_frame.append(detections)
            
            mag_small = None
            if prev_gray_small is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray_small, gray_small, None, 
                    pyr_scale=0.5, levels=3, winsize=15, 
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                mag_small, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_mag_per_frame.append(mag_small)
            
            prev_gray_small = gray_small
            
        precomputed_data.append({
            'detections_per_frame': detections_per_frame,
            'flow_mag_per_frame': flow_mag_per_frame,
            'height': height,
            'width': width,
            'fps': fps,
            'gt_accident_frame': gt_accident_frame
        })
        
    return precomputed_data

def objective(trial, precomputed_data, valid_classes, data_cfg, model_cfg, predict_cfg):
    trial_cfg = predict_cfg.copy()
    
    trial_cfg['accl_thres'] = trial.suggest_float('accl_thres', -10.0, -0.01)
    trial_cfg['angle_thres'] = trial.suggest_float('angle_thres', 0.2, np.pi/3)
    trial_cfg['min_movement_pixels'] = trial.suggest_float('min_movement_pixels', 1.0, 10.0)
    trial_cfg['closing_speed_thres'] = trial.suggest_float('closing_speed_thres', 1.0, 10.0)
    trial_cfg['min_history_len'] = trial.suggest_int('min_history_len', 5, 20)
    trial_cfg['accel_window'] = trial.suggest_int('accel_window', 3, 10)
    trial_cfg['angle_step'] = trial.suggest_int('angle_step', 2, 10)
    flow_energy_thresh = trial.suggest_float('flow_energy_thresh', 1.0, 10.0)
    flow_percentile = trial.suggest_int('flow_percentile', 85, 95)
    trial_cfg['proximity_ratio'] = trial.suggest_float('proximity_ratio', 0.1, 0.8)
    trial_cfg['flow_spike_thres'] = trial.suggest_float('flow_spike_thres', 1.0, 6.0)
    trial_cfg['alignment'] = trial.suggest_float('alignment', 0.80, 0.99)
    trial_cfg['anomaly_threshold'] = trial.suggest_float('anomaly_threshold', 0.5, 3.0)

    tp, fp, fn = 0, 0, 0
    inv_class_map = {v: k for k, v in data_cfg.get('class_map', {}).items()}

    for video_data in precomputed_data:
        gt_frame = video_data['gt_accident_frame']
        fps = video_data['fps']
        
        tracker = BYTETracker(predict_cfg=trial_cfg, model_cfg=model_cfg)
        crash_detector = CrashDetector(accident_cooldown=0, inv_class_map=inv_class_map,
                                       proximity_ratio=trial_cfg['proximity_ratio'])
        
        if video_data['flow_mag_per_frame'] and video_data['flow_mag_per_frame'][0] is not None:
            flow_w = video_data['flow_mag_per_frame'][0].shape[1]
            width = video_data['width']
            flow_scale = flow_w / width
        else:
            flow_scale = 0.5
        
        predicted_accidents = evaluate_model(
            video_data, tracker, crash_detector, valid_classes, 
            flow_energy_thresh, flow_percentile, flow_scale
        )
        
        matched_tp = False
        is_accident = gt_frame is not None and gt_frame > 0 
        for accident in predicted_accidents:
            event_frame = accident.get('frame', int(accident.get('time', 0) * fps))
            if is_accident and abs(event_frame - gt_frame) <= (0.5 * fps) :
                if not matched_tp: 
                    tp += 1
                    matched_tp = True
            else:
                fp += 1
        if is_accident and not matched_tp:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    trial.set_user_attr("precision", precision)
    trial.set_user_attr("recall", recall)
    trial.set_user_attr("tp", tp)
    trial.set_user_attr("fp", fp)
    trial.set_user_attr("fn", fn)

    return f1_score

def run_optimization(model, dataloader, valid_classes, data_cfg, model_cfg, predict_cfg):

    print("--- Hyperparameter Optimization for Accident Prediction ---")
    
    precomputed_data = precompute_data(dataloader, model, detection_thresh=0.4, flow_scale=0.5)

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name="accident_prediction_f1_study"
    )
    
    obj_fn = lambda trial: objective(trial, precomputed_data, valid_classes, data_cfg, model_cfg, predict_cfg)
    
    study.optimize(obj_fn, n_trials=100, show_progress_bar=True)

    print("\n Optimization completed.")
    print(f"Best F1 Score: {study.best_value:.4f}")
    
    print("\nBest Parameters found:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
        
    best_trial = study.best_trial
    print("\nMetrics for Best Trial:")
    print(f"  - Precision: {best_trial.user_attrs.get('precision', 'N/A'):.4f}")
    print(f"  - Recall: {best_trial.user_attrs.get('recall', 'N/A'):.4f}")
    print(f"  - TP: {best_trial.user_attrs.get('tp', 'N/A')}, FP: {best_trial.user_attrs.get('fp', 'N/A')}, FN: {best_trial.user_attrs.get('fn', 'N/A')}")
