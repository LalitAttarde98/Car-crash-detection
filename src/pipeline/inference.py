import cv2
import numpy as np
import copy
import supervision as sv
from PIL import Image
from collections import deque
from pathlib import Path

from bytetrack.byte_tracker import BYTETracker
from models.crash_detector import CrashDetector
from models.vlm_infer import classify_crash_images

def tlwhs_to_xyxys(tlwhs):
    xyxys = tlwhs.copy()
    xyxys[:, 2:] += xyxys[:, :2]
    return xyxys

def calculate_optical_flow(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 
        pyr_scale=0.5, levels=3, winsize=15, 
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag

def process_video(
    video_data, model, tracker, crash_detector, box_annotator, label_annotator, 
    valid_classes, inv_class_map, predict_cfg, vlm_model, vlm_processor
):
    video_frames = video_data['video'][0]
    video_path = Path(video_data['rgb_path'][0])
    height, width = video_data['height'][0].item(), video_data['width'][0].item()

    fps = (video_data['total_frames'] / video_data['total_duration']).item() if video_data['total_duration'] > 0 else 30
    
    image_buffer = deque(maxlen=int(fps) + 1) 
    
    output_path = Path('inference_videos') / f"{video_path.stem}_tracked.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    print(f"Processing video: {video_path}")
    
    tracker.reset()
    prev_gray_small = None
    motion_history = deque(maxlen=int(fps)) 
    
    flow_scale = predict_cfg['flow_scale']
    flow_energy_thresh = predict_cfg['flow_energy_thresh']
    flow_w, flow_h = int(width * flow_scale), int(height * flow_scale)

    curr_accident = "No Accident"
    for frame_id, frame_chw in enumerate(video_frames):
        frame = np.ascontiguousarray(frame_chw.permute(1, 2, 0).numpy()) 
        image_pil = Image.fromarray(frame)
        image_buffer.append(image_pil)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
        gray_small = cv2.resize(gray, (flow_w, flow_h))
        
        detections = model.predict(image_pil, threshold=predict_cfg['detection_thresh'])
        mask = np.isin(detections.class_id, list(valid_classes.keys()))
        
        mag_small = None
        optical_flow_trigger = False
        if prev_gray_small is not None:
            mag_small = calculate_optical_flow(prev_gray_small, gray_small)
            curr_motion_score = np.percentile(mag_small, predict_cfg['flow_percentile'])
            
            if len(motion_history) == motion_history.maxlen:
                baseline_mean, baseline_std = np.mean(motion_history), np.std(motion_history)
                if curr_motion_score > (baseline_mean + 6.0 * baseline_std):
                    optical_flow_trigger = True
            
            motion_history.append(curr_motion_score)
        
        prev_gray_small = gray_small
        
        detections_for_tracker = np.empty((0, 5))
        if mask.sum() > 0:
            scores = detections.confidence.reshape(-1, 1)
            detections_for_tracker = np.hstack((detections.xyxy, scores))[mask]

        online_targets = tracker.update(detections_for_tracker, (height, width), (height, width), fps)
        
        annotated_frame = frame.copy()            
        if online_targets:
            online_tlwhs, online_ids, labels = [], [], []
            
            flow_energies = {}
            for t in online_targets:
                if t.track_id is None: continue
                box_energy = 0.0 
                if mag_small is not None:
                    x1, y1, w, h = t.tlwh
                    if not any(v is None for v in [x1, y1, w, h]):
                        x1_s, y1_s = max(0, int(x1 * flow_scale)), max(0, int(y1 * flow_scale))
                        x2_s, y2_s = min(flow_w, int((x1 + w) * flow_scale)), min(flow_h, int((y1 + h) * flow_scale))
                        
                        roi_mag = mag_small[y1_s:y2_s, x1_s:x2_s]
                        box_energy = np.percentile(roi_mag, predict_cfg['flow_percentile']) if roi_mag.size > 0 else 0.0
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

            detected_accidents = crash_detector.update(online_targets, frame_id, fps)
            confirmed_accidents = []

            if detected_accidents or optical_flow_trigger:
                
                if vlm_model is not None:
                    before_image = image_buffer[0] 
                    impact_image = image_pil.copy() 

                    curr_accident = classify_crash_images(vlm_model, vlm_processor, before_image, impact_image)
                elif detected_accidents:
                    acc = detected_accidents[0]
                    cx, cy = acc["location"]
                    curr_accident = detected_accidents[0]["type"] 
                print(f"Accident detected! (Flow: {optical_flow_trigger}, Physics: {bool(detected_accidents)}) \
                      Predicted type: {curr_accident}")

            
            for t in online_targets:
                box_energy = getattr(t, 'curr_flow_energy', 0.0)
                if t.is_anomaly:
                    score = getattr(t, 'anomaly_score', 0.0)
                    labels.append(f"ANOMALY! S: {score:.2f} | E: {box_energy:.1f}")
                else:
                    labels.append(f"ID:{t.track_id} | E:{box_energy:.1f}")
                
                online_tlwhs.append(t.tlwh)
                online_ids.append(t.track_id)

            if online_tlwhs:
                detections_sv = sv.Detections(
                    xyxy=tlwhs_to_xyxys(np.array(online_tlwhs)), 
                    tracker_id=np.array(online_ids, dtype=int),
                    class_id=np.zeros(len(online_ids), dtype=int) 
                )
                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections_sv)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections_sv, labels=labels)
            
            for t in online_targets:
                    if getattr(t, 'is_anomaly', False):
                        x1, y1, w, h = t.tlwh
                        x2, y2 = int(x1 + w), int(y1 + h)
                        x1, y1 = int(x1), int(y1)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 6)
            
            if optical_flow_trigger:
                cv2.putText(annotated_frame, "Optical flow anomaly detected!", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            if predict_cfg.get('use_ml_predictor', False):
                is_anomaly_present = any(t.is_anomaly for t in online_targets)
                color = (0, 0, 255) if is_anomaly_present else (0, 255, 0)
                predicted_type = confirmed_accidents[0]['type'] if confirmed_accidents else "No Accident"
                display_text = f"Accident Type: {predicted_type}" if is_anomaly_present else "No Accident Detected"
                cv2.putText(annotated_frame, display_text, (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

        if frame_id % 30 == 0:
            print(f"  Processed frame {frame_id}/{len(video_frames)}")

    out.release()
    print(f"Inference video saved to {output_path}\n")


def run_inference(model, dataloader, valid_classes, inv_class_map, predict_cfg, model_cfg, vlm_model, vlm_processor):
    predict_cfg_poor = copy.deepcopy(predict_cfg)
    if "poor_quality" in predict_cfg_poor:
        predict_cfg_poor.update(predict_cfg_poor["poor_quality"])
    
    tracker_good = BYTETracker(predict_cfg=predict_cfg, model_cfg=model_cfg)
    crash_detector_good = CrashDetector(
        accident_cooldown=0, inv_class_map=inv_class_map,
        proximity_ratio=predict_cfg["proximity_ratio"],
    )
    tracker_poor = BYTETracker(predict_cfg=predict_cfg_poor, model_cfg=model_cfg)
    crash_detector_poor = CrashDetector(
        accident_cooldown=0, inv_class_map=inv_class_map,
        proximity_ratio=predict_cfg_poor["proximity_ratio"],
    )

    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)

    for data in dataloader:
        quality = data.get("quality", "Good")
        is_poor = "Poor" in quality
        process_video(
            video_data=data,
            model=model,
            tracker=tracker_poor if is_poor else tracker_good,
            crash_detector=crash_detector_poor if is_poor else crash_detector_good,
            box_annotator=box_annotator,
            label_annotator=label_annotator,
            valid_classes=valid_classes,
            inv_class_map=inv_class_map,
            predict_cfg=predict_cfg_poor if is_poor else predict_cfg,
            vlm_model=vlm_model,
            vlm_processor=vlm_processor
        )