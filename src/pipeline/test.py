import cv2
import csv
import copy
import numpy as np
from PIL import Image
from collections import deque
from pathlib import Path

from bytetrack.byte_tracker import BYTETracker
from models.crash_detector import CrashDetector
from models.optical_flow_analyser import OpticalFlowAnalyser
from models.vlm_infer import classify_crash_images

def calculate_optical_flow(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag

def process_video(
    video_data, model, tracker, crash_detector,
    valid_classes, inv_class_map, predict_cfg,
    vlm_model, vlm_processor,
):
    video_frames = video_data["video"][0]
    video_path = Path(video_data["rgb_path"][0])
    height = video_data["height"][0].item()
    width = video_data["width"][0].item()
    fps = ((video_data["total_frames"] / video_data["total_duration"]).item()
        if video_data["total_duration"] > 0 else 30.0
    )

    print(f"Processing: {video_path.name}  ({width}x{height} @ {fps:.1f}fps)")

    tracker.reset()

    buffer_len = max(int(fps * 2), 10)
    image_buffer = deque(maxlen=buffer_len)

    flow_scale = predict_cfg["flow_scale"]
    flow_w = int(width  * flow_scale)
    flow_h = int(height * flow_scale)
    flow_energy_veto = predict_cfg.get("flow_energy_veto", 0.5)

    flow_analyser = OpticalFlowAnalyser(fps, predict_cfg)
    prev_gray_small = None

    # Priority ladder:
    # 0 = physics anomaly 
    # 1 = physics + flow corroboration
    # 2 = optical flow
    # 3 = peak-flow safety net
    best_priority = 99
    final_prediction   = {"time": 0.0, "cx": 0.5, "cy": 0.5, "type": "single"}
    abs_max_flow_score = -1.0
    abs_max_flow_data = None
    physics_candidate = None
    physics_confirm_frames = predict_cfg.get("physics_confirm_frames", 2)
    physics_frame_count = 0

    for frame_id, frame_chw in enumerate(video_frames):
        frame = frame_chw.permute(1, 2, 0).numpy()
        image_pil = Image.fromarray(frame)
        image_buffer.append(image_pil)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_small = cv2.resize(gray, (flow_w, flow_h))

        detections = model.predict(image_pil, threshold=predict_cfg["detection_thresh"])
        mask = np.isin(detections.class_id, list(valid_classes.keys()))

        mag_small = None
        flow_triggered = False
        flow_cx, flow_cy = 0.5 * width, 0.5 * height
        curr_motion_score= 0.0

        if prev_gray_small is not None:
            mag_small = calculate_optical_flow(prev_gray_small, gray_small)

            flow_triggered, flow_cx, flow_cy, curr_motion_score = flow_analyser.update(
                mag_small, frame_id, flow_scale, width, height
            )

            if curr_motion_score > abs_max_flow_score:
                abs_max_flow_score = curr_motion_score
                abs_max_flow_data  = {"time": frame_id / fps,
                    "cx":flow_cx,"cy": flow_cy,
                    "before_img": image_buffer[0],
                    "impact_img": image_pil.copy(),
                }

        prev_gray_small = gray_small

        detections_for_tracker = np.empty((0, 5))
        if mask.sum() > 0:
            scores = detections.confidence.reshape(-1, 1)
            detections_for_tracker = np.hstack((detections.xyxy, scores))[mask]

        online_targets = tracker.update(
            detections_for_tracker, (height, width), (height, width), fps
        )

        #Per-track flow energy 
        flow_energies = {}
        if online_targets:
            for t in online_targets:
                if t.track_id is None:
                    continue
                box_energy = 0.0
                if mag_small is not None:
                    x1, y1, bw, bh = t.tlwh
                    x1_s = max(0, int(x1 * flow_scale))
                    y1_s = max(0, int(y1 * flow_scale))
                    x2_s = min(flow_w, int((x1 + bw) * flow_scale))
                    y2_s = min(flow_h, int((y1 + bh) * flow_scale))
                    roi  = mag_small[y1_s:y2_s, x1_s:x2_s]
                    box_energy = (
                        float(np.percentile(roi, predict_cfg["flow_percentile"])) * 100.0
                        / (np.hypot(bw, bh) + 1e-6)
                        if roi.size > 0 else 0.0
                    )
                t.curr_flow_energy  = box_energy
                flow_energies[t.track_id] = box_energy

        #Physics Anomaly Detector 
        anomalous_tracks = {}
        if online_targets:
            anomalous_tracks = tracker.anomaly_detector.update(tracker, fps, flow_energies)
            for t in online_targets:
                t.is_anomaly     = t.track_id in anomalous_tracks
                t.anomaly_score  = anomalous_tracks.get(t.track_id, 0.0)
                if t.is_anomaly and t.curr_flow_energy < flow_energy_veto:
                    t.anomaly_score *= 0.5   #soft veto logic for low flow energy

        # Find the type of crash if physics anomaly is detected
        detected_accidents = crash_detector.update(online_targets, frame_id, fps)

        current_priority = 99
        cx, cy = 0.5 * width, 0.5 * height
        heuristic_type = "single"

        if detected_accidents:
            acc_cx, acc_cy = detected_accidents[0]["location"]

            if flow_triggered:
                current_priority = 0
                cx, cy = acc_cx, acc_cy
                heuristic_type = detected_accidents[0]["type"]
                physics_frame_count = physics_confirm_frames  # instant confirm
            else:
                if physics_candidate is None:
                    physics_candidate = {
                        "cx": acc_cx, "cy": acc_cy,
                        "type": detected_accidents[0]["type"],
                        "frame_id": frame_id,
                    }
                    physics_frame_count = 1
                else:
                    physics_frame_count += 1

                if physics_frame_count >= physics_confirm_frames:
                    current_priority = 1
                    cx, cy = physics_candidate["cx"], physics_candidate["cy"]
                    heuristic_type = physics_candidate["type"]
        else:
            physics_candidate   = None
            physics_frame_count = 0

            if flow_triggered:
                current_priority = 2
                best_track = max(
                    (t for t in online_targets if getattr(t, "curr_flow_energy", 0) > flow_energy_veto),
                    key=lambda t: t.curr_flow_energy,
                    default=None,
                )
                if best_track:
                    bx, by, bw, bh = best_track.tlwh
                    cx, cy = bx + bw / 2, by + bh / 2
                else:
                    cx, cy = flow_cx, flow_cy


        if current_priority < best_priority:
            best_priority  = current_priority
            accident_time  = frame_id / fps

            before_img = image_buffer[0]  

            if vlm_model is not None:
                final_type = classify_crash_images(vlm_model, vlm_processor, before_img, image_pil.copy())
            else:
                final_type = heuristic_type

            norm_cx = float(np.clip(cx / width,  0.0, 1.0))
            norm_cy = float(np.clip(cy / height, 0.0, 1.0))
            final_prediction = {
                "time": accident_time, "cx": norm_cx, "cy": norm_cy, "type": final_type,
            }
            print(
                f"New Best | t={accident_time:.2f}s  type={final_type} "
                f"priority={best_priority}  loc=({norm_cx:.3f},{norm_cy:.3f})"
            )

            if best_priority == 0:
                break

    if best_priority == 99 and abs_max_flow_data is not None:
        print("  Safety net: using absolute peak-flow frame.")
        if vlm_model is not None:
            final_type = classify_crash_images(
                vlm_model, vlm_processor,
                abs_max_flow_data["before_img"],
                abs_max_flow_data["impact_img"],
            )
        else:
            final_type = "single"

        final_prediction = {
            "time": abs_max_flow_data["time"],
            "cx":   float(np.clip(abs_max_flow_data["cx"] / width,  0.0, 1.0)),
            "cy":   float(np.clip(abs_max_flow_data["cy"] / height, 0.0, 1.0)),
            "type": final_type,
        }

    return (
        f"videos/{video_path.name}",
        final_prediction["time"],
        final_prediction["cx"],
        final_prediction["cy"],
        final_prediction["type"],
    )

def run_test(model, dataloader, valid_classes, inv_class_map, predict_cfg, model_cfg, vlm_model, vlm_processor):

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

    submission_data = []

    for data in dataloader:
        q_raw = data.get("quality", ["Good"])
        if isinstance(q_raw, (list, tuple)): q_raw  = q_raw[0]
        quality_str = q_raw.decode("utf-8") if isinstance(q_raw, bytes) else str(q_raw)
        print(f"\n--- Processing Video with Quality: {quality_str} ---")
        is_poor = "Poor" in quality_str

        result = process_video(video_data= data, model = model,
            tracker = tracker_poor if is_poor else tracker_good,
            crash_detector = crash_detector_poor if is_poor else crash_detector_good,
            valid_classes = valid_classes, inv_class_map = inv_class_map,
            predict_cfg = predict_cfg_poor if is_poor else predict_cfg,
            vlm_model = vlm_model, vlm_processor= vlm_processor,
        )
        submission_data.append(result)

    csv_file_path = "submission.csv"
    with open(csv_file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "accident_time", "center_x", "center_y", "type"])
        for path, acc_time, cx, cy, acc_type in submission_data:
            writer.writerow([path, f"{acc_time:.2f}", f"{cx:.3f}", f"{cy:.3f}", acc_type])

    print(f"\nSubmission written: {csv_file_path}  ({len(submission_data)} videos)")
