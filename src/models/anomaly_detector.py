import numpy as np
from collections import defaultdict, deque

class AnomalyDetector:

    def __init__(self, predict_cfg):
        self.fps = 30
        self.dt = 1.0 / self.fps
        
        self.accl_thres = predict_cfg.get('accl_thres', -4.0) 
        self.angle_thres = predict_cfg.get('angle_thres', np.pi / 6)  
        self.min_history_len     = predict_cfg.get('min_history_len', 20)
        self.accel_window        = predict_cfg.get('accel_window', 5)
        self.closing_speed_thres = predict_cfg.get('closing_speed_thres', 10.0)
        self.angle_step          = predict_cfg.get('angle_step', 5)
        self.proximity_ratio     = predict_cfg.get('proximity_ratio', 0.3)
        self.flow_spike_thres    = predict_cfg.get('flow_spike_thres', 3.0)
        self.alignment = predict_cfg.get('alignment', 0.90)

        self.anomaly_threshold = predict_cfg.get('anomaly_threshold', 1.5)
        self.score_decay = predict_cfg.get('score_decay', 0.9) 
        
        self.weights = {
            'traj': 1.2, 
            'ang': 1.0,  
            'flow': 0.8, 
            'acc': 0.6,   
        }
        
        self.track_history = defaultdict(lambda: deque(maxlen=predict_cfg['track_buffer']))

        self.temporal_scores = defaultdict(float)


    def update(self, tracker, fps, flow_energies):
        self.fps = fps
        self.dt = 1.0 / self.fps
        frame_num = tracker.frame_id
        
        for track in tracker.tracked_stracks:
            if not track.is_activated:
                continue
            c_x, c_y = track.xywh[:2]
            vx, vy = track.mean[4:6]
            
            h = track.mean[3]
            w = track.mean[2] * h 
            
            e = flow_energies.get(track.track_id, 0.0)
            self.track_history[track.track_id].append((c_x, c_y, frame_num, vx, vy, w, h, e))

        anomalies = self.detect_anomalies(tracker.tracked_stracks)
        
        self.clean_up_history(tracker.removed_stracks)
        return anomalies

    def _get_candidate_pairs(self, tracks):
        n = len(tracks)
        if n < 2: 
            return []
            
        boxes = np.array([t.tlwh for t in tracks])
        
        margin_w = boxes[:, 2] * self.proximity_ratio
        margin_h = boxes[:, 3] * self.proximity_ratio

        x1 = boxes[:, 0] - margin_w
        y1 = boxes[:, 1] - margin_h
        x2 = boxes[:, 0] + boxes[:, 2] + margin_w
        y2 = boxes[:, 1] + boxes[:, 3] + margin_h

        xx1 = np.maximum(x1[:, None], x1)
        yy1 = np.maximum(y1[:, None], y1)
        xx2 = np.minimum(x2[:, None], x2)
        yy2 = np.minimum(y2[:, None], y2)

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        proximity_matrix = (w > 0) & (h > 0)
        i, j = np.where(np.triu(proximity_matrix, k=1))
        return list(zip(i, j))

    def detect_anomalies(self, tracked_stracks):
        anomalous_tracks = {}
        if len(tracked_stracks) == 0: return anomalous_tracks
        
        curr_evidence = defaultdict(float)

        if len(tracked_stracks) >= 2:
            candidate_pairs = self._get_candidate_pairs(tracked_stracks)
            for i, j in candidate_pairs:
                track1, track2 = tracked_stracks[i], tracked_stracks[j]
                tid1, tid2 = track1.track_id, track2.track_id
                
                beta = self.calculate_trajectory_anomaly(track1, track2)
                
                curr_evidence[tid1] = max(curr_evidence[tid1], beta)
                curr_evidence[tid2] = max(curr_evidence[tid2], beta)

        for track in tracked_stracks:
            tid = track.track_id

            acc = self.calculate_acceleration_anomaly(tid)
            ang = self.calculate_angle_anomaly(tid)
            flow = self.calculate_flow_anomaly(tid)
            traj = curr_evidence[tid] 
            
            frame_score = (
                traj * self.weights['traj'] +
                ang * self.weights['ang'] +
                flow * self.weights['flow'] +
                acc * self.weights['acc']
            )
            
            new_score = (self.temporal_scores[tid] * self.score_decay) + frame_score
            self.temporal_scores[tid] = new_score
            
            if new_score >= self.anomaly_threshold:
                anomalous_tracks[tid] = new_score

        return anomalous_tracks

    def calculate_acceleration_anomaly(self, track_id):
        history = self.track_history[track_id]
        if len(history) < self.min_history_len:
            return 0.0

        recent_len = self.accel_window + 1
        recent_hist = list(history)[-recent_len:]
        
        vx = np.array([item[3] for item in recent_hist])
        vy = np.array([item[4] for item in recent_hist])
        w = np.array([item[5] for item in recent_hist])
        h = np.array([item[6] for item in recent_hist])
        
        depth_scale = np.hypot(w, h) + 1e-6
        norm_vx = vx / depth_scale 
        norm_vy = vy / depth_scale
        
        speeds = np.hypot(norm_vx, norm_vy)
        accelerations = np.diff(speeds) / self.dt

        if np.min(accelerations) < self.accl_thres:
            return 1.0
        return 0.0
    
    def calculate_trajectory_anomaly(self, track1, track2):
        
        p1, p2 = track1.mean[:2], track2.mean[:2]
        v1, v2 = track1.mean[4:6], track2.mean[4:6]
        
        h1 = track1.mean[3]
        w1 = track1.mean[2] * h1
        h2 = track2.mean[3]
        w2 = track2.mean[2] * h2

        dx_curr = abs(p1[0] - p2[0])
        dy_curr = abs(p1[1] - p2[1])
        min_dx = (w1 + w2) / 2.0
        min_dy = (h1 + h2) / 2.0
        
        if dx_curr < (min_dx * 0.5) and dy_curr < (min_dy * 0.5):
            return 1.0

        dist_vec = p2 - p1
        rel_vel = v1 - v2
        v_rel_sq = np.dot(rel_vel, rel_vel)
        
        if v_rel_sq < 1e-6: 
            return 0.0

        closing_speed = np.dot(rel_vel, dist_vec) / (np.linalg.norm(dist_vec) + 1e-6)
        if closing_speed <= self.closing_speed_thres:
            return 0.0

        t_closest = np.dot(dist_vec, rel_vel) / v_rel_sq

        if 0 < t_closest < (self.fps*1.5):
            pos1_at_t = p1 + v1 * t_closest
            pos2_at_t = p2 + v2 * t_closest
            
            dx = abs(pos1_at_t[0] - pos2_at_t[0])
            dy = abs(pos1_at_t[1] - pos2_at_t[1])
            
            if dx < min_dx and dy < min_dy:
                
                rel_vel_norm = rel_vel / (np.linalg.norm(rel_vel) + 1e-6)
                dist_vec_norm = dist_vec / (np.linalg.norm(dist_vec) + 1e-6) 
                
                alignment = np.dot(rel_vel_norm, dist_vec_norm)

                if alignment > 0.90:
                    return 1.0

        return 0.0
        
    def calculate_angle_anomaly(self, track_id):
        history = self.track_history[track_id]
        
        if len(history) < (2 * self.angle_step + 1):
            return 0.0

        pos1 = np.array(history[-1 - 2 * self.angle_step][:2])
        pos2 = np.array(history[-1 - self.angle_step][:2])
        pos3 = np.array(history[-1][:2])

        vec1 = pos2 - pos1
        vec2 = pos3 - pos2

        dist1 = np.linalg.norm(vec1)
        dist2 = np.linalg.norm(vec2)
        
        w = history[-1][5]
        h = history[-1][6]
        depth_scale = np.hypot(w, h) + 1e-6
        
        dynamic_min_movement = max(1.0, depth_scale * 0.03)

        if dist1 < dynamic_min_movement or dist2 < dynamic_min_movement:
            return 0.0

        unit_vec1 = vec1 / dist1
        unit_vec2 = vec2 / dist2
        dot_product = np.dot(unit_vec1, unit_vec2)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

        if angle > self.angle_thres:
            return 1.0
            
        return 0.0
    
    def calculate_flow_anomaly(self, track_id):
        history = self.track_history[track_id]
        
        if len(history) < self.min_history_len:
            return 0.0
            
        recent_energy = history[-1][7]

        baseline_history = [item[7] for item in list(history)[:-1]]
        baseline_mean = np.mean(baseline_history)
        baseline_std = np.std(baseline_history) + 1e-6 
        
        baseline_mean = max(baseline_mean, 1.0) 
        
        z_score = (recent_energy - baseline_mean) / baseline_std
        
        if z_score > self.flow_spike_thres:
            return 1.0
            
        return 0.0

    def clean_up_history(self, removed_stracks):
        for track in removed_stracks:
            self.track_history.pop(track.track_id, None)