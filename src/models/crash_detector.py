import math

class CrashDetector:
    def __init__(self, accident_cooldown, inv_class_map, proximity_ratio=0.3):
        self.accident_cooldown = accident_cooldown
        self.inv_class_map = inv_class_map
        self.last_global_accident_time = -self.accident_cooldown
        self.proximity_ratio = proximity_ratio 

    def classify_accident_type(self, main_track, surrounding_tracks):
        mx, my, mw, mh = main_track.tlwh
        main_center_x, main_center_y = mx + (mw / 2), my + (mh / 2)
        
        margin_w = mw * self.proximity_ratio
        margin_h = mh * self.proximity_ratio
        
        mx1, mx2 = mx - margin_w, mx + mw + margin_w
        my1, my2 = my - margin_h, my + mh + margin_h
        
        candidates = []
        
        for other_t in surrounding_tracks:
            if other_t.track_id == main_track.track_id: continue
            
            ox, oy, ow, oh = other_t.tlwh
            ox1, ox2 = ox, ox + ow
            oy1, oy2 = oy, oy + oh
            
            if (mx1 < ox2 and mx2 > ox1 and my1 < oy2 and my2 > oy1):
                other_center_x, other_center_y = ox + (ow / 2), oy + (oh / 2)
                dist = math.hypot(main_center_x - other_center_x, main_center_y - other_center_y)
                candidates.append((dist, other_t))
                
        if not candidates:
            return 3 # Single 

        candidates.sort(key=lambda x: x[0])
        colliding_track = candidates[0][1]

        v1_x, v1_y = getattr(main_track, 'velocity_x', 0), getattr(main_track, 'velocity_y', 0)
        v2_x, v2_y = getattr(colliding_track, 'velocity_x', 0), getattr(colliding_track, 'velocity_y', 0)
        
        angle1 = math.degrees(math.atan2(v1_y, v1_x))
        angle2 = math.degrees(math.atan2(v2_y, v2_x))
        
        angle_diff = abs((angle1 - angle2 + 180) % 360 - 180)
        
        if angle_diff > 135: 
            return 0  # Head-on
        elif 45 <= angle_diff <= 135:
            return 4  # T-bone
        else:
            # Parallel impacts
            ox, oy, ow, oh = colliding_track.tlwh
            other_center_x = ox + (ow / 2)
            
            lateral_distance = abs(main_center_x - other_center_x)
            
            if lateral_distance < (mw * 0.75): 
                return 1  # Rear-end
            else:
                return 2  # Sideswipe

    def update(self, online_targets, frame_id, fps):
        detected_accidents = []
        current_time = frame_id / fps
        
        if current_time - self.last_global_accident_time < self.accident_cooldown:
            return detected_accidents

        for t in online_targets:
            if t.is_anomaly:
                accident_class_id = self.classify_accident_type(t, online_targets)
                predicted_type = self.inv_class_map.get(accident_class_id, "Unknown")
                
                x1, y1, w, h = t.tlwh
                center_x, center_y = x1 + w/2, y1 + h/2
                
                accident_info = {
                    "type": predicted_type,
                    "time": current_time,
                    "location": (center_x, center_y),
                    "track_id": t.track_id
                }
                detected_accidents.append(accident_info)
        
        if detected_accidents:
            self.last_global_accident_time = current_time
            return [detected_accidents[0]]
            
        return []