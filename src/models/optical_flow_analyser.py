import cv2
import numpy as np
from collections import deque

class OpticalFlowAnalyser:

    def __init__(self, fps, cfg):
        self.fps = fps
        baseline_secs         = cfg.get("flow_baseline_secs", 4.0)
        self.baseline_maxlen  = max(int(baseline_secs * fps), 10)
        self.motion_history   = deque(maxlen=self.baseline_maxlen)

        self.flow_spike_thres   = cfg.get("flow_spike_thres", 3.5)     
        self.sustain_frames     = cfg.get("flow_sustain_frames", 3)  
        self.flow_percentile    = cfg.get("flow_percentile", 85)
        self.concentration_ratio = cfg.get("flow_concentration_ratio", 2.5)
        self.min_time_s         = cfg.get("flow_min_time_s", 2.0)

        self._consecutive_spikes = 0
        self._current_score      = 0.0

    def update(self, mag_small, frame_id, flow_scale, width, height):

        curr_score  = float(np.percentile(mag_small, self.flow_percentile))
        peak_score  = float(np.percentile(mag_small, 95))
        self._current_score = curr_score

        _, _, _, max_loc = cv2.minMaxLoc(mag_small)
        cx = max_loc[0] / flow_scale
        cy = max_loc[1] / flow_scale

        if frame_id / self.fps < self.min_time_s:
            self.motion_history.append(curr_score)
            self._consecutive_spikes = 0
            return False, cx, cy, 0.0

        is_spike = False
        if len(self.motion_history) >= max(int(self.fps), 5):
            baseline_arr  = np.array(self.motion_history)

            clip_val      = np.percentile(baseline_arr, 90)
            baseline_clean = baseline_arr[baseline_arr <= clip_val]

            baseline_mean = float(np.mean(baseline_clean)) if len(baseline_clean) else float(np.mean(baseline_arr))
            baseline_std  = float(np.std(baseline_clean))  + 1e-6

            z = (curr_score - baseline_mean) / baseline_std

            mean_energy = float(np.mean(mag_small)) + 1e-6
            concentration = peak_score / mean_energy

            is_spike = (z > self.flow_spike_thres) and (concentration > self.concentration_ratio)

        self.motion_history.append(curr_score)

        if is_spike:
            self._consecutive_spikes += 1
        else:
            self._consecutive_spikes = 0

        triggered = self._consecutive_spikes >= self.sustain_frames
        return triggered, cx, cy, curr_score

    def reset(self, fps):
        self.fps = fps
        self.motion_history.clear()
        self._consecutive_spikes = 0
        self._current_score = 0.0
