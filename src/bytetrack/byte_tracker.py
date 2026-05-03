import numpy as np
from .kalman_filter import KalmanFilter
from . import matching
from .base_track import BaseTrack, TrackState
from models.anomaly_detector import AnomalyDetector
from collections import deque
import torch
from models.ml_detector import AccidentTrajectoryPredictor

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    
    def __init__(self, tlwh, score, window_size=30):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.is_anomaly = False
        self.anomaly_score = 0.0
        self.score = score
        self.tracklet_len = 0
        self.history = deque(maxlen=window_size)
        self.accident_type = -1
        self.accident_type_score = 0.0

    def predict(self, fps):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance, 1/fps)

    @staticmethod
    def multi_predict(stracks, fps):
        if len(stracks) > 0:
            multi_mean = np.array([st.mean for st in stracks])
            multi_covariance = np.array([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance, 1/fps
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xywh(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, predict_cfg=None, model_cfg=None):
        self.track_thresh = predict_cfg['track_thresh']
        self.track_buffer = predict_cfg['track_buffer']
        self.match_thresh = predict_cfg['match_thresh']
        self.fuse_score = predict_cfg['fuse_score']
                 
        self.tracked_stracks = []  
        self.lost_stracks = []  
        self.removed_stracks = []  

        self.frame_id = 0
        frame_rate = 30
        self.det_thresh = self.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        
        self.anomaly_detector = None
        self.model = None
        self.use_ml_predictor = predict_cfg['use_ml_predictor']

        if not self.use_ml_predictor:
            self.anomaly_detector = AnomalyDetector(predict_cfg)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = AccidentTrajectoryPredictor(**model_cfg)
            self.model.load_state_dict(torch.load(predict_cfg['ckpt_path'], map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

    def update(self, output_results, img_info, img_size, fps):
        self.frame_id += 1
        # Dynamically scale allowed lost frames based on the current video's FPS
        self.max_time_lost = int(fps / 30.0 * self.track_buffer)
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
            
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, self.track_buffer) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        unconfirmed = []
        tracked_stracks = [] 
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool, fps)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.fuse_score:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, self.track_buffer) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
            
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.fuse_score:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
            
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        
        inv_w, inv_h = 1.0 / img_w, 1.0 / img_h
        for track in self.tracked_stracks:
            if track.is_activated and track.mean is not None:
                x, y, a, h, vx, vy, va, vh = track.mean
                
                w = a * h
                vw = va * h + a * vh
                
                if self.use_ml_predictor:
                    feature = np.array([
                        x * inv_w, y * inv_h, w * inv_w, h * inv_h,
                        vx * inv_w, vy * inv_h, vw * inv_w, vh * inv_h
                    ], dtype=np.float32)
                else:
                    feature = np.array([x, y, w, h, vx, vy, vw, vh], dtype=np.float32)
                    
                track.history.append(feature)

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # if not self.use_ml_predictor:
        #     if self.anomaly_detector:
        #         anomalous_tracks = self.anomaly_detector.update(self, fps)
        #         for track in output_stracks:
        #             if track.track_id in anomalous_tracks:
        #                 track.is_anomaly = True
        #                 track.anomaly_score = anomalous_tracks.get(track.track_id, 0.0)
        if self.use_ml_predictor:
            if self.model and len(output_stracks) > 0:
                candidate_tracks = [track for track in output_stracks if len(track.history) == self.track_buffer]

                if len(candidate_tracks) > 0:
                    histories = [list(track.history) for track in candidate_tracks]
                    x = torch.tensor(histories, dtype=torch.float32, device=self.device).unsqueeze(0)

                    with torch.no_grad():
                        outputs = self.model(x)

                    time_logits = outputs['time_logits'].squeeze(0)
                    window_anomaly_prob = torch.sigmoid(time_logits.max()).item()

                    if window_anomaly_prob > self.track_thresh:
                        location_logits = outputs['location_logits'].squeeze(0)
                        location_probs = torch.sigmoid(location_logits).cpu().numpy()

                        type_logits = outputs['type_logits'].squeeze(0)
                        type_probs = torch.softmax(type_logits, dim=-1).cpu().numpy()
                        
                        accident_type = np.argmax(type_probs)
                        accident_type_score = np.max(type_probs)

                        for i, track in enumerate(candidate_tracks):
                            track.anomaly_score = location_probs[i]
                            if track.anomaly_score > self.track_thresh:
                                track.is_anomaly = True
                                track.accident_type = accident_type
                                track.accident_type_score = accident_type_score
                            else: 
                                track.is_anomaly = False
                                track.accident_type = -1
                                track.accident_type_score = 0.0
                    else: 
                        for track in candidate_tracks:
                            track.is_anomaly = False
                            track.anomaly_score = 0.0
                            track.accident_type = -1
                            track.accident_type_score = 0.0

        return output_stracks

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        STrack.reset_id()


def joint_stracks(tlista, tlistb):
    # O(1) set lookup instead of dict manipulation
    exists = {t.track_id for t in tlista}
    return tlista + [t for t in tlistb if t.track_id not in exists]


def sub_stracks(tlista, tlistb):
    stracks_ids = {t.track_id for t in tlistb}
    return [t for t in tlista if t.track_id not in stracks_ids]


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    
    # Using sets for O(1) lookups instead of lists
    dupa, dupb = set(), set()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.add(q)
        else:
            dupa.add(p)
            
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb