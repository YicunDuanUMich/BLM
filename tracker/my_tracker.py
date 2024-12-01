import numpy as np

from .kalman_filter import KalmanFilter
from . import matching
from .track import BaseTrack, TrackState
from .reid_estimator import FastReIDEstimator
from .track import EnhancedTrack


class MyTracker(object):
    """
    This is a tracker based on ByteTrack.
    """

    def __init__(self, args, frame_rate=30, use_reid=False, use_level=False, use_optical_flow=False):
        self.tracked_stracks = []  # type: list[EnhancedTrack]
        self.lost_stracks = []  # type: list[EnhancedTrack]
        self.removed_stracks = []  # type: list[EnhancedTrack]
        BaseTrack.clear_count()

        self.frame_id = 0

        self.use_reid = use_reid
        self.use_level = use_level
        self.use_optical_flow = use_optical_flow

        assert not (self.use_level and not self.use_reid), \
            "if you set use_level to True, you should also set use_reid to True"

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        if self.use_reid:
            self.reid_estimator = FastReIDEstimator(config_file="fast-reid/configs/MOT17/sbs_S50.yml",
                                                    weights_path="assets/weights/mot17_sbs_S50.pth",
                                                    device="cuda")

        if self.use_level:
            self.detection_thresh_levels = [(0.0, 0.1), (0.1, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 1.0)]
            self.reid_factors = [0.85, 0.95, 0.98]
        else:
            self.detection_thresh_levels = [(0.0, 0.1), (0.1, 0.5), (0.5, 1.0)]
            self.reid_factors = [0.98]

    def _estimate_dists(self, tracks, detections, use_reid, reid_factor):
        # This comment contains the code for element-wise cost minimization method proposed in BoT-SORT
        # If you are interested in this method, you can uncomment the below codes and slightly adjust the program
        # iou_dists = matching.iou_distance(tracks, detections)
        # iou_dists_mask = (iou_dists > 0.5)
        # iou_dists = matching.fuse_score(iou_dists, detections)

        # emb_dists = matching.embedding_distance(tracks, detections) / 2.0
        # emb_dists[emb_dists > 0.25] = 1.0
        # emb_dists[iou_dists_mask] = 1.0
        #
        # return np.minimum(iou_dists, emb_dists)

        if use_reid:
            fuse_dists = matching.embedding_distance(tracks, detections)
            return matching.fuse_motion(self.kalman_filter, fuse_dists, tracks, detections, lambda_=reid_factor)
        else:
            iou_dists = matching.iou_distance(tracks, detections)
            return matching.fuse_score(iou_dists, detections)

    def update(self, detector_outputs, img_info, test_img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        unconfirmed_stracks = []  # type: list[EnhancedTrack]
        confirmed_stracks = []  # type: list[EnhancedTrack]

        # Find pedestrians
        if len(detector_outputs) > 0:
            if detector_outputs.shape[1] == 5:
                detector_outputs = detector_outputs[detector_outputs[:, -1].astype(int) == 0]
            else:
                detector_outputs = detector_outputs.cpu().numpy()
                detector_outputs = detector_outputs[detector_outputs[:, -1].astype(int) == 0]

        if len(detector_outputs) > 0:
            if detector_outputs.shape[1] == 5:
                scores = detector_outputs[:, 4]
                bboxes = detector_outputs[:, :4]
            else:
                scores = detector_outputs[:, 4] * detector_outputs[:, 5]
                bboxes = detector_outputs[:, :4]  # x1y1x2y2

            # Resize bboxes
            img_h, img_w = img_info["height"], img_info["width"]
            scale = min(test_img_size[0] / float(img_h),
                        test_img_size[1] / float(img_w))
            bboxes /= scale

            detection_levels = []
            for level_idx, detection_thresh_values in enumerate(self.detection_thresh_levels):
                if level_idx == 0:
                    continue

                lower_bound = detection_thresh_values[0]
                higher_bound = detection_thresh_values[1]
                not_high_items = (scores <= higher_bound)
                not_low_items = (scores > lower_bound)
                inter_items = np.logical_and(not_low_items, not_high_items)
                inter_bboxes = bboxes[inter_items]
                inter_scores = scores[inter_items]

                if len(inter_bboxes) > 0:
                    if self.use_reid:
                        inter_features = self.reid_estimator.inference(image=img_info["raw_img"], bboxes=inter_bboxes)
                        inter_detections = [EnhancedTrack(EnhancedTrack.tlbr_to_tlwh(tlbr), s, f)
                                            for (tlbr, s, f) in zip(inter_bboxes, inter_scores, inter_features)]
                    else:
                        inter_detections = [EnhancedTrack(EnhancedTrack.tlbr_to_tlwh(tlbr), s)
                                            for (tlbr, s) in zip(inter_bboxes, inter_scores)]
                else:
                    inter_detections = []

                detection_levels.append(inter_detections)

            detection_levels.reverse()
            self.reid_factors.reverse()

            ''' Add newly detected tracklets to unconfirmed_stracks'''
            for track in self.tracked_stracks:
                if not track.is_activated:
                    unconfirmed_stracks.append(track)
                else:
                    confirmed_stracks.append(track)

            confirmed_unmatched_stracks = joint_stracks(confirmed_stracks, self.lost_stracks)
            # Predict the current location with KF
            EnhancedTrack.multi_predict(confirmed_unmatched_stracks)

            ''' Level Matching '''
            high_u_detections = []
            for level_idx, detections in enumerate(detection_levels[:-1]):
                dists = self._estimate_dists(confirmed_unmatched_stracks, detections,
                                             use_reid=self.use_reid, reid_factor=self.reid_factors[level_idx])
                matches_idx, u_track_idx, u_detection_idx = matching.linear_assignment(dists, thresh=0.8)

                for track_idx, det_idx in matches_idx:
                    track = confirmed_unmatched_stracks[track_idx]
                    det = detections[det_idx]
                    if track.state == TrackState.Tracked:
                        track.update(det, self.frame_id, frame=img_info["raw_img"] if self.use_optical_flow else None)
                        activated_starcks.append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id=False,
                                          frame=img_info["raw_img"] if self.use_optical_flow else None)
                        refind_stracks.append(track)

                confirmed_unmatched_stracks = [confirmed_unmatched_stracks[i] for i in u_track_idx]
                for u_det_idx in u_detection_idx:
                    high_u_detections.append(detections[u_det_idx])

            ''' The end of Level Matching '''
            # Associate the unmatched tracks to the low score detections
            remaining_confirmed_unmatched_tracked_stracks = \
                [track for track in confirmed_unmatched_stracks if track.state == TrackState.Tracked]

            dists = matching.iou_distance(remaining_confirmed_unmatched_tracked_stracks, detection_levels[-1])
            matches_idx, u_track_idx, u_detection_idx = matching.linear_assignment(dists, thresh=0.5)

            for track_idx, det_idx in matches_idx:
                track = remaining_confirmed_unmatched_tracked_stracks[track_idx]
                det = detection_levels[-1][det_idx]
                # if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, frame=img_info["raw_img"] if self.use_optical_flow else None)
                activated_starcks.append(track)
                # else:
                #     track.re_activate(det, self.frame_id, new_id=False)
                #     refind_stracks.append(track)

            for track_idx in u_track_idx:
                track = remaining_confirmed_unmatched_tracked_stracks[track_idx]
                # if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
                # else:
                #     print("WARNING")

            ''' Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            dists = self._estimate_dists(unconfirmed_stracks, high_u_detections, use_reid=False, reid_factor=0.0)
            matches_idx, u_track_idx, u_detection_idx = matching.linear_assignment(dists, thresh=0.7)

            for track_idx, det_idx in matches_idx:
                unconfirmed_stracks[track_idx].update(high_u_detections[det_idx], self.frame_id,
                                                      frame=img_info["raw_img"] if self.use_optical_flow else None)
                activated_starcks.append(unconfirmed_stracks[track_idx])
            for track_idx in u_track_idx:
                track = unconfirmed_stracks[track_idx]
                track.mark_removed()
                removed_stracks.append(track)

            """ Init new tracks """
            for new_idx in u_detection_idx:
                track = high_u_detections[new_idx]
                if track.score < 0.6:
                    continue

                track.activate(self.kalman_filter, self.frame_id,
                               frame=img_info["raw_img"] if self.use_optical_flow else None)
                activated_starcks.append(track)

        else:
            for track in self.tracked_stracks:
                if not track.is_activated:
                    unconfirmed_stracks.append(track)
                else:
                    confirmed_stracks.append(track)

            confirmed_unmatched_stracks = joint_stracks(confirmed_stracks, self.lost_stracks)
            # Predict the current location with KF
            EnhancedTrack.multi_predict(confirmed_unmatched_stracks)

        """ Update state"""
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

        return [track for track in self.tracked_stracks if track.is_activated]


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
