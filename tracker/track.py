import numpy as np
from collections import deque
import cv2
from .kalman_filter import KalmanFilter
from .track_utils import lucas_kanad_pyramid

TRACK_FLAG = True


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 4


class BaseTrack(object):
    _count = 0

    def __init__(self):
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New

        # self.history = OrderedDict()
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0

        # multi-camera
        # self.location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def clear_count():
        BaseTrack._count = 0


class EnhancedTrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, init_appearance_feature=None, feature_hold_time=50):
        super().__init__()

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        # appearance features related
        self.smooth_appearance_feature = None
        self.cur_appearance_feature = None
        self.appearance_features = deque([], maxlen=feature_hold_time)
        if init_appearance_feature is not None:
            self.update_appearance_features(init_appearance_feature)
        self.smooth_appearance_feature_factor = 0.9

        # optical flow tracking related
        self.corner_points = None
        self.pre_corner_points = None
        self.pre_frame_gray = None

    def update_appearance_features(self, new_appearance_feature):
        new_appearance_feature /= np.linalg.norm(new_appearance_feature)
        self.cur_appearance_feature = new_appearance_feature

        if self.smooth_appearance_feature is None:
            self.smooth_appearance_feature = new_appearance_feature
        else:
            self.smooth_appearance_feature = self.smooth_appearance_feature_factor * self.smooth_appearance_feature + \
                                             (1 - self.smooth_appearance_feature_factor) * new_appearance_feature
        self.smooth_appearance_feature /= np.linalg.norm(self.smooth_appearance_feature)

        self.appearance_features.append(new_appearance_feature)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = EnhancedTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def _corner_points_refresh(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corner_points_mask = np.zeros(frame_gray.shape, dtype=np.uint8)

        def clamp_int(x, min_value, max_value):
            return int(max(min(max_value, x), min_value))

        top_left_point = (clamp_int(self.tlbr[0], 0.0, corner_points_mask.shape[1]),
                          clamp_int(self.tlbr[1], 0.0, corner_points_mask.shape[0]))
        bottom_right_point = (clamp_int(self.tlbr[2], 0.0, corner_points_mask.shape[1]),
                              clamp_int(self.tlbr[3], 0.0, corner_points_mask.shape[0]))

        corner_points_mask[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0]] += 1
        self.corner_points = cv2.goodFeaturesToTrack(frame_gray, mask=corner_points_mask,
                                                     maxCorners=100, qualityLevel=0.3, minDistance=7)
        self.pre_corner_points = None
        self.pre_frame_gray = frame_gray.copy()

    def _corner_points_update(self, frame):
        if TRACK_FLAG:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            updated_corner_points, status, errors = cv2.calcOpticalFlowPyrLK(self.pre_frame_gray, frame_gray,
                                                                             self.corner_points,
                                                                             None, winSize=(15, 15), maxLevel=2,
                                                                             criteria=(
                                                                                 cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                                                 10, 0.03))
            if updated_corner_points is not None:
                good_new_corner_points = updated_corner_points[status == 1]
                self.pre_corner_points = self.corner_points[status == 1]
                self.corner_points = good_new_corner_points.reshape(-1, 1, 2)
                self.pre_frame_gray = frame_gray.copy()

        else:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            updated_corner_points = lucas_kanad_pyramid(self.pre_frame_gray, frame_gray,
                                                        self.corner_points, window_size=3, num_levels=3)
            self.pre_corner_points = self.corner_points
            self.corner_points = updated_corner_points
            self.pre_frame_gray = frame_gray.copy()

    def activate(self, kalman_filter, frame_id, frame=None):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked

        if frame_id == 1:
            self.is_activated = True

        if frame is not None:
            self._corner_points_refresh(frame)

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False, frame=None):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        if new_track.cur_appearance_feature is not None:
            self.update_appearance_features(new_track.cur_appearance_feature)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        if frame is not None:
            self._corner_points_refresh(frame)

    def update(self, new_track, frame_id, frame=None):
        """
        Update a matched track
        :type new_track: EnhancedTrack
        :type frame_id: int
        :type frame: np.ndarray
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))

        if new_track.cur_appearance_feature is not None:
            self.update_appearance_features(new_track.cur_appearance_feature)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        if frame is not None:
            self._corner_points_update(frame)

    def mark_lost(self):
        self.state = TrackState.Lost
        self.corner_points = None
        self.pre_corner_points = None
        self.pre_frame_gray = None

    def mark_removed(self):
        self.state = TrackState.Removed
        self.corner_points = None
        self.pre_corner_points = None
        self.pre_frame_gray = None

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)
