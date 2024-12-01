import numpy as np
import argparse
import time
import os
import cv2
import gc
import sys

from video_processor.video_reader import VideoReader
from detector.yolo_detector import YOLODetector
from tracker.my_tracker import MyTracker
from video_processor.draw_trajectory_utils import plot_tracking, plot_trajectories
from loguru import logger

from ui_related.displayer_window import Ui_displayer_window

from PyQt6.QtWidgets import QWidget, QApplication, QDialog, QMessageBox, QFileDialog
from PyQt6.QtCore import Qt, QUrl, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QIcon


parser = argparse.ArgumentParser("Yicun Duan STATS 507 Final Project")

# YOLOX arguments
parser.add_argument("-dn", "--detector_name", default="yolox-m", type=str, choices=["yolox-m"],
                    help="model name [yolox-m]")
parser.add_argument("-w", "--detector_pretrained_weight", default=os.path.join(".", "assets", "weights", "yolox_m.pth"),
                    type=str, help="pretrained weights file for the model")
parser.add_argument("-t", "--detector_thresh", type=float, default=0.01, help="detector's confidence threshold")
parser.add_argument("-nms", "--detector_nms_thresh", type=float, default=0.4,
                    help="YOLOX's non-maximum suppression threshold threshold")

# MyTrack arguments
parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                    help="threshold for filtering out boxes of which aspect ratio are above the given value.")
parser.add_argument("--use_bbox_filters", default=False, action="store_true",
                    help="use ByteTrack bbox size and dimensions ratio filters")

args = parser.parse_args()

DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 600


class TrackerMethod:
    ByteTrackMethod = 0
    ByteTrackReIDMethod = 1
    ByteTrackReIDLevelMethod = 2


class ThreadState:
    Running = 0
    Stop = 1


class TrackingThread(QThread):
    change_pixmap = pyqtSignal(QImage)

    def __init__(self, parent):
        super().__init__(parent)
        self.window = parent  # type: Window
        self.cur_state = ThreadState.Stop

    def run(self) -> None:
        self.cur_state = ThreadState.Running
        while True:
            if self.window.program_exit_flag:
                self.window.video_writer.release()
                self.window.release_resource()
                self.cur_state = ThreadState.Stop
                break

            if self.window.pause_flag:
                self.cur_state = ThreadState.Stop
                break
                # while True:
                #     if not self.window.pause_flag:
                #         break

            if self.window.stop_flag:
                self.window.video_writer.release()
                self.window.release_resource()
                self.change_pixmap.emit(
                    QImage(os.path.join(".", "assets", "init_photo.jpg")).scaled(DISPLAY_WIDTH, DISPLAY_HEIGHT))
                self.cur_state = ThreadState.Stop
                break

            frame = self.window.generate_one_tracking_output()
            if frame is not None:
                self.window.video_writer.write(frame)
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_format_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                rescaled_qt_image = qt_format_image.scaled(DISPLAY_WIDTH, DISPLAY_HEIGHT)
                self.change_pixmap.emit(rescaled_qt_image)
            else:
                self.window.video_writer.release()
                self.window.release_resource()
                self.change_pixmap.emit(
                    QImage(os.path.join(".", "assets", "init_photo.jpg")).scaled(DISPLAY_WIDTH, DISPLAY_HEIGHT))
                self.cur_state = ThreadState.Stop

                self.window.ui.start_button.setEnabled(True)
                self.window.ui.pause_button.setEnabled(False)
                self.window.ui.continue_button.setEnabled(False)
                self.window.ui.stop_button.setEnabled(False)
                self.window.ui.tracker_combo_box.setEnabled(True)
                self.window.ui.optical_flow_check_box.setEnabled(True)
                self.window.ui.select_video_button.setEnabled(True)

                break


class Window(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_displayer_window()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon(os.path.join(".", "assets", "pedestrian_icon.jpg")))
        # with open(os.path.join(".", "ui_related", "resources", "QSS-master", "Aqua.qss"), "r") as f:
        #     self.setStyleSheet(f.read())
        self.ui.start_button.setEnabled(False)
        self.ui.pause_button.setEnabled(False)
        self.ui.continue_button.setEnabled(False)
        self.ui.stop_button.setEnabled(False)
        self.connect_signals_slots()

        self.detector = None
        self.tracker = None
        self.video_input_path = None
        self.video_output_path = None
        if not os.path.exists(os.path.join(".", "outputs")):
            os.makedirs(os.path.join(".", "outputs"))
        self.video_reader = None
        self.video_writer = None
        self.frame_id = 0
        self.trajectories_canvas = None

        self.tracker_method = TrackerMethod.ByteTrackMethod
        self.use_optical_flow_tracking = False

        self.pause_flag = False
        self.stop_flag = False
        self.program_exit_flag = False

        self.ui.image_label.setPixmap(
            QPixmap(os.path.join(".", "assets", "init_photo.jpg")).scaled(DISPLAY_WIDTH, DISPLAY_HEIGHT))
        self.display_thread = TrackingThread(self)
        self.display_thread.change_pixmap.connect(self.set_display_image)

    def connect_signals_slots(self):
        self.ui.tracker_combo_box.currentIndexChanged.connect(self.change_tracker)
        self.ui.optical_flow_check_box.stateChanged.connect(self.change_optical_flow_tracking)
        self.ui.select_video_button.clicked.connect(self.select_video_button_clicked)
        self.ui.start_button.clicked.connect(self.start_button_clicked)
        self.ui.pause_button.clicked.connect(self.pause_button_clicked)
        self.ui.continue_button.clicked.connect(self.continue_button_clicked)
        self.ui.stop_button.clicked.connect(self.stop_button_clicked)

    def init_engine(self):
        self.detector = YOLODetector(pretrained_weight_path=args.detector_pretrained_weight,
                                     model_name=args.detector_name,
                                     confthre=args.detector_thresh, nmsthre=args.detector_nms_thresh)
        self.video_reader = VideoReader(self.video_input_path)
        if self.video_reader.video_cap is None:
            logger.info("loading video fails")
            exit(-1)

        if self.tracker_method == TrackerMethod.ByteTrackMethod:
            use_reid = False
            use_level = False
        elif self.tracker_method == TrackerMethod.ByteTrackReIDMethod:
            use_reid = True
            use_level = False
        else:
            use_reid = True
            use_level = True

        self.tracker = MyTracker(args, frame_rate=self.video_reader.fps,
                                 use_reid=use_reid, use_level=use_level,
                                 use_optical_flow=self.use_optical_flow_tracking)

        self.frame_id = 0
        self.trajectories_canvas = None

        self.video_writer = cv2.VideoWriter(self.video_output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                            self.video_reader.fps,
                                            (self.video_reader.width, self.video_reader.height))

    def change_tracker(self, updated_index):
        if updated_index == 0:
            self.tracker_method = TrackerMethod.ByteTrackMethod
        elif updated_index == 1:
            self.tracker_method = TrackerMethod.ByteTrackReIDMethod
        elif updated_index == 2:
            self.tracker_method = TrackerMethod.ByteTrackReIDLevelMethod

    def change_optical_flow_tracking(self, updated_state):
        if self.ui.optical_flow_check_box.isChecked():
            self.use_optical_flow_tracking = True
        else:
            self.use_optical_flow_tracking = False

    def select_video_button_clicked(self):
        response, _ = QFileDialog.getOpenFileName(parent=self,
                                                  caption="Select a Video",
                                                  directory=os.getcwd(),
                                                  filter="MP4 (*.mp4);; AVI (*.avi)",
                                                  initialFilter="MP4 (*.mp4)")
        self.video_input_path = response
        url = QUrl.fromLocalFile(response)
        self.video_output_path = os.path.join(".", "outputs", url.fileName().split(".")[0] + "_output.mp4")
        logger.info("video input path: {}".format(self.video_input_path))
        logger.info("video output path: {}".format(self.video_output_path))
        self.ui.path_to_video_line_edit.setText(self.video_input_path)
        if os.path.isfile(self.video_input_path):
            self.ui.start_button.setEnabled(True)
        else:
            self.ui.start_button.setEnabled(False)

    def start_button_clicked(self):
        self.ui.start_button.setEnabled(False)
        self.ui.pause_button.setEnabled(True)
        self.ui.continue_button.setEnabled(False)
        self.ui.stop_button.setEnabled(True)
        self.ui.tracker_combo_box.setEnabled(False)
        self.ui.optical_flow_check_box.setEnabled(False)
        self.ui.select_video_button.setEnabled(False)

        self.init_engine()

        self.stop_flag = False
        self.pause_flag = False

        self.display_thread.start()

    def pause_button_clicked(self):
        self.ui.start_button.setEnabled(False)
        self.ui.pause_button.setEnabled(False)
        self.ui.continue_button.setEnabled(True)
        self.ui.stop_button.setEnabled(True)
        self.ui.tracker_combo_box.setEnabled(False)
        self.ui.optical_flow_check_box.setEnabled(False)
        self.ui.select_video_button.setEnabled(False)

        self.pause_flag = True

    def continue_button_clicked(self):
        self.ui.start_button.setEnabled(False)
        self.ui.pause_button.setEnabled(True)
        self.ui.continue_button.setEnabled(False)
        self.ui.stop_button.setEnabled(True)
        self.ui.tracker_combo_box.setEnabled(False)
        self.ui.optical_flow_check_box.setEnabled(False)
        self.ui.select_video_button.setEnabled(False)

        self.pause_flag = False

        self.display_thread.start()

    def stop_button_clicked(self):
        self.ui.start_button.setEnabled(True)
        self.ui.pause_button.setEnabled(False)
        self.ui.continue_button.setEnabled(False)
        self.ui.stop_button.setEnabled(False)
        self.ui.tracker_combo_box.setEnabled(True)
        self.ui.optical_flow_check_box.setEnabled(True)
        self.ui.select_video_button.setEnabled(True)

        self.stop_flag = True
        self.pause_flag = False

    def generate_one_tracking_output(self):
        ret, frame = self.video_reader.get_frame()
        if ret is False:
            return None

        outputs, img_info = self.detector.inference(frame)
        time_detector = self.detector.det_time

        if outputs is not None:
            time_start_tracker = time.time()
            online_targets = self.tracker.update(outputs,
                                                 img_info,
                                                 self.detector.exp.test_size)
            time_tracker = time.time() - time_start_tracker
            online_tlwhs = []
            online_ids = []
            online_scores = []
            if self.use_optical_flow_tracking:
                online_pre_corner_points = []
                online_corner_points = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                if self.use_optical_flow_tracking:
                    pre_corner_points = t.pre_corner_points
                    corner_points = t.corner_points
                if args.use_bbox_filters:
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] < args.min_box_area and vertical:
                        continue
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                if self.use_optical_flow_tracking:
                    if pre_corner_points is not None:
                        online_pre_corner_points.append(pre_corner_points)
                    else:
                        online_pre_corner_points.append(np.ndarray(shape=(0, 1, 2)))
                    online_corner_points.append(corner_points)

            frame_fps = 1.0 / (time_tracker + time_detector) if (time_tracker + time_detector) > 0.0 else 30.0
            online_im = plot_tracking(img_info["raw_img"],
                                      online_tlwhs, online_ids, frame_id=self.frame_id + 1, fps=frame_fps)

            if self.use_optical_flow_tracking:
                if self.trajectories_canvas is None:
                    self.trajectories_canvas = np.zeros_like(online_im)

                self.trajectories_canvas = plot_trajectories(self.trajectories_canvas, online_ids,
                                                             online_pre_corner_points, online_corner_points)
                online_im = cv2.add(online_im, self.trajectories_canvas)

        else:
            online_im = img_info["raw_img"]

            if self.use_optical_flow_tracking:
                if self.trajectories_canvas is None:
                    self.trajectories_canvas = np.zeros_like(online_im)

                online_im = cv2.add(online_im, self.trajectories_canvas)

        self.frame_id += 1

        return online_im

    @pyqtSlot(QImage)
    def set_display_image(self, image):
        self.ui.image_label.setPixmap(QPixmap.fromImage(image))

    def release_resource(self):
        del self.video_reader
        del self.video_writer
        del self.trajectories_canvas
        del self.detector
        del self.tracker
        gc.collect()

    def closeEvent(self, event):
        if self.display_thread.cur_state is ThreadState.Running:
            self.program_exit_flag = True

            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setWindowIcon(self.windowIcon())
            msg_box.setText("Please wait to exit")
            msg_box.setWindowTitle("Please wait")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            return_value = msg_box.exec()

            while True:
                if self.display_thread.cur_state is ThreadState.Stop:
                    break

        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec())
