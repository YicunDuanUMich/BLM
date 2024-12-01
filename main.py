import numpy as np
import cv2
import argparse
import time
import os

from video_processor.video_reader import VideoReader
from detector.yolo_detector import YOLODetector
from tracker.my_tracker import MyTracker
from video_processor.draw_trajectory_utils import plot_tracking, plot_trajectories
from video_processor.show_image_utils import show_image_with_resize
from loguru import logger


parser = argparse.ArgumentParser("STATS 507 Final Project")

# video input and outputs
parser.add_argument("-i", required=True, help="path to input video file")
parser.add_argument("-o", type=str, default=os.path.join(".", "output", "output.avi"), help="path to output")
parser.add_argument("--use_optical_flow", action="store_true", default=False)

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


if __name__ == '__main__':
    detector = YOLODetector(pretrained_weight_path=args.detector_pretrained_weight, model_name=args.detector_name,
                            confthre=args.detector_thresh, nmsthre=args.detector_nms_thresh)
    video_input_path = args.i
    video_output_path = os.path.join(".", "outputs", args.o)

    if not os.path.exists(os.path.join(".", "outputs")):
        os.makedirs(os.path.join(".", "outputs"))

    video_reader = VideoReader(video_input_path)
    if video_reader.video_cap is None:
        exit(-1)

    video_writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   video_reader.fps,
                                   (video_reader.width, video_reader.height))

    use_optical_flow = args.use_optical_flow
    if use_optical_flow:
        trajectories_canvas = None
    tracker = MyTracker(args, frame_rate=video_reader.fps,
                        use_reid=False, use_level=False,
                        use_optical_flow=use_optical_flow)

    frame_id = 0

    while True:
        ret, frame = video_reader.get_frame()
        if ret is False:
            break

        outputs, img_info = detector.inference(frame)
        time_detector = detector.det_time

        if outputs is not None:
            time_start_tracker = time.time()
            online_targets = tracker.update(outputs,
                                            img_info,
                                            detector.exp.test_size)
            time_tracker = time.time() - time_start_tracker
            online_tlwhs = []
            online_ids = []
            online_scores = []
            if use_optical_flow:
                online_pre_corner_points = []
                online_corner_points = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                if use_optical_flow:
                    pre_corner_points = t.pre_corner_points
                    corner_points = t.corner_points
                if args.use_bbox_filters:
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] < args.min_box_area and vertical:
                        continue
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                if use_optical_flow:
                    if pre_corner_points is not None:
                        online_pre_corner_points.append(pre_corner_points)
                    else:
                        online_pre_corner_points.append(np.ndarray(shape=(0, 1, 2)))
                    online_corner_points.append(corner_points)

            frame_fps = 1.0 / (time_tracker + time_detector) if (time_tracker + time_detector) > 0.0 else 30.0
            online_im = plot_tracking(img_info["raw_img"],
                                      online_tlwhs, online_ids, frame_id=frame_id + 1, fps=frame_fps)

            if use_optical_flow:
                if trajectories_canvas is None:
                    trajectories_canvas = np.zeros_like(online_im)

                trajectories_canvas = plot_trajectories(trajectories_canvas, online_ids,
                                                        online_pre_corner_points, online_corner_points)
                online_im = cv2.add(online_im, trajectories_canvas)

        else:
            online_im = img_info["raw_img"]

            if use_optical_flow:
                if trajectories_canvas is None:
                    trajectories_canvas = np.zeros_like(online_im)

                online_im = cv2.add(online_im, trajectories_canvas)

        if online_im.shape[1] > 1280:
            show_image_with_resize("frame", online_im, (1280, -1))
        else:
            cv2.imshow("frame", online_im)
        video_writer.write(online_im)

        cv_key = cv2.waitKey(1)
        if cv_key is ord('q'):
            break
        frame_id += 1

    video_writer.release()
    cv2.destroyAllWindows()
    logger.info("program exits")
