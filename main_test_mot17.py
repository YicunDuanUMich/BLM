import numpy as np
import argparse
import os.path

from tracker.my_tracker import MyTracker
from tracker.ori_tracker import BYTETracker
from norfair import metrics, video

parser = argparse.ArgumentParser("STATS 507 Final Project")

parser.add_argument(
    "dataset_path",
    type=str,
    nargs="?",
    help="path to the MOT Challenge train dataset folder (test dataset doesn't provide labels)",
)
parser.add_argument(
    "--save-pred",
    dest="save_pred",
    action="store_true",
    help="generate a text file with your predictions",
)
parser.add_argument(
    "--save-metrics",
    dest="save_metrics",
    action="store_true",
    help="generate a text file with your MOTChallenge metrics results",
)
parser.add_argument(
    "--output-path",
    dest="output_path",
    type=str,
    nargs="?",
    default="./outputs",
    help="output path",
)
parser.add_argument(
    "--select-sequences",
    dest="select_sequences",
    type=str,
    nargs="+",
    help="if you want to select a subset of sequences in your dataset path. "
         "Insert the names of the sequences you want to process.",
)
parser.add_argument(
    "--use-my-tracker",
    dest="use_my_tracker",
    action="store_true",
    help="whether to use my tracker.",
)
parser.add_argument(
    "--use-reid",
    dest="use_reid",
    action="store_true",
    help="whether to enable the reid for my tracker.",
)
parser.add_argument(
    "--use-level-match",
    dest="use_level_match",
    action="store_true",
    help="whether to enable the level match for my tracker.",
)

# MyTrack arguments
parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                    help="threshold for filtering out boxes of which aspect ratio are above the given value.")
parser.add_argument("--use_bbox_filters", default=False, action="store_true",
                    help="use ByteTrack bbox size and dimensions ratio filters")

args = parser.parse_args()

output_path = args.output_path

if args.save_metrics:
    print("Saving metrics file at " + output_path)
if args.save_pred:
    print("Saving predictions files at " + output_path)
if args.use_my_tracker:
    print("Using my tracker")
else:
    print("Using original ByteTrack")
if args.use_reid:
    print("Using ReID")
if args.use_level_match:
    print("Using Level Match")


class PartialStaticTracker:
    def __init__(self, estimate, obj_id, live_points):
        self.estimate = estimate
        self.id = obj_id
        self.live_points = live_points


class ArgsByte:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self.mot20 = False
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh


if __name__ == '__main__':
    if args.select_sequences is None:
        sequences_paths = [f.path for f in os.scandir(args.dataset_path) if f.is_dir()]
    else:
        sequences_paths = [
            os.path.join(args.dataset_path, f) for f in args.select_sequences
        ]

    accumulator = metrics.Accumulators()

    for input_path in sequences_paths:
        # Search vertical resolution in seqinfo.ini
        seqinfo_path = os.path.join(input_path, "seqinfo.ini")
        info_file = metrics.InformationFile(file_path=seqinfo_path)

        all_detections = metrics.DetectionFileParser(
            input_path=input_path, information_file=info_file
        )

        video_file = video.VideoFromFrames(
            input_path=input_path, save_path=output_path, information_file=info_file,
            make_video=False
        )

        if args.save_pred:
            predictions_text_file = metrics.PredictionsTextFile(
                input_path=input_path, save_path=output_path, information_file=info_file
            )

        if args.use_my_tracker:
            tracker = MyTracker(args,
                                frame_rate=info_file.search("frameRate"),
                                use_reid=args.use_reid,
                                use_level=args.use_level_match,
                                use_optical_flow=False,
                                only_pedestrian=False)
        else:
            args_byte = ArgsByte(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
            tracker = BYTETracker(args_byte, info_file.search("frameRate"))

        accumulator.create_accumulator(input_path=input_path, information_file=info_file)

        img_size = [info_file.search("imHeight"), info_file.search("imWidth")]
        img_info = {
            "height": info_file.search("imHeight"),
            "width": info_file.search("imWidth"),
        }

        byte_tracked_objects = []
        for frame_number, detections in enumerate(all_detections):
            byte_detections = []
            img_info["raw_img"] = next(video_file)
            for det in detections:
                byte_det = np.append(det.points.reshape((1, -1)), det.scores[0])
                byte_detections.append(byte_det)

            if len(byte_detections) > 0:
                if args.use_my_tracker:
                    byte_tracked_objects = tracker.update(
                        np.array(byte_detections), img_info, tuple(img_size)
                    )
                else:
                    byte_tracked_objects = tracker.update(
                        np.array(byte_detections), img_size, tuple(img_size)
                    )

            tracked_objects = []
            for obj in byte_tracked_objects:
                box = obj.tlbr.reshape((2, 2))
                tracked_objects.append(
                    PartialStaticTracker(box, obj.track_id, np.array([True]))
                )

            # Update output text file
            if args.save_pred:
                predictions_text_file.update(predictions=tracked_objects)

            accumulator.update(predictions=tracked_objects)

    accumulator.compute_metrics()
    accumulator.print_metrics()

    if args.save_metrics:
        if args.use_my_tracker:
            metrics_folder = "metrics_my_tracker"
            if args.use_reid:
                metrics_folder += "_reid"
            if args.use_level_match:
                metrics_folder += "_level_match"
        else:
            metrics_folder = "metrics_bytetrack"

        accumulator.save_metrics(save_path=os.path.join(output_path, metrics_folder))
