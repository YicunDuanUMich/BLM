import cv2
import numpy as np


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def plot_trajectories(trajectories_canvas,
                      obj_ids,
                      pre_corner_points_list, corner_points_list):
    canvas = np.ascontiguousarray(np.copy(trajectories_canvas))
    for i, id in enumerate(obj_ids):
        for new_corner_point, old_corner_point in zip(corner_points_list[i], pre_corner_points_list[i]):
            a, b = new_corner_point.ravel()
            c, d = old_corner_point.ravel()
            cv2.line(canvas, (int(a), int(b)), (int(c), int(d)), get_color(abs(id)), 2)

    return canvas


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))

    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 0),
                   thickness=text_thickness*3, lineType=cv2.LINE_AA)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, color,
                   thickness=text_thickness, lineType=cv2.LINE_AA)

    cv2.putText(im, 'Frame: %d FPS: %.2f Num: %d' % (frame_id, fps, len(tlwhs)),
               (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), thickness=6, lineType=cv2.LINE_AA)
    cv2.putText(im, 'Frame: %d FPS: %.2f Num: %d' % (frame_id, fps, len(tlwhs)),
               (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    return im
