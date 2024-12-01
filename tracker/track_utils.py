import cv2
import numpy as np


def reduce(image, level=1):
    result = np.copy(image)

    for _ in range(level - 1):
        result = cv2.pyrDown(result)

    return result


def expand(image, level=1):
    return cv2.pyrUp(np.copy(image))


def lucas_kanad_pyramid(old_frame, new_frame, feature_list, window_size=3, num_levels=3):
    current_level = num_levels

    while current_level > 0:
        old_frame_reduced = reduce(old_frame, current_level)
        new_frame_reduced = reduce(new_frame, current_level)

        # if current_level == 1:
        #     feature_list_reduced = feature_list
        # else:
        #     feature_list_reduced = np.floor(feature_list * (0.5 ** (current_level - 1)))

        if current_level == num_levels:
            u = np.zeros(old_frame_reduced.shape)
            v = np.zeros(old_frame_reduced.shape)
        else:
            u = 2 * expand(u)
            v = 2 * expand(v)

        if current_level != 1:
            dx, dy = lucas_kanade(old_frame_reduced, new_frame_reduced,
                                  window_size=window_size, feature_list=None)
        else:
            dx, dy = lucas_kanade(old_frame_reduced, new_frame_reduced,
                                  window_size=window_size, feature_list=feature_list)

        # dx, dy = horn_schunk(old_frame_reduced, new_frame_reduced)

        u = u + dx
        v = v + dy

        current_level -= 1

    new_feature_list = feature_list.copy()

    for feature_idx, feature in enumerate(feature_list):
        i, j = feature.ravel()
        i, j = int(i), int(j)  # i,j are floats initially

        x_change = u[j, i]
        y_change = v[j, i]

        new_feature_list[feature_idx, 0, 0] += int(x_change)
        new_feature_list[feature_idx, 0, 1] += int(y_change)

    return new_feature_list


def lucas_kanade(old_frame, new_frame, feature_list, window_size=3):
    if feature_list is None:
        Ix_full = np.zeros(old_frame.shape)
        Iy_full = np.zeros(old_frame.shape)
        It_full = np.zeros(old_frame.shape)

        Ix_full[1:-1, 1:-1] = (old_frame[1:-1, 2:] - old_frame[1:-1, :-2]) / 2
        Iy_full[1:-1, 1:-1] = (old_frame[2:, 1:-1] - old_frame[:-2, 1:-1]) / 2
        It_full[1:-1, 1:-1] = old_frame[1:-1, 1:-1] - new_frame[1:-1, 1:-1]

        params = np.zeros(old_frame.shape + (5,))
        params[..., 0] = cv2.GaussianBlur(Ix_full * Ix_full, (5, 5), 3)
        params[..., 1] = cv2.GaussianBlur(Iy_full * Iy_full, (5, 5), 3)
        params[..., 2] = cv2.GaussianBlur(Ix_full * Iy_full, (5, 5), 3)
        params[..., 3] = cv2.GaussianBlur(Ix_full * It_full, (5, 5), 3)
        params[..., 4] = cv2.GaussianBlur(Iy_full * It_full, (5, 5), 3)

        cumulated_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
        win_params = (cumulated_params[2 * window_size + 1:, 2 * window_size + 1:] -
                      cumulated_params[2 * window_size + 1:, :-1 - 2 * window_size] -
                      cumulated_params[:-1 - 2 * window_size, 2 * window_size + 1:] +
                      cumulated_params[:-1 - 2 * window_size, :-1 - 2 * window_size])

        u = np.zeros(old_frame.shape)
        v = np.zeros(old_frame.shape)

        Ix_Ix_sum = win_params[..., 0]
        Iy_Iy_sum = win_params[..., 1]
        Ix_Iy_sum = win_params[..., 2]
        Ix_It_sum = -win_params[..., 3]
        Iy_It_sum = -win_params[..., 4]

        M_det = Ix_Ix_sum * Iy_Iy_sum - Ix_Iy_sum ** 2
        temp_u = Iy_Iy_sum * (-Ix_It_sum) + (-Ix_Iy_sum) * (-Iy_It_sum)
        temp_v = (-Ix_Iy_sum) * (-Ix_It_sum) + Ix_Ix_sum * (-Iy_It_sum)
        # op_flow_x = np.where(M_det != 0, temp_u / M_det, 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(temp_u, M_det)
            c[c == np.inf] = 0
            op_flow_x = np.nan_to_num(c)
        # op_flow_y = np.where(M_det != 0, temp_v / M_det, 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(temp_v, M_det)
            c[c == np.inf] = 0
            op_flow_y = np.nan_to_num(c)

        u[window_size + 1: -1 - window_size, window_size + 1: -1 - window_size] = op_flow_x[:-1, :-1]
        v[window_size + 1: -1 - window_size, window_size + 1: -1 - window_size] = op_flow_y[:-1, :-1]

        return u, v

    else:
        w = int(window_size / 2)

        # old_frame = old_frame / 255
        # new_frame = new_frame / 255

        # Convolve to get gradients w.r.t. X, Y and T dimensions
        Ix_full = np.zeros(old_frame.shape)
        Iy_full = np.zeros(old_frame.shape)
        It_full = np.zeros(old_frame.shape)

        # Ix_full = cv2.Sobel(old_frame, cv2.CV_64F, 1, 0, ksize=3)
        # Iy_full = cv2.Sobel(old_frame, cv2.CV_64F, 0, 1, ksize=3)
        # It_full = new_frame.astype(np.float64) - old_frame.astype(np.float64)

        Ix_full[1:-1, 1:-1] = (old_frame[1:-1, 2:] - old_frame[1:-1, :-2]) / 2
        Iy_full[1:-1, 1:-1] = (old_frame[2:, 1:-1] - old_frame[:-2, 1:-1]) / 2
        It_full[1:-1, 1:-1] = old_frame[1:-1, 1:-1] - new_frame[1:-1, 1:-1]

        u = np.zeros(old_frame.shape)
        v = np.zeros(old_frame.shape)

        for feature in feature_list:  # for every corner
            i, j = feature.ravel()
            i, j = int(i), int(j)  # i,j are floats initially

            if i - w >= 0 and i + w + 1 < old_frame.shape[1] and j - w >= 0 and j + w + 1 < old_frame.shape[0]:
                Ix = Ix_full[j - w:j + w + 1, i - w: i + w + 1]
                Iy = Iy_full[j - w:j + w + 1, i - w:i + w + 1]
                It = It_full[j - w:j + w + 1, i - w:i + w + 1]

                Ix_Ix_sum = np.sum(Ix * Ix)
                Ix_Iy_sum = np.sum(Ix * Iy)
                Iy_Iy_sum = np.sum(Iy * Iy)
                Ix_It_sum = np.sum(Ix * It)
                Iy_It_sum = np.sum(Iy * It)

                A_matrix = np.array([[Ix_Ix_sum, Ix_Iy_sum],
                                     [Ix_Iy_sum, Iy_Iy_sum]])
                b = np.array([[Ix_It_sum, ],
                              [Iy_It_sum, ]])
                uv = np.matmul(np.linalg.pinv(A_matrix), -b)

                u[j, i] = uv[0][0]
                v[j, i] = uv[1][0]
            else:
                continue

            return u, v


# copy from lab, just for testing
def horn_schunk(old_frame, new_frame):
    iterations = 10
    lam = 0.001

    Ix = cv2.Sobel(old_frame, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(old_frame, cv2.CV_64F, 0, 1, ksize=3)
    It = new_frame.astype(np.float64) - old_frame.astype(np.float64)

    kenelAvg = np.array([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]], dtype=np.float64)

    directionU = np.zeros(old_frame.shape)  # to store the x component of the direction of an image patch
    directionV = np.zeros(old_frame.shape)  # to store the y component of the direction of an image patch
    uAvg = np.zeros(old_frame.shape)  # to store the x component of the direction of an image patch
    vAvg = np.zeros(old_frame.shape)  # to store the y component of the direction of an image patch

    for i in range(iterations):
        uAvg = cv2.filter2D(directionU, cv2.CV_64F, kenelAvg)
        vAvg = cv2.filter2D(directionV, cv2.CV_64F, kenelAvg)
        directionU = uAvg - lam * Ix * (Ix * uAvg + Iy * vAvg + It) / (1 + lam * (Ix ** 2 + Iy ** 2))
        directionV = vAvg - lam * Iy * (Ix * uAvg + Iy * vAvg + It) / (1 + lam * (Ix ** 2 + Iy ** 2))

    [X, Y] = np.meshgrid(np.arange(old_frame.shape[1]), np.arange(new_frame.shape[0]))
    U = np.zeros(X.shape)
    V = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            U[i, j] = directionU[old_frame.shape[0] - Y[i, j] - 1, X[i, j]]
            V[i, j] = -directionV[old_frame.shape[0] - Y[i, j] - 1, X[i, j]]

    return U, V
