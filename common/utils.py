import cv2
import numpy as np
from sklearn.cluster import KMeans


def equalize(image):
    if len(image.shape) == 2:
        hist_equalized_image = cv2.equalizeHist(image)
    else:
        hist_equalized_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        hist_equalized_image[:, :, 0] = cv2.equalizeHist(hist_equalized_image[:, :, 0])
        hist_equalized_image = cv2.cvtColor(hist_equalized_image, cv2.COLOR_YCrCb2BGR)
    return hist_equalized_image


def normalize_channels(image):
    _, _, channels = image.shape
    for c in range(channels):
        image[:, :, c] -= np.min(image[:, :, c])
    return image


def generous_difference(a, b):
    a_b = np.expand_dims(np.abs(a - b), axis=0)
    a_b_l = np.expand_dims(np.abs(a - shift(b, (0, -1))), axis=0)
    a_b_r = np.expand_dims(np.abs(a - shift(b, (0, 1))), axis=0)
    a_b_u = np.expand_dims(np.abs(a - shift(b, (-1, 0))), axis=0)
    a_b_d = np.expand_dims(np.abs(a - shift(b, (1, 0))), axis=0)
    a_b_lu = np.expand_dims(np.abs(a - shift(b, (-1, -1))), axis=0)
    a_b_ld = np.expand_dims(np.abs(a - shift(b, (1, -1))), axis=0)
    a_b_ru = np.expand_dims(np.abs(a - shift(b, (-1, 1))), axis=0)
    a_b_rd = np.expand_dims(np.abs(a - shift(b, (1, 1))), axis=0)
    composite = np.concatenate([a_b, a_b_l, a_b_r, a_b_u, a_b_d, a_b_lu, a_b_ld, a_b_ru, a_b_rd], axis=0)
    composite[composite > 1] = 1
    generous_difference_image = np.min(composite, axis=0)
    return generous_difference_image


def shift(original_image, shift_tuple: tuple):
    shift_y, shift_x = shift_tuple
    shifted_image = np.zeros_like(original_image)
    shifted_image[:, :, :] = np.inf

    siy_from, siy_to = max(0, shift_y), (min(0, shift_y) if min(0, shift_y) < 0 else None)
    oiy_from, oiy_to = max(0, -shift_y), (min(0, -shift_y) if min(0, -shift_y) < 0 else None)
    six_from, six_to = max(0, shift_x), (min(0, shift_x) if min(0, shift_x) < 0 else None)
    oix_from, oix_to = max(0, -shift_x), (min(0, -shift_x) if min(0, -shift_x) < 0 else None)
    shifted_image[siy_from:siy_to, six_from:six_to, :] = original_image[oiy_from:oiy_to, oix_from:oix_to, :]
    return shifted_image


def fast_clustering(image, n_clusters=5, reduced_dimension: int = 640):
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    h, w, _ = image.shape
    reduced_size_image = cv2.resize(image, (reduced_dimension, int(reduced_dimension * h / w)),
                                    interpolation=cv2.INTER_AREA)
    x_train = reduced_size_image.reshape(-1, image.shape[2])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=3).fit(x_train)
    cc = kmeans.cluster_centers_

    img_batches = None
    for center in cc:
        img_batch = np.linalg.norm(image - center, axis=2)
        if img_batches is None:
            img_batches = np.expand_dims(img_batch, axis=2)
        else:
            img_batches = np.append(img_batches, np.expand_dims(img_batch, axis=2), axis=2)
    img_batches = np.argmin(img_batches, axis=2)
    new_image = cc[img_batches].astype(np.uint8)
    return new_image


def laplacian_blend(src_image, target_image, flag=False):
    hs, ws = src_image.shape[:2]
    ht, wt = target_image.shape[:2]
    min_dim = min(hs, ws, ht, wt)
    src_image = src_image[:min_dim, :min_dim]
    target_image = target_image[:min_dim, :min_dim]
    lp_scr, sizes = get_laplacian_pyramid(src_image)
    lp_target, _ = get_laplacian_pyramid(target_image)
    lp_merge = merge_laplacians(lp_scr, lp_target, flag)
    return reconstruct_from_laplacian_pyramid(lp_merge, sizes)


def get_laplacian_pyramid(image, levels=6):
    # generate Gaussian pyramid
    gaussian_pyramid = image.copy()
    gp = [gaussian_pyramid]
    sizes = []
    for i in range(levels):
        sizes.append((gaussian_pyramid.shape[0], gaussian_pyramid.shape[1]))
        gaussian_pyramid = cv2.pyrDown(gaussian_pyramid)
        gp.append(gaussian_pyramid)

    # generate Laplacian Pyramid
    lp = [gp[levels - 1]]
    for i in range(levels - 1, 0, -1):
        gaussian_pyramid_expanded = cv2.pyrUp(gp[i], dstsize=sizes[i - 1])
        laplacian_pyramid = cv2.subtract(gp[i - 1], gaussian_pyramid_expanded)
        lp.append(laplacian_pyramid)

    return lp, sizes


def merge_laplacians(lp1, lp2, flag):
    lp = []
    for la, lb in zip(lp1, lp2):
        if flag:
            ls = (np.min(np.stack((la, lb), axis=-1), axis=-1)).astype(np.uint8)
        else:
            ls = (np.mean(np.stack((la, lb), axis=-1), axis=-1)).astype(np.uint8)
        lp.append(ls)
    return lp


def reconstruct_from_laplacian_pyramid(lp, sizes=None, levels=6):
    ls_ = lp[0]
    for i in range(1, levels):
        ls_ = cv2.pyrUp(ls_, dstsize=sizes[levels - i - 1] if sizes else None)
        ls_ = cv2.add(ls_, lp[i])

    return ls_


# Define functions
def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return (theta, rho)