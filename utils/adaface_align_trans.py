# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:43:29 2017
@author: zhaoy
"""
import numpy as np
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank
import cv2


# reference facial points, a list of coordinates (x,y)
REFERENCE_FACIAL_POINTS = [
    [30.29459953,  51.69630051],
    [65.53179932,  51.50139999],
    [48.02519989,  71.73660278],
    [33.54930115,  92.3655014],
    [62.72990036,  92.20410156]
]

DEFAULT_CROP_SIZE = (96, 112)


class FaceWarpException(Exception):
    def __str__(self):
        return "In File {}:{}".format(
            __file__, super.__str__(self))


class MatlabCp2tormException(Exception):
    def __str__(self):
        return "In File {}:{}".format(
                __file__, super.__str__(self))


def tformfwd(trans, uv):
    uv = np.hstack((
        uv, np.ones((uv.shape[0], 1))
    ))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]
    return xy


def tforminv(trans, uv):
    Tinv = inv(trans)
    xy = tformfwd(Tinv, uv)
    return xy


def findNonreflectiveSimilarity(uv, xy, options=None):

    options = {"K": 2}

    K = options["K"]
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))

    # We know that X * r = U
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U, rcond=-1)
        r = np.squeeze(r)
    else:
        raise Exception("cp2tform:twoUniquePointsReq")

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss,  sc, 0],
        [tx,  ty, 1]
    ])

    T = inv(Tinv)

    T[:, 2] = np.array([0, 0, 1])

    return T, Tinv


def findSimilarity(uv, xy, options=None):

    options = {"K": 2}

    # Solve for trans1
    trans1, trans1_inv = findNonreflectiveSimilarity(uv, xy, options)

    # Solve for trans2

    # manually reflect the xy data across the Y-axis
    xyR = xy
    xyR[:, 0] = -1 * xyR[:, 0]

    trans2r, trans2r_inv = findNonreflectiveSimilarity(uv, xyR, options)

    # manually reflect the tform to undo the reflection done on xyR
    TreflectY = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    trans2 = np.dot(trans2r, TreflectY)

    # Figure out if trans1 or trans2 is better
    xy1 = tformfwd(trans1, uv)
    norm1 = norm(xy1 - xy)

    xy2 = tformfwd(trans2, uv)
    norm2 = norm(xy2 - xy)

    if norm1 <= norm2:
        return trans1, trans1_inv
    else:
        trans2_inv = inv(trans2)
        return trans2, trans2_inv


def get_similarity_transform(src_pts, dst_pts):
    trans, trans_inv = findSimilarity(src_pts, dst_pts)
    return trans, trans_inv


def cvt_tform_mat_for_cv2(trans):
    cv2_trans = trans[:, 0:2].T
    return cv2_trans


def get_similarity_transform_for_cv2(src_pts, dst_pts):
    trans, _ = get_similarity_transform(src_pts, dst_pts)
    cv2_trans = cvt_tform_mat_for_cv2(trans)
    return cv2_trans


def get_reference_facial_points(output_size=None,
                                inner_padding_factor=0.0,
                                outer_padding=(0, 0),
                                default_square=False):
    scaling = (output_size[0] / 112) if output_size is not None else 1

    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS) * scaling
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE) * scaling

    # 0) make the inner region a square
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    if (output_size and
            output_size[0] == tmp_crop_size[0] and
            output_size[1] == tmp_crop_size[1]):
        return tmp_5pts

    if (inner_padding_factor == 0 and
            outer_padding == (0, 0)):
        if output_size is None:
            return tmp_5pts
        else:
            raise FaceWarpException(
                "No paddings to do, output_size must be None or {}".format(tmp_crop_size))

    # check output size
    if not (0 <= inner_padding_factor <= 1.0):
        raise FaceWarpException("Not (0 <= inner_padding_factor <= 1.0)")

    if ((inner_padding_factor > 0 or outer_padding[0] > 0 or outer_padding[1] > 0)
            and output_size is None):
        output_size = tmp_crop_size * \
            (1 + inner_padding_factor * 2).astype(np.int32)
        output_size += np.array(outer_padding)

    if not (outer_padding[0] < output_size[0]
            and outer_padding[1] < output_size[1]):
        raise FaceWarpException("Not (outer_padding[0] < output_size[0]"
                                "and outer_padding[1] < output_size[1])")

    # 1) pad the inner region according inner_padding_factor
    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += np.round(size_diff).astype(np.int32)

    # 2) resize the padded inner region
    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2

    if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[1] * tmp_crop_size[0]:
        raise FaceWarpException("Must have (output_size - outer_padding)"
                                "= some_scale * (crop_size * (1.0 + inner_padding_factor)")

    scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]

    tmp_5pts = tmp_5pts * scale_factor
    tmp_crop_size = size_bf_outer_pad

    # 3) add outer_padding to make output_size
    reference_5point = tmp_5pts + np.array(outer_padding)
    tmp_crop_size = output_size

    return reference_5point


def get_affine_transform_matrix(src_pts, dst_pts):
    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    A, _, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)

    if rank == 3:
        tfm = np.float32([
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]]
        ])
    elif rank == 2:
        tfm = np.float32([
            [A[0, 0], A[1, 0], 0],
            [A[0, 1], A[1, 1], 0]
        ])

    return tfm


def warp_and_crop_face(
    src_img,
    facial_pts,
    reference_pts,
    crop_size=(96, 112),
    align_type="smilarity"
):
    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
        raise FaceWarpException(
            "reference_pts.shape must be (K,2) or (2,K) and K>2")

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
        raise FaceWarpException(
            "facial_pts.shape must be (K,2) or (2,K) and K>2")

    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException(
            "facial_pts and reference_pts must have the same shape")

    if align_type == "cv2_affine":
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
    elif align_type == "affine":
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
    else:
        tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)

    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

    return face_img, tfm