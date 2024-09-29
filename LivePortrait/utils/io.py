import cv2
import numpy as np


def contiguous(obj):
    if not obj.flags.c_contiguous:
        obj = obj.copy(order="C")
    return obj


def resize_to_limit(img: np.ndarray, max_dim=1920, division=2):
    """
    ajust the size of the image so that the maximum dimension does not exceed max_dim, and the width and the height of the image are multiples of n.
    :param img: the image to be processed.
    :param max_dim: the maximum dimension constraint.
    :param n: the number that needs to be multiples of.
    :return: the adjusted image.
    """
    h, w = img.shape[:2]

    # ajust the size of the image according to the maximum dimension
    if max_dim > 0 and max(h, w) > max_dim:
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        img = cv2.resize(img, (new_w, new_h))

    # ensure that the image dimensions are multiples of n
    division = max(division, 1)
    new_h = img.shape[0] - (img.shape[0] % division)
    new_w = img.shape[1] - (img.shape[1] % division)

    if new_h == 0 or new_w == 0:
        # when the width or height is less than n, no need to process
        return img

    if new_h != img.shape[0] or new_w != img.shape[1]:
        img = img[:new_h, :new_w]

    return img


def load_img_online(img, mode="bgr", **kwargs):
    max_dim = kwargs.get("max_dim", 1920)
    n = kwargs.get("n", 2)

    # Resize image to satisfy constraints
    img = resize_to_limit(img, max_dim=max_dim, division=n)

    if mode.lower() == "bgr":
        return contiguous(img)
    elif mode.lower() == "rgb":
        return contiguous(img[..., ::-1])
    else:
        raise Exception(f"Unknown mode {mode}")