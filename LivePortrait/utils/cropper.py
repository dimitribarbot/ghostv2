import cv2
import numpy as np

from LivePortrait.utils.landmark_runner import LandmarkRunner
from LivePortrait.utils.crop import crop_image


class Cropper(object):
    def __init__(self, **kwargs) -> None:
        self.human_landmark_runner = LandmarkRunner(
            ckpt_path=kwargs.get("landmark_ckpt_path"),
            device=kwargs.get("device"),
        )


    def crop_source_image(self, img_rgb: np.ndarray, original_lmk: np.ndarray):
        cropped_img_rgb = img_rgb.copy()

        ret_dct = crop_image(
            cropped_img_rgb,
            original_lmk,
            dsize=512,
            scale=3.5,
            vx_ratio=0,
            vy_ratio=-0.125,
            flag_do_rot=True,
        )

        ret_dct["img_crop_256x256"] = cv2.resize(ret_dct["img_crop"], (256, 256), interpolation=cv2.INTER_AREA)
        lmk = self.human_landmark_runner.run(cropped_img_rgb, original_lmk)
        ret_dct["lmk_crop"] = lmk
        ret_dct["lmk_crop_256x256"] = ret_dct["lmk_crop"] * 256 / 512

        return ret_dct


    def calc_lmk_from_cropped_image(self, img_rgb: np.ndarray, original_lmk: np.ndarray):
        return self.human_landmark_runner.run(img_rgb, original_lmk)
