import os

import cv2
import numpy as np
import torch

from LivePortrait.utils.camera import get_rotation_matrix
from LivePortrait.utils.crop import prepare_paste_back, paste_back
from LivePortrait.utils.cropper import Cropper
from LivePortrait.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio
from LivePortrait.utils.io import load_img_online
from LivePortrait.wrapper import LivePortraitWrapper


class LivePortraitPipeline(object):

    def __init__(
            self,
            landmark_ckpt_path: str,
            checkpoint_F: str,
            checkpoint_M: str,
            checkpoint_W: str,
            checkpoint_G: str,
            checkpoint_S: str,
            device: torch.device
    ):
        self.device = device

        mask_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", "mask_template.png")
        self.mask_crop = cv2.imread(mask_file, cv2.IMREAD_COLOR)

        self.cropper: Cropper = Cropper(
            landmark_ckpt_path=landmark_ckpt_path,
            device=device,
        )
        self.wrapper: LivePortraitWrapper = LivePortraitWrapper(
            checkpoint_F,
            checkpoint_M,
            checkpoint_W,
            checkpoint_G,
            checkpoint_S,
            device,
        )


    @torch.no_grad()
    def update_delta_new_eyeball_direction(self, eyeball_direction_x, eyeball_direction_y, delta_new, **kwargs):
        if eyeball_direction_x > 0:
                delta_new[0, 11, 0] += eyeball_direction_x * 0.0007
                delta_new[0, 15, 0] += eyeball_direction_x * 0.001
        else:
            delta_new[0, 11, 0] += eyeball_direction_x * 0.001
            delta_new[0, 15, 0] += eyeball_direction_x * 0.0007

        delta_new[0, 11, 1] += eyeball_direction_y * -0.001
        delta_new[0, 15, 1] += eyeball_direction_y * -0.001
        blink = -eyeball_direction_y / 2.

        delta_new[0, 11, 1] += blink * -0.001
        delta_new[0, 13, 1] += blink * 0.0003
        delta_new[0, 15, 1] += blink * -0.001
        delta_new[0, 16, 1] += blink * 0.0003

        return delta_new

    @torch.no_grad()
    def update_delta_new_smile(self, smile, delta_new, **kwargs):
        delta_new[0, 20, 1] += smile * -0.01
        delta_new[0, 14, 1] += smile * -0.02
        delta_new[0, 17, 1] += smile * 0.0065
        delta_new[0, 17, 2] += smile * 0.003
        delta_new[0, 13, 1] += smile * -0.00275
        delta_new[0, 16, 1] += smile * -0.00275
        delta_new[0, 3, 1] += smile * -0.0035
        delta_new[0, 7, 1] += smile * -0.0035

        return delta_new

    @torch.no_grad()
    def update_delta_new_wink(self, wink, delta_new, **kwargs):
        delta_new[0, 11, 1] += wink * 0.001
        delta_new[0, 13, 1] += wink * -0.0003
        delta_new[0, 17, 0] += wink * 0.0003
        delta_new[0, 17, 1] += wink * 0.0003
        delta_new[0, 3, 1] += wink * -0.0003

        return delta_new

    @torch.no_grad()
    def update_delta_new_eyebrow(self, eyebrow, delta_new, **kwargs):
        if eyebrow > 0:
            delta_new[0, 1, 1] += eyebrow * 0.001
            delta_new[0, 2, 1] += eyebrow * -0.001
        else:
            delta_new[0, 1, 0] += eyebrow * -0.001
            delta_new[0, 2, 0] += eyebrow * 0.001
            delta_new[0, 1, 1] += eyebrow * 0.0003
            delta_new[0, 2, 1] += eyebrow * -0.0003
        return delta_new

    @torch.no_grad()
    def update_delta_new_lip_variation_zero(self, lip_variation_zero, delta_new, **kwargs):
        delta_new[0, 19, 0] += lip_variation_zero

        return delta_new

    @torch.no_grad()
    def update_delta_new_lip_variation_one(self, lip_variation_one, delta_new, **kwargs):
        delta_new[0, 14, 1] += lip_variation_one * 0.001
        delta_new[0, 3, 1] += lip_variation_one * -0.0005
        delta_new[0, 7, 1] += lip_variation_one * -0.0005
        delta_new[0, 17, 2] += lip_variation_one * -0.0005

        return delta_new

    @torch.no_grad()
    def update_delta_new_lip_variation_two(self, lip_variation_two, delta_new, **kwargs):
        delta_new[0, 20, 2] += lip_variation_two * -0.001
        delta_new[0, 20, 1] += lip_variation_two * -0.001
        delta_new[0, 14, 1] += lip_variation_two * -0.001

        return delta_new

    @torch.no_grad()
    def update_delta_new_lip_variation_three(self, lip_variation_three, delta_new, **kwargs):
        delta_new[0, 19, 1] += lip_variation_three * 0.001
        delta_new[0, 19, 2] += lip_variation_three * 0.0001
        delta_new[0, 17, 1] += lip_variation_three * -0.0001

        return delta_new

    @torch.no_grad()
    def update_delta_new_mov_x(self, mov_x, delta_new, **kwargs):
        delta_new[0, 5, 0] += mov_x

        return delta_new

    @torch.no_grad()
    def update_delta_new_mov_y(self, mov_y, delta_new, **kwargs):
        delta_new[0, 5, 1] += mov_y

        return delta_new


    @torch.no_grad()
    def init_retargeting_image(self, input_image: cv2.typing.MatLike, original_lmk: np.ndarray, do_crop: bool):
        img_rgb = load_img_online(input_image, mode='rgb', max_dim=1280, n=16)
        if do_crop:
            crop_info = self.cropper.crop_source_image(img_rgb, original_lmk)
            lmk = crop_info['lmk_crop']
        else:
            lmk = self.cropper.calc_lmk_from_cropped_image(img_rgb, original_lmk)
        if lmk is None:
            return None, None
        source_eye_ratio = calc_eye_close_ratio(lmk[None])
        source_lip_ratio = calc_lip_close_ratio(lmk[None])
        source_eye_ratio = round(float(source_eye_ratio.mean()), 2)
        source_lip_ratio = round(float(source_lip_ratio[0][0]), 2)
        return source_eye_ratio, source_lip_ratio
    

    @torch.no_grad()
    def prepare_retargeting_image(
        self,
        input_image: cv2.typing.MatLike,
        original_lmk: np.ndarray,
        input_head_pitch_variation,
        input_head_yaw_variation,
        input_head_roll_variation,
        do_crop: bool,
    ):
        img_rgb = load_img_online(input_image, mode='rgb', max_dim=1280, n=2)
        if do_crop:
            crop_info = self.cropper.crop_source_image(img_rgb, original_lmk)
            I_s = self.wrapper.prepare_source(crop_info['img_crop_256x256'])
            source_lmk_user = crop_info['lmk_crop']
            crop_M_c2o = crop_info['M_c2o']
            mask_ori = prepare_paste_back(self.mask_crop, crop_M_c2o, dsize=(img_rgb.shape[1], img_rgb.shape[0]))
        else:
            I_s = self.wrapper.prepare_source(img_rgb)
            source_lmk_user = self.cropper.calc_lmk_from_cropped_image(img_rgb, original_lmk)
            crop_M_c2o = None
            mask_ori = None
        x_s_info = self.wrapper.get_kp_info(I_s)
        x_d_info_user_pitch = x_s_info['pitch'] + input_head_pitch_variation
        x_d_info_user_yaw = x_s_info['yaw'] + input_head_yaw_variation
        x_d_info_user_roll = x_s_info['roll'] + input_head_roll_variation
        R_s_user = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        R_d_user = get_rotation_matrix(x_d_info_user_pitch, x_d_info_user_yaw, x_d_info_user_roll)
        f_s_user = self.wrapper.extract_feature_3d(I_s)
        x_s_user = self.wrapper.transform_keypoint(x_s_info)
        return f_s_user, x_s_user, R_s_user, R_d_user, x_s_info, source_lmk_user, crop_M_c2o, mask_ori, img_rgb
    

    @torch.no_grad()
    def execute_image_retargeting(
        self,
        input_image: cv2.typing.MatLike,
        original_lmk: np.ndarray,
        source_eye_ratio: float,
        source_lip_ratio: float,
        input_eye_ratio: float,
        input_lip_ratio: float,
        input_head_pitch_variation: float,
        input_head_yaw_variation: float,
        input_head_roll_variation: float,
        mov_x: float,
        mov_y: float,
        mov_z: float,
        lip_variation_zero: float,
        lip_variation_one: float,
        lip_variation_two: float,
        lip_variation_three: float,
        smile: float,
        wink: float,
        eyebrow: float,
        eyeball_direction_x: float,
        eyeball_direction_y: float,
        do_crop: bool,
    ):
        f_s_user, x_s_user, R_s_user, R_d_user, x_s_info, source_lmk_user, crop_M_c2o, mask_ori, img_rgb = \
            self.prepare_retargeting_image(
                input_image, original_lmk, input_head_pitch_variation, input_head_yaw_variation, input_head_roll_variation, do_crop)
        if source_lmk_user is None:
            return None

        x_s_user = x_s_user.to(self.device)
        f_s_user = f_s_user.to(self.device)
        R_s_user = R_s_user.to(self.device)
        R_d_user = R_d_user.to(self.device)
        mov_x = torch.tensor(mov_x).to(self.device)
        mov_y = torch.tensor(mov_y).to(self.device)
        mov_z = torch.tensor(mov_z).to(self.device)
        eyeball_direction_x = torch.tensor(eyeball_direction_x).to(self.device)
        eyeball_direction_y = torch.tensor(eyeball_direction_y).to(self.device)
        smile = torch.tensor(smile).to(self.device)
        wink = torch.tensor(wink).to(self.device)
        eyebrow = torch.tensor(eyebrow).to(self.device)
        lip_variation_zero = torch.tensor(lip_variation_zero).to(self.device)
        lip_variation_one = torch.tensor(lip_variation_one).to(self.device)
        lip_variation_two = torch.tensor(lip_variation_two).to(self.device)
        lip_variation_three = torch.tensor(lip_variation_three).to(self.device)

        x_c_s = x_s_info['kp'].to(self.device)
        delta_new = x_s_info['exp'].to(self.device)
        scale_new = x_s_info['scale'].to(self.device)
        t_new = x_s_info['t'].to(self.device)
        R_d_new = (R_d_user @ R_s_user.permute(0, 2, 1)) @ R_s_user

        if eyeball_direction_x != 0 or eyeball_direction_y != 0:
            delta_new = self.update_delta_new_eyeball_direction(eyeball_direction_x, eyeball_direction_y, delta_new)
        if smile != 0:
            delta_new = self.update_delta_new_smile(smile, delta_new)
        if wink != 0:
            delta_new = self.update_delta_new_wink(wink, delta_new)
        if eyebrow != 0:
            delta_new = self.update_delta_new_eyebrow(eyebrow, delta_new)
        if lip_variation_zero != 0:
            delta_new = self.update_delta_new_lip_variation_zero(lip_variation_zero, delta_new)
        if lip_variation_one !=  0:
            delta_new = self.update_delta_new_lip_variation_one(lip_variation_one, delta_new)
        if lip_variation_two != 0:
            delta_new = self.update_delta_new_lip_variation_two(lip_variation_two, delta_new)
        if lip_variation_three != 0:
            delta_new = self.update_delta_new_lip_variation_three(lip_variation_three, delta_new)
        if mov_x != 0:
            delta_new = self.update_delta_new_mov_x(-mov_x, delta_new)
        if mov_y !=0 :
            delta_new = self.update_delta_new_mov_y(mov_y, delta_new)

        x_d_new = mov_z * scale_new * (x_c_s @ R_d_new + delta_new) + t_new
        eyes_delta, lip_delta = None, None
        if input_eye_ratio != source_eye_ratio:
            combined_eye_ratio_tensor = self.wrapper.calc_combined_eye_ratio([[float(input_eye_ratio)]], source_lmk_user)
            eyes_delta = self.wrapper.retarget_eye(x_s_user, combined_eye_ratio_tensor)
        if input_lip_ratio != source_lip_ratio:
            combined_lip_ratio_tensor = self.wrapper.calc_combined_lip_ratio([[float(input_lip_ratio)]], source_lmk_user)
            lip_delta = self.wrapper.retarget_lip(x_s_user, combined_lip_ratio_tensor)
            # print(lip_delta)
        x_d_new = x_d_new + \
                (eyes_delta if eyes_delta is not None else 0) + \
                (lip_delta if lip_delta is not None else 0)

        x_d_new = self.wrapper.stitching(x_s_user, x_d_new)
        out = self.wrapper.warp_decode(f_s_user, x_s_user, x_d_new)
        out = self.wrapper.parse_output(out['out'])[0]
        if do_crop:
            out_to_ori_blend = paste_back(out, crop_M_c2o, img_rgb, mask_ori)
        else:
            out_to_ori_blend = out
        return out, out_to_ori_blend