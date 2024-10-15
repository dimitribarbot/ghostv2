import os
from dataclasses import dataclass
from typing import cast, List, Tuple

import cv2
import numpy as np
import torch

from LivePortrait.utils.camera import get_rotation_matrix
from LivePortrait.utils.crop import prepare_paste_back, paste_back
from LivePortrait.utils.cropper import Cropper
from LivePortrait.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio
from LivePortrait.wrapper import LivePortraitWrapper


@dataclass
class RetargetingParameters:
    input_eye_ratio: float = 0
    input_lip_ratio: float = 0
    input_head_pitch_variation: float = 0
    input_head_yaw_variation: float = 0
    input_head_roll_variation: float = 0
    mov_x: float = 0
    mov_y: float = 0
    mov_z: float = 1
    lip_variation_zero: float = 0
    lip_variation_one: float = 0
    lip_variation_two: float = 0
    lip_variation_three: float = 0
    smile: float = 0
    wink: float = 0
    eyebrow: float = 0
    eyeball_direction_x: float = 0
    eyeball_direction_y: float = 0


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
    def update_delta_new_eyeball_direction_multi(self, eyeball_direction_x, eyeball_direction_y, delta_new):
        i = torch.argwhere(eyeball_direction_x + eyeball_direction_y)

        delta_new[i, 11, 0] += torch.where(eyeball_direction_x[i] > 0, eyeball_direction_x[i] * 0.0007, eyeball_direction_x[i] * 0.001)
        delta_new[i, 15, 0] += torch.where(eyeball_direction_x[i] > 0, eyeball_direction_x[i] * 0.001, eyeball_direction_x[i] * 0.0007)

        delta_new[i, 11, 1] += eyeball_direction_y[i] * -0.001
        delta_new[i, 15, 1] += eyeball_direction_y[i] * -0.001
        
        blink = -eyeball_direction_y / 2.

        delta_new[i, 11, 1] += blink[i] * -0.001
        delta_new[i, 13, 1] += blink[i] * 0.0003
        delta_new[i, 15, 1] += blink[i] * -0.001
        delta_new[i, 16, 1] += blink[i] * 0.0003

        return delta_new

    @torch.no_grad()
    def update_delta_new_smile_multi(self, smile, delta_new):
        i = torch.argwhere(smile)

        delta_new[i, 20, 1] += smile[i] * -0.01
        delta_new[i, 14, 1] += smile[i] * -0.02
        delta_new[i, 17, 1] += smile[i] * 0.0065
        delta_new[i, 17, 2] += smile[i] * 0.003
        delta_new[i, 13, 1] += smile[i] * -0.00275
        delta_new[i, 16, 1] += smile[i] * -0.00275
        delta_new[i, 3, 1] += smile[i] * -0.0035
        delta_new[i, 7, 1] += smile[i] * -0.0035

        return delta_new

    @torch.no_grad()
    def update_delta_new_wink_multi(self, wink, delta_new):
        i = torch.argwhere(wink)

        delta_new[i, 11, 1] += wink[i] * 0.001
        delta_new[i, 13, 1] += wink[i] * -0.0003
        delta_new[i, 17, 0] += wink[i] * 0.0003
        delta_new[i, 17, 1] += wink[i] * 0.0003
        delta_new[i, 3, 1] += wink[i] * -0.0003

        return delta_new

    @torch.no_grad()
    def update_delta_new_eyebrow_multi(self, eyebrow, delta_new):
        i = torch.argwhere(eyebrow)

        delta_new[i, 1, 0] += torch.where(eyebrow[i] > 0, 0, eyebrow[i] * -0.001)
        delta_new[i, 2, 0] += torch.where(eyebrow[i] > 0, 0, eyebrow[i] * 0.001)
        delta_new[i, 1, 1] += torch.where(eyebrow[i] > 0, eyebrow[i] * 0.001, eyebrow[i] * 0.0003)
        delta_new[i, 2, 1] += torch.where(eyebrow[i] > 0, eyebrow[i] * -0.001, eyebrow[i] * -0.0003)

        return delta_new

    @torch.no_grad()
    def update_delta_new_lip_variation_zero_multi(self, lip_variation_zero, delta_new):
        i = torch.argwhere(lip_variation_zero)

        delta_new[i, 19, 0] += lip_variation_zero[i]

        return delta_new

    @torch.no_grad()
    def update_delta_new_lip_variation_one_multi(self, lip_variation_one, delta_new):
        i = torch.argwhere(lip_variation_one)

        delta_new[i, 14, 1] += lip_variation_one[i] * 0.001
        delta_new[i, 3, 1] += lip_variation_one[i] * -0.0005
        delta_new[i, 7, 1] += lip_variation_one[i] * -0.0005
        delta_new[i, 17, 2] += lip_variation_one[i] * -0.0005

        return delta_new

    @torch.no_grad()
    def update_delta_new_lip_variation_two_multi(self, lip_variation_two, delta_new):
        i = torch.argwhere(lip_variation_two)

        delta_new[i, 20, 2] += lip_variation_two[i] * -0.001
        delta_new[i, 20, 1] += lip_variation_two[i] * -0.001
        delta_new[i, 14, 1] += lip_variation_two[i] * -0.001

        return delta_new

    @torch.no_grad()
    def update_delta_new_lip_variation_three_multi(self, lip_variation_three, delta_new):
        i = torch.argwhere(lip_variation_three)

        delta_new[i, 19, 1] += lip_variation_three[i] * 0.001
        delta_new[i, 19, 2] += lip_variation_three[i] * 0.0001
        delta_new[i, 17, 1] += lip_variation_three[i] * -0.0001

        return delta_new

    @torch.no_grad()
    def update_delta_new_mov_x_multi(self, mov_x, delta_new):
        i = torch.argwhere(mov_x)

        delta_new[i, 5, 0] += mov_x[i]

        return delta_new

    @torch.no_grad()
    def update_delta_new_mov_y_multi(self, mov_y, delta_new):
        i = torch.argwhere(mov_y)

        delta_new[i, 5, 1] += mov_y[i]

        return delta_new

    @torch.no_grad()
    def calc_combined_eye_ratio_multi(self, c_d_eyes_i: torch.Tensor, source_lmk: np.ndarray):
        c_s_eyes = calc_eye_close_ratio(source_lmk[None])
        c_s_eyes_tensor = torch.from_numpy(c_s_eyes).float().to(self.device).repeat(c_d_eyes_i.size(0), 1)
        c_d_eyes_i_tensor = c_d_eyes_i.unsqueeze(1) # bsx1
        combined_eye_ratio_tensor = torch.cat([c_s_eyes_tensor, c_d_eyes_i_tensor], dim=1) # bsx3
        return combined_eye_ratio_tensor

    @torch.no_grad()
    def calc_combined_lip_ratio_multi(self, c_d_lip_i: torch.Tensor, source_lmk: np.ndarray):
        c_s_lip = calc_lip_close_ratio(source_lmk[None])
        c_s_lip_tensor = torch.from_numpy(c_s_lip).float().to(self.device).repeat(c_d_lip_i.size(0), 1)
        c_d_lip_i_tensor = c_d_lip_i.unsqueeze(1) # bsx1
        combined_lip_ratio_tensor = torch.cat([c_s_lip_tensor, c_d_lip_i_tensor], dim=1) # bsx2
        return combined_lip_ratio_tensor
    

    @torch.no_grad()
    def prepare_retargeting_image_multi(
        self,
        img_rgb: cv2.typing.MatLike,
        original_lmk: np.ndarray,
        input_head_pitch_variation: torch.Tensor,
        input_head_yaw_variation: torch.Tensor,
        input_head_roll_variation: torch.Tensor,
        do_crop: bool,
        crop_scale: float,
    ):
        if do_crop:
            crop_info = self.cropper.crop_source_image(img_rgb, original_lmk, crop_scale)
            source_lmk_user = crop_info['lmk_crop']
            I_s = self.wrapper.prepare_source(crop_info['img_crop_256x256'])
            crop_M_c2o = crop_info['M_c2o']
            mask_ori = prepare_paste_back(self.mask_crop, crop_M_c2o, dsize=(img_rgb.shape[1], img_rgb.shape[0]))
        else:
            source_lmk_user = self.cropper.calc_lmk_from_cropped_image(img_rgb, original_lmk)
            I_s = self.wrapper.prepare_source(img_rgb)
            crop_M_c2o = None
            mask_ori = None
        x_s_info = self.wrapper.get_kp_info(I_s)
        source_eye_ratio = calc_eye_close_ratio(source_lmk_user[None])
        source_lip_ratio = calc_lip_close_ratio(source_lmk_user[None])
        source_eye_ratio = round(float(source_eye_ratio.mean()), 2)
        source_lip_ratio = round(float(source_lip_ratio[0][0]), 2)
        x_s_info_pitch = x_s_info['pitch'].repeat(input_head_pitch_variation.size(0), 1)
        x_s_info_yaw = x_s_info['yaw'].repeat(input_head_yaw_variation.size(0), 1)
        x_s_info_roll = x_s_info['roll'].repeat(input_head_roll_variation.size(0), 1)
        x_d_info_user_pitch = x_s_info_pitch + input_head_pitch_variation.unsqueeze(-1)
        x_d_info_user_yaw = x_s_info_yaw + input_head_yaw_variation.unsqueeze(-1)
        x_d_info_user_roll = x_s_info_roll + input_head_roll_variation.unsqueeze(-1)
        R_s_user = get_rotation_matrix(x_s_info_pitch, x_s_info_yaw, x_s_info_roll)
        R_d_user = get_rotation_matrix(x_d_info_user_pitch, x_d_info_user_yaw, x_d_info_user_roll)
        f_s_user = self.wrapper.extract_feature_3d(I_s)
        x_s_user = self.wrapper.transform_keypoint(x_s_info)
        return f_s_user, x_s_user, R_s_user, R_d_user, x_s_info, source_lmk_user, source_eye_ratio, source_lip_ratio, crop_M_c2o, mask_ori
    

    @torch.no_grad()
    def execute_image_retargeting_multi(
        self,
        img_rgb: cv2.typing.MatLike,
        original_lmks: List[np.ndarray],
        all_parameters: List[List[RetargetingParameters]],
        do_crop: bool,
        crop_scale: float,
    ):
        if len(original_lmks) > 1 and not do_crop:
            raise ValueError("Cannot handle multiple face when do_crop is False.")

        if len(original_lmks) != len(all_parameters) or len(all_parameters) == 0:
            return None

        paste_back_info: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for face_index in range(len(original_lmks)):
            original_lmk = original_lmks[face_index]
            parameters = all_parameters[face_index]

            input_head_pitch_variation = torch.tensor(list(map(lambda variation: variation.input_head_pitch_variation, parameters))).to(self.device)
            input_head_yaw_variation = torch.tensor(list(map(lambda variation: variation.input_head_yaw_variation, parameters))).to(self.device)
            input_head_roll_variation = torch.tensor(list(map(lambda variation: variation.input_head_roll_variation, parameters))).to(self.device)

            f_s_user, x_s_user, R_s_user, R_d_user, x_s_info, source_lmk_user, source_eye_ratio, source_lip_ratio, crop_M_c2o, mask_ori = \
                self.prepare_retargeting_image_multi(
                    img_rgb, original_lmk, input_head_pitch_variation, input_head_yaw_variation, input_head_roll_variation, do_crop, crop_scale)
            if source_lmk_user is None:
                return None

            x_s_user = cast(torch.Tensor, x_s_user.repeat(len(parameters), 1, 1).to(self.device))
            f_s_user = f_s_user.repeat(len(parameters), 1, 1, 1, 1).to(self.device)
            R_s_user = R_s_user.to(self.device)
            R_d_user = R_d_user.to(self.device)
            
            input_eye_ratio = torch.tensor(list(map(lambda parameter: parameter.input_eye_ratio, parameters))).to(self.device)
            input_lip_ratio = torch.tensor(list(map(lambda parameter: parameter.input_lip_ratio, parameters))).to(self.device)
            mov_x = torch.tensor(list(map(lambda parameter: parameter.mov_x, parameters))).to(self.device)
            mov_y = torch.tensor(list(map(lambda parameter: parameter.mov_y, parameters))).to(self.device)
            mov_z = torch.tensor(list(map(lambda parameter: parameter.mov_z, parameters))).to(self.device)
            eyeball_direction_x = torch.tensor(list(map(lambda parameter: parameter.eyeball_direction_x, parameters))).to(self.device)
            eyeball_direction_y = torch.tensor(list(map(lambda parameter: parameter.eyeball_direction_y, parameters))).to(self.device)
            smile = torch.tensor(list(map(lambda parameter: parameter.smile, parameters))).to(self.device)
            wink = torch.tensor(list(map(lambda parameter: parameter.wink, parameters))).to(self.device)
            eyebrow = torch.tensor(list(map(lambda parameter: parameter.eyebrow, parameters))).to(self.device)
            lip_variation_zero = torch.tensor(list(map(lambda parameter: parameter.lip_variation_zero, parameters))).to(self.device)
            lip_variation_one = torch.tensor(list(map(lambda parameter: parameter.lip_variation_one, parameters))).to(self.device)
            lip_variation_two = torch.tensor(list(map(lambda parameter: parameter.lip_variation_two, parameters))).to(self.device)
            lip_variation_three = torch.tensor(list(map(lambda parameter: parameter.lip_variation_three, parameters))).to(self.device)

            x_c_s = cast(torch.Tensor, x_s_info['kp'].to(self.device))
            delta_new = cast(torch.Tensor, x_s_info['exp'].repeat(len(parameters), 1, 1).to(self.device))
            scale_new = cast(torch.Tensor, x_s_info['scale'].to(self.device))
            t_new = cast(torch.Tensor, x_s_info['t'].to(self.device))

            R_d_new = (R_d_user @ R_s_user.permute(0, 2, 1)) @ R_s_user

            delta_new = self.update_delta_new_eyeball_direction_multi(eyeball_direction_x, eyeball_direction_y, delta_new)
            delta_new = self.update_delta_new_smile_multi(smile, delta_new)
            delta_new = self.update_delta_new_wink_multi(wink, delta_new)
            delta_new = self.update_delta_new_eyebrow_multi(eyebrow, delta_new)
            delta_new = self.update_delta_new_lip_variation_zero_multi(lip_variation_zero, delta_new)
            delta_new = self.update_delta_new_lip_variation_one_multi(lip_variation_one, delta_new)
            delta_new = self.update_delta_new_lip_variation_two_multi(lip_variation_two, delta_new)
            delta_new = self.update_delta_new_lip_variation_three_multi(lip_variation_three, delta_new)
            delta_new = self.update_delta_new_mov_x_multi(-mov_x, delta_new)
            delta_new = self.update_delta_new_mov_y_multi(mov_y, delta_new)

            x_d_new = mov_z.unsqueeze(1).unsqueeze(2) * scale_new * (x_c_s @ R_d_new + delta_new) + t_new

            eyes_delta = torch.zeros_like(x_d_new, device=self.device)
            lip_delta = torch.zeros_like(x_d_new, device=self.device)
            i_eye = torch.argwhere(input_eye_ratio - source_eye_ratio)
            i_lip = torch.argwhere(input_lip_ratio - source_lip_ratio)

            combined_eye_ratio_tensor = self.calc_combined_eye_ratio_multi(input_eye_ratio, source_lmk_user)
            combined_lip_ratio_tensor = self.calc_combined_lip_ratio_multi(input_lip_ratio, source_lmk_user)

            eyes_delta[i_eye] = self.wrapper.retarget_eye(x_s_user[i_eye].squeeze(1), combined_eye_ratio_tensor[i_eye].squeeze(1)).unsqueeze(1)
            lip_delta[i_lip] = self.wrapper.retarget_lip(x_s_user[i_lip].squeeze(1), combined_lip_ratio_tensor[i_lip].squeeze(1)).unsqueeze(1)
            
            x_d_new = x_d_new + eyes_delta + lip_delta

            x_d_new = self.wrapper.stitching(x_s_user, x_d_new)
            out = self.wrapper.warp_decode(f_s_user, x_s_user, x_d_new)
            out = self.wrapper.parse_output(out['out'])
            
            paste_back_info.append((out, crop_M_c2o, mask_ori))

        if do_crop:
            retargeting_count = len(all_parameters[0])
            out_to_ori_blend = [img_rgb] * retargeting_count
            for out, crop_M_c2o, mask_ori in paste_back_info:
                for out_img_index in range(out.shape[0]):
                    paste_back_out = paste_back(out[out_img_index], crop_M_c2o, out_to_ori_blend[out_img_index], mask_ori)
                    out_to_ori_blend[out_img_index] = paste_back_out
        else:
            out_to_ori_blend = [[out_img for out_img in out] for out, _, _ in paste_back_info]

        return out_to_ori_blend