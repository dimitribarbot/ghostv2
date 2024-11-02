import os
import cv2
import random
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

from BiSeNet.bisenet import BiSeNet
from CVLFace.differentiable_face_aligner import DifferentiableFaceAligner
from GFPGAN.gfpganv1_clean_arch import GFPGANv1Clean
from LivePortrait.pipeline import LivePortraitPipeline, RetargetingParameters
from LivePortrait.utils.io import contiguous, resize_to_limit
from RetinaFace.detector import RetinaFace
from face_alignment import FaceAlignment
from utils.preprocessing.preprocess_arguments import PreprocessArguments
from utils.image_processing import get_aligned_face_and_affine_matrix, paste_face_back_facexlib, \
    enhance_face, sort_faces_by_coordinates, random_horizontal_flip, trans_points2d


@torch.no_grad()
def enhance_faces_in_original_image(
    gfpgan: GFPGANv1Clean,
    face_parser: BiSeNet,
    rgb_image: cv2.typing.MatLike,
    lmks: np.ndarray,
    image_name: str,
    device: str,
):
    upsample_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    for lmk in lmks:
        cropped_face, affine_matrix = get_aligned_face_and_affine_matrix(upsample_img, lmk)
        restored_face = enhance_face(gfpgan, cropped_face, image_name, device)
        upsample_img = paste_face_back_facexlib(face_parser, upsample_img, restored_face, affine_matrix, True, device)

    if np.max(upsample_img) > 256:  # 16-bit image
        upsample_img = upsample_img.astype(np.uint16)
    else:
        upsample_img = upsample_img.astype(np.uint8)

    upsample_img = cv2.cvtColor(upsample_img, cv2.COLOR_BGR2RGB)
        
    return upsample_img


def get_all_face_retargeting_parameters(number_of_faces: int, number_of_variants_per_face: int):
    all_parameters: List[List[RetargetingParameters]] = []
    for _ in range(number_of_faces):
        parameters: List[RetargetingParameters] = []
        for _ in range(number_of_variants_per_face):
            parameters.append(RetargetingParameters(
                input_eye_ratio=round(random.uniform(0.34, 0.42), 2),
                input_lip_ratio=round(random.uniform(0, 0.3), 2),
                input_head_pitch_variation=round(random.uniform(-8, 8), 0),
                input_head_yaw_variation=round(random.uniform(-15, 15), 0),
                input_head_roll_variation=round(random.uniform(-8, 8), 0),
                mov_x=0,
                mov_y=0,
                mov_z=1,
                lip_variation_zero=round(random.uniform(-0.01, 0.01), 2),
                lip_variation_one=round(random.uniform(-10, 10), 2),
                lip_variation_two=round(random.uniform(0, 5), 2),
                lip_variation_three=round(random.uniform(-10, 20), 0),
                smile=round(random.uniform(-0.2, 0.6), 2),
                wink=0.0,
                eyebrow=round(random.uniform(-7.5, 7.5), 2),
                eyeball_direction_x=round(random.uniform(-10, 10), 2),
                eyeball_direction_y=round(random.uniform(-10, 10), 2),
            ))
        all_parameters.append(parameters)
    return all_parameters


@torch.no_grad()
def get_retargeted_images(
    live_portrait_pipeline: LivePortraitPipeline,
    rgb_image: cv2.typing.MatLike,
    bgr_image: cv2.typing.MatLike,
    image_id: int,
    faces: List[Dict[str, np.ndarray]],
    number_of_variants_per_face: int,
    do_crop: bool,
    crop_scale: float,
    filter_valid_faces: bool,
    save_retargeted: bool,
    output_dir_retargeted: str,
):
    if save_retargeted and do_crop:
        os.makedirs(os.path.join(output_dir_retargeted, f"{image_id}"), exist_ok=True)
        cv2.imwrite(
            os.path.join(output_dir_retargeted, f"{image_id}", "0.jpg"),
            bgr_image
        )

    landmarks = [np.array(face["kps"]) for face in faces]
    faces_rgb = [contiguous(cv2.cvtColor(face["cropped_bgr"], cv2.COLOR_BGR2RGB)) if face["cropped_bgr"] is not None else None for face in faces]

    all_parameters: List[List[RetargetingParameters]] = get_all_face_retargeting_parameters(
        len(faces),
        number_of_variants_per_face
    )

    retargeted_image_variations = live_portrait_pipeline.execute_image_retargeting_multi(
        contiguous(rgb_image),
        landmarks,
        faces_rgb,
        all_parameters,
        do_crop,
        crop_scale,
        filter_valid_faces,
    )
    
    if retargeted_image_variations is not None:
        if do_crop:
            retargeted_images = [bgr_image]

            retargeted_image_variations = [
                cv2.cvtColor(variation, cv2.COLOR_RGB2BGR) for variation in retargeted_image_variations if variation is not None
            ]

            if save_retargeted:
                for i in range(len(retargeted_image_variations)):
                    cv2.imwrite(
                        os.path.join(output_dir_retargeted, f"{image_id}", f"{i + 1}.jpg"),
                        retargeted_image_variations[i]
                    )

            retargeted_images += retargeted_image_variations
        else:
            retargeted_images = [[face["cropped_bgr"]] for face in faces]

            for face_index in range(len(retargeted_image_variations)):
                if len(retargeted_image_variations[face_index]) > 0:
                    retargeted_image_variations[face_index] = [
                        cv2.cvtColor(variation, cv2.COLOR_RGB2BGR) for variation in retargeted_image_variations[face_index] if variation is not None
                    ]
                    retargeted_images[face_index] += retargeted_image_variations[face_index]
                else:
                    retargeted_images[face_index] = []
    else:
        retargeted_images = []
    
    return retargeted_images


@torch.no_grad()
def align_and_save(
    gfpgan: GFPGANv1Clean,
    bgr_image: cv2.typing.MatLike,
    lmk: np.ndarray,
    image_index: int,
    final_crop_size: Tuple[int, int],
    cropped_face_path:str,
    cropped_face_path_resized: str,
    image_name: str,
    align_mode: str,
    aligner: Optional[DifferentiableFaceAligner],
    device: str,
    save_full_size: bool,
    should_enhance_face: bool
):
    if save_full_size:
        save_path = os.path.join(cropped_face_path, image_name)
        os.makedirs(save_path, exist_ok=True)
    save_path_resized = os.path.join(cropped_face_path_resized, image_name)
    os.makedirs(save_path_resized, exist_ok=True)

    if save_full_size:
        cv2.imwrite(os.path.join(save_path, f"{image_index}.jpg"), bgr_image)
    
    if should_enhance_face:
        if align_mode == "facexlib":
            cropped_face, _ = get_aligned_face_and_affine_matrix(bgr_image, lmk)
            cropped_face = enhance_face(gfpgan, cropped_face, image_name, device)
            cropped_face = cv2.resize(cropped_face, (final_crop_size, final_crop_size), interpolation=cv2.INTER_CUBIC)
        else:
            cropped_face, affine_matrix = get_aligned_face_and_affine_matrix(bgr_image, lmk)
            cropped_face = enhance_face(gfpgan, cropped_face, image_name, device)
            transformed_lmk = trans_points2d(lmk, affine_matrix)
            cropped_face, _ = get_aligned_face_and_affine_matrix(cropped_face, transformed_lmk, final_crop_size, align_mode, aligner, device)
    else:
        cropped_face, _ = get_aligned_face_and_affine_matrix(bgr_image, lmk, final_crop_size, align_mode, aligner, device)

    cv2.imwrite(os.path.join(save_path_resized, f"{image_index}.jpg"), cropped_face)


@torch.no_grad()
def get_faces_from_landmarks(
    gfpgan: GFPGANv1Clean,
    bgr_image: cv2.typing.MatLike,
    landmarks_68: List[np.ndarray],
    landmarks_5: List[np.ndarray],
    do_crop: bool,
    image_name: str,
    device: str,
):
    faces = []
    for landmark_68_index, landmark_68 in enumerate(landmarks_68):
        if do_crop:
            face = {
                "kps": landmark_68,
                "cropped_bgr": None
            }
        else:
            landmark_5 = landmarks_5[landmark_68_index]
            cropped_face, affine_matrix = get_aligned_face_and_affine_matrix(bgr_image, landmark_5)
            cropped_face = enhance_face(gfpgan, cropped_face, image_name, device)
            cropped_face_landmark = trans_points2d(landmark_68, affine_matrix)
            face = {
                "kps": cropped_face_landmark,
                "cropped_bgr": cropped_face
            }
        faces.append(face)
    return faces


def is_face_size_ok(face: Dict[str, np.ndarray], eye_dist_threshold: int):
    if face["kps"] is None:
        return False

    eye_dist = np.linalg.norm([face["kps"][0][0] - face["kps"][1][0], face["kps"][0][1] - face["kps"][1][1]])

    return eye_dist >= eye_dist_threshold


def filter_faces(faces: List[Dict[str, np.ndarray]], eye_dist_threshold: int):
    return list(filter(lambda face: is_face_size_ok(face, eye_dist_threshold), faces))


def verify_retargeted_faces_have_same_length(all_faces: List[List[Dict[str, np.ndarray]]], eye_dist_threshold: int):
    if len(all_faces) > 0:
        length = len(filter_faces(all_faces[0], eye_dist_threshold))
        return all(len(filter_faces(l, eye_dist_threshold)) == length for l in all_faces)
    return True


def preprocess(
    id: str,
    rgb_image: cv2.typing.MatLike,
    face_detector: RetinaFace,
    face_alignment: FaceAlignment,
    live_portrait_pipeline: LivePortraitPipeline,
    gfpgan: GFPGANv1Clean,
    face_parser: BiSeNet,
    aligner: Optional[DifferentiableFaceAligner],
    args: PreprocessArguments,
    cropped_face_path:str,
    cropped_face_path_resized: str,
    output_dir_retargeted: str,
    device: str
):
    if not rgb_image.shape[0] > args.min_original_image_size or not rgb_image.shape[1] > args.min_original_image_size:
        return

    rgb_image = resize_to_limit(rgb_image, max_dim=1280, division=2)

    detected_faces = face_detector(rgb_image, threshold=0.97, return_dict=True)
    detected_faces = filter_faces(detected_faces, args.eye_dist_threshold)
    if len(detected_faces) == 0:
        return

    bboxes = [detected_face["box"] for detected_face in detected_faces]
    kpss = [detected_face["kps"] for detected_face in detected_faces]

    if args.enhance_before_retargeting:
        rgb_image = enhance_faces_in_original_image(gfpgan, face_parser, rgb_image, kpss, id, device)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    if not args.retargeting:
        image_index = 0

        for face_index, kps in enumerate(kpss):
            image_name = f'{id}_{face_index:02d}'

            align_and_save(
                gfpgan,
                bgr_image,
                kps,
                image_index,
                args.final_crop_size,
                cropped_face_path,
                cropped_face_path_resized,
                image_name,
                args.align_mode,
                aligner,
                device,
                args.save_full_size,
                should_enhance_face=True
            )
        return

    landmarks = face_alignment.get_landmarks_from_image(
        rgb_image,
        detected_faces=bboxes,
    )

    if landmarks is None or len(landmarks) == 0:
        return

    faces = get_faces_from_landmarks(gfpgan, bgr_image, landmarks, kpss, args.retargeting_do_crop, id, device)

    retargeted_images = get_retargeted_images(
        live_portrait_pipeline,
        rgb_image,
        bgr_image,
        id,
        faces,
        args.number_of_variants_per_face,
        args.retargeting_do_crop,
        args.retargeting_crop_scale,
        args.filter_valid_faces,
        args.save_retargeted,
        output_dir_retargeted
    )

    if args.retargeting_do_crop:
        if len(retargeted_images) == 0:
            return

        retargeted_images_faces = face_detector(retargeted_images, threshold=0.97, return_dict=True, cv=True)

        if not verify_retargeted_faces_have_same_length(retargeted_images_faces, args.eye_dist_threshold):
            return

        for image_index, retargeted_image in enumerate(retargeted_images):
            retargeted_image_faces = filter_faces(
                retargeted_images_faces[image_index],
                args.eye_dist_threshold
            )

            if len(retargeted_image_faces) == 0:
                continue

            sort_faces_by_coordinates(retargeted_image_faces)
            
            for retargeted_face_index, retargeted_face in enumerate(retargeted_image_faces):
                image_name = f'{id}_{retargeted_face_index:02d}'

                align_and_save(
                    gfpgan,
                    retargeted_image,
                    retargeted_face["kps"],
                    image_index,
                    args.final_crop_size,
                    cropped_face_path,
                    cropped_face_path_resized,
                    image_name,
                    args.align_mode,
                    aligner,
                    device,
                    args.save_full_size,
                    should_enhance_face=True
                )
    else:
        for retargeted_face_index in range(len(retargeted_images)):
            if len(retargeted_images[retargeted_face_index]) == 0:
                continue

            image_name = f'{id}_{retargeted_face_index:02d}'

            face_retargeted_images = [random_horizontal_flip(retargeted_image) for retargeted_image in retargeted_images[retargeted_face_index]]
            retargeted_faces = face_detector(face_retargeted_images, threshold=0.97, return_dict=True, cv=True)

            for image_index, retargeted_face in enumerate(retargeted_faces):
                if len(retargeted_face) == 0:
                    continue

                align_and_save(
                    gfpgan,
                    face_retargeted_images[image_index],
                    retargeted_face[0]["kps"],
                    image_index,
                    args.final_crop_size,
                    cropped_face_path,
                    cropped_face_path_resized,
                    image_name,
                    args.align_mode,
                    aligner,
                    device,
                    args.save_full_size,
                    should_enhance_face=False
                )