import os
import cv2
from tqdm import tqdm
import random
from typing import cast, Dict, List, Literal, Tuple
import traceback

from simple_parsing import ArgumentParser

import pyarrow.parquet as pq
import webdataset as wds

import torch
from safetensors.torch import load_file
import numpy as np

from BiSeNet.bisenet import BiSeNet
from GFPGAN.gfpganv1_clean_arch import GFPGANv1Clean
from LivePortrait.pipeline import LivePortraitPipeline, RetargetingParameters
from LivePortrait.utils.io import contiguous, resize_to_limit
from RetinaFace.detector import RetinaFace
from face_alignment import FaceAlignment, LandmarksType
from utils.training.preprocess_arguments import PreprocessArguments
from utils.image_processing import get_aligned_face_and_affine_matrix, paste_face_back, \
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
        upsample_img = paste_face_back(face_parser, upsample_img, restored_face, affine_matrix, device)

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
                wink=round(random.uniform(0, 10), 2),
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
    )

    if do_crop:
        retargeted_images = [bgr_image]

        if retargeted_image_variations is not None:
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

        if retargeted_image_variations is not None:
            for face_index in range(len(retargeted_image_variations)):
                retargeted_image_variations[face_index] = [
                    cv2.cvtColor(variation, cv2.COLOR_RGB2BGR) for variation in retargeted_image_variations[face_index] if variation is not None
                ]
                retargeted_images[face_index] += retargeted_image_variations[face_index]
    
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
    device: str,
    save_full_size: bool,
    should_enhance_face: bool
):
    if save_full_size:
        save_path = os.path.join(cropped_face_path, image_name)
        os.makedirs(save_path, exist_ok=True)
    save_path_resized = os.path.join(cropped_face_path_resized, image_name)
    os.makedirs(save_path_resized, exist_ok=True)
    
    if should_enhance_face:
        if align_mode == "facexlib":
            cropped_face, _ = get_aligned_face_and_affine_matrix(bgr_image, lmk)
            cropped_face = enhance_face(gfpgan, cropped_face, image_name, device)
        else:
            cropped_face, affine_matrix = get_aligned_face_and_affine_matrix(bgr_image, lmk)
            cropped_face = enhance_face(gfpgan, cropped_face, image_name, device)
            transformed_lmk = trans_points2d(lmk, affine_matrix)
            cropped_face, _ = get_aligned_face_and_affine_matrix(bgr_image, transformed_lmk, 512, align_mode)
    else:
        cropped_face, _ = get_aligned_face_and_affine_matrix(bgr_image, lmk, 512, align_mode)
    if save_full_size:
        cv2.imwrite(os.path.join(save_path, f"{image_index}.jpg"), cropped_face)
    cropped_face = cv2.resize(cropped_face, final_crop_size, interpolation=cv2.INTER_LINEAR)
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


def preprocess_data(sample):
    image, json = sample
    id = int(json["SAMPLE_ID"])
    return image, id


def extract_parquet_files(laion_data_dir: str, output_dir: str, split_folder: str):
    parquet_files = []
    for subdir, _, files in os.walk(os.path.join(laion_data_dir, split_folder)):
        for file in sorted(files):
            ext = os.path.splitext(file)[-1].lower()
            if ext == ".parquet":
                parquet_file = os.path.join(subdir, file)
                cropped_face_path = os.path.join(output_dir, split_folder, os.path.basename(os.path.splitext(parquet_file)[0]))
                if not os.path.exists(cropped_face_path):
                    parquet_files.append(parquet_file)
    return parquet_files


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


@torch.no_grad()
def process(
    face_detector: RetinaFace,
    face_alignment: FaceAlignment,
    live_portrait_pipeline: LivePortraitPipeline,
    gfpgan: GFPGANv1Clean,
    face_parser: BiSeNet,
    args: PreprocessArguments,
    device: str
):
    laion_face_base_dir = args.laion_face_base_dir
    output_dir = args.output_dir
    output_dir_resized = args.output_dir_resized

    laion_data_dir = os.path.join(laion_face_base_dir, "laion_face_data")

    final_crop_size = (args.final_crop_size, args.final_crop_size)

    for laion_face_part_index in args.laion_face_part_indices:
        split_folder = f"split_{laion_face_part_index:05d}"

        parquet_files = extract_parquet_files(laion_data_dir, output_dir_resized, split_folder)

        for parquet_file in tqdm(parquet_files, total=len(parquet_files), desc="tar files"):
            base_pathname = os.path.splitext(parquet_file)[0]
            base_filename = os.path.basename(base_pathname)

            cropped_face_path = os.path.join(output_dir, split_folder, base_filename)
            cropped_face_path_resized = os.path.join(output_dir_resized, split_folder, base_filename)
            if os.path.exists(cropped_face_path_resized):
                continue

            df = pq.read_table(parquet_file).to_pandas()
            df = df[df["status"] == "success"]

            dataset = (
                wds.WebDataset(f"{base_pathname}.tar", shardshuffle=False)
                    .decode("rgb8")
                    .to_tuple("jpg;png", "json")
                    .map(preprocess_data)
            )

            for rgb_image, id in tqdm(dataset, total=len(df), desc=f"images {base_filename}", leave=False):
                try:
                    if not rgb_image.shape[0] > args.min_original_image_size or not rgb_image.shape[1] > args.min_original_image_size:
                        continue

                    rgb_image = resize_to_limit(rgb_image, max_dim=1280, division=2)

                    detected_faces = face_detector(rgb_image, threshold=0.97, return_dict=True)
                    detected_faces = filter_faces(detected_faces, args.eye_dist_threshold)
                    if len(detected_faces) == 0:
                        continue

                    bboxes = [detected_face["box"] for detected_face in detected_faces]
                    kpss = [detected_face["kps"] for detected_face in detected_faces]
                    
                    if args.enhance_before_retargeting:
                        rgb_image = enhance_faces_in_original_image(gfpgan, face_parser, rgb_image, kpss, id, device)

                    landmarks = face_alignment.get_landmarks_from_image(
                        rgb_image,
                        detected_faces=bboxes,
                    )

                    if landmarks is None or len(landmarks) == 0:
                        continue
                    
                    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

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
                        args.save_retargeted,
                        args.output_dir_retargeted
                    )

                    if args.retargeting_do_crop:
                        retargeted_images_faces = face_detector(retargeted_images, threshold=0.97, return_dict=True, cv=True)

                        if not verify_retargeted_faces_have_same_length(retargeted_images_faces, args.eye_dist_threshold):
                            continue

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
                                    final_crop_size,
                                    cropped_face_path,
                                    cropped_face_path_resized,
                                    image_name,
                                    args.align_mode,
                                    device,
                                    args.save_full_size,
                                    should_enhance_face=True
                                )
                    else:
                        for retargeted_face_index in range(len(retargeted_images)):
                            image_name = f'{id}_{retargeted_face_index:02d}'

                            face_retargeted_images = [random_horizontal_flip(retargeted_image) for retargeted_image in retargeted_images[retargeted_face_index]]
                            retargeted_faces = face_detector(face_retargeted_images, threshold=0.97, return_dict=True, cv=True)

                            for image_index, retargeted_face in enumerate(retargeted_faces):
                                align_and_save(
                                    gfpgan,
                                    face_retargeted_images[image_index],
                                    retargeted_face[0]["kps"],
                                    image_index,
                                    final_crop_size,
                                    cropped_face_path,
                                    cropped_face_path_resized,
                                    image_name,
                                    args.align_mode,
                                    device,
                                    args.save_full_size,
                                    should_enhance_face=False
                                )
                except Exception as ex:
                    print(f"An error occurred for sample {id}: {ex}")
                    traceback.print_tb(ex.__traceback__)


def main(args: PreprocessArguments):
    try:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda:" + str(args.device_id)
        else:
            device = "cpu"
    except:
        device = "cpu"

    if device == "cpu":
        print("Nor Cuda nor MPS are available, using CPU. Check if it's ok.")

    gfpgan = GFPGANv1Clean(
        out_size=512,
        num_style_feat=512,
        channel_multiplier=2,
        decoder_load_path=None,
        fix_decoder=False,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True
    )
    gfpgan.load_state_dict(load_file(args.gfpgan_model_path), strict=True)
    gfpgan.eval()
    gfpgan = gfpgan.to(device)

    face_parser = BiSeNet(num_class=19)
    face_parser.load_state_dict(load_file(args.face_parser_model_path), strict=True)
    face_parser.eval()
    face_parser = face_parser.to(device)

    face_detector = RetinaFace(
        gpu_id=0,
        fp16=True,
        model_path=args.retina_face_model_path
    )

    face_alignment = FaceAlignment(
        LandmarksType.TWO_D,
        flip_input=False,
        device=device,
        dtype=torch.float16,
        face_detector="sfd",
        face_detector_kwargs={
            "path_to_detector": args.face_alignment_model_path
        }
    )

    live_portrait_pipeline = LivePortraitPipeline(
        args.live_portrait_landmark_model_path,
        args.live_portrait_F_model_path,
        args.live_portrait_M_model_path,
        args.live_portrait_W_model_path,
        args.live_portrait_G_model_path,
        args.live_portrait_S_model_path,
        device
    )

    process(
        face_detector,
        face_alignment,
        live_portrait_pipeline,
        gfpgan,
        face_parser,
        args,
        device
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(PreprocessArguments, dest="arguments")
    args = cast(PreprocessArguments, parser.parse_args().arguments)
    
    main(args)