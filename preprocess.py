import os
import cv2
from tqdm import tqdm
import functools
import random
from typing import cast, Dict, List, Tuple, Union
import traceback

from simple_parsing import ArgumentParser

import pyarrow.parquet as pq
import webdataset as wds

import torch
from safetensors.torch import load_file
from torchvision.transforms.functional import normalize
import numpy as np

from GFPGAN.gfpganv1_clean_arch import GFPGANv1Clean
from LivePortrait.pipeline import LivePortraitPipeline, RetargetingParameters
from LivePortrait.utils.io import contiguous, resize_to_limit
from RetinaFace.detector import RetinaFace
from face_alignment import FaceAlignment, LandmarksType
from utils.training.preprocess_arguments import PreprocessArguments
from utils.image_processing import align_warp_face, img2tensor, tensor2img


@torch.no_grad()
def enhance(gfpgan: GFPGANv1Clean, img: cv2.typing.MatLike, image_name: str, device: torch.device, weight=0.5):
    cropped_face_t = img2tensor(img / 255., bgr2rgb=True, float32=True)
    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

    try:
        output = gfpgan(cropped_face_t, return_rgb=False, weight=weight)[0]
        restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
    except RuntimeError as error:
        print(f"Failed inference for GFPGAN for image {image_name}: {error}.")
        restored_face = img

    restored_face = restored_face.astype('uint8')
    
    return restored_face


def get_retargeted_image_ratios_and_landmarks(
    live_portrait_pipeline: LivePortraitPipeline,
    image: cv2.typing.MatLike,
    faces: List[Dict[str, np.ndarray]]
):
    landmarks: List[np.ndarray] = []
    eye_and_lip_ratios: List[Union[Tuple[None, None] | Tuple[float, float]]] = []

    for face_index in range(len(faces)):
        lmk = np.array(faces[face_index]["kps"])
        ratios = live_portrait_pipeline.init_retargeting_image(
            image,
            lmk,
            do_crop=args.retargeting_do_crop,
            crop_scale=args.retargeting_crop_scale
        )
        landmarks.append(lmk)
        eye_and_lip_ratios.append(ratios)

    return landmarks, eye_and_lip_ratios


def get_all_face_retargeting_parameters(faces: List[Dict[str, np.ndarray]], number_of_variants_per_face: int):
    all_parameters: List[List[RetargetingParameters]] = []
    for _ in range(len(faces)):
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


def get_retargeted_images(
    live_portrait_pipeline: LivePortraitPipeline,
    image: cv2.typing.MatLike,
    image_id: int,
    faces: List[Dict[str, np.ndarray]],
    number_of_variants_per_face: int,
    do_crop: bool,
    crop_scale: float,
    save_retargeted: bool,
    output_dir_retargeted: str,
):
    bgr_image = contiguous(image[..., ::-1])

    retargeted_images = [bgr_image]

    if save_retargeted:
        os.makedirs(os.path.join(output_dir_retargeted, f"{image_id}"), exist_ok=True)
        cv2.imwrite(
            os.path.join(output_dir_retargeted, f"{image_id}", "0.jpg"),
            bgr_image
        )

    landmarks, eye_and_lip_ratios = get_retargeted_image_ratios_and_landmarks(
        live_portrait_pipeline,
        image,
        faces
    )

    all_parameters: List[List[RetargetingParameters]] = get_all_face_retargeting_parameters(
        faces,
        number_of_variants_per_face
    )

    retargeted_image_variations = live_portrait_pipeline.execute_image_retargeting_multi(
        bgr_image.copy(),
        landmarks,
        eye_and_lip_ratios,
        all_parameters,
        do_crop,
        crop_scale,
    )

    if retargeted_image_variations is not None:
        retargeted_image_variations = [cv2.cvtColor(variation, cv2.COLOR_RGB2BGR) for variation in retargeted_image_variations if variation is not None]

        if save_retargeted:
            for i in range(len(retargeted_image_variations)):
                cv2.imwrite(
                    os.path.join(output_dir_retargeted, f"{image_id}", f"{i + 1}.jpg"),
                    retargeted_image_variations[i]
                )

        retargeted_images += retargeted_image_variations
    
    return retargeted_images


def align_and_save(
    gfpgan: GFPGANv1Clean,
    image: cv2.typing.MatLike,
    lmk: np.ndarray,
    image_index: int,
    final_crop_size: Tuple[int, int],
    cropped_face_path:str,
    cropped_face_path_resized: str,
    image_name: str,
    device: str,
    save_full_size: bool
):
    if save_full_size:
        save_path = os.path.join(cropped_face_path, image_name)
        os.makedirs(save_path, exist_ok=True)
    save_path_resized = os.path.join(cropped_face_path_resized, image_name)
    os.makedirs(save_path_resized, exist_ok=True)
    
    cropped_face = align_warp_face(image, lmk)
    cropped_face = enhance(gfpgan, cropped_face, image_name, device)
    if save_full_size:
        cv2.imwrite(os.path.join(save_path, f"{image_index}.jpg"), cropped_face)
    cropped_face = cv2.resize(cropped_face, final_crop_size)
    cv2.imwrite(os.path.join(save_path_resized, f"{image_index}.jpg"), cropped_face)


def preprocess_data(sample):
    image, json = sample
    id = int(json["SAMPLE_ID"])
    return image, id


def extract_parquet_files(laion_data_dir: str, output_dir: str, split_folder: str):
    parquet_files = []
    for subdir, dirs, files in os.walk(os.path.join(laion_data_dir, split_folder)):
        for file in sorted(files):
            ext = os.path.splitext(file)[-1].lower()
            if ext == ".parquet":
                parquet_file = os.path.join(subdir, file)
                cropped_face_path = os.path.join(output_dir, split_folder, os.path.basename(os.path.splitext(parquet_file)[0]))
                if not os.path.exists(cropped_face_path):
                    parquet_files.append(parquet_file)
    return parquet_files


def is_face_size_ok(face: Dict[str, np.ndarray], min_original_face_size: int):
    if face["box"] is None:
        return False
    
    # (x1, y1)
    #
    #
    #
    #                       (x2, y2)

    x1 = face["box"][0]
    y1 = face["box"][1]

    x2 = face["box"][2]
    y2 = face["box"][3]
    
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    return width >= min_original_face_size and height >= min_original_face_size


def get_face_sort_key(face1: Dict[str, np.ndarray], face2: Dict[str, np.ndarray]):
    if face1["box"] is None:
        if face2["box"] is None:
            return 0
        return -1
    
    if face2["box"] is None:
        return 1
    
    # (x1, y1)
    #
    #          (c1, c2)
    #
    #                       (x2, y2)

    x1_1 = face1["box"][0]
    y1_1 = face1["box"][1]
    x2_1 = face1["box"][2]
    y2_1 = face1["box"][3]
    cx_1 = (x1_1 + x2_1) / 2
    cy_1 = (y1_1 + y2_1) / 2

    x1_2 = face2["box"][0]
    y1_2 = face2["box"][1]
    x2_2 = face2["box"][2]
    y2_2 = face2["box"][3]
    cx_2 = (x1_2 + x2_2) / 2
    cy_2 = (y1_2 + y2_2) / 2

    d1 = cx_1 * cx_1 + cy_1 * cy_1
    d2 = cx_2 * cx_2 + cy_2 * cy_2

    if d1 < d2:
        return -1
    
    if d1 > d2:
        return 1

    return 0


def filter_faces(faces: List[Dict[str, np.ndarray]], min_original_face_size: int):
    return list(filter(lambda face: is_face_size_ok(face, min_original_face_size), faces))


def verify_retargeted_faces_have_same_length(all_faces: List[List[Dict[str, np.ndarray]]], min_original_face_size: int):
    if len(all_faces) > 0:
        length = len(filter_faces(all_faces[0], min_original_face_size))
        return all(len(filter_faces(l, min_original_face_size)) == length for l in all_faces)
    return True


def process(
    face_detector: RetinaFace,
    face_alignment: FaceAlignment,
    live_portrait_pipeline: LivePortraitPipeline,
    gfpgan: GFPGANv1Clean,
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

            for image, id in tqdm(dataset, total=len(df), desc=f"images {base_filename}", leave=False):
                try:
                    if not image.shape[0] > args.min_original_image_size or not image.shape[1] > args.min_original_image_size:
                        continue

                    image = resize_to_limit(image, max_dim=1280, division=2)

                    detected_faces = face_detector(image, threshold=0.99, return_dict=True)
                    detected_faces = filter_faces(detected_faces, args.min_original_face_size)
                    if len(detected_faces) == 0:
                        continue

                    bboxes = [detected_face["box"] for detected_face in detected_faces]

                    landmarks = face_alignment.get_landmarks_from_image(
                        image,
                        detected_faces=bboxes,
                    )

                    if landmarks is None or len(landmarks) == 0:
                        continue

                    faces = []
                    for landmark in landmarks:
                        faces.append({
                            "kps": landmark
                        })

                    retargeted_images = get_retargeted_images(
                        live_portrait_pipeline,
                        image,
                        id,
                        faces,
                        args.number_of_variants_per_face,
                        args.retargeting_do_crop,
                        args.retargeting_crop_scale,
                        args.save_retargeted,
                        args.output_dir_retargeted
                    )

                    retargeted_images_faces = face_detector(retargeted_images, threshold=0.99, return_dict=True, cv=True)

                    if not verify_retargeted_faces_have_same_length(retargeted_images_faces, args.min_original_face_size):
                        continue

                    for image_index, retargeted_image in enumerate(retargeted_images):
                        retargeted_image_faces = filter_faces(
                            retargeted_images_faces[image_index],
                            args.min_original_face_size
                        )

                        if len(retargeted_image_faces) == 0:
                            continue

                        retargeted_image_faces.sort(key=functools.cmp_to_key(get_face_sort_key))
                        
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
                                device,
                                args.save_full_size
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

    process(face_detector, face_alignment, live_portrait_pipeline, gfpgan, args, device)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(PreprocessArguments, dest="arguments")
    args = cast(PreprocessArguments, parser.parse_args().arguments)
    
    main(args)