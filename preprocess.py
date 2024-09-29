import os
import cv2
from tqdm import tqdm
import random
from typing import cast

from simple_parsing import ArgumentParser

import pyarrow.parquet as pq
import webdataset as wds

import torch
from safetensors.torch import load_file
from torchvision.transforms.functional import normalize
import numpy as np

from GFPGAN.gfpganv1_clean_arch import GFPGANv1Clean
from LivePortrait.pipeline import LivePortraitPipeline
from RetinaFace.detector import RetinaFace
from utils.training.preprocess_arguments import PreprocessArguments
from utils.training.image_processing import align_warp_face, img2tensor, tensor2img


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


def get_retargeted_images(
        live_portrait_pipeline: LivePortraitPipeline,
        image: cv2.typing.MatLike,
        lmk: np.ndarray,
        number_of_variants_per_face: int,
        do_crop: bool,
        crop_scale: float,
        save_retargeted: bool,
        retargeted_path: str,
):
    if save_retargeted:
        retargeted_path = "/home/dimitribarbot/datasets/LAION-Face/laion_face_retargeted_data"
        os.makedirs(retargeted_path, exist_ok=True)

        retargeted_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(retargeted_path, "0.jpg"), retargeted_image)

    retargeted_images = []

    source_eye_ratio, source_lip_ratio = live_portrait_pipeline.init_retargeting_image(
        image, lmk,
        do_crop=do_crop,
        crop_scale=crop_scale
    )
    if source_eye_ratio is not None and source_lip_ratio is not None:
        for i in range(number_of_variants_per_face):
            _, retargeted_image = live_portrait_pipeline.execute_image_retargeting(
                image,
                lmk,
                source_eye_ratio,
                source_lip_ratio,
                input_eye_ratio=round(random.uniform(0.32, 0.42), 2),
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
                do_crop=do_crop,
                crop_scale=crop_scale,
            )
            if retargeted_image is not None:
                if save_retargeted:
                    cv2.imwrite(os.path.join(retargeted_path, f"{i +  1}.jpg"), retargeted_image)
                retargeted_image = cv2.cvtColor(retargeted_image, cv2.COLOR_RGB2BGR)
                retargeted_images.append(retargeted_image)

    return retargeted_images


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


def process(
        face_detector: RetinaFace,
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
            if not os.path.exists(cropped_face_path_resized):
                os.makedirs(cropped_face_path, exist_ok=True)
                os.makedirs(cropped_face_path_resized)

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
                        if image.shape[0] > 250 and image.shape[1] > 250:
                            faces = face_detector(image, threshold=0.99, return_dict=True)
                            for face_index in range(len(faces)):
                                lmk = faces[face_index]["kps"]
                                image_name = f'{id}_{face_index:02d}'
                                retargeted_images = get_retargeted_images(
                                    live_portrait_pipeline,
                                    image,
                                    np.array(lmk),
                                    args.number_of_variants_per_face,
                                    args.retargeting_do_crop,
                                    args.retargeting_crop_scale,
                                    args.save_retargeted,
                                    args.output_dir_retargeted
                                )
                                if len(retargeted_images) > 0:
                                    save_path = os.path.join(cropped_face_path, image_name)
                                    save_path_resized = os.path.join(cropped_face_path_resized, image_name)
                                    os.makedirs(save_path, exist_ok=True)
                                    os.makedirs(save_path_resized, exist_ok=True)

                                    cropped_face = align_warp_face(image, lmk)
                                    cropped_face = enhance(gfpgan, cropped_face, image_name, device)
                                    cv2.imwrite(os.path.join(save_path, "0.jpg"), cropped_face)
                                    cropped_face = cv2.resize(cropped_face, final_crop_size)
                                    cv2.imwrite(os.path.join(save_path_resized, "0.jpg"), cropped_face)

                                    retargeted_faces = face_detector(retargeted_images, threshold=0.99, return_dict=True)

                                    for i, retargeted_image in enumerate(retargeted_images):
                                        if retargeted_faces[i] is not None and len(retargeted_faces[i]) > face_index:
                                            if len(faces) != len(retargeted_faces[i]):
                                                print(f"Found {len(faces)} faces in original image {id} and {len(retargeted_faces[i])} in retargeted one. Retargeted images will not be saved.")
                                                continue
                                            cropped_face = align_warp_face(retargeted_image, retargeted_faces[i][face_index]["kps"])
                                            cropped_face = enhance(gfpgan, cropped_face, image_name, device)
                                            cv2.imwrite(os.path.join(save_path, f"{i + 1}.jpg"), cropped_face)
                                            cropped_face = cv2.resize(cropped_face, final_crop_size)
                                            cv2.imwrite(os.path.join(save_path_resized, f"{i + 1}.jpg"), cropped_face)
                    except Exception as ex:
                        print(f"An error occurred for sample {id}: {ex}")


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

    live_portrait_pipeline = LivePortraitPipeline(
        args.live_portrait_landmark_model_path,
        args.live_portrait_F_model_path,
        args.live_portrait_M_model_path,
        args.live_portrait_W_model_path,
        args.live_portrait_G_model_path,
        args.live_portrait_S_model_path,
        device
    )

    process(face_detector, live_portrait_pipeline, gfpgan, args, device)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(PreprocessArguments, dest="arguments")
    args = cast(PreprocessArguments, parser.parse_args().arguments)
    
    main(args)