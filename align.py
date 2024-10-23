import os
from typing import cast, Optional

from simple_parsing import ArgumentParser
from tqdm import tqdm

import cv2
import torch

from CVLFace import get_aligner
from CVLFace.differentiable_face_aligner import DifferentiableFaceAligner
from RetinaFace.detector import RetinaFace
from utils.training.align_arguments import AlignArguments
from utils.image_processing import get_aligned_face_and_affine_matrix


def get_save_path(root_folder: str, source_image: str, aligned_folder: str):
    image_name = os.path.splitext(os.path.basename(source_image))[0]
    relative_path = os.path.relpath(os.path.dirname(source_image), root_folder)
    save_folder = os.path.join(aligned_folder, relative_path)
    save_path = os.path.join(save_folder, f"{image_name}.png")
    return save_path


def process_one_image(
    source_image: str,
    save_path: str,
    face_detector: RetinaFace,
    final_crop_size: int,
    align_mode: str,
    aligner: Optional[DifferentiableFaceAligner],
    device: Optional[torch.device]=None,
):
    image = cv2.imread(source_image, cv2.IMREAD_COLOR)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    detected_faces = face_detector(image, threshold=0.97, return_dict=True, cv=True)
    if len(detected_faces) == 0:
        return None

    cropped_face, _ = get_aligned_face_and_affine_matrix(
        image,
        detected_faces[0]["kps"],
        final_crop_size,
        align_mode,
        aligner,
        device
    )

    cv2.imwrite(save_path, cropped_face)

    return save_path


def process(
    source_image: Optional[str],
    source_folder: Optional[str],
    aligned_folder: str,
    face_detector: RetinaFace,
    final_crop_size: int,
    overwrite: bool,
    align_mode: str,
    aligner: Optional[DifferentiableFaceAligner],
    device: Optional[torch.device]=None,
):
    if source_image is None and source_folder is None:
        raise ValueError("Arguments 'source_image' and 'source_folder' cannot be both empty.")

    if source_folder is None:
        if not os.path.exists(source_image):
            raise ValueError(f"Arguments 'source_image' {source_image} points to a file that does not exist.")
        
        root_folder = os.path.dirname(source_image)
        save_path = get_save_path(root_folder, source_image, aligned_folder)
        if overwrite or not os.path.exists(save_path):
            print(f"Processing image {source_image}.")
            save_path = process_one_image(
                source_image,
                save_path,
                face_detector,
                final_crop_size,
                align_mode,
                aligner,
                device
            )
            if save_path is not None:
                print(f"Saving cropped face to {save_path}.")
            else:
                print(f"No face detected in source image {source_image}.")
        else:
            print(f"Not processing image {source_image} as target {save_path} already exists and overwrite is false.")
    else:
        if not os.path.exists(source_folder):
            raise ValueError(f"Arguments 'source_folder' {source_folder} points to a folder that does not exist.")
    
        print(f"Processing images in folder {source_folder}.")
        print("Counting number of files to process.")
        total = sum([len(list(filter(lambda file: overwrite or not os.path.exists(get_save_path(source_folder, os.path.join(root, file), aligned_folder)), files))) \
                     for root, _, files in os.walk(source_folder)])
        print(f"Number of files to process: {total}.")
        with tqdm(total=total) as pbar:
            for root, _, files in os.walk(source_folder):
                for file in files:
                    source_image = os.path.join(root, file)
                    save_path = get_save_path(source_folder, source_image, aligned_folder)
                    if overwrite or not os.path.exists(save_path):
                        save_path = process_one_image(
                            source_image,
                            save_path,
                            face_detector,
                            final_crop_size,
                            align_mode,
                            aligner,
                            device
                        )
                        if save_path is None:
                            print(f"No face detected in source image {source_image}.")
                        pbar.update()


def main(args: AlignArguments):
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

    face_detector = RetinaFace(
        gpu_id=args.device_id,
        fp16=True,
        model_path=args.retina_face_model_path
    )

    aligner = None
    if args.align_mode == "cvlface":
        aligner = get_aligner(args.cvlface_aligner_model_path, device)

    process(
        args.source_image,
        args.source_folder,
        args.aligned_folder,
        face_detector,
        args.final_crop_size,
        args.overwrite,
        args.align_mode,
        aligner,
        device
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(AlignArguments, dest="arguments")
    args = cast(AlignArguments, parser.parse_args().arguments)
    
    main(args)