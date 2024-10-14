import os
import cv2
from typing import cast, Optional

from simple_parsing import ArgumentParser

from RetinaFace.detector import RetinaFace
from utils.training.example_arguments import ExampleArguments
from utils.image_processing import align_warp_face


def process_one_image(
    source_image: str,
    aligned_folder: str,
    face_detector: RetinaFace,
    final_crop_size: int,
):
    print(f"Processing image {source_image}.")
    image = cv2.imread(source_image, cv2.IMREAD_COLOR)
    image_name = os.path.splitext(os.path.basename(source_image))[0]

    save_path = os.path.join(aligned_folder, f"{image_name}.png")
    os.makedirs(aligned_folder, exist_ok=True)

    detected_faces = face_detector(image, threshold=0.99, return_dict=True, cv=True)
    if len(detected_faces) == 0:
        raise ValueError(f"No face detected in source image {source_image}.")

    cropped_face = align_warp_face(image, detected_faces[0]["kps"])
    cropped_face = cv2.resize(cropped_face, (final_crop_size, final_crop_size))
    print(f"Saving cropped face to {save_path}.")
    cv2.imwrite(save_path, cropped_face)


def process(
    source_image: Optional[str],
    source_folder: Optional[str],
    aligned_folder: str,
    face_detector: RetinaFace,
    final_crop_size: int,
):
    if source_image is None and source_folder is None:
        raise ValueError("Arguments 'source_image' and 'source_folder' cannot be both empty.")

    if source_folder is None:
        if not os.path.exists(source_image):
            raise ValueError(f"Arguments 'source_image' {source_image} points to a file that does not exist.")

        process_one_image(source_image, aligned_folder, face_detector, final_crop_size)
    else:
        if not os.path.exists(source_folder):
            raise ValueError(f"Arguments 'source_folder' {source_folder} points to a folder that does not exist.")
    
        print(f"Processing images in folder {source_folder}.")        
        for dirpath, _, source_folder_images in os.walk(source_folder):
            for source_folder_image in source_folder_images:
                process_one_image(os.path.join(dirpath, source_folder_image), aligned_folder, face_detector, final_crop_size)


def main(args: ExampleArguments):
    face_detector = RetinaFace(
        gpu_id=args.device_id,
        fp16=True,
        model_path=args.retina_face_model_path
    )

    process(
        args.source_image,
        args.source_folder,
        args.aligned_folder,
        face_detector,
        args.final_crop_size
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ExampleArguments, dest="arguments")
    args = cast(ExampleArguments, parser.parse_args().arguments)
    
    main(args)