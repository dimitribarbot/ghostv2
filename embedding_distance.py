import os
from typing import cast, Any, Optional

from simple_parsing import ArgumentParser

import cv2
import torch
import torch.nn.functional as F

from CVLFace import get_aligner
from CVLFace.differentiable_face_aligner import DifferentiableFaceAligner
from CVLFace.utils import load_model_by_repo_id, tensor_to_numpy
from RetinaFace.detector import RetinaFace
from utils.image_processing import convert_to_batch_tensor, get_aligned_face_and_affine_matrix, get_face_embeddings, initialize_embedding_model
from utils.preprocessing.embedding_distance_arguments import EmbeddingDistanceArguments


@torch.no_grad()
def process(
    source_embedding_model: Any,
    target_embedding_model: Any,
    face_detector: RetinaFace,
    original_aligner: Optional[DifferentiableFaceAligner],
    aligner: Optional[DifferentiableFaceAligner],
    args: EmbeddingDistanceArguments,
    device: str
):
    bgr_image = cv2.imread(args.source_image_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    detected_faces = face_detector(rgb_image, threshold=args.detection_threshold, return_dict=True)
    if len(detected_faces) == 0:
        return
    
    lmk = detected_faces[0]["kps"]
    
    if args.source_align_mode == "original_cvlface":
        source_cropped_face, _, _, _, _, _ = original_aligner(convert_to_batch_tensor(rgb_image, device))
        source_cropped_face = F.interpolate(source_cropped_face, [args.target_crop_size, args.target_crop_size], mode="bilinear")
        source_cropped_face = tensor_to_numpy(source_cropped_face)
    else:
        source_cropped_face, _ = get_aligned_face_and_affine_matrix(bgr_image, lmk, args.source_crop_size, args.source_align_mode, aligner, device)
    
    if args.target_align_mode == "original_cvlface":
        target_cropped_face, _, _, _, _, _ = original_aligner(convert_to_batch_tensor(rgb_image, device))
        target_cropped_face = F.interpolate(target_cropped_face, [args.target_crop_size, args.target_crop_size], mode="bilinear")
        target_cropped_face = tensor_to_numpy(target_cropped_face)
    else:
        target_cropped_face, _ = get_aligned_face_and_affine_matrix(bgr_image, lmk, args.target_crop_size, args.target_align_mode, aligner, device)

    if args.debug:
        os.makedirs(os.path.dirname(args.debug_source_face_path), exist_ok=True)
        os.makedirs(os.path.dirname(args.debug_target_face_path), exist_ok=True)
        cv2.imwrite(args.debug_source_face_path, source_cropped_face)
        cv2.imwrite(args.debug_target_face_path, target_cropped_face)

    source_image_face = convert_to_batch_tensor(source_cropped_face, device)
    target_image_face = convert_to_batch_tensor(target_cropped_face, device)

    source_embeddings = get_face_embeddings(source_image_face, source_embedding_model, args.source_face_embeddings)
    target_embeddings = get_face_embeddings(target_image_face, target_embedding_model, args.target_face_embeddings)

    distance = 1 - F.cosine_similarity(source_embeddings, target_embeddings).item()

    print(f"Distance between embeddings is: {distance}")


def main(args: EmbeddingDistanceArguments):
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
        gpu_id=0,
        fp16=True,
        model_path=args.retina_face_model_path
    )

    original_aligner = None
    if args.source_align_mode == "original_cvlface" or args.target_align_mode == "original_cvlface":
        original_aligner = load_model_by_repo_id(
            args.cvlface_original_aligner_repository,
            args.cvlface_original_aligner_model_path
        )
        original_aligner = original_aligner.to(device)

    aligner = None
    if args.source_align_mode == "cvlface" or args.target_align_mode == "cvlface":
        aligner = get_aligner(args.cvlface_aligner_model_path, device)

    source_embedding_model = initialize_embedding_model(args.source_face_embeddings, args, device)
    if args.target_face_embeddings != args.source_face_embeddings:
        target_embedding_model = initialize_embedding_model(args.target_face_embeddings, args, device)
    else:
        target_embedding_model = source_embedding_model

    process(
        source_embedding_model,
        target_embedding_model,
        face_detector,
        original_aligner,
        aligner,
        args,
        device
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(EmbeddingDistanceArguments, dest="arguments")
    args = cast(EmbeddingDistanceArguments, parser.parse_args().arguments)
    
    main(args)