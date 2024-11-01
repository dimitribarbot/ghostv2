import os
from typing import cast, Any, Optional

from simple_parsing import ArgumentParser

import cv2
import torch
from safetensors.torch import load_file

from CVLFace import get_aligner
from CVLFace.differentiable_face_aligner import DifferentiableFaceAligner
from RetinaFace.detector import RetinaFace
from utils.image_processing import convert_to_batch_tensor, get_aligned_face_and_affine_matrix, get_face_embeddings
from utils.preprocessing.embedding_distance_arguments import EmbeddingDistanceArguments


@torch.no_grad()
def process(
    embedding_model: Any,
    face_detector: RetinaFace,
    aligner: Optional[DifferentiableFaceAligner],
    args: EmbeddingDistanceArguments,
    device: str
):
    bgr_image = cv2.imread(args.source_image_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    detected_faces = face_detector(rgb_image, threshold=0.97, return_dict=True)
    if len(detected_faces) == 0:
        return
    
    lmk = detected_faces[0]["kps"]

    source_cropped_face, _ = get_aligned_face_and_affine_matrix(bgr_image, lmk, args.source_crop_size, args.source_align_mode, aligner, device)
    target_cropped_face, _ = get_aligned_face_and_affine_matrix(bgr_image, lmk, args.target_crop_size, args.target_align_mode, aligner, device)

    if args.debug:
        os.makedirs(os.path.dirname(args.debug_source_face_path), exist_ok=True)
        os.makedirs(os.path.dirname(args.debug_target_face_path), exist_ok=True)
        cv2.imwrite(args.debug_source_face_path, source_cropped_face)
        cv2.imwrite(args.debug_target_face_path, target_cropped_face)

    source_image_face = convert_to_batch_tensor(source_cropped_face, device)
    target_image_face = convert_to_batch_tensor(target_cropped_face, device)

    source_embeddings = get_face_embeddings(source_image_face, embedding_model, args.face_embeddings)
    target_embeddings = get_face_embeddings(target_image_face, embedding_model, args.face_embeddings)

    distance = 1 - torch.nn.CosineSimilarity(dim=1, eps=1e-6)(source_embeddings, target_embeddings).item()

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

    aligner = None
    if args.source_align_mode == "cvlface" or args.target_align_mode == "cvlface":
        aligner = get_aligner(args.cvlface_aligner_model_path, device)

    if args.face_embeddings == "arcface":
        print("Initializing ArcFace model")
        from ArcFace.iresnet import iresnet100
        embedding_model = iresnet100()
        embedding_model.load_state_dict(load_file("./weights/ArcFace/backbone.safetensors"))
        embedding_model.eval()
    elif args.face_embeddings == "adaface":
        print("Initializing AdaFace model")
        from AdaFace.net import build_model
        embedding_model = build_model("ir_101")
        embedding_model.load_state_dict(load_file("./weights/AdaFace/adaface_ir101_webface12m.safetensors"))
        embedding_model.eval()
    elif args.face_embeddings == "cvl_arcface":
        print("Initializing CVL ArcFace model")
        from CVLFace import get_arcface_model
        embedding_model = get_arcface_model("./weights/CVLFace/cvlface_arcface_ir101_webface4m.safetensors")
    elif args.face_embeddings == "cvl_adaface":
        print("Initializing CVL AdaFace model")
        from CVLFace import get_adaface_model
        embedding_model = get_adaface_model("./weights/CVLFace/cvlface_adaface_ir101_webface12m.safetensors")
    elif args.face_embeddings == "cvl_vit":
        print("Initializing CVL ViT model")
        from CVLFace import get_vit_model
        embedding_model = get_vit_model("./weights/CVLFace/cvlface_adaface_vit_base_webface4m.safetensors")
    else:
        print("Initializing Facenet model")
        from facenet.inception_resnet_v1 import InceptionResnetV1
        embedding_model = InceptionResnetV1()
        embedding_model.load_state_dict(load_file("./weights/Facenet/facenet_pytorch.safetensors"))
        embedding_model.eval()

    embedding_model = embedding_model.to(device)

    process(
        embedding_model,
        face_detector,
        aligner,
        args,
        device
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(EmbeddingDistanceArguments, dest="arguments")
    args = cast(EmbeddingDistanceArguments, parser.parse_args().arguments)
    
    main(args)