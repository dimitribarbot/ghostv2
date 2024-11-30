import os
from dataclasses import dataclass
from typing import Optional

from simple_parsing import choice, flag

from utils.embedding_models_arguments import EmbeddingModelsArguments


def make_real_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", relative_path)

@dataclass
class InferenceArguments(EmbeddingModelsArguments):

    """ Data arguments """
    source_file_path: str = make_real_path("./examples/images/base/source1.jpg")
    target_file_path: str = make_real_path("./examples/images/base/target1.jpg")
    output_file_path: str = make_real_path("./examples/results/inference/source1_target1.jpg")

    """ Model params """
    G_path: str = make_real_path("./experiments/saved_models_ghost_v2_5_sch_part2/G_latest.safetensors")
    gfpgan_model_path: str = make_real_path("./weights/GFPGAN/GFPGANv1.4.safetensors")
    face_parser_model_path: str = make_real_path("./weights/BiSeNet/79999_iter.safetensors")
    retina_face_model_path: str = make_real_path("./weights/RetinaFace/Resnet50_Final.safetensors")
    cvlface_aligner_model_path: str = make_real_path("./weights/CVLFace/cvlface_DFA_mobilenet.safetensors")
    face_embeddings: str = choice("facenet", "arcface", "adaface", "cvl_arcface", "cvl_adaface", "cvl_vit", default="cvl_vit")
    backbone: str = choice("unet", "linknet", "resnet", default="unet")
    num_blocks: int = 2
    align_corners: bool = flag(default=False, negative_prefix="--no-")
    precision: Optional[str] = choice(
        None,
        "64",
        "32",
        "16",
        "bf16",
        "transformer-engine",
        "transformer-engine-float16",
        "16-true",
        "16-mixed",
        "bf16-true",
        "bf16-mixed",
        "32-true",
        "64-true",
        default=None
    )

    """ Run arguments """
    device_id: int = 0
    enhance_output: bool = flag(default=True, negative_prefix="--no-")
    source_face_index: int = 0
    target_face_index: int = 0
    align_mode: str = choice("facexlib", "insightface_v1", "insightface_v2", "mtcnn", "cvlface", default="insightface_v2")
    paste_back_mode: str = choice("facexlib_with_parser", "facexlib_without_parser", "insightface", "ghost", "basic", "none", default="ghost")

    """ Debug arguments """
    debug: bool = flag(default=False, negative_prefix="--no-")
    debug_ghost_landmarks: bool = flag(default=False, negative_prefix="--no-")
    debug_source_face_path: str = make_real_path("./examples/results/inference/source_face.jpg")
    debug_target_face_path: str = make_real_path("./examples/results/inference/target_face.jpg")
    debug_swapped_face_path: str = make_real_path("./examples/results/inference/swapped_face.jpg")
    debug_enhanced_face_path: str = make_real_path("./examples/results/inference/enhanced_face.jpg")