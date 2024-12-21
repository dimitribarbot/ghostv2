import os
from typing import Optional
from dataclasses import dataclass

from simple_parsing import choice, flag

from utils.embedding_models_arguments import EmbeddingModelsArguments


def make_real_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", relative_path)

@dataclass
class EmbeddingDistanceArguments(EmbeddingModelsArguments):

    """ Data arguments """
    source_image_path: Optional[str] = make_real_path("./examples/images/base/source1.jpg")

    """ Model arguments """
    retina_face_model_path: str = make_real_path("./weights/RetinaFace/Resnet50_Final.safetensors")
    cvlface_original_aligner_model_path: str = make_real_path("./weights/CVLFace/cvlface_DFA_mobilenet")  # First run "pip install timm huggingface_hub transformers" to use this
    cvlface_original_aligner_repository: str = "minchul/cvlface_DFA_mobilenet"  # First run "pip install timm huggingface_hub transformers" to use this
    cvlface_aligner_model_path: str = make_real_path("./weights/CVLFace/cvlface_DFA_mobilenet.safetensors")
    source_face_embeddings: str = choice("facenet", "arcface", "adaface", "cvl_arcface", "cvl_adaface", "cvl_vit", default="cvl_vit")
    target_face_embeddings: str = choice("facenet", "arcface", "adaface", "cvl_arcface", "cvl_adaface", "cvl_vit", default="cvl_vit")

    """ Run arguments """
    device_id: int = 0
    detection_threshold: float = 0.97
    source_crop_size: int = 256
    target_crop_size: int = 112
    source_align_mode: str = choice("facexlib", "insightface_v1", "insightface_v2", "mtcnn", "cvlface", "original_cvlface", default="insightface_v1")
    target_align_mode: str = choice("facexlib", "insightface_v1", "insightface_v2", "mtcnn", "cvlface", "original_cvlface", default="insightface_v2")

    """ Debug arguments """
    debug: bool = flag(default=False, negative_prefix="--no-")
    debug_source_face_path: str = make_real_path("./examples/results/inference/source_face.jpg")
    debug_target_face_path: str = make_real_path("./examples/results/inference/target_face.jpg")