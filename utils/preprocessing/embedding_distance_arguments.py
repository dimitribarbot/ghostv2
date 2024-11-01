import os
from typing import Optional
from dataclasses import dataclass

from simple_parsing import choice, flag

def make_real_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", relative_path)

@dataclass
class EmbeddingDistanceArguments:

    """ Data arguments """
    source_image_path: Optional[str] = make_real_path("./examples/images/base/source1.jpg")

    """ Model arguments """
    retina_face_model_path: str = make_real_path("./weights/RetinaFace/Resnet50_Final.safetensors")
    cvlface_aligner_model_path: str = make_real_path("./weights/CVLFace/cvlface_DFA_mobilenet.safetensors")
    face_embeddings: str = choice("facenet", "arcface", "adaface", "cvl_arcface", "cvl_adaface", "cvl_vit", default="cvl_adaface")

    """ Run arguments """
    device_id: int = 0
    source_crop_size: int = 256
    target_crop_size: int = 112
    source_align_mode: str = choice("facexlib", "insightface", "mtcnn", "cvlface", default="insightface")
    target_align_mode: str = choice("facexlib", "insightface", "mtcnn", "cvlface", default="cvlface")

    """ Debug arguments """
    debug: bool = flag(default=False, negative_prefix="--no-")
    debug_source_face_path: str = make_real_path("./examples/results/inference/source_face.jpg")
    debug_target_face_path: str = make_real_path("./examples/results/inference/target_face.jpg")