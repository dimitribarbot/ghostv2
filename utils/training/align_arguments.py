import os
from typing import Optional
from dataclasses import dataclass

from simple_parsing import choice
from simple_parsing.helpers import flag

def make_real_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", relative_path)

@dataclass
class AlignArguments:

    """ Data arguments """
    source_image: Optional[str] = None
    source_folder: Optional[str] = None
    aligned_folder: str = make_real_path("./examples/images/training_facexlib")
    output_extension: str = choice(".png", ".jpg", "same_as_source", default="same_as_source")

    """ Model arguments """
    retina_face_model_path: str = make_real_path("./weights/RetinaFace/Resnet50_Final.safetensors")
    cvlface_aligner_model_path: str = make_real_path("./weights/CVLFace/cvlface_DFA_mobilenet.safetensors")

    """ Run arguments """
    device_id: int = 0
    final_crop_size: int = 256
    align_mode: str = choice("facexlib", "insightface", "mtcnn", "cvlface", default="facexlib")
    overwrite: bool = flag(default=False, negative_prefix="--no-")