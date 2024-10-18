import os
from typing import Optional
from dataclasses import dataclass

from simple_parsing import choice

def make_real_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", relative_path)

@dataclass
class ExampleArguments:

    """ Data arguments """
    source_image: Optional[str] = None
    source_folder: Optional[str] = None
    aligned_folder: str = make_real_path("./examples/images/training_facexlib")

    """ Model arguments """
    retina_face_model_path: str = make_real_path("./weights/RetinaFace/Resnet50_Final.safetensors")

    """ Run arguments """
    device_id: int = 0
    final_crop_size: int = 224
    align_mode: str = choice("facexlib", "insightface", "mtcnn", default="facexlib")