import os
from dataclasses import dataclass
from typing import List

from simple_parsing.helpers import list_field

from utils.preprocessing.preprocess_arguments import PreprocessArguments

def make_real_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", relative_path)

@dataclass
class PreprocessLaionArguments(PreprocessArguments):

    """ Data arguments """
    laion_face_base_dir: str = "/home/dimitribarbot/datasets/LAION-Face"
    laion_face_part_indices: List[int] = list_field(default=[0])
    output_dir: str = "/home/dimitribarbot/datasets/LAION-Face/laion_face_cropped_512x512_unaligned"
    output_dir_resized: str = "/home/dimitribarbot/datasets/LAION-Face/laion_face_cropped_256x256"
    output_dir_retargeted: str = "/home/dimitribarbot/datasets/LAION-Face/laion_face_retargeted_data"