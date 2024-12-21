import os
from dataclasses import dataclass

from simple_parsing.helpers import flag

from utils.preprocessing.preprocess_arguments import PreprocessArguments


def make_real_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", relative_path)

@dataclass
class PreprocessICartoonFaceArguments(PreprocessArguments):
    """ Model arguments """
    dass_face_model_path: str = make_real_path("./weights/DASS/xl_mixdata_finetuned_stage3.safetensors")

    """ Data arguments """
    icartoonface_base_dir: str = make_real_path("./datasets/icartoonface/personai_icartoonface_dettrain/icartoonface_dettrain")
    output_dir: str = make_real_path("./datasets/icartoonface/icartoonface_face_cropped_512x512_unaligned_v2")
    output_dir_resized: str = make_real_path("./datasets/icartoonface/icartoonface_face_cropped_256x256_v2")
    output_dir_retargeted: str = make_real_path("./datasets/icartoonface/icartoonface_face_retargeted_data_v2")

    """ Run arguments """
    detection_threshold: float = 0.99
    has_persona_folders: bool = flag(default=False, negative_prefix="--no-")