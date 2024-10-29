import os
from dataclasses import dataclass
from typing import List

from simple_parsing import choice
from simple_parsing.helpers import flag, list_field

def make_real_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", relative_path)

@dataclass
class PreprocessArguments:

    """ Data arguments """
    laion_face_base_dir: str = "/home/dimitribarbot/datasets/LAION-Face"
    laion_face_part_indices: List[int] = list_field(default=[0])
    output_dir: str = "/home/dimitribarbot/datasets/LAION-Face/laion_face_cropped_512x512_unaligned"
    output_dir_resized: str = "/home/dimitribarbot/datasets/LAION-Face/laion_face_cropped_256x256"
    output_dir_retargeted: str = "/home/dimitribarbot/datasets/LAION-Face/laion_face_retargeted_data"

    """ Model arguments """
    gfpgan_model_path: str = make_real_path("./weights/GFPGAN/GFPGANCleanv1-NoCE-C2.safetensors")
    face_parser_model_path: str = make_real_path("./weights/BiSeNet/79999_iter.safetensors")
    retina_face_model_path: str = make_real_path("./weights/RetinaFace/Resnet50_Final.safetensors")
    face_alignment_model_path: str = make_real_path("./weights/FaceAlignment/s3fd-619a316812.safetensors")
    cvlface_aligner_model_path: str = make_real_path("./weights/CVLFace/cvlface_DFA_mobilenet.safetensors")
    live_portrait_landmark_model_path: str = make_real_path("./weights/LivePortrait/landmark_model.pth")
    live_portrait_F_model_path: str = make_real_path("./weights/LivePortrait/appearance_feature_extractor.safetensors")
    live_portrait_M_model_path: str = make_real_path("./weights/LivePortrait/motion_extractor.safetensors")
    live_portrait_W_model_path: str = make_real_path("./weights/LivePortrait/warping_module.safetensors")
    live_portrait_G_model_path: str = make_real_path("./weights/LivePortrait/spade_generator.safetensors")
    live_portrait_S_model_path: str = make_real_path("./weights/LivePortrait/stitching_retargeting_module.safetensors")

    """ Run arguments """
    device_id: int = 0
    min_original_image_size: int = 250
    eye_dist_threshold: int = 5
    final_crop_size: int = 256
    align_mode: str = choice("facexlib", "insightface", "mtcnn", "cvlface", default="insightface")

    """ Retargeting options """
    number_of_variants_per_face: int = 9
    enhance_before_retargeting: bool = flag(default=False, negative_prefix="--no-")
    retargeting_do_crop: bool = flag(default=False, negative_prefix="--no-")
    retargeting_crop_scale: float = 2.3
    filter_valid_faces: bool = flag(default=True, negative_prefix="--no-")
    save_full_size: bool = flag(default=False, negative_prefix="--no-")
    save_retargeted: bool = flag(default=False, negative_prefix="--no-")