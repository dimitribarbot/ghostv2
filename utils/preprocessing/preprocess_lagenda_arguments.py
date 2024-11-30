import os
from dataclasses import dataclass

from utils.preprocessing.preprocess_arguments import PreprocessArguments


def make_real_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", relative_path)

@dataclass
class PreprocessLagendaArguments(PreprocessArguments):

    """ Data arguments """
    lagenda_base_dir: str = make_real_path("./datasets/Lagenda/lag_benchmark")
    output_dir: str = make_real_path("./datasets/Lagenda/lag_benchmark_face_cropped_512x512_unaligned")
    output_dir_resized: str = make_real_path("./datasets/Lagenda/lag_benchmark_face_cropped_256x256_2")
    output_dir_retargeted: str = make_real_path("./datasets/Lagenda/lag_benchmark_retargeted_data")