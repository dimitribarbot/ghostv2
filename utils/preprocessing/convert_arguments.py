import os
from typing import Optional
from dataclasses import dataclass

from simple_parsing import choice
from simple_parsing.helpers import flag


def make_real_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", relative_path)

@dataclass
class ConvertArguments:

    """ Data arguments """
    source_image: Optional[str] = None
    source_folder: Optional[str] = make_real_path("./examples/images/training_facexlib_png")
    output_folder: str = make_real_path("./examples/images/training_facexlib_jpg")
    output_extension: str = choice(".png", ".jpg", default=".jpg")

    """ Run arguments """
    overwrite: bool = flag(default=False, negative_prefix="--no-")