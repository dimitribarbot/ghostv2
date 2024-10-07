import os
from dataclasses import dataclass
from typing import Optional

from simple_parsing import choice

def make_real_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", relative_path)

@dataclass
class InferenceArguments:

    """ Model params """
    G_path: str = make_real_path("./weights/G.safetensors")                 # Path to pretrained weights for G. Only used if pretrained=True

    """ Training params you may want to change """
    backbone: str = choice("unet", "linknet", "resnet", default="unet")     # Backbone for attribute encoder
    num_blocks: int = 2                                                     # Numbers of AddBlocks at AddResblock

    """ Training params you probably don't want to change """
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