import os
from dataclasses import dataclass

def make_real_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", relative_path)

@dataclass
class EmbeddingModelsArguments:

    """ Model arguments """
    arcface_model_path: str = make_real_path("./weights/ArcFace/backbone.safetensors")
    adaface_model_path: str = make_real_path("./weights/AdaFace/adaface_ir101_webface12m.safetensors")
    cvl_arcface_model_path: str = make_real_path("./weights/CVLFace/cvlface_arcface_ir101_webface4m.safetensors")
    cvl_adaface_model_path: str = make_real_path("./weights/CVLFace/cvlface_adaface_ir101_webface12m.safetensors")
    cvl_vit_model_path: str = make_real_path("./weights/CVLFace/cvlface_adaface_vit_base_webface4m.safetensors")
    facenet_model_path: str = make_real_path("./weights/Facenet/facenet_pytorch.safetensors")