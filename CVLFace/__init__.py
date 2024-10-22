from typing import Optional

from safetensors.torch import load_file

from CVLFace.differentiable_face_aligner import DifferentiableFaceAligner
from CVLFace.differentiable_face_aligner.dfa import get_landmark_predictor, get_preprocessor


def get_aligner(model_path: str, device: Optional[str] = None):
    net, prior_box = get_landmark_predictor(network='mobile0.25', use_aggregator=True, input_size=160)
    preprocessor = get_preprocessor(output_size=160, padding=0, padding_val='zero')

    for param in net.parameters():
        param.requires_grad = False

    checkpoint = load_file(model_path)
    checkpoint = { k.replace("model.", ""): v for k,v in checkpoint.items() }

    aligner = DifferentiableFaceAligner(net, prior_box, preprocessor)
    aligner.load_state_dict(checkpoint)
    if device is not None:
        aligner = aligner.to(device)
    aligner.eval()

    for param in aligner.parameters():
        param.requires_grad = False

    return aligner