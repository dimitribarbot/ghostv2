import torch
from safetensors.torch import load_file

from LivePortrait.modules.appearance_feature_extractor import AppearanceFeatureExtractor
from LivePortrait.modules.motion_extractor import MotionExtractor
from LivePortrait.modules.spade_generator import SPADEDecoder
from LivePortrait.modules.stitching_retargeting_network import StitchingRetargetingNetwork
from LivePortrait.modules.warping_network import WarpingNetwork


def filter_checkpoint_for_model(checkpoint, prefix):
    """Filter and adjust the checkpoint dictionary for a specific model based on the prefix."""
    # Create a new dictionary where keys are adjusted by removing the prefix and the model name
    filtered_checkpoint = {key.replace(prefix + "_module.", ""): value for key, value in checkpoint.items() if key.startswith(prefix)}
    return filtered_checkpoint


def load_model(ckpt_path, model_config, device, model_type):
    model_params = model_config['model_params'][f'{model_type}_params']

    if model_type == 'appearance_feature_extractor':
        model = AppearanceFeatureExtractor(**model_params).to(device)
    elif model_type == 'motion_extractor':
        model = MotionExtractor(**model_params).to(device)
    elif model_type == 'warping_module':
        model = WarpingNetwork(**model_params).to(device)
    elif model_type == 'spade_generator':
        model = SPADEDecoder(**model_params).to(device)
    elif model_type == 'stitching_retargeting_module':
        # Special handling for stitching and retargeting module
        config = model_config['model_params']['stitching_retargeting_module_params']
        checkpoint = load_file(ckpt_path, device=device)

        stitcher_prefix = 'retarget_shoulder'
        stitcher_checkpoint = filter_checkpoint_for_model(checkpoint, stitcher_prefix)
        stitcher = StitchingRetargetingNetwork(**config.get('stitching'))
        stitcher.load_state_dict(stitcher_checkpoint)
        stitcher = stitcher.to(device)
        stitcher.eval()

        retargetor_lip_prefix = 'retarget_mouth'
        retargetor_lip_checkpoint = filter_checkpoint_for_model(checkpoint, retargetor_lip_prefix)
        retargetor_lip = StitchingRetargetingNetwork(**config.get('lip'))
        retargetor_lip.load_state_dict(retargetor_lip_checkpoint)
        retargetor_lip = retargetor_lip.to(device)
        retargetor_lip.eval()

        retargetor_eye_prefix = 'retarget_eye'
        retargetor_eye_checkpoint = filter_checkpoint_for_model(checkpoint, retargetor_eye_prefix)
        retargetor_eye = StitchingRetargetingNetwork(**config.get('eye'))
        retargetor_eye.load_state_dict(retargetor_eye_checkpoint)
        retargetor_eye = retargetor_eye.to(device)
        retargetor_eye.eval()

        return {
            'stitching': stitcher,
            'lip': retargetor_lip,
            'eye': retargetor_eye
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(load_file(ckpt_path, device=device))
    model.eval()
    return model


def concat_feat(kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
    """
    kp_source: (bs, k, 3)
    kp_driving: (bs, k, 3)
    Return: (bs, 2k*3)
    """
    bs_src = kp_source.shape[0]
    bs_dri = kp_driving.shape[0]
    assert bs_src == bs_dri, 'batch size must be equal'

    feat = torch.cat([kp_source.view(bs_src, -1), kp_driving.view(bs_dri, -1)], dim=1)
    return feat