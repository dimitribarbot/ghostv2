from typing import Literal, Optional

from safetensors.torch import load_file


def get_aligner(model_path: str, device: Optional[str] = None):
    from CVLFace.differentiable_face_aligner import DifferentiableFaceAligner
    from CVLFace.differentiable_face_aligner.dfa import get_landmark_predictor, get_preprocessor

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


def get_arcface_model(model_path: str, model_name: Literal["ir18", "ir50", "ir101"] = "ir101", device: Optional[str] = None):
    from CVLFace.iresnet import IResNetModel
    from ArcFace.iresnet import iresnet100, iresnet50, iresnet18

    if model_name == 'ir50':
        net = iresnet50(input_size=(112,112), output_dim=512)
    elif model_name == 'ir101':
        net = iresnet100()
    elif model_name == 'ir18':
        net = iresnet18(input_size=(112,112), output_dim=512)
    else:
        raise NotImplementedError
    
    checkpoint = load_file(model_path)
    checkpoint = { k.replace("model.", ""): v for k,v in checkpoint.items() }
    
    model = IResNetModel(net)
    model.load_state_dict(checkpoint)
    if device is not None:
        model = model.to(device)
    model.eval()

    return model


def get_adaface_model(model_path: str, model_name: Literal["ir18", "ir50", "ir101"] = "ir101", device: Optional[str] = None):
    from CVLFace.iresnet import IResNetModel
    from AdaFace.net import IR_101, IR_50, IR_18

    if model_name == 'ir50':
        net = IR_50(input_size=(112,112), output_dim=512)
    elif model_name == 'ir101':
        net = IR_101(input_size=(112,112), output_dim=512)
    elif model_name == 'ir18':
        net = IR_18(input_size=(112,112), output_dim=512)
    else:
        raise NotImplementedError
    
    checkpoint = load_file(model_path)
    checkpoint = { k.replace("model.", ""): v for k,v in checkpoint.items() }
    
    model = IResNetModel(net)
    model.load_state_dict(checkpoint)
    if device is not None:
        model = model.to(device)
    model.eval()

    return model


def get_vit_model(model_path: str, device: Optional[str] = None):
    from CVLFace.vit import ViTModel
    from CVLFace.vit.vit import VisionTransformer

    net = VisionTransformer(img_size=112, patch_size=8, num_classes=512, embed_dim=512, depth=24,
                                    mlp_ratio=3, num_heads=16, drop_path_rate=0.1, norm_layer="ln",
                                    mask_ratio=0.0)
    
    checkpoint = load_file(model_path)
    checkpoint = { k.replace("model.", ""): v for k,v in checkpoint.items() }
    if "net.vit" in list(net.state_dict().keys())[-1]:
        checkpoint = { k.replace("net", "net.vit"): v for k,v in checkpoint.items() }
    
    model = ViTModel(net)
    model.load_state_dict(checkpoint)
    if device is not None:
        model = model.to(device)
    model.eval()

    return model