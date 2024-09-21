import cv2
import numpy as np
from PIL import Image
from typing import Any

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F


transformer_Facenet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def torch2image(torch_image: torch.tensor) -> np.ndarray:
    batch = False
    
    if torch_image.dim() == 4:
        torch_image = torch_image[:8]
        batch = True
    
    device = torch_image.device
    # mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
    # std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
    mean = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2).to(device)
    std = torch.tensor([0.5, 0.5, 0.5]).unsqueeze(1).unsqueeze(2).to(device)
    
    denorm_image = (std * torch_image) + mean
    
    if batch:
        denorm_image = denorm_image.permute(0, 2, 3, 1)
    else:
        denorm_image = denorm_image.permute(1, 2, 0)
    
    np_image = denorm_image.detach().cpu().numpy()
    np_image = np.clip(np_image*255., 0, 255).astype(np.uint8)
    
    if batch:
        return np.concatenate(np_image, axis=1)
    else:
        return np_image


def make_image_list(images) -> np.ndarray:    
    np_images = []
    
    for torch_image in images:
        np_img = torch2image(torch_image)
        np_images.append(np_img)
    
    return np.concatenate(np_images, axis=0)


def read_torch_image(path: str) -> torch.tensor:
    
    image = cv2.imread(path)
    image = cv2.resize(image, (256, 256))
    image = Image.fromarray(image[:, :, ::-1])
    image = transformer_Facenet(image)
    image = image.view(-1, image.shape[0], image.shape[1], image.shape[2])
    
    return image


def get_faceswap(source_path: str,
                 target_path: str,
                 G: str,
                 facenet: Any, 
                 device: str) -> np.array:
    '''G: generator model, facenet: Facenet model, device: torch device'''
    source = read_torch_image(source_path)
    source = source.to(device)

    embeds = facenet(F.interpolate(source, [160, 160], mode='bilinear', align_corners=False))
    # embeds = F.normalize(embeds, p=2, dim=1)

    target = read_torch_image(target_path)
    target = target.cuda()

    with torch.no_grad():
        Yt, _ = G(target, embeds)
        Yt = torch2image(Yt)

    source = torch2image(source)
    target = torch2image(target)

    return np.concatenate((cv2.resize(source, (256, 256)), target, Yt), axis=1)  
        