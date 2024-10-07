import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
from .image_processing import torch2image


transforms_base = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


def get_preds_fromhm(hm: torch.Tensor):
    _, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)
    
    
    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-0.5)

    return preds


def detect_landmarks(inputs: torch.Tensor, model_ft):
    mean = torch.tensor([0.5, 0.5, 0.5], device=inputs.device).unsqueeze(1).unsqueeze(2)
    std = torch.tensor([0.5, 0.5, 0.5], device=inputs.device).unsqueeze(1).unsqueeze(2)
    inputs = (std * inputs) + mean

    outputs, _ = model_ft(inputs)
    pred_heatmap = outputs[-1][:, :-1, :, :]
    pred_landmarks, _ = get_preds_fromhm(pred_heatmap.cpu()).to(inputs.device)
    landmarks = pred_landmarks * 4.0
    eyes = torch.cat((landmarks[:, 96, :], landmarks[:, 97, :]), 1)
    return eyes, pred_heatmap[:, 96, :, :], pred_heatmap[:, 97, :, :]


def paint_eyes(images, eyes):
    list_eyes = []
    for i in range(len(images)):
        mask = torch2image(images[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) 
        
        cv2.circle(mask, (int(eyes[i][0]),int(eyes[i][1])), radius=3, color=(0,255,255), thickness=-1)
        cv2.circle(mask, (int(eyes[i][2]),int(eyes[i][3])), radius=3, color=(0,255,255), thickness=-1)
        
        mask = mask[:, :, ::-1]
        mask = transforms_base(Image.fromarray(mask))
        list_eyes.append(mask)
    tensor_eyes = torch.stack(list_eyes)
    return tensor_eyes