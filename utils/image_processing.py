import os
import functools
import cv2
import math
import random
from typing import cast, Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import normalize
from torchvision.utils import make_grid

import numpy as np
from PIL import Image
from skimage import transform as trans

from BiSeNet.bisenet import BiSeNet
from CVLFace.differentiable_face_aligner import DifferentiableFaceAligner
from GFPGAN.gfpganv1_clean_arch import GFPGANv1Clean
from utils.adaface_align_trans import get_reference_facial_points, warp_and_crop_face


transformer_embeddings = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
#<--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

#---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

#-->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

#-->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

# arcface_dst = np.array(
#     [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
#      [41.5493, 92.3655], [70.7299, 92.2041]],
#     dtype=np.float32)


# standard 5 landmarks for FFHQ faces with 512 x 512
# facexlib
face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935], [201.26117, 371.41043], [313.08905, 371.15118]])
# face_template_src = np.expand_dims(face_template, axis=0)
# arcface_src = np.expand_dims(arcface_dst, axis=0)
default_template_src = np.array([src1, src2, src3, src4, src5])

adaface_reference = None


def torch2image(torch_image: torch.tensor) -> np.ndarray:
    batch = False
    
    if torch_image.dim() == 4:
        torch_image = torch_image[:8]
        batch = True
    
    device = torch_image.device
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


def img2tensor(imgs, bgr2rgb=True, float32=True):
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)
    

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


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
    image = transformer_embeddings(image)
    image = image.view(-1, image.shape[0], image.shape[1], image.shape[2])
    
    return image


def convert_to_batch_tensor(bgr_image: cv2.typing.MatLike, device: torch.device):
    input = img2tensor(bgr_image / 255., bgr2rgb=True, float32=True)
    normalize(input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    return input.unsqueeze(0).to(device)


def get_face_embeddings(source: torch.Tensor, model: Any, face_embeddings: str):
    model_size = [112, 112] if face_embeddings == "arcface" or face_embeddings == "adaface" else [160, 160]
    return model(F.interpolate(source, model_size, mode="bicubic"))


def get_faceswap(source_path: str,
                 target_path: str,
                 G: Any,
                 model: Any,
                 face_embeddings: str,
                 device: str) -> np.array:
    '''G: generator model, facenet: Facenet model, device: torch device'''
    source = read_torch_image(source_path)
    source = source.to(device)

    embeds = get_face_embeddings(source, model, face_embeddings)

    target = read_torch_image(target_path)
    target = target.to(device)

    with torch.no_grad():
        Yt, _ = G(target, embeds)
        Yt = torch2image(Yt)

    source = torch2image(source)
    target = torch2image(target)

    return np.concatenate((cv2.resize(source, (256, 256)), target, Yt), axis=1)


# Modified from https://github.com/deepinsight/insightface/blob/e896172e45157d5101448b5b9e327073073bfb1b/python-package/insightface/utils/face_align.py
def estimate_norm(lmk: np.ndarray, image_size=224):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    # src = face_template_src * (image_size / 512)
    src = default_template_src * (image_size / 112)
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


# Modified from https://github.com/deepinsight/insightface/blob/e896172e45157d5101448b5b9e327073073bfb1b/python-package/insightface/utils/face_align.py
def norm_crop(bgr_image: cv2.typing.MatLike, landmark, face_size=224):
    M, _ = estimate_norm(landmark, face_size)
    warped = cv2.warpAffine(bgr_image, M, (face_size, face_size), borderValue=0.0)
    return warped, M


def align_warp_face(bgr_image: cv2.typing.MatLike, landmarks: np.ndarray, face_size=512):
    """Align and warp faces with face template.
    """
    # use 5 landmarks to get affine matrix
    # use cv2.LMEDS method for the equivalence to skimage transform
    # ref: https://blog.csdn.net/yichxi/article/details/115827338
    affine_matrix = cv2.estimateAffinePartial2D(landmarks, face_template * (face_size / 512.0), method=cv2.LMEDS)[0]
    # warp and crop faces
    cropped_face = cv2.warpAffine(
        bgr_image, affine_matrix, (face_size, face_size), borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))  # gray
    return cropped_face, affine_matrix


def get_aligned_face(bgr_image: cv2.typing.MatLike, landmarks: np.ndarray, face_size=512):
    global adaface_reference
    if adaface_reference is None:
        adaface_reference = get_reference_facial_points(output_size=(face_size, face_size), default_square=True)
    return warp_and_crop_face(bgr_image, landmarks, adaface_reference, crop_size=(face_size, face_size))


def get_cvl_aligned_face(
    bgr_image: cv2.typing.MatLike,
    landmarks: np.ndarray,
    face_size=512,
    aligner: Optional[DifferentiableFaceAligner]=None,
    device: Optional[torch.device]=None,
):
    if aligner is None:
        raise ValueError("CVL Face aligner is not optional.")
    if device is None:
        raise ValueError("CVL Face device is not optional.")
    
    input, M1 = align_warp_face(bgr_image, landmarks, face_size=face_size)
    
    _, _, _, _, _, _, cv2_tfms = aligner(convert_to_batch_tensor(input, device), output_size=face_size)
    M2 = cv2_tfms.squeeze()

    affine_matrix = np.matmul(np.vstack([M2, [0, 0, 1]]), np.vstack([M1, [0, 0, 1]]))[:2]

    output = cv2.warpAffine(bgr_image, affine_matrix, (face_size, face_size), borderValue=0.0)

    return output, affine_matrix


def get_aligned_face_and_affine_matrix(
    bgr_image: cv2.typing.MatLike,
    landmarks: np.ndarray,
    face_size=512,
    align_mode="facexlib",
    aligner: Optional[DifferentiableFaceAligner] = None,
    device: Optional[torch.device]=None,
):
    if align_mode == "insightface":
        return norm_crop(bgr_image, landmarks, face_size=face_size)
    elif align_mode == "mtcnn":
        return get_aligned_face(bgr_image, landmarks, face_size=face_size)
    elif align_mode == "cvlface":
        return get_cvl_aligned_face(bgr_image, landmarks, face_size=face_size, aligner=aligner, device=device)
    return align_warp_face(bgr_image, landmarks, face_size=face_size)


# Modified from https://github.com/deepinsight/insightface/blob/e896172e45157d5101448b5b9e327073073bfb1b/python-package/insightface/utils/face_align.py
def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = M @ new_pt
        new_pts[i] = new_pt[0:2]
    return new_pts


def get_face_sort_key(face1: Dict[str, np.ndarray], face2: Dict[str, np.ndarray]):
    if face1["kps"] is None:
        if face2["kps"] is None:
            return 0
        return -1
    
    if face2["kps"] is None:
        return 1

    if face1["kps"][0][0] < face2["kps"][0][0]:
        return -1
    
    if face1["kps"][0][0] > face2["kps"][0][0]:
        return 1

    if face1["kps"][0][1] < face2["kps"][0][1]:
        return -1
    
    if face1["kps"][0][1] > face2["kps"][0][1]:
        return 1

    return 0


def sort_faces_by_coordinates(faces: List[Dict[str, np.ndarray]]):
    faces.sort(key=functools.cmp_to_key(get_face_sort_key))


@torch.no_grad()
def paste_face_back(
    face_parser: BiSeNet,
    original_image: cv2.typing.MatLike,
    restored_face: cv2.typing.MatLike,
    affine_matrix: np.ndarray,
    device: torch.device,
):
    original_size = (original_image.shape[1], original_image.shape[0])

    inverse_affine = cv2.invertAffineTransform(affine_matrix)
    inv_restored = cv2.warpAffine(restored_face, inverse_affine, original_size)

    face_input = cv2.resize(restored_face, (512, 512))
    face_input = convert_to_batch_tensor(restored_face, device)

    with torch.no_grad():
        out = cast(torch.Tensor, face_parser(face_input)[0])
    out = out.argmax(dim=1).squeeze().cpu().numpy()
    
    mask = np.zeros(out.shape)
    MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
    for idx, color in enumerate(MASK_COLORMAP):
        mask[out == idx] = color
    #  blur the mask
    mask = cv2.GaussianBlur(mask, (101, 101), 11)
    mask = cv2.GaussianBlur(mask, (101, 101), 11)
    # remove the black borders
    thres = 10
    mask[:thres, :] = 0
    mask[-thres:, :] = 0
    mask[:, :thres] = 0
    mask[:, -thres:] = 0
    mask = mask / 255.

    mask = cv2.resize(mask, restored_face.shape[:2])
    mask = cv2.warpAffine(mask, inverse_affine, original_size, flags=3)
    inv_soft_mask = mask[:, :, None]

    if len(original_image.shape) == 3 and original_image.shape[2] == 4:  # alpha channel
        alpha = original_image[:, :, 3:]
        original_image = inv_soft_mask * inv_restored + (1 - inv_soft_mask) * original_image[:, :, 0:3]
        original_image = np.concatenate((original_image, alpha), axis=2)
    else:
        original_image = inv_soft_mask * inv_restored + (1 - inv_soft_mask) * original_image
    
    return original_image


@torch.no_grad()
def enhance_face(
    gfpgan: GFPGANv1Clean,
    img_bgr: cv2.typing.MatLike,
    image_name: str,
    device: torch.device,
    weight=0.5
):
    cropped_face_t = convert_to_batch_tensor(img_bgr, device)

    try:
        output = cast(torch.Tensor, gfpgan(cropped_face_t, return_rgb=False, weight=weight)[0])
        restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
    except RuntimeError as error:
        print(f"Failed GFPGAN inference for {image_name} image: {error}.")
        restored_face = img_bgr

    restored_face = restored_face.astype('uint8')
    
    return restored_face


def save_image_with_landmarks(bgr_image: cv2.typing.MatLike, landmark: np.ndarray, save_path: str):
    for point in landmark:
        bgr_image = cv2.circle(bgr_image, (int(point[0]), int(point[1])), radius=2, color=(0, 0, 255), thickness=-1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, bgr_image)


def random_horizontal_flip(bgr_image: cv2.typing.MatLike):
    random_flip = bool(random.getrandbits(1))
    if random_flip:
        return cv2.flip(bgr_image, 1)
    return bgr_image