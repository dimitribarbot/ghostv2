import os
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm

import torch
from safetensors.torch import load_file
from torchvision.transforms.functional import normalize

from GFPGAN.gfpganv1_clean_arch import GFPGANv1Clean
from utils.training.image_processing import img2tensor, tensor2img


@torch.no_grad()
def enhance(gfpgan: GFPGANv1Clean, img: cv2.typing.MatLike, image_name: str, device: torch.device, weight=0.5):
    # prepare data
    cropped_face_t = img2tensor(img / 255., bgr2rgb=True, float32=True)
    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

    try:
        output = gfpgan(cropped_face_t, return_rgb=False, weight=weight)[0]
        # convert to image
        restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
    except RuntimeError as error:
        print(f"Failed inference for GFPGAN for image {image_name}: {error}.")
        restored_face = img

    restored_face = restored_face.astype('uint8')
    
    return restored_face


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print('cuda is not available. using cpu. check if it\'s ok')

    gfpgan = GFPGANv1Clean(
        out_size=512,
        num_style_feat=512,
        channel_multiplier=2,
        decoder_load_path=None,
        fix_decoder=False,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True
    )
    gfpgan.load_state_dict(load_file(args.gfpgan_model_path), strict=True)
    gfpgan.eval()
    gfpgan = gfpgan.to(device)

    crop_size = 224

    for _, dirs, _ in os.walk(args.path_to_dataset, topdown=True):
        dirs = sorted(dirs)
        for i in tqdm(range(len(dirs))):
            d = os.path.join(args.path_to_dataset, dirs[i])
            dir_to_save = os.path.join(args.save_path, dirs[i])
            Path(dir_to_save).mkdir(parents=True, exist_ok=True)
            
            image_names = os.listdir(d)
            for j in tqdm(range(len(image_names)), leave=False):
                try:
                    image_path = os.path.join(d, image_names[j])
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    image = enhance(gfpgan, image, image_names[j], device)
                    image = cv2.resize(image, (crop_size, crop_size))
                    cv2.imwrite(os.path.join(dir_to_save, image_names[j]), image)
                except Exception as ex:
                    print(f"An error has occurred while resizing image {image_names[j]}:", ex)
                break
            break
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path_to_dataset', default='/home/dimitribarbot/datasets/LAION-Face/laion_cropped_face_data/split_00000', type=str)
    parser.add_argument('--save_path', default='/home/dimitribarbot/datasets/LAION-Face/laion_restored_face_224x224/split_00000', type=str)
    parser.add_argument('--gfpgan_model_path', default='./weights/GFPGAN/GFPGANv1.4.safetensors', type=str)
    
    args = parser.parse_args()
    
    main(args)