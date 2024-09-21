import os
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm


def main(args):
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
                    image = cv2.imread(image_path)
                    image = cv2.resize(image, (crop_size, crop_size))
                    cv2.imwrite(os.path.join(dir_to_save, image_names[j]), image)
                except Exception as ex:
                    print("An error has occurred while resizing image:", ex)
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path_to_dataset', default='/home/dimitribarbot/datasets/LAION-Face/laion_cropped_face_data/split_00000', type=str)
    parser.add_argument('--save_path', default='/home/dimitribarbot/datasets/LAION-Face/laion_cropped_face_224x224/split_00000', type=str)
    
    args = parser.parse_args()
    
    main(args)