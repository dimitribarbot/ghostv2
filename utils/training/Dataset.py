from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from PIL import Image
import glob
import random
import cv2
from itertools import groupby
from os.path import dirname
from tqdm import tqdm
import sys
sys.path.append('..')


def read_image(image_path: str):
    x = cv2.imread(image_path)[:, :, ::-1]
    x = Image.fromarray(x)
    return x


def get_image_folder_name(image_path: str):
    return '/'.join(image_path.split('/')[:-1])


class FaceEmbed(TensorDataset):
    def __init__(self, data_path, same_prob=0.2):

        self.same_prob = same_prob
        
        self.images_list = glob.glob(f'{data_path}/**/*.*g', recursive=True)

        self.N = len(self.images_list)

        self.transforms_facenet = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.transforms_base = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item: int):

        Xs_image_path = self.images_list[item]
        Xs = read_image(Xs_image_path)

        if random.random() > self.same_prob:
            Xt_image_path = random.choice(self.images_list)
            Xt = read_image(Xt_image_path)
            same_person = int(Xs_image_path != Xt_image_path)
        else:
            Xt = Xs.copy()
            same_person = 1
            
        return self.transforms_facenet(Xs), self.transforms_base(Xs),  self.transforms_base(Xt), same_person

    def __len__(self):
        return self.N


class FaceEmbedLaion(FaceEmbed):
    def __init__(self, data_path, same_prob=0.2, same_identity=False):

        super().__init__(data_path, same_prob)

        self.same_identity = same_identity
        
        folder_groups = groupby(self.images_list, key=dirname)
        folder_len = len(folder_groups)

        self.folder2imgs = {}
        for k, g in tqdm(folder_groups, total=folder_len):
            self.folder2imgs[k] = list(g)

    def __getitem__(self, item: int):
            
        Xs_image_path = self.images_list[item]
        Xs_folder_name = get_image_folder_name(Xs_image_path)
        Xs = read_image(Xs_image_path)

        if random.random() > self.same_prob:
            Xt_image_path = random.choice(self.images_list)
            Xt_folder_name = get_image_folder_name(Xt_image_path)
            Xt = read_image(Xt_image_path)
            same_person = int(Xt_folder_name != Xs_folder_name)
        else:
            if self.same_identity:
                Xt_image_path = random.choice(self.folder2imgs[Xs_folder_name])
                Xt = read_image(Xt_image_path)
            else:
                Xt = Xs.copy()
            same_person = 1
            
        return self.transforms_facenet(Xs), self.transforms_base(Xs),  self.transforms_base(Xt), same_person