from torch.utils.data import Dataset

import cv2


class FaceEmbed(Dataset):
    def __init__(self, source_file_path: str, target_file_path: str):

        self.Xs_image_path = source_file_path
        self.Xt_image_path = target_file_path

    def __getitem__(self, item: int):

        Xs_image = cv2.imread(self.Xs_image_path)
        Xt_image = cv2.imread(self.Xt_image_path)
            
        return Xs_image, Xt_image

    def __len__(self):
        return 1