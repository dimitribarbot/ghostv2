import os

import cv2
import torch

from DASS.detector import DassDetFace
from FaceAlignment.api import FaceAlignment, LandmarksType
from utils.image_processing import get_aligned_face_and_affine_matrix, save_image_with_bbox_and_landmark, save_image, save_image_with_landmarks


def make_real_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), relative_path)


def main():
    device_id = 0

    try:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda:" + str(device_id)
        else:
            device = "cpu"
    except:
        device = "cpu"

    if device == "cpu":
        print("Nor Cuda nor MPS are available, using CPU. Check if it's ok.")

    face_detector = DassDetFace(
        gpu_id=0,
        fp16=True,
        model_path=make_real_path("./weights/DASS/xl_mixdata_finetuned_stage3.safetensors"),
        landmark_detector_model_path=make_real_path("./weights/DASS/checkpoint_landmark_191116.safetensors"),
    )

    image_path = "/home/dimitribarbot/stable-diffusion/ghostv2/test.png"
    image = cv2.imread(image_path)

    outputs = face_detector(image, threshold=0.99, return_dict=True, cv=True)

    for index, output in enumerate(outputs):
        bbox = output["box"]
        score = output["score"]
        landmark5 = output["kps"]
        landmark24 = output["kps24"]

        if landmark5 is not None:
            cropped_face, _ = get_aligned_face_and_affine_matrix(image, landmark5, 256, "insightface_v2", None, device)

            output_path = f"/home/dimitribarbot/stable-diffusion/ghostv2/test_bbox_{index + 1}.png"
            output_cropped_path = f"/home/dimitribarbot/stable-diffusion/ghostv2/test_aligned_{index + 1}.png"

            save_image_with_bbox_and_landmark(image, bbox, landmark5, output_path)
            save_image(cropped_face, output_cropped_path)


if __name__ == "__main__":    
    main()