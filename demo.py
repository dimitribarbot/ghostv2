print("started imports")

import os
from typing import cast

from simple_parsing import ArgumentParser

import cv2
import numpy as np
import torch
import lightning as L

from CVLFace import get_aligner
from RetinaFace.detector import RetinaFace
from inference import GhostV2DataModule, GhostV2Module
from utils.image_processing import get_aligned_face_and_affine_matrix
from utils.inference.inference_arguments import InferenceArguments

print("finished imports")


torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

def make_real_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), relative_path)


def main(args: InferenceArguments):
    try:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda:" + str(args.device_id)
        else:
            device = "cpu"
    except:
        device = "cpu"

    if device == "cpu":
        print("Nor Cuda nor MPS are available, using CPU. Check if it's ok.")

    print("Creating PyTorch Lightning trainer")
    trainer = L.Trainer(precision=args.precision, logger=[])

    demo_path: str = make_real_path("./examples/results/training")

    print("Creating GhostV2 Module")
    with trainer.init_module():
        model = GhostV2Module(args)

    corner_image = cv2.imread(os.path.join(demo_path, "source_target.png"))

    image_grid = [
        [corner_image, None, None, None, None, None],
        [None, None, None, None, None, None],
        [None, None, None, None, None, None],
        [None, None, None, None, None, None],
        [None, None, None, None, None, None],
        [None, None, None, None, None, None]
    ]

    for i in range(5):
        source_i = os.path.join(demo_path, f"source_{i + 1}_full.jpg")

        image_grid[i + 1][0] = cv2.imread(source_i)

        for j in range(5):
            target_j = os.path.join(demo_path, f"target_{j + 1}_full.jpg")

            if image_grid[0][j + 1] is None:
                image_grid[0][j + 1] = cv2.imread(target_j)

            dm = GhostV2DataModule(source_i, target_j)
            
            print(f"Starting inference for source {i + 1} and target {j + 1}")
            image_grid[i + 1][j + 1] = trainer.predict(model, dm)[0]

    face_detector = RetinaFace(
        gpu_id=0,
        fp16=True,
        model_path=args.retina_face_model_path
    )

    aligner = None
    if args.align_mode == "cvlface":
        aligner = get_aligner(args.cvlface_aligner_model_path)

    for i in range(6):
        for j in range(6):
            if i > 0 or j > 0:
                if image_grid[i][j] is not None:
                    if args.paste_back_mode != "none" or i == 0 or j == 0:
                        detected_faces = face_detector(image_grid[i][j], threshold=args.detection_threshold, return_dict=True, cv=True)
                        if len(detected_faces) == 0:
                            raise ValueError(f"No face detected in image for source {i} and target {j}!")
                        face_kps = detected_faces[0]["kps"]
                        face, _ = get_aligned_face_and_affine_matrix(
                            image_grid[i][j], face_kps, face_size=256, align_mode=args.align_mode, aligner=aligner, device=device)
                        print(f"Setting face at indice ({i}, {j}) with shape {face.shape}")
                        image_grid[i][j] = face
                else:
                    image_grid[i][j] = np.zeros_like(corner_image)

    output_0 = np.concatenate((image_grid[0][0], image_grid[1][0], image_grid[2][0], image_grid[3][0], image_grid[4][0], image_grid[5][0]), axis=0)
    output_1 = np.concatenate((image_grid[0][1], image_grid[1][1], image_grid[2][1], image_grid[3][1], image_grid[4][1], image_grid[5][1]), axis=0)
    output_2 = np.concatenate((image_grid[0][2], image_grid[1][2], image_grid[2][2], image_grid[3][2], image_grid[4][2], image_grid[5][2]), axis=0)
    output_3 = np.concatenate((image_grid[0][3], image_grid[1][3], image_grid[2][3], image_grid[3][3], image_grid[4][3], image_grid[5][3]), axis=0)
    output_4 = np.concatenate((image_grid[0][4], image_grid[1][4], image_grid[2][4], image_grid[3][4], image_grid[4][4], image_grid[5][4]), axis=0)
    output_5 = np.concatenate((image_grid[0][5], image_grid[1][5], image_grid[2][5], image_grid[3][5], image_grid[4][5], image_grid[5][5]), axis=0)

    output = np.concatenate((output_0, output_1, output_2, output_3, output_4, output_5), axis=1)

    print("Saving result to output")
    cv2.imwrite(os.path.join(demo_path, "source_target_grid.png"), output)

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(InferenceArguments, dest="arguments")  # add arguments for the dataclass
    args = cast(InferenceArguments, parser.parse_args().arguments)
    
    main(args)