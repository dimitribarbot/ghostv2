import os
import cv2
from itertools import groupby
from tqdm import tqdm
from typing import cast, Optional
import traceback

from simple_parsing import ArgumentParser

import torch
from safetensors.torch import load_file

from BiSeNet.bisenet import BiSeNet
from CVLFace import get_aligner
from CVLFace.differentiable_face_aligner import DifferentiableFaceAligner
from DASS.detector import DassDetFace
from GFPGAN.gfpganv1_clean_arch import GFPGANv1Clean
from LivePortrait.pipeline import LivePortraitPipeline
from RetinaFace.detector import RetinaFace
from FaceAlignment import FaceAlignment, LandmarksType
from utils.preprocessing.preprocess import preprocess
from utils.preprocessing.preprocess_icartoonface_arguments import PreprocessICartoonFaceArguments


def output_file_exists(cropped_face_path_resized: str, file: str):
    return os.path.exists(os.path.join(cropped_face_path_resized, f"{os.path.splitext(file)[0]}_00"))


def regroup_same_personas(cropped_face_path: str):
    if os.path.exists(cropped_face_path):
        folders = list(sorted(os.listdir(cropped_face_path)))
        
        grouped_folders_by_persona = [list(i) for _, i in groupby(folders, lambda f: f.split('_')[-3])]
        grouped_folders_by_persona = [i for i in grouped_folders_by_persona if len(i) > 1]

        for grouped_folder_by_persona in grouped_folders_by_persona:
            grouped_persona_folder = os.path.join(cropped_face_path, grouped_folder_by_persona[0])
            max_file_number = max(map(lambda f: int(os.path.splitext(f)[0]), os.listdir(grouped_persona_folder)))
            for persona_folder_index in range(1, len(grouped_folder_by_persona)):
                persona_folder = os.path.join(cropped_face_path, grouped_folder_by_persona[persona_folder_index])
                for persona_file in sorted(os.listdir(persona_folder), key=lambda f: int(os.path.splitext(f)[0])):
                    max_file_number += 1
                    _, extension = os.path.splitext(persona_file)
                    source_persona_file = os.path.join(persona_folder, persona_file)
                    target_persona_file = os.path.join(grouped_persona_folder, f"{max_file_number}{extension}")
                    os.rename(source_persona_file, target_persona_file)
                os.rmdir(persona_folder)


@torch.no_grad()
def process(
    face_detector: RetinaFace,
    face_alignment: FaceAlignment,
    live_portrait_pipeline: LivePortraitPipeline,
    gfpgan: GFPGANv1Clean,
    face_parser: BiSeNet,
    aligner: Optional[DifferentiableFaceAligner],
    args: PreprocessICartoonFaceArguments,
    device: str
):
    icartoonface_base_dir = args.icartoonface_base_dir
    cropped_face_path = args.output_dir
    cropped_face_path_resized = args.output_dir_resized

    image_files = list(sorted(filter(lambda file: not output_file_exists(cropped_face_path_resized, file), os.listdir(icartoonface_base_dir))))

    for image_file in tqdm(image_files, total=len(image_files)):
        try:
            image_file_path = os.path.join(icartoonface_base_dir, image_file)

            id = os.path.basename(os.path.splitext(image_file)[0])
            rgb_image = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)

            preprocess(
                id,
                rgb_image,
                face_detector,
                face_alignment,
                live_portrait_pipeline,
                gfpgan,
                face_parser,
                aligner,
                args,
                cropped_face_path,
                cropped_face_path_resized,
                args.output_dir_retargeted,
                device
            )
        except Exception as ex:
            print(f"An error occurred for sample {id}: {ex}")
            traceback.print_tb(ex.__traceback__)
            if args.stop_if_error:
                raise ex


@torch.no_grad()
def process_with_personas(
    face_detector: RetinaFace,
    face_alignment: FaceAlignment,
    live_portrait_pipeline: LivePortraitPipeline,
    gfpgan: GFPGANv1Clean,
    face_parser: BiSeNet,
    aligner: Optional[DifferentiableFaceAligner],
    args: PreprocessICartoonFaceArguments,
    device: str
):
    icartoonface_base_dir = args.icartoonface_base_dir
    cropped_face_path = args.output_dir
    cropped_face_path_resized = args.output_dir_resized

    folders = list(sorted(os.listdir(icartoonface_base_dir)))
    for folder in tqdm(folders, total=len(folders), desc="folders"):
        image_files = list(filter(lambda file: not output_file_exists(cropped_face_path_resized, file), os.listdir(os.path.join(icartoonface_base_dir, folder))))

        for image_file in tqdm(image_files, total=len(image_files), desc=f"images {folder}", leave=False):
            try:
                image_file_path = os.path.join(icartoonface_base_dir, folder, image_file)

                id = os.path.basename(os.path.splitext(image_file)[0])
                rgb_image = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)

                preprocess(
                    id,
                    rgb_image,
                    face_detector,
                    face_alignment,
                    live_portrait_pipeline,
                    gfpgan,
                    face_parser,
                    aligner,
                    args,
                    cropped_face_path,
                    cropped_face_path_resized,
                    args.output_dir_retargeted,
                    device
                )
            except Exception as ex:
                print(f"An error occurred for sample {id}: {ex}")
                traceback.print_tb(ex.__traceback__)
                if args.stop_if_error:
                    raise ex

    regroup_same_personas(cropped_face_path)
    regroup_same_personas(cropped_face_path_resized)


def main(args: PreprocessICartoonFaceArguments):
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

    gfpgan = None
    if args.enhance_faces_in_original_image or args.enhance_output_face:
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

    face_parser = BiSeNet(num_class=19)
    face_parser.load_state_dict(load_file(args.face_parser_model_path), strict=True)
    face_parser.eval()
    face_parser = face_parser.to(device)

    face_alignment = FaceAlignment(
        LandmarksType.TWO_D,
        safetensors_file_path=args.face_alignment_model_path,
        flip_input=False,
        device=device,
        dtype=torch.float16,
        face_detector="default",
    )

    face_detector = DassDetFace(
        gpu_id=0,
        fp16=True,
        model_path=args.dass_face_model_path,
        landmark_detector_model_path=args.dass_face_landmark_detector_model_path,
    )

    live_portrait_pipeline = LivePortraitPipeline(
        args.live_portrait_landmark_model_path,
        args.live_portrait_F_model_path,
        args.live_portrait_M_model_path,
        args.live_portrait_W_model_path,
        args.live_portrait_G_model_path,
        args.live_portrait_S_model_path,
        device
    )

    aligner = None
    if args.align_mode == "cvlface":
        aligner = get_aligner(args.cvlface_aligner_model_path, device)

    if args.has_persona_folders:
        process_with_personas(
            face_detector,
            face_alignment,
            live_portrait_pipeline,
            gfpgan,
            face_parser,
            aligner,
            args,
            device
        )
    else:
        process(
            face_detector,
            face_alignment,
            live_portrait_pipeline,
            gfpgan,
            face_parser,
            aligner,
            args,
            device
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(PreprocessICartoonFaceArguments, dest="arguments")
    args = cast(PreprocessICartoonFaceArguments, parser.parse_args().arguments)
    
    main(args)