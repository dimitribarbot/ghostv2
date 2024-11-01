import os
import cv2
from tqdm import tqdm
from typing import cast, Optional
import traceback

from simple_parsing import ArgumentParser

import torch
from safetensors.torch import load_file

from BiSeNet.bisenet import BiSeNet
from CVLFace import get_aligner
from CVLFace.differentiable_face_aligner import DifferentiableFaceAligner
from GFPGAN.gfpganv1_clean_arch import GFPGANv1Clean
from LivePortrait.pipeline import LivePortraitPipeline
from RetinaFace.detector import RetinaFace
from face_alignment import FaceAlignment, LandmarksType
from utils.preprocessing.preprocess import preprocess
from utils.preprocessing.preprocess_lagenda_arguments import PreprocessLagendaArguments


@torch.no_grad()
def process(
    face_detector: RetinaFace,
    face_alignment: FaceAlignment,
    live_portrait_pipeline: LivePortraitPipeline,
    gfpgan: GFPGANv1Clean,
    face_parser: BiSeNet,
    aligner: Optional[DifferentiableFaceAligner],
    args: PreprocessLagendaArguments,
    device: str
):
    lagenda_base_dir = args.lagenda_base_dir
    cropped_face_path = args.output_dir
    cropped_face_path_resized = args.output_dir_resized

    image_files = os.listdir(lagenda_base_dir)

    for image_file in tqdm(image_files, total=len(image_files)):
        try:
            image_file_path = os.path.join(lagenda_base_dir, image_file)

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


def main(args: PreprocessLagendaArguments):
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

    face_detector = RetinaFace(
        gpu_id=0,
        fp16=True,
        model_path=args.retina_face_model_path
    )

    face_alignment = FaceAlignment(
        LandmarksType.TWO_D,
        flip_input=False,
        device=device,
        dtype=torch.float16,
        face_detector="sfd",
        face_detector_kwargs={
            "path_to_detector": args.face_alignment_model_path
        }
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
    parser.add_arguments(PreprocessLagendaArguments, dest="arguments")
    args = cast(PreprocessLagendaArguments, parser.parse_args().arguments)
    
    main(args)