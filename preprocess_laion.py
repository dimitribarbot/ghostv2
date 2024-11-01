import os
from tqdm import tqdm
from typing import cast, Optional
import traceback

from simple_parsing import ArgumentParser

import pyarrow.parquet as pq
import webdataset as wds

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
from utils.preprocessing.preprocess_laion_arguments import PreprocessLaionArguments


def preprocess_data(sample):
    image, json = sample
    id = int(json["SAMPLE_ID"])
    return image, id


def extract_parquet_files(laion_data_dir: str, output_dir: str, split_folder: str):
    parquet_files = []
    for subdir, _, files in os.walk(os.path.join(laion_data_dir, split_folder)):
        for file in sorted(files):
            ext = os.path.splitext(file)[-1].lower()
            if ext == ".parquet":
                parquet_file = os.path.join(subdir, file)
                cropped_face_path = os.path.join(output_dir, split_folder, os.path.basename(os.path.splitext(parquet_file)[0]))
                if not os.path.exists(cropped_face_path):
                    parquet_files.append(parquet_file)
    return parquet_files


@torch.no_grad()
def process(
    face_detector: RetinaFace,
    face_alignment: FaceAlignment,
    live_portrait_pipeline: LivePortraitPipeline,
    gfpgan: GFPGANv1Clean,
    face_parser: BiSeNet,
    aligner: Optional[DifferentiableFaceAligner],
    args: PreprocessLaionArguments,
    device: str
):
    laion_face_base_dir = args.laion_face_base_dir
    output_dir = args.output_dir
    output_dir_resized = args.output_dir_resized

    laion_data_dir = os.path.join(laion_face_base_dir, "laion_face_data")

    for laion_face_part_index in args.laion_face_part_indices:
        split_folder = f"split_{laion_face_part_index:05d}"

        parquet_files = extract_parquet_files(laion_data_dir, output_dir_resized, split_folder)

        for parquet_file in tqdm(parquet_files, total=len(parquet_files), desc="tar files"):
            base_pathname = os.path.splitext(parquet_file)[0]
            base_filename = os.path.basename(base_pathname)

            cropped_face_path = os.path.join(output_dir, split_folder, base_filename)
            cropped_face_path_resized = os.path.join(output_dir_resized, split_folder, base_filename)
            if os.path.exists(cropped_face_path_resized):
                continue

            df = pq.read_table(parquet_file).to_pandas()
            df = df[df["status"] == "success"]

            dataset = (
                wds.WebDataset(f"{base_pathname}.tar", shardshuffle=False)
                    .decode("rgb8")
                    .to_tuple("jpg;png", "json")
                    .map(preprocess_data)
            )

            for rgb_image, id in tqdm(dataset, total=len(df), desc=f"images {base_filename}", leave=False):
                try:
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
                        device
                    )
                except Exception as ex:
                    print(f"An error occurred for sample {id}: {ex}")
                    traceback.print_tb(ex.__traceback__)
                    if args.stop_if_error:
                        raise ex


def main(args: PreprocessLaionArguments):
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
    parser.add_arguments(PreprocessLaionArguments, dest="arguments")
    args = cast(PreprocessLaionArguments, parser.parse_args().arguments)
    
    main(args)