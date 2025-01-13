# Heavily modified from https://github.com/ai-forever/ghost/blob/main/inference.py

print("started imports")

import os
from typing import cast

from simple_parsing import ArgumentParser

import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file
import lightning as L
from diffusers import AutoPipelineForInpainting

from FaceAlignment.api import FaceAlignment, LandmarksType
from Ghost.AEI_Net import *
from CVLFace import get_aligner
from BiSeNet.bisenet import BiSeNet
from GFPGAN.gfpganv1_clean_arch import GFPGANv1Clean
from RetinaFace.detector import RetinaFace
from utils.image_processing import (
    get_edge_mask,
    get_face_embeddings,
    convert_to_batch_tensor,
    get_padding_to_fit_resolution_multiple,
    initialize_embedding_model,
    paste_face_back_basic,
    paste_face_back_ghost,
    trans_points2d,
    torch2image,
    paste_face_back_facexlib,
    paste_face_back_insightface,
    enhance_face,
    sort_faces_by_coordinates,
    get_aligned_face_and_affine_matrix
)
from utils.inference.dataset import FaceEmbed
from utils.inference.inference_arguments import InferenceArguments

print("finished imports")


torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


class GhostV2DataModule(L.LightningDataModule):
    def __init__(
        self,
        source_file_path: str,
        target_file_path: str,
    ):
        super().__init__()
        self.source_file_path = source_file_path
        self.target_file_path = target_file_path


    def setup(self, stage=None):
        self.dataset = FaceEmbed(self.source_file_path, target_file_path=self.target_file_path)


    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            collate_fn=lambda l: l[0],
        )


class GhostV2Module(L.LightningModule):
    def __init__(self, args: InferenceArguments):
        super().__init__()

        self.detection_threshold = args.detection_threshold
        self.source_face_index = args.source_face_index
        self.target_face_index = args.target_face_index
        self.enhance_output = args.enhance_output
        self.align_mode = args.align_mode
        self.face_embeddings = args.face_embeddings
        self.paste_back_mode = args.paste_back_mode
        self.inpaint_output = args.inpaint_output

        self.debug = args.debug
        self.debug_ghost_landmarks = args.debug_ghost_landmarks
        self.debug_source_face_path = args.debug_source_face_path
        self.debug_target_face_path = args.debug_target_face_path
        self.debug_swapped_face_path = args.debug_swapped_face_path
        self.debug_enhanced_face_path = args.debug_enhanced_face_path
        
        checkpoint = load_file(args.G_path)
        checkpoint = { k.replace("_orig_mod.", ""): v for k,v in checkpoint.items() }

        self.G = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512, align_corners=args.align_corners)
        self.G.load_state_dict(checkpoint, strict=True)
        self.G.eval()

        self.embedding_model = initialize_embedding_model(args.face_embeddings, args)

        self.gfpgan = GFPGANv1Clean(
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
        self.gfpgan.load_state_dict(load_file(args.gfpgan_model_path), strict=True)
        self.gfpgan.eval()

        self.face_parser = None
        if args.paste_back_mode == "facexlib_with_parser":
            self.face_parser = BiSeNet(num_class=19)
            self.face_parser.load_state_dict(load_file(args.face_parser_model_path), strict=True)
            self.face_parser.eval()

        self.face_detector = RetinaFace(
            gpu_id=0,
            fp16=True,
            model_path=args.retina_face_model_path
        )

        self.face_alignment = None
        if args.paste_back_mode == "ghost":
            self.face_alignment = FaceAlignment(
                LandmarksType.TWO_D,
                safetensors_file_path=args.face_alignment_model_path,
                flip_input=False,
                dtype=torch.float16,
                face_detector="default",
            )

        self.aligner = None
        if args.align_mode == "cvlface":
            self.aligner = get_aligner(args.cvlface_aligner_model_path)

        self.sd_pipe = None

    
    def setup(self, stage: str):
        if self.inpaint_output and self.sd_pipe is None:
            self.sd_pipe = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            self.sd_pipe.enable_xformers_memory_efficient_attention()
            self.sd_pipe.enable_vae_tiling()
            self.sd_pipe.to(self.device)

    
    def predict_step(self, batch):
        print("Running predict step.")

        Xs_image, Xt_image = batch

        Xs_detected_faces = self.face_detector(Xs_image, threshold=self.detection_threshold, return_dict=True, cv=True)
        Xt_detected_faces = self.face_detector(Xt_image, threshold=self.detection_threshold, return_dict=True, cv=True)

        if len(Xs_detected_faces) == 0:
            raise ValueError("No face detected in source image!")

        if len(Xt_detected_faces) == 0:
            raise ValueError("No face detected in target image!")

        if len(Xs_detected_faces) <= self.source_face_index:
            raise ValueError(f"Only {len(Xs_detected_faces)} faces detected in source image, cannot select face with index {self.source_face_index}!")

        if len(Xt_detected_faces) <= self.target_face_index:
            raise ValueError(f"Only {len(Xt_detected_faces)} faces detected in target image, cannot select face with index {self.target_face_index}!")
        
        sort_faces_by_coordinates(Xs_detected_faces)
        sort_faces_by_coordinates(Xt_detected_faces)

        Xs_face_kps = Xs_detected_faces[self.source_face_index]["kps"]
        Xt_face_kps = Xt_detected_faces[self.target_face_index]["kps"]

        print(f"Aligning source and target images using {self.align_mode} align mode")
        Xs_face, Xs_affine_matrix = get_aligned_face_and_affine_matrix(
            Xs_image, Xs_face_kps, face_size=256, align_mode=self.align_mode, aligner=self.aligner, device=self.device)
        Xt_face, Xt_affine_matrix = get_aligned_face_and_affine_matrix(
            Xt_image, Xt_face_kps, face_size=256, align_mode=self.align_mode, aligner=self.aligner, device=self.device)

        Xs_face_landmarks_68 = None
        Xt_face_landmarks_68 = None
        if self.paste_back_mode == "ghost":
            Xs_face_box = Xs_detected_faces[self.source_face_index]["box"]
            Xt_face_box = Xt_detected_faces[self.target_face_index]["box"]

            Xs_landmarks_68 = self.face_alignment.get_landmarks_from_image(
                Xs_image[:, :, ::-1],
                detected_faces=[Xs_face_box],
            )[0]
            Xt_landmarks_68 = self.face_alignment.get_landmarks_from_image(
                Xt_image[:, :, ::-1],
                detected_faces=[Xt_face_box],
            )[0]
            Xs_face_landmarks_68 = trans_points2d(Xs_landmarks_68, Xs_affine_matrix)
            Xt_face_landmarks_68 = trans_points2d(Xt_landmarks_68, Xt_affine_matrix)
        
        if self.debug:
            os.makedirs(os.path.dirname(self.debug_source_face_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.debug_target_face_path), exist_ok=True)

            Xs_face_debug = Xs_face.copy()
            if self.debug_ghost_landmarks and Xs_face_landmarks_68 is not None:
                for index, point in enumerate(Xs_face_landmarks_68):
                    cv2.putText(Xs_face_debug, f"{index}", (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0))

            Xt_face_debug = Xt_face.copy()
            if self.debug_ghost_landmarks and Xt_face_landmarks_68 is not None:
                for index, point in enumerate(Xt_face_landmarks_68):
                    cv2.putText(Xt_face_debug, f"{index}", (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0))

            cv2.imwrite(self.debug_source_face_path, Xs_face_debug)
            cv2.imwrite(self.debug_target_face_path, Xt_face_debug)

        Xs_face_tensor = convert_to_batch_tensor(Xs_face, self.device)
        Xt_face_tensor = convert_to_batch_tensor(Xt_face, self.device)

        with torch.no_grad():
            Xs_embed = get_face_embeddings(Xs_face_tensor, self.embedding_model, self.face_embeddings)
            Yt_face, _ = self.G(Xt_face_tensor, Xs_embed)
            Yt_face = torch2image(Yt_face)[:, :, ::-1]

        if self.debug:
            os.makedirs(os.path.dirname(self.debug_swapped_face_path), exist_ok=True)
            cv2.imwrite(self.debug_swapped_face_path, Yt_face)

        if self.enhance_output:
            Yt_face_enhanced = cv2.resize(Yt_face, (512, 512), interpolation=cv2.INTER_LINEAR)
            Yt_face_enhanced = enhance_face(self.gfpgan, Yt_face_enhanced, "output", self.device)
            Yt_face_enhanced = cv2.resize(Yt_face_enhanced, (256, 256), interpolation=cv2.INTER_LINEAR)
        else:
            Yt_face_enhanced = Yt_face

        if self.debug:
            os.makedirs(os.path.dirname(self.debug_enhanced_face_path), exist_ok=True)
            cv2.imwrite(self.debug_enhanced_face_path, Yt_face_enhanced)

        if self.paste_back_mode == "facexlib_with_parser":
            Yt_image, Yt_mask = paste_face_back_facexlib(self.face_parser, Xt_image, Yt_face_enhanced, Xt_affine_matrix, True, self.device)
        elif self.paste_back_mode == "facexlib_without_parser":
            Yt_image, Yt_mask = paste_face_back_facexlib(self.face_parser, Xt_image, Yt_face_enhanced, Xt_affine_matrix, False, self.device)
        elif self.paste_back_mode == "insightface":
            Yt_image = paste_face_back_insightface(Xt_image, Xt_face, Yt_face_enhanced, Xt_affine_matrix)
            Yt_mask = None
        elif self.paste_back_mode == "ghost":
            Yt_image, Yt_mask = paste_face_back_ghost(Xt_image, Xs_face, Yt_face_enhanced, Xs_face_landmarks_68, Xt_face_landmarks_68, Xt_affine_matrix)
        elif self.paste_back_mode == "basic":
            Yt_image = paste_face_back_basic(Xt_image, Yt_face_enhanced, Xt_affine_matrix)
            Yt_mask = None
        else:
            Yt_image = Yt_face_enhanced
            Yt_mask = None
        
        if self.inpaint_output and Yt_mask is not None:
            Yt_image_rgba = cv2.cvtColor(Yt_image, cv2.COLOR_BGR2RGBA)
            Yt_edge_mask = get_edge_mask(Yt_mask)

            H_raw, W_raw, _ = Yt_image_rgba.shape
            W_pad, H_pad = get_padding_to_fit_resolution_multiple((W_raw, H_raw))

            Yt_image_rgba_padded = np.pad(Yt_image_rgba, [[0, H_pad], [0, W_pad], [0, 0]], mode="edge")
            Yt_edge_mask_padded = np.pad(Yt_edge_mask, [[0, H_pad], [0, W_pad]], mode="edge")

            Yt_image_rgba_padded = Image.fromarray(Yt_image_rgba_padded)
            Yt_edge_mask_padded = Image.fromarray(Yt_edge_mask_padded)

            blurred_mask = self.sd_pipe.mask_processor.blur(Yt_edge_mask_padded, blur_factor=4)

            Yt_inpainted = self.sd_pipe(
                prompt="a person's face",
                negative_prompt="deformed, glitch, noise, noisy, cross-eyed, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated",
                image=Yt_image_rgba_padded,
                mask_image=blurred_mask,
                width=Yt_image_rgba_padded.width,
                height=Yt_image_rgba_padded.height,
                num_inference_steps=20,
                guidance_scale=8.0,
                strength=0.4,
                padding_mask_crop=64,
            ).images[0]

            Yt_inpainted = cv2.cvtColor(np.ascontiguousarray(np.array(Yt_inpainted)[:H_raw, :W_raw].copy()).copy(), cv2.COLOR_RGBA2BGR)

            return Yt_inpainted

        return Yt_image


def main(args: InferenceArguments):
    if not torch.cuda.is_available():
        print("Cuda is not available, using CPU. Check if it's ok.")

    print("Creating PyTorch Lightning trainer")
    trainer = L.Trainer(precision=args.precision, logger=[])

    print("Creating GhostV2 Data Module")
    dm = GhostV2DataModule(
        args.source_file_path,
        args.target_file_path
    )

    print("Creating GhostV2 Module")
    with trainer.init_module():
        model = GhostV2Module(args)
    
    print("Starting inference")
    output = trainer.predict(model, dm)[0]

    print("Saving result to output")
    os.makedirs(os.path.dirname(args.output_file_path), exist_ok=True)
    cv2.imwrite(args.output_file_path, output)

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(InferenceArguments, dest="arguments")  # add arguments for the dataclass
    args = cast(InferenceArguments, parser.parse_args().arguments)
    
    main(args)