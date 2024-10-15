print("started imports")

import os
from typing import cast

from simple_parsing import ArgumentParser

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import normalize
from safetensors.torch import load_file
import lightning as L

from network.AEI_Net import *
from network.MultiscaleDiscriminator import *
from utils.inference.inference_arguments import InferenceArguments
from utils.image_processing import align_warp_face, img2tensor, torch2image, paste_face_back, enhance_face, sort_faces_by_coordinates, norm_crop
from utils.inference.Dataset import FaceEmbed
from facenet.inception_resnet_v1 import InceptionResnetV1
from BiSeNet.bisenet import BiSeNet
from GFPGAN.gfpganv1_clean_arch import GFPGANv1Clean
from RetinaFace.detector import RetinaFace

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

        self.source_face_index = args.source_face_index
        self.target_face_index = args.target_face_index
        self.enhance_output = args.enhance_output
        self.align_mode = args.align_mode
        
        checkpoint = load_file(args.G_path)
        checkpoint = { k.replace("_orig_mod.", ""): v for k,v in checkpoint.items() }

        self.G = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512)
        self.G.load_state_dict(checkpoint, strict=True)
        self.G.eval()

        self.facenet = InceptionResnetV1()
        self.facenet.load_state_dict(load_file("./weights/Facenet/facenet_pytorch.safetensors"))
        self.facenet.eval()

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

        self.face_parser = BiSeNet(num_class=19)
        self.face_parser.load_state_dict(load_file(args.face_parser_model_path), strict=True)
        self.face_parser.eval()

        self.face_detector = RetinaFace(
            gpu_id=0,
            fp16=True,
            model_path=args.retina_face_model_path
        )

    
    def predict_step(self, batch):
        print("Running predict step.")

        Xs_image, Xt_image = batch

        Xs_detected_faces = self.face_detector(Xs_image, threshold=0.97, return_dict=True, cv=True)
        Xt_detected_faces = self.face_detector(Xt_image, threshold=0.97, return_dict=True, cv=True)

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

        # Xs_image_size = Xs_image.shape[1::-1]

        # Xs_face_box = Xs_detected_faces[self.source_face_index]["box"]
        Xs_face_kps = Xs_detected_faces[self.source_face_index]["kps"]
        Xt_face_kps = Xt_detected_faces[self.target_face_index]["kps"]

        # Xs_face_for_facenet = Xs_image[
        #     int(max(Xs_face_box[1], 0)):int(min(Xs_face_box[3], Xs_image_size[1])),
        #     int(max(Xs_face_box[0], 0)):int(min(Xs_face_box[2], Xs_image_size[0]))
        # ]
        # Xs_face_for_facenet = cv2.resize(Xs_face_for_facenet, (160, 160), interpolation=cv2.INTER_AREA).copy()
        # Xs_face_for_facenet = img2tensor(Xs_face_for_facenet, bgr2rgb=True, float32=True)
        # Xs_face_for_facenet = (Xs_face_for_facenet - 127.5) / 128.0
        # Xs_face_for_facenet = Xs_face_for_facenet.unsqueeze(0).to(self.device)

        if self.align_mode == "insightface":
            Xs_face, _ = norm_crop(Xs_image, Xs_face_kps, face_size=256)
            Xt_face, Xt_affine_matrix = norm_crop(Xt_image, Xt_face_kps, face_size=256)
        else:
            Xs_face, _ = align_warp_face(Xs_image, Xs_face_kps, face_size=256)
            Xt_face, Xt_affine_matrix = align_warp_face(Xt_image, Xt_face_kps, face_size=256)

        Xs_face = img2tensor(Xs_face / 255., bgr2rgb=True, float32=True)
        normalize(Xs_face, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        Xs_face = Xs_face.unsqueeze(0).to(self.device)

        Xt_face = img2tensor(Xt_face / 255., bgr2rgb=True, float32=True)
        normalize(Xt_face, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        Xt_face = Xt_face.unsqueeze(0).to(self.device)

        with torch.no_grad():
            Xs_embed = self.facenet(F.interpolate(Xs_face, [160, 160], mode="bilinear", align_corners=False))
            # Xs_embed = self.facenet(Xs_face_for_facenet)
            Yt_face, _ = self.G(Xt_face, Xs_embed)
            Yt_face = torch2image(Yt_face)[:, :, ::-1]

        if self.enhance_output:
            Yt_face_enhanced = cv2.resize(Yt_face, (512, 512), interpolation=cv2.INTER_LINEAR)
            Yt_face_enhanced = enhance_face(self.gfpgan, Yt_face_enhanced, "output", self.device)
            Yt_face_enhanced = cv2.resize(Yt_face_enhanced, (256, 256), interpolation=cv2.INTER_LINEAR)
        else:
            Yt_face_enhanced = Yt_face

        Yt_image = paste_face_back(self.face_parser, Xt_image, Yt_face_enhanced, Xt_affine_matrix, self.device)
        if np.max(Yt_image) > 256:  # 16-bit image
            Yt_image = Yt_image.astype(np.uint16)
        else:
            Yt_image = Yt_image.astype(np.uint8)

        return Yt_image


def main(args: InferenceArguments):
    if not torch.cuda.is_available():
        print("Cuda is not available, using CPU. Check if it's ok.")

    print("Creating PyTorch Lightning trainer")
    trainer = L.Trainer(precision=args.precision)

    print("Creating GhostV2 Data Module")
    dm = GhostV2DataModule(
        args.source_file_path,
        args.target_file_path
    )

    print("Creating GhostV2 Module")
    with trainer.init_module():
        model = GhostV2Module(args)
    
    print("Starting training")
    output = trainer.predict(model, dm)[0]

    print("Saving result to output")
    os.makedirs(os.path.dirname(args.output_file_path), exist_ok=True)
    cv2.imwrite(args.output_file_path, output)

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(InferenceArguments, dest="arguments")  # add arguments for the dataclass
    args = cast(InferenceArguments, parser.parse_args().arguments)
    
    main(args)