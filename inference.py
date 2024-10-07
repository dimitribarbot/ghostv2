print("started imports")

import cv2
from typing import cast

from simple_parsing import ArgumentParser

import torch
from safetensors.torch import load_file
import lightning as L

from network.AEI_Net import *
from network.MultiscaleDiscriminator import *
from utils.inference.inference_arguments import InferenceArguments
from utils.training.image_processing import get_faceswap
from facenet.inception_resnet_v1 import InceptionResnetV1

print("finished imports")


torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


class GhostV2Module(L.LightningModule):
    def __init__(self, args: InferenceArguments):
        super().__init__()
        self.G_path = args.G_path
        self.G = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512)
        self.G = cast(AEI_Net, torch.compile(self.G))


    def setup(self, stage=None):
        try:
            self.G.load_state_dict(load_file(self.G_path), strict=False)
            self.G = self.G.to(self.device)
            self.G.eval()
            print("Loaded pretrained weights for G")
        except FileNotFoundError:
            raise ValueError(f"Pretrained weights for G not found at {self.G_path}.")

        self.facenet = InceptionResnetV1()
        self.facenet.load_state_dict(load_file("./weights/Facenet/facenet_pytorch.safetensors"))
        self.facenet = self.facenet.to(self.device)
        self.facenet.eval()

    
    def predict_step(self, batch, batch_idx):
        print("Running predict step.")

        res1 = get_faceswap("examples/images/training/source1.png", "examples/images/training/target1.png", self.G, self.facenet, self.device)
        res2 = get_faceswap("examples/images/training/source2.png", "examples/images/training/target2.png", self.G, self.facenet, self.device)  
        res3 = get_faceswap("examples/images/training/source3.png", "examples/images/training/target3.png", self.G, self.facenet, self.device)
        
        res4 = get_faceswap("examples/images/training/source4.png", "examples/images/training/target4.png", self.G, self.facenet, self.device)
        res5 = get_faceswap("examples/images/training/source5.png", "examples/images/training/target5.png", self.G, self.facenet, self.device)  
        res6 = get_faceswap("examples/images/training/source6.png", "examples/images/training/target6.png", self.G, self.facenet, self.device)
        
        output1 = np.concatenate((res1, res2, res3), axis=0)
        output2 = np.concatenate((res4, res5, res6), axis=0)
        
        output = np.concatenate((output1, output2), axis=1)

        cv2.imwrite("./examples/results/our_images.jpg", output[:,:,::-1])

        return output


def main(args: InferenceArguments):
    if not torch.cuda.is_available():
        print("Cuda is not available, using CPU. Check if it's ok.")

    print("Creating PyTorch Lightning trainer")
    trainer = L.Trainer(precision=args.precision)

    print("Creating GhostV2 Module")
    with trainer.init_module():
        model = GhostV2Module(args)
    
    print("Starting training")
    trainer.predict(model, dataloaders=[[""]])

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(InferenceArguments, dest="arguments")  # add arguments for the dataclass
    args = cast(InferenceArguments, parser.parse_args().arguments)
    
    main(args)