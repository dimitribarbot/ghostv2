print("started imports")

import time
import cv2
import wandb
import os
from typing import cast, List

from simple_parsing import ArgumentParser

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.optim.lr_scheduler as scheduler
from safetensors.torch import load_file, save_file
import lightning as L
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch import loggers as pl_loggers

from network.AEI_Net import *
from network.MultiscaleDiscriminator import *
from AdaptiveWingLoss.core import models
from utils.training.arguments import TrainingArguments
from utils.training.Dataset import FaceEmbedVGG2, FaceEmbed
from utils.training.image_processing import make_image_list, get_faceswap
from utils.training.losses import compute_discriminator_loss, compute_generator_losses
from utils.training.detector import detect_landmarks, paint_eyes
from facenet.inception_resnet_v1 import InceptionResnetV1

print("finished imports")


torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


class GhostV2DataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        same_person=0.2,
        vgg=True,
        same_identity=True,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.same_person = same_person
        self.vgg = vgg
        self.same_identity = same_identity
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory


    def setup(self, stage=None):
        if self.vgg:
            self.dataset = FaceEmbedVGG2(self.dataset_path, same_prob=self.same_person, same_identity=self.same_identity)
        else:
            self.dataset = FaceEmbed([self.dataset_path], same_prob=self.same_person)


    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )


class GhostV2Module(L.LightningModule):
    def __init__(self, args: TrainingArguments):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.args = args

        self.G = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512)
        self.D = MultiscaleDiscriminator(input_nc=3, n_layers=5, norm_layer=torch.nn.InstanceNorm2d)
        self.G = cast(AEI_Net, torch.compile(self.G))
        self.D = cast(MultiscaleDiscriminator, torch.compile(self.D))
        self.G.train()
        self.D.train()

        self.register_buffer("loss_adv_accumulated", torch.tensor(20.))


    def setup(self, stage=None):
        if self.args.pretrained:
            try:
                self.G.load_state_dict(load_file(self.args.G_path, device=self.device), strict=False)
                self.D.load_state_dict(load_file(self.args.D_path, device=self.device), strict=False)
                print("Loaded pretrained weights for G and D")
            except FileNotFoundError as e:
                print("Not found pretrained weights. Continue without any pretrained weights.")

        self.facenet = InceptionResnetV1()
        self.facenet.load_state_dict(load_file("./weights/Facenet/facenet_pytorch.safetensors"))
        self.facenet = self.facenet.to(self.device)
        self.facenet.eval()

        if self.args.eye_detector_loss:
            self.model_ft = models.FAN(4, "False", "False", 98)
            checkpoint = load_file("./weights/AdaptiveWingLoss/WFLW_4HG.safetensors")
            if "state_dict" not in checkpoint:
                self.model_ft.load_state_dict(checkpoint)
            else:
                pretrained_weights = checkpoint["state_dict"]
                model_weights = self.model_ft.state_dict()
                pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                    if k in model_weights}
                model_weights.update(pretrained_weights)
                self.model_ft.load_state_dict(model_weights)
            self.model_ft = self.model_ft.to(self.device)
            self.model_ft.eval()
        else:
            self.model_ft=None

    
    def training_step(self, batch):
        Xs_orig, Xs, Xt, same_person = batch

        opt_G, opt_D = cast(List[LightningOptimizer], self.optimizers())

        # Hack to avoid double count of global_step
        # See https://github.com/Lightning-AI/pytorch-lightning/issues/17958 for more information
        opt_D._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_D._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")

        if self.args.use_scheduler:
            scheduler_G, scheduler_D = self.lr_schedulers()

        # get the identity embeddings of Xs
        with torch.no_grad():
            embed = self.facenet(F.interpolate(Xs_orig, [160, 160], mode="bilinear", align_corners=False))

        diff_person = torch.ones_like(same_person, device=self.device)

        if self.args.diff_eq_same:
            same_person = diff_person

        # generator training
        opt_G.optimizer.zero_grad()

        Y, Xt_attr = self.G(Xt, embed)
        Di = self.D(Y)
        ZY = self.facenet(F.interpolate(Y, [160, 160], mode="bilinear", align_corners=False))
        
        if self.args.eye_detector_loss:
            Xt_eyes, Xt_heatmap_left, Xt_heatmap_right = detect_landmarks(Xt, self.model_ft)
            Y_eyes, Y_heatmap_left, Y_heatmap_right = detect_landmarks(Y, self.model_ft)
            eye_heatmaps = [Xt_heatmap_left, Xt_heatmap_right, Y_heatmap_left, Y_heatmap_right]
        else:
            eye_heatmaps = None
        
        lossG, self.loss_adv_accumulated, L_adv, L_attr, L_id, L_rec, L_l2_eyes = compute_generator_losses(
            self.G, Y, Xt, Xt_attr, Di, embed, ZY, eye_heatmaps, self.loss_adv_accumulated, diff_person, same_person, self.args)
        
        self.manual_backward(lossG)
        opt_G.step()
        if self.args.use_scheduler:
            scheduler_G.step()

        # discriminator training
        opt_D.optimizer.zero_grad()
        lossD = compute_discriminator_loss(self.D, Y, Xs, diff_person)
        self.manual_backward(lossD)
        if (not self.args.discr_force) or (self.loss_adv_accumulated < 4.):
            opt_D.step()
        if self.args.use_scheduler:
            scheduler_D.step()

        return {
            "Xs": Xs,
            "Xt": Xt,
            "Xt_eyes": Xt_eyes if self.args.eye_detector_loss else None,
            "Y": Y,
            "Y_eyes": Y_eyes if self.args.eye_detector_loss else None,
            "loss_eyes": L_l2_eyes if self.args.eye_detector_loss else None,
            "loss_id": L_id,
            "lossD": lossD,
            "lossG": lossG,
            "loss_adv": L_adv,
            "loss_attr": L_attr,
            "loss_rec": L_rec,
        }


    def configure_optimizers(self):
        opt_G = optim.Adam(self.G.parameters(), lr=self.args.lr_G, betas=(self.args.b1_G, self.args.b2_G), weight_decay=self.args.wd_G)
        opt_D = optim.Adam(self.D.parameters(), lr=self.args.lr_D, betas=(self.args.b1_D, self.args.b2_D), weight_decay=self.args.wd_D)

        if self.args.use_scheduler:
            scheduler_G = scheduler.StepLR(opt_G, step_size=self.args.scheduler_step, gamma=self.args.scheduler_gamma)
            scheduler_D = scheduler.StepLR(opt_D, step_size=self.args.scheduler_step, gamma=self.args.scheduler_gamma)
            return [opt_G, opt_D], [scheduler_G, scheduler_D]

        return [opt_G, opt_D], []
    

class GhostV2EvalCallback(L.Callback):
    def on_train_batch_end(self, trainer, pl_module: GhostV2Module, outputs, batch, batch_idx):
        if (batch_idx % 250 == 0):
            # Let's see how the swap looks on three specific photos to track the dynamics
            pl_module.G.eval()

            res1 = get_faceswap("examples/images/training/source1.png", "examples/images/training/target1.png", pl_module.G, pl_module.facenet, pl_module.device)
            res2 = get_faceswap("examples/images/training/source2.png", "examples/images/training/target2.png", pl_module.G, pl_module.facenet, pl_module.device)  
            res3 = get_faceswap("examples/images/training/source3.png", "examples/images/training/target3.png", pl_module.G, pl_module.facenet, pl_module.device)
            
            res4 = get_faceswap("examples/images/training/source4.png", "examples/images/training/target4.png", pl_module.G, pl_module.facenet, pl_module.device)
            res5 = get_faceswap("examples/images/training/source5.png", "examples/images/training/target5.png", pl_module.G, pl_module.facenet, pl_module.device)  
            res6 = get_faceswap("examples/images/training/source6.png", "examples/images/training/target6.png", pl_module.G, pl_module.facenet, pl_module.device)
            
            output1 = np.concatenate((res1, res2, res3), axis=0)
            output2 = np.concatenate((res4, res5, res6), axis=0)
            
            output = np.concatenate((output1, output2), axis=1)

            if pl_module.args.use_wandb:
                wandb.log({"our_images":wandb.Image(output, caption=f"{pl_module.current_epoch:03}_{batch_idx:06}")})
            else:
                cv2.imwrite("./results/images/our_images.jpg", output[:,:,::-1])

            pl_module.G.train()
    

class GhostV2ShowStepCallback(L.Callback):
    def on_train_batch_end(self, trainer, pl_module: GhostV2Module, outputs, batch, batch_idx):
        if batch_idx % pl_module.args.show_step == 0:
            images = [outputs["Xs"], outputs["Xt"], outputs["Y"]]
            if pl_module.args.eye_detector_loss:
                Xt_eyes_img = paint_eyes(outputs["Xt"], outputs["Xt_eyes"])
                Yt_eyes_img = paint_eyes(outputs["Y"], outputs["Y_eyes"])
                images.extend([Xt_eyes_img, Yt_eyes_img])
            image = make_image_list(images)
            if pl_module.args.use_wandb:
                wandb.log({"gen_images":wandb.Image(image, caption=f"{pl_module.current_epoch:03}_{batch_idx:06}")})
            else:
                cv2.imwrite("./results/images/generated_image.jpg", image[:,:,::-1])
    

class GhostV2LoggingCallback(L.Callback):
    def on_train_batch_start(self, trainer, pl_module: GhostV2Module, batch, batch_idx):
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module: GhostV2Module, outputs, batch, batch_idx):        
        if batch_idx % 50 == 0:
            batch_time = time.time() - self.start_time
            print(f"epoch: {pl_module.current_epoch}    {batch_idx} / {pl_module.trainer.num_training_batches}")
            print(f"lossD: {outputs['lossD']}    lossG: {outputs['lossG']} batch_time: {batch_time}s")
            print(f"L_adv: {outputs['loss_adv']} L_id: {outputs['loss_id']} L_attr: {outputs['loss_attr']} L_rec: {outputs['loss_rec']}")
            if pl_module.args.eye_detector_loss:
                print(f"L_l2_eyes: {outputs['loss_eyes']}")
            print(f"loss_adv_accumulated: {pl_module.loss_adv_accumulated}")
            if pl_module.args.use_scheduler:
                scheduler_G, scheduler_D = pl_module.lr_schedulers()
                print(f"scheduler_G lr: {scheduler_G.get_last_lr()} scheduler_D lr: {scheduler_D.get_last_lr()}")
    

class GhostV2CheckpointCallback(L.Callback):
    def on_train_batch_end(self, trainer, pl_module: GhostV2Module, outputs, batch, batch_idx):
        iteration = pl_module.global_step

        if iteration > 0 and iteration % 20000 == 0:
            save_file(pl_module.G.state_dict(), f"./results/saved_models_{pl_module.args.run_name}/G_latest.safetensors")
            save_file(pl_module.D.state_dict(), f"./results/saved_models_{pl_module.args.run_name}/D_latest.safetensors")

            save_file(pl_module.G.state_dict(), f"./results/current_models_{pl_module.args.run_name}/G_{str(pl_module.current_epoch)}_{batch_idx:06}.safetensors")
            save_file(pl_module.D.state_dict(), f"./results/current_models_{pl_module.args.run_name}/D_{str(pl_module.current_epoch)}_{batch_idx:06}.safetensors")
    

class GhostV2WandbCallback(L.Callback):
    def on_train_batch_end(self, trainer, pl_module: GhostV2Module, outputs, batch, batch_idx):
        if pl_module.args.eye_detector_loss:
            wandb.log({"loss_eyes": outputs["loss_eyes"].item(),
                       "trainer/global_step": pl_module.global_step}, commit=False)
        wandb.log({"loss_id": outputs["loss_id"].item(),
                   "lossD": outputs["lossD"].item(),
                   "lossG": outputs["lossG"].item(),
                   "loss_adv": outputs["loss_adv"].item(),
                   "loss_attr": outputs["loss_attr"].item(),
                   "loss_rec": outputs["loss_rec"].item(),
                   "trainer/global_step": pl_module.global_step})


def main(args: TrainingArguments):
    if not torch.cuda.is_available():
        print("Cuda is not available, using CPU. Check if it's ok.")

    callbacks = [
        GhostV2ShowStepCallback(),
        GhostV2LoggingCallback(),
        GhostV2CheckpointCallback(),
        GhostV2EvalCallback(),
    ]

    if args.use_wandb:
        logger = pl_loggers.WandbLogger(project=args.wandb_project)
        logger.experiment.config.update({
            "dataset_path": args.dataset_path,
            "weight_adv": args.weight_adv,
            "weight_attr": args.weight_attr,
            "weight_id": args.weight_id,
            "weight_rec": args.weight_rec,
            "weight_eyes": args.weight_eyes,
            "same_person": args.same_person,
            "vgg_to_face": args.vgg,
            "same_identity": args.same_identity,
            "diff_eq_same": args.diff_eq_same,
            "discr_force": args.discr_force,
            "scheduler": args.use_scheduler,
            "scheduler_step": args.scheduler_step,
            "scheduler_gamma": args.scheduler_gamma,
            "eye_detector_loss": args.eye_detector_loss,
            "pretrained": args.pretrained,
            "run_name": args.run_name,
            "ckpt_path": args.ckpt_path,
            "G_path": args.G_path,
            "D_path": args.D_path,
            "batch_size": args.batch_size,
            "lr_G": args.lr_G,
            "lr_D": args.lr_D,
            "precision": args.precision,
        })
        callbacks.append(GhostV2WandbCallback())
    else:
        logger = pl_loggers.TensorBoardLogger(save_dir=os.getcwd(), version=args.run_name)

    print("Creating PyTorch Lightning trainer")
    trainer = L.Trainer(
        max_epochs=args.max_epoch,
        limit_val_batches=0,
        default_root_dir=f"./results/current_models_{args.run_name}",
        logger=logger,
        callbacks=callbacks,
        precision=args.precision,
    )

    print("Creating GhostV2 Data Module")
    dm = GhostV2DataModule(
        args.dataset_path,
        same_person=args.same_person,
        vgg=args.vgg,
        same_identity=args.same_identity,
        batch_size=args.batch_size,
    )

    print("Creating GhostV2 Module")
    with trainer.init_module():
        model = GhostV2Module(args)
    
    print("Starting training")
    trainer.fit(model, dm, ckpt_path=args.ckpt_path)

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(TrainingArguments, dest="arguments")  # add arguments for the dataclass
    args = cast(TrainingArguments, parser.parse_args().arguments)
    
    if args.vgg==False and args.same_identity==True:
        raise ValueError("Sorry, you can't use some other dataset than VGG2 Faces with param same_identity=True")
    
    if not os.path.exists("./results/images"):
        os.makedirs("./results/images")
    
    # Create folders to store the latest model weights, as well as weights from each era
    if not os.path.exists(f"./results/saved_models_{args.run_name}"):
        os.makedirs(f"./results/saved_models_{args.run_name}")
    if not os.path.exists(f"./results/current_models_{args.run_name}"):
        os.makedirs(f"./results/current_models_{args.run_name}")
    
    main(args)