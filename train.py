print("started imports")

import argparse
import time
import cv2
import wandb
import os

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.optim.lr_scheduler as scheduler
from safetensors.torch import load_file, save_file
import lightning as L

from network.AEI_Net import *
from network.MultiscaleDiscriminator import *
from AdaptiveWingLoss.core import models
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
        if args.vgg:
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
    def __init__(
        self,
        args,
        backbone="unet",
        num_blocks=2,
        lr_G=4e-4,
        lr_D=4e-4,
        b1_G=0,
        b2_G=0.999,
        b1_D=0,
        b2_D=0.999,
        wd_G=1e-4,
        wd_D=1e-4,
        batch_size=16,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.args = args

        self.G = AEI_Net(backbone, num_blocks=num_blocks, c_id=512)
        self.D = MultiscaleDiscriminator(input_nc=3, n_layers=5, norm_layer=torch.nn.InstanceNorm2d)
        self.G.train()
        self.D.train()


    def setup(self, stage=None):
        self.loss_adv_accumulated = 20.
        
        if self.args.pretrained:
            try:
                self.G.load_state_dict(load_file(self.args.G_path, device=self.device), strict=False)
                self.D.load_state_dict(load_file(self.args.D_path, device=self.device), strict=False)
                print("Loaded pretrained weights for G and D")
            except FileNotFoundError as e:
                print("Not found pretrained weights. Continue without any pretrained weights.")

        self.facenet = InceptionResnetV1()
        self.facenet.load_state_dict(load_file('./weights/Facenet/facenet_pytorch.safetensors'))
        self.facenet = self.facenet.to(self.device)
        self.facenet.eval()

        if args.eye_detector_loss:
            self.model_ft = models.FAN(4, "False", "False", 98)
            checkpoint = load_file('./weights/AdaptiveWingLoss/WFLW_4HG.safetensors')
            if 'state_dict' not in checkpoint:
                self.model_ft.load_state_dict(checkpoint)
            else:
                pretrained_weights = checkpoint['state_dict']
                model_weights = self.model_ft.state_dict()
                pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                    if k in model_weights}
                model_weights.update(pretrained_weights)
                self.model_ft.load_state_dict(model_weights)
            self.model_ft = self.model_ft.to(self.device)
            self.model_ft.eval()
        else:
            self.model_ft=None


    def forward(self, Xt, embed):
        return self.G(Xt, embed)

    
    def on_train_batch_start(self, batch, batch_idx):
        self.start_time = time.time()

    
    def training_step(self, batch):
        Xs_orig, Xs, Xt, same_person = batch

        opt_G, opt_D = self.optimizers()
        if self.args.scheduler:
            scheduler_G, scheduler_D = self.lr_schedulers()

        # get the identity embeddings of Xs
        with torch.no_grad():
            embed = self.facenet(F.interpolate(Xs_orig, [160, 160], mode='bilinear', align_corners=False))

        diff_person = torch.ones_like(same_person, device=self.device)

        if self.args.diff_eq_same:
            same_person = diff_person

        # generator training
        Y, Xt_attr = self(Xt, embed)
        Di = self.D(Y)
        with torch.no_grad():
            ZY = self.facenet(F.interpolate(Y, [160, 160], mode='bilinear', align_corners=False))
        
        if self.args.eye_detector_loss:
            Xt_eyes, Xt_heatmap_left, Xt_heatmap_right = detect_landmarks(Xt, self.model_ft)
            Y_eyes, Y_heatmap_left, Y_heatmap_right = detect_landmarks(Y, self.model_ft)
            eye_heatmaps = [Xt_heatmap_left, Xt_heatmap_right, Y_heatmap_left, Y_heatmap_right]
        else:
            eye_heatmaps = None
        
        lossG, self.loss_adv_accumulated, L_adv, L_attr, L_id, L_rec, L_l2_eyes = compute_generator_losses(
            self.G, Y, Xt, Xt_attr, Di, embed, ZY, eye_heatmaps, self.loss_adv_accumulated, diff_person, same_person, self.args)
        
        opt_G.zero_grad()
        self.manual_backward(lossG)
        opt_G.step()
        if self.args.scheduler:
            scheduler_G.step()

        # discriminator training
        lossD = compute_discriminator_loss(self.D, Y, Xs, diff_person)

        opt_D.zero_grad()
        self.manual_backward(lossD)
        if (not self.args.discr_force) or (self.loss_adv_accumulated < 4.):
            opt_D.step()
        if self.args.scheduler:
            scheduler_D.step()

        return {
            "Xs": Xs,
            "Xt": Xt,
            "Xt_eyes": Xt_eyes if self.args.eye_detector_loss else None,
            "Y": Y,
            "Y_eyes": Y_eyes if self.args.eye_detector_loss else None,
            "loss_eyes": L_l2_eyes.item() if self.args.eye_detector_loss else None,
            "loss_id": L_id.item(),
            "lossD": lossD.item(),
            "lossG": lossG.item(),
            "loss_adv": L_adv.item(),
            "loss_attr": L_attr.item(),
            "loss_rec": L_rec.item(),
        }


    def configure_optimizers(self):
        lr_G = self.hparams.lr_G
        lr_D = self.hparams.lr_D
        b1_G = self.hparams.b1_G
        b2_G = self.hparams.b2_G
        b1_D = self.hparams.b1_D
        b2_D = self.hparams.b2_D
        wd_G = self.hparams.wd_G
        wd_D = self.hparams.wd_D

        opt_G = optim.Adam(self.G.parameters(), lr=lr_G, betas=(b1_G, b2_G), weight_decay=wd_G)
        opt_D = optim.Adam(self.D.parameters(), lr=lr_D, betas=(b1_D, b2_D), weight_decay=wd_D)

        if self.args.scheduler:
            scheduler_G = scheduler.StepLR(opt_G, step_size=self.args.scheduler_step, gamma=self.args.scheduler_gamma)
            scheduler_D = scheduler.StepLR(opt_D, step_size=self.args.scheduler_step, gamma=self.args.scheduler_gamma)
            return [opt_G, opt_D], [scheduler_G, scheduler_D]

        return [opt_G, opt_D], []

    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_time = time.time() - self.start_time

        if batch_idx % self.args.show_step == 0:
            images = [outputs["Xs"], outputs["Xt"], outputs["Y"]]
            if self.args.eye_detector_loss:
                Xt_eyes_img = paint_eyes(outputs["Xt"], outputs["Xt_eyes"])
                Yt_eyes_img = paint_eyes(outputs["Y"], outputs["Y_eyes"])
                images.extend([Xt_eyes_img, Yt_eyes_img])
            image = make_image_list(images)
            if self.args.use_wandb:
                wandb.log({"gen_images":wandb.Image(image, caption=f"{self.current_epoch:03}" + '_' + f"{batch_idx:06}")})
            else:
                cv2.imwrite('./results/images/generated_image.jpg', image[:,:,::-1])
        
        if batch_idx % 10 == 0:
            print(f'epoch: {self.current_epoch}    {batch_idx} / {self.trainer.num_training_batches}')
            print(f'lossD: {outputs["lossD"]}    lossG: {outputs["lossG"]} batch_time: {batch_time}s')
            print(f'L_adv: {outputs["loss_adv"]} L_id: {outputs["loss_id"]} L_attr: {outputs["loss_attr"]} L_rec: {outputs["loss_rec"]}')
            if self.args.eye_detector_loss:
                print(f'L_l2_eyes: {outputs["loss_eyes"]}')
            print(f'loss_adv_accumulated: {self.loss_adv_accumulated}')
            if self.args.scheduler:
                scheduler_G, scheduler_D = self.lr_schedulers()
                print(f'scheduler_G lr: {scheduler_G.get_last_lr()} scheduler_D lr: {scheduler_D.get_last_lr()}')
        
        if self.args.use_wandb:
            if self.args.eye_detector_loss:
                wandb.log({"loss_eyes": outputs["loss_eyes"]}, commit=False)
            wandb.log({"loss_id": outputs["loss_id"],
                       "lossD": outputs["lossD"],
                       "lossG": outputs["lossG"],
                       "loss_adv": outputs["loss_adv"],
                       "loss_attr": outputs["loss_attr"],
                       "loss_rec": outputs["loss_rec"]})
        
        if batch_idx % 5000 == 0:
            save_file(self.G.state_dict(), f'./results/saved_models_{self.args.run_name}/G_latest.safetensors')
            save_file(self.D.state_dict(), f'./results/saved_models_{self.args.run_name}/D_latest.safetensors')

            save_file(self.G.state_dict(), f'./results/current_models_{self.args.run_name}/G_' + str(self.current_epoch)+ '_' + f"{batch_idx:06}" + '.safetensors')
            save_file(self.D.state_dict(), f'./results/current_models_{self.args.run_name}/D_' + str(self.current_epoch)+ '_' + f"{batch_idx:06}" + '.safetensors')

        if (batch_idx % 250 == 0):
            # Let's see how the swap looks on three specific photos to track the dynamics
            self.G.eval()

            res1 = get_faceswap('examples/images/training/source1.png', 'examples/images/training/target1.png', self.G, self.facenet, self.device)
            res2 = get_faceswap('examples/images/training/source2.png', 'examples/images/training/target2.png', self.G, self.facenet, self.device)  
            res3 = get_faceswap('examples/images/training/source3.png', 'examples/images/training/target3.png', self.G, self.facenet, self.device)
            
            res4 = get_faceswap('examples/images/training/source4.png', 'examples/images/training/target4.png', self.G, self.facenet, self.device)
            res5 = get_faceswap('examples/images/training/source5.png', 'examples/images/training/target5.png', self.G, self.facenet, self.device)  
            res6 = get_faceswap('examples/images/training/source6.png', 'examples/images/training/target6.png', self.G, self.facenet, self.device)
            
            output1 = np.concatenate((res1, res2, res3), axis=0)
            output2 = np.concatenate((res4, res5, res6), axis=0)
            
            output = np.concatenate((output1, output2), axis=1)

            if self.args.use_wandb:
                wandb.log({"our_images":wandb.Image(output, caption=f"{self.current_epoch:03}" + '_' + f"{batch_idx:06}")})
            else:
                cv2.imwrite('./results/images/our_images.jpg', output[:,:,::-1])

            self.G.train()


def main(args):
    if not torch.cuda.is_available():
        print('cuda is not available. using cpu. check if it\'s ok')

    print("Creating PyTorch Lightning trainer")
    trainer = L.Trainer(
        max_epochs=args.max_epoch,
        limit_val_batches=0,
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
        model = GhostV2Module(
            args,
            backbone=args.backbone,
            num_blocks=args.num_blocks,
            lr_G=args.lr_G,
            lr_D=args.lr_D,
            batch_size=args.batch_size,
        )
    
    print("Starting training")
    trainer.fit(model, dm)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # dataset params
    parser.add_argument('--dataset_path', default='/VggFace2-crop/', help='Path to the dataset. If not VGG2 dataset is used, param --vgg should be set False')
    parser.add_argument('--G_path', default='./weights/G.safetensors', help='Path to pretrained weights for G. Only used if pretrained=True')
    parser.add_argument('--D_path', default='./weights/D.safetensors', help='Path to pretrained weights for D. Only used if pretrained=True')
    parser.add_argument('--vgg', action=argparse.BooleanOptionalAction, default=True, type=bool, help='When using VGG2 dataset (or any other dataset with several photos for one identity)')
    # weights for loss
    parser.add_argument('--weight_adv', default=1, type=float, help='Adversarial Loss weight')
    parser.add_argument('--weight_attr', default=10, type=float, help='Attributes weight')
    parser.add_argument('--weight_id', default=20, type=float, help='Identity Loss weight')
    parser.add_argument('--weight_rec', default=10, type=float, help='Reconstruction Loss weight')
    parser.add_argument('--weight_eyes', default=0., type=float, help='Eyes Loss weight')
    # training params you may want to change
    
    parser.add_argument('--backbone', default='unet', const='unet', nargs='?', choices=['unet', 'linknet', 'resnet'], help='Backbone for attribute encoder')
    parser.add_argument('--num_blocks', default=2, type=int, help='Numbers of AddBlocks at AddResblock')
    parser.add_argument('--same_person', default=0.2, type=float, help='Probability of using same person identity during training')
    parser.add_argument('--same_identity', action=argparse.BooleanOptionalAction, default=True, type=bool, help='Using simswap approach, when source_id = target_id. Only possible with vgg=True')
    parser.add_argument('--diff_eq_same', action=argparse.BooleanOptionalAction, default=False, type=bool, help='Don\'t use info about where is defferent identities')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True, type=bool, help='If using the pretrained weights for training or not')
    parser.add_argument('--discr_force', action=argparse.BooleanOptionalAction, default=False, type=bool, help='If True Discriminator would not train when adversarial loss is high')
    parser.add_argument('--scheduler', action=argparse.BooleanOptionalAction, default=False, type=bool, help='If True decreasing LR is used for learning of generator and discriminator')
    parser.add_argument('--scheduler_step', default=5000, type=int)
    parser.add_argument('--scheduler_gamma', default=0.2, type=float, help='It is value, which shows how many times to decrease LR')
    parser.add_argument('--eye_detector_loss', action=argparse.BooleanOptionalAction, default=False, type=bool, help='If True eye loss with using AdaptiveWingLoss detector is applied to generator')
    # info about this run
    parser.add_argument('--use_wandb', action=argparse.BooleanOptionalAction, default=False, type=bool, help='Use wandb to track your experiments or not')
    parser.add_argument('--run_name', required=True, type=str, help='Name of this run. Used to create folders where to save the weights.')
    parser.add_argument('--wandb_project', default='your-project-name', type=str)
    parser.add_argument('--wandb_entity', default='your-login', type=str)
    # training params you probably don't want to change
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr_G', default=4e-4, type=float)
    parser.add_argument('--lr_D', default=4e-4, type=float)
    parser.add_argument('--max_epoch', default=2000, type=int)
    parser.add_argument('--show_step', default=500, type=int)
    parser.add_argument('--save_epoch', default=1, type=int)

    args = parser.parse_args()
    
    if args.vgg==False and args.same_identity==True:
        raise ValueError("Sorry, you can't use some other dataset than VGG2 Faces with param same_identity=True")
    
    if args.use_wandb==True:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, settings=wandb.Settings(start_method='fork'))

        config = wandb.config
        config.dataset_path = args.dataset_path
        config.weight_adv = args.weight_adv
        config.weight_attr = args.weight_attr
        config.weight_id = args.weight_id
        config.weight_rec = args.weight_rec
        config.weight_eyes = args.weight_eyes
        config.same_person = args.same_person
        config.vgg_to_face = args.vgg
        config.same_identity = args.same_identity
        config.diff_eq_same = args.diff_eq_same
        config.discr_force = args.discr_force
        config.scheduler = args.scheduler
        config.scheduler_step = args.scheduler_step
        config.scheduler_gamma = args.scheduler_gamma
        config.eye_detector_loss = args.eye_detector_loss
        config.pretrained = args.pretrained
        config.run_name = args.run_name
        config.G_path = args.G_path
        config.D_path = args.D_path
        config.batch_size = args.batch_size
        config.lr_G = args.lr_G
        config.lr_D = args.lr_D
    elif not os.path.exists('./results/images'):
        os.makedirs('./results/images')
    
    # Create folders to store the latest model weights, as well as weights from each era
    if not os.path.exists(f'./results/saved_models_{args.run_name}'):
        os.makedirs(f'./results/saved_models_{args.run_name}')
    if not os.path.exists(f'./results/current_models_{args.run_name}'):
        os.makedirs(f'./results/current_models_{args.run_name}')
    
    main(args)