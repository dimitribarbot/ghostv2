from dataclasses import dataclass

from simple_parsing import choice
from simple_parsing.helpers import flag

@dataclass
class TrainingArguments:
    """ Dataset params """
    dataset_path: str = "/VggFace2-crop/"                                   # Path to the dataset. If not VGG2 dataset is used, param --vgg should be set False
    G_path: str = "./weights/G.safetensors"                                 # Path to pretrained weights for G. Only used if pretrained=True
    D_path: str = "./weights/D.safetensors"                                 # Path to pretrained weights for D. Only used if pretrained=True
    vgg: bool = flag(default=True, negative_prefix="--no-")                 # When using VGG2 dataset (or any other dataset with several photos for one identity)

    """ Weights for loss """
    weight_adv: float = 1                                                   # Adversarial Loss weight
    weight_attr: float = 10                                                 # Attributes weight
    weight_id: float = 15                                                   # Identity Loss weight
    weight_rec: float = 10                                                  # Reconstruction Loss weight
    weight_eyes: float = 0                                                  # Eyes Loss weight

    """ Training params you may want to change """
    backbone: str = choice("unet", "linknet", "resnet", default="unet")     # Backbone for attribute encoder
    num_blocks: int = 2                                                     # Numbers of AddBlocks at AddResblock
    same_person: float = 0.2                                                # Probability of using same person identity during training
    same_identity: bool = flag(default=True, negative_prefix="--no-")       # Using simswap approach, when source_id = target_id. Only possible with vgg=True
    diff_eq_same: bool = flag(default=False, negative_prefix="--no-")       # Don't use info about where is defferent identities
    pretrained: bool = flag(default=True, negative_prefix="--no-")          # If using the pretrained weights for training or not
    discr_force: bool = flag(default=False, negative_prefix="--no-")        # If True Discriminator would not train when adversarial loss is high
    scheduler: bool = flag(default=False, negative_prefix="--no-")          # If True decreasing LR is used for learning of generator and discriminator
    scheduler_step: int = 5000
    scheduler_gamma: float = 0.2                                            # It is value, which shows how many times to decrease LR
    eye_detector_loss: bool = flag(default=False, negative_prefix="--no-")  # If True eye loss with using AdaptiveWingLoss detector is applied to generator

    """ Info about this run """
    use_wandb: bool = flag(default=False, negative_prefix="--no-")          # Use wandb to track your experiments or not
    run_name: str = "ghost-v2"                                              # Name of this run. Used to create folders where to save the weights.
    wandb_project: str = "GhostV2"
    wandb_entity: str = "your-login"

    """ Training params you probably don't want to change """
    batch_size: int = 16
    lr_G: float = 4e-4
    lr_D: float = 4e-4
    max_epoch: int = 2000
    show_step: int = 500
    save_epoch: int = 1