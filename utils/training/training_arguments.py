import os
from dataclasses import dataclass
from typing import Optional

from simple_parsing import choice
from simple_parsing.helpers import flag

from utils.embedding_models_arguments import EmbeddingModelsArguments


def make_real_path(relative_path):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", relative_path)

@dataclass
class TrainingArguments(EmbeddingModelsArguments):
    """ Info about this run """
    run_name: str = "ghost_v2"                                              # Name of this run. Used to create folders where to save the weights.

    """ Dataset params """
    dataset_path: str = make_real_path("./datasets/LAION-Face/laion_face_cropped_256x256") # Path to the dataset. If not LAION dataset is used, param --laion should be set False
    ckpt_path: Optional[str] = None                                         # Path to checkpoint to resume training.
    example_images_path: str = make_real_path("./examples/images/training_insightface_v2") # Path to source1.png..source6.png and target1.png..target6.png training images.
    G_path: Optional[str] = None                                            # Path to pretrained weights for G. Only used if pretrained=True
    D_path: Optional[str] = None                                            # Path to pretrained weights for D. Only used if pretrained=True
    laion: bool = flag(default=True, negative_prefix="--no-")               # When using LAION dataset (or any other dataset with several photos for one identity)

    """ Weights for loss """
    weight_adv: float = 1                                                   # Adversarial Loss weight
    weight_attr: float = 10                                                 # Attributes weight
    weight_id: float = 20                                                   # Identity Loss weight
    weight_rec: float = 10                                                  # Reconstruction Loss weight
    weight_eyes: float = 0                                                  # Eyes Loss weight

    """ Training params you may want to change """
    adaptive_wing_loss_model_path: str = make_real_path("./weights/AdaptiveWingLoss/WFLW_4HG.safetensors")
    face_embeddings: str = choice("facenet", "arcface", "adaface", "cvl_arcface", "cvl_adaface", "cvl_vit", default="cvl_vit")  # Model used for face embeddings
    backbone: str = choice("unet", "linknet", "resnet", default="unet")     # Backbone for attribute encoder
    num_blocks: int = 2                                                     # Numbers of AddBlocks at AddResblock
    align_corners: bool = flag(default=True, negative_prefix="--no-")       # If we should align corners when upsampling
    same_person: float = 0.2                                                # Probability of using same person identity during training
    same_identity: bool = flag(default=True, negative_prefix="--no-")       # Using simswap approach, when source_id = target_id. Only possible with laion=True
    diff_eq_same: bool = flag(default=False, negative_prefix="--no-")       # Don't use info about where is different identities
    pretrained: bool = flag(default=True, negative_prefix="--no-")          # If using the pretrained weights for training or not
    discr_force: bool = flag(default=False, negative_prefix="--no-")        # If True, discriminator would not train when adversarial loss is high
    initial_loss_adv_accumulated: float = 20.0                              # Initial accumulated adverserial loss
    loss_adv_accumulated_threshold: float = 4.0                             # Accumulated adverserial loss threshold
    use_scheduler: bool = flag(default=False, negative_prefix="--no-")      # If True, decreasing LR is used for learning of generator and discriminator
    scheduler_type_G: str = choice("step", "one_cycle", default="step")
    scheduler_step_G: int = 25000                                            # Parameter for StepLR scheduler for G
    scheduler_gamma_G: float = 0.5                                          # Parameter for StepLR scheduler for G, value which shows how many times to decrease LR
    scheduler_total_steps_G: int = -1                                       # Parameter for OneCycleLR scheduler for G, leave to -1 to automatically compute it from max_epoch
    scheduler_last_batches_G: int = -1                                      # Parameter for OneCycleLR scheduler for G, leave to -1 to start from scratch
    scheduler_type_D: str = choice("step", "one_cycle", default="step")
    scheduler_step_D: int = 25000                                            # Parameter for StepLR scheduler for D
    scheduler_gamma_D: float = 0.5                                          # Parameter for StepLR scheduler for D, value which shows how many times to decrease LR
    scheduler_total_steps_D: int = -1                                       # Parameter for OneCycleLR scheduler for D, leave to -1 to automatically compute it from max_epoch
    scheduler_last_batches_D: int = -1                                      # Parameter for OneCycleLR scheduler for D, leave to -1 to start from scratch
    eye_detector_loss: bool = flag(default=False, negative_prefix="--no-")  # If True eye loss with using AdaptiveWingLoss detector is applied to generator

    """ W&B logging """
    use_wandb: bool = flag(default=False, negative_prefix="--no-")          # Use wandb to track your experiments or not
    wandb_project: str = "GhostV2"

    """ Training params you probably don't want to change """
    batch_size: int = 16
    lr_G: float = 4e-4
    lr_D: float = 4e-4
    b1_G: float = 0
    b1_D: float = 0
    b2_G: float = 0.999
    b2_D: float = 0.999
    wd_G: float = 1e-4
    wd_D: float = 1e-4
    max_epoch: int = 20
    show_step: int = 2500
    save_epoch: int = 1
    precision: Optional[str] = choice(
        None,
        "64",
        "32",
        "16",
        "bf16",
        "transformer-engine",
        "transformer-engine-float16",
        "16-true",
        "16-mixed",
        "bf16-true",
        "bf16-mixed",
        "32-true",
        "64-true",
        default=None
    )