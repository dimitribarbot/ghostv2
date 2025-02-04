# Ghost V2

The goal of this project is to try to implement [Ghost](https://github.com/ai-forever/ghost) but with other face detection and recognition models than InsightFace RetinaFace and ArcFace to allow a more permissive licence than the InsightFace ones. It includes a full rewrite of the original [Ghost](https://github.com/ai-forever/ghost) repository code, integrating Pytorch Lightning to boost training and using other datasets than VGGFace2.

## Image Swap Results

![](/examples/results/training/source_target_grid.png)

## Reminder of GHOST Ethics

Here you can find the ethic chart written by the original authors of Ghost which still holds today:

"Deepfake stands for a face swapping algorithm where the source and target can be an image or a video. Researchers have investigated sophisticated generative adversarial networks (GAN), autoencoders, and other approaches to establish precise and robust algorithms for face swapping. However, the achieved results are far from perfect in terms of human and visual evaluation. In this study, we propose a new one-shot pipeline for image-to-image and image-to-video face swap solutions - GHOST (Generative High-fidelity One Shot Transfer).

Deep fake synthesis methods have been improved a lot in quality in recent years. The research solutions were wrapped in easy-to-use API, software and different plugins for people with a little technical knowledge. As a result, almost anyone is able to make a deepfake image or video by just doing a short list of simple operations. At the same time, a lot of people with malicious intent are able to use this technology in order to produce harmful content. High distribution of such a content over the web leads to caution, disfavor and other negative feedback to deepfake synthesis or face swap research.

As a group of researchers, we are not trying to denigrate celebrities and statesmen or to demean anyone. We are computer vision researchers, we are engineers, we are activists, we are hobbyists, we are human beings. To this end, we feel that it's time to come out with a standard statement of what this technology is and isn't as far as us researchers are concerned.
* GHOST is not for creating inappropriate content.
* GHOST is not for changing faces without consent or with the intent of hiding its use.
* GHOST is not for any illicit, unethical, or questionable purposes.
* GHOST exists to experiment and discover AI techniques, for social or political commentary, for movies, and for any number of ethical and reasonable uses.

We are very troubled by the fact that GHOST can be used for unethical and disreputable things. However, we support the development of tools and techniques that can be used ethically as well as provide education and experience in AI for anyone who wants to learn it hands-on. Now and further, we take a **zero-tolerance approach** and **total disregard** to anyone using this software for any unethical purposes and will actively discourage any such uses."

## Disclaimer

We understand the unethical potential of GhostV2 and are committed to protecting against such behavior. The repository has been modified to prevent the processing of inappropriate content, including nudity, graphic content, and sensitive content. Collaboration with websites that promote the use of unauthorized software is strictly prohibited. Those who intend to engage in such activities will be subject to repercussions, such as being reported to authorities for violating the law.

## Installation
  
1. Clone this repository
```bash
git clone https://github.com/dimitribarbot/ghostv2.git
cd ghostv2
```

2. Install dependent packages
```bash
pip install -r requirements.txt
```

3. Download weights

To only download the needed models for inference run this script from the root folder of the repository:
```bash
sh download_inference_models.sh
```

To download all the needed models for inference, dataset preprocessing and training, run this script from the root folder of the repository:
```bash
sh download_all_models.sh
```

## Usage

For the moment, face swap only works for single images containing a single face (in case of multiple faces, the first face will be used, sorted by left eye and then right eye coordinates).

Run inference using our GhostV2 pretrained model by specifying the path to a source file containing the face to be swapped into the target at target file path. The output image will be created at output file path:
```bash
python inference.py --source_file_path={PATH_TO_IMAGE} --target_file_path={PATH_TO_IMAGE} --output_file_path={PATH_TO_IMAGE}
```

Note that an [NSFW filter](https://huggingface.co/AdamCodd/vit-base-nsfw-detector) has been added to prevent the creation of malicious content.

### Inference Options

By default, after main model inference, an enhancing step using [GFPGAN v1.4](https://github.com/TencentARC/GFPGAN) model will be performed, followed by a face paste back step. For this last step, we provide multiple options:
- `ghost`: adapted from the GhostV1, this version uses [FaceAlignment](https://github.com/1adrianb/face-alignment) to get facial landmarks in the source and target images in order to paste the output face into the target image. This option is the default.
- `facexlib_with_parser`: the code was largely inspired by [facexlib](https://github.com/xinntao/facexlib). This version uses [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) internally to parse the output face and paste it into the target image.
- `facexlib_without_parser`: the code was largely inspired by [facexlib](https://github.com/xinntao/facexlib). This version only uses code to paste the output face into the target image.
- `insightface`: the code was largely inspired by [insightface](https://github.com/deepinsight/insightface). This version only uses code to paste the output face into the target image.
- `basic`: this option directly uses the output of the main model inference to paste the output face into the target image.
- `none`: no paste back will be done, the returned image will be the swapped face only (256x256), not the face swapped in the target image.

Eventually, after paste back, an extra step may be done when choosing either `ghost` or `facexlib_with_parser` paste back option. We propose to inpaint face edges using the [SDXL inpainting model](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) to improve the output results.

All command line optional parameters can be found in this [argument file](./utils/inference/inference_arguments.py).

### GhostV1 Inference

It is still possible to run inference with the original version of Ghost for comparison. To do that, first run the `download_all_models.sh` script and then run the inference script with the following parameters:
- `--G_path=./weights/GhostV1/G_unet_2blocks.safetensors`
- `--face_embeddings=arcface`
- `--align_mode=insightface_v1`

Note however that the ArcFace model used internally follows the [InsightFace licence](https://github.com/deepinsight/insightface/tree/master?tab=readme-ov-file#license).

### Demo

It is possible to replicate the source/target matrix of the [Image Swap Results](#image-swap-results) section by running the following script:

```bash
python demo.py
```

Internally, it uses the same command line parameters as for inference. Options can be found in this [argument file](./utils/inference/inference_arguments.py).

## Dataset Preprocessing

We provide scripts to prepare the datasets used for training. We mainly use two datasets for our training stage:
- [Laion Face](https://github.com/FacePerceiver/LAION-Face) dataset: this dataset contains 50 million images with faces. For our pretrained model, we only downloaded the first part out of the 32 parts it contains.
- [Lagenda](https://wildchlamydia.github.io/lagenda/) dataset: originally used for age and gender recognition tasks, this dataset is well suited for our face swap task. It can be used to train a model faster than with the Laion-Face dataset.

We experimented a lot with the dataset preprocessing and we come up with the following proposed solution:
- We exclude images that are too small and contain faces that are too small,
- We use [FaceAlignment](https://github.com/1adrianb/face-alignment) and [Live Portrait](https://github.com/KwaiVGI/LivePortrait) landmark models and code to exclude faces which are not fully visible,
- We use [Live Portrait](https://github.com/KwaiVGI/LivePortrait) to generate various versions of the same face with random facial expressions,
- We optionnaly use [GFPGAN v1.2](https://github.com/TencentARC/GFPGAN) to enhance face quality.

Specific arguments for the Laion-Face dataset preprocessing, such as the dataset location, can be found in this [argument file](./utils/preprocessing/preprocess_laion_arguments.py).
Specific arguments for the Lagenda dataset preprocessing, such as the dataset location, can be found in this [argument file](./utils/preprocessing/preprocess_lagenda_arguments.py).

N.B.: For the Laion Face dataset, you may want to download it using the explanations given [here](https://github.com/FacePerceiver/LAION-Face/issues/10#issuecomment-1694407485).

### Face Alignment

We tried several alignment techniques while preprocessing the datasets, and we found that the latest version of the InsightFace alignment code gives the best results. The list of the distinct alignment techniques and other preprocessing parameters can be found in this [argument file](./utils/preprocessing/preprocess_arguments.py).

It is also possible to compare the various alignment modes by running the following command:

For a single image:
```bash
python align.py --source_image={PATH_TO_IMAGE} --aligned_folder={OUTPUT_PATH} --align_mode={ALIGN_OPTION}
```

Or for an entire folder:
```bash
python align.py --source_folder={PATH_TO_IMAGES} --aligned_folder={OUTPUT_PATH} --align_mode={ALIGN_OPTION}
```

All alignment command line parameters can be found in this [argument file](./utils/preprocessing/align_arguments.py).

### Dataset Image Format

You may want to convert your images to a common .jpg or .png format. To do this for a single file or recursively on a large amount of images, you can use the following script:

For a single image:
```bash
python convert.py --source_image={PATH_TO_IMAGE} --output_folder={OUTPUT_PATH} --output_extension={EXTENSION_OPTION}
```

Or for an entire folder:
```bash
python convert.py --source_folder={PATH_TO_IMAGES} --output_folder={OUTPUT_PATH} --output_extension={EXTENSION_OPTION}
```

Where `EXTENSION_OPTION` is either `.png` or `.jpg`.

All conversion command line parameters can be found in this [argument file](./utils/preprocessing/convert_arguments.py).

## Training
  
To train GhostV2, you can run the following script:
```bash
python train.py
```

### Training Options

We provide a lot of different options for training.

Internally, we detect faces using the [Pytorch RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) model. We then compute face embeddings using one of the available face recognition models:
- The original [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) model, used by the initial version of Ghost (beware, the model is available for non-commercial research purposes only),
- [AdaFace](https://github.com/mk-minchul/AdaFace), a concurrent model of ArcFace,
- [CVLFace](https://github.com/mk-minchul/CVLface), by the author of AdaFace, proposing various face recognition models,
- [Facenet Pytorch](https://github.com/timesler/facenet-pytorch), the Pytorch version of David Sandberg's tensorflow facenet.

By default we use ViT AdaFace, which apparently gives the best results, especially in terms of identity preservation.

More information regarding each option can be found in this [argument file](./utils/training/training_arguments.py). If you want to use wandb logging for your experiments, you should login to wandb first  `--wandb login`.

N.B.: The `--example_images_path` must points to a folder containing test images cropped using the alignment method used to generate your training dataset.

### Face Embeddings

It is possible to calculate the distance between embeddings computed using distinct face recognition models or distinct face alignment modes or both. This is useful if you want to know whether you can replace a face recognition model or face alignment algorithm with a given face swap model.

To do this, you can run the following script:

```bash
python embedding_distance.py
```

And play with the `--source_face_embeddings`, `--target_face_embeddings`, `--source_crop_size`, `--target_crop_size`, `--source_align_mode` and `--target_align_mode` parameters.

All command line parameters can be found in this [argument file](./utils/preprocessing/embedding_distance_arguments.py).

### Our Experiments

Our pretrained model was trained on a single RTX 4090 card using FP16 mixed precision and the Laion-Face dataset preprocessed as explained in the [Dataset Preprocessing](#dataset-preprocessing) section above (around 300000 faces, each one with 10 distinct facial expressions, using insightface_v2 as aligment algorithm) and the CVL ViT face embedding model.

It consisted of two phases:
- A first run of 4 epochs (~20 hours) with a batch size of 32 and default parameters set in the training arguments file (no scheduler),
- A second run of 1 epoch (~20 hours as well) with a batch size of 16 (due to the 24GB memory limit of the RTX 4090 card), `--eye_detector_loss` enabled, `--weight_id=70` and `--weight_eyes=1200`, and using the G and D files of the previous run. We also use a scheduler for both the G and D models by setting the `--use_scheduler` flag and the default scheduler parameters of the training arguments file.

### Tips
1. In case of finetuning you can variate losses coefficients to make the output look similar to the source identity, or vice versa, to save features and attributes of target face.
2. You can change the backbone of the attribute encoder and num_blocks of AAD ResBlk using parameters `--backbone` and `--num_blocks`.
3. During the finetuning stage you can use our pretrain weights for generator and discriminator that are located in the `weights` folder. We provide the weights for models with U-Net backbone with 2 blocks in AAD ResBlk.

## Discussion & Improvements

### Known Caveats

The output are not as good as InsightFace and post-processing is needed to achieve the best results.

Currently, we propose 2 optional post-processing steps :
- Face restoration using GFPGAN v1.4 (or v1.2),
- Face edge inpainting using diffusers SDXL inpainting model

Here are comparisons with and without post-processing:

- Without face restoration and without face edge inpainting:

![](/examples/results/training/source_2_full_target_2_full_no_inpaint_no_enhance.png)

- Without face edge inpainting but with face restoration:

![](/examples/results/training/source_2_full_target_2_full_no_inpaint.png)

- Without face restoration but with face edge inpainting:

![](/examples/results/training/source_2_full_target_2_full_no_enhance.png)

- With face restoration and face edge inpainting:

![](/examples/results/training/source_2_full_target_2_full.png)

### Known Improvements

This project can still be improved. Here is a list of known topics:
- Add video face swap as in original Ghost repository.
- Use Pytorch Lightning CLI to train the model using various configurations.
- And of course, improve the face swap result!

## License

The pretrained models and source code of this repository are under the BSD-3 Clause license.

|file|source|license|
|----|------|-------|
| [GhostV2 Discriminator](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/GhostV2_D_unet_2blocks.safetensors) | [dimitribarbot/ghostv2](https://github.com/dimitribarbot/ghostv2) | ![license](https://img.shields.io/badge/license-BSD_3-green.svg) |
| [GhostV2 Generator](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/GhostV2_G_unet_2blocks.safetensors) | [dimitribarbot/ghostv2](https://github.com/dimitribarbot/ghostv2) | ![license](https://img.shields.io/badge/license-BSD_3-green.svg) |

## Thanks

The models and code used in this repository are:

|file|source|license|
|----|------|-------|
| [Ghost](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/GhostV1_G_unet_2blocks.safetensors) (v1) | [ai-forever/ghost](https://github.com/ai-forever/ghost) | ![license](https://img.shields.io/badge/license-Apache_2.0-green.svg) |
| [AdaFace](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/adaface_ir101_webface12m.safetensors) | [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace) | ![license](https://img.shields.io/badge/license-MIT-green.svg) |
| [CVL ArcFace IR101](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/cvlface_arcface_ir101_webface4m.safetensors) | [mk-minchul/CVLface](https://github.com/mk-minchul/CVLface) | ![license](https://img.shields.io/badge/license-MIT-green.svg) |
| [CVL AdaFace IR101](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/cvlface_adaface_ir101_webface12m.safetensors) | [mk-minchul/CVLface](https://github.com/mk-minchul/CVLface) | ![license](https://img.shields.io/badge/license-MIT-green.svg) |
| [CVL AdaFace ViT](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/cvlface_adaface_vit_base_webface4m.safetensors) | [mk-minchul/CVLface](https://github.com/mk-minchul/CVLface) | ![license](https://img.shields.io/badge/license-MIT-green.svg) |
| [CVL AdaFace DFA mobilenet](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/cvlface_DFA_mobilenet.safetensors) | [mk-minchul/CVLface](https://github.com/mk-minchul/CVLface) | ![license](https://img.shields.io/badge/license-MIT-green.svg) |
| [Facenet Pytorch](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/facenet_pytorch.safetensors) | [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch) | ![license](https://img.shields.io/badge/license-MIT-green.svg) |
| [AdaptiveWingLoss](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/WFLW_4HG.safetensors) | [protossw512/AdaptiveWingLoss](https://github.com/protossw512/AdaptiveWingLoss) | ![license](https://img.shields.io/badge/license-Apache_2.0-green.svg) |
| [Pytorch RetinaFace](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/RetinaFace_Resnet50_Final.safetensors) | [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) | ![license](https://img.shields.io/badge/license-MIT-green.svg) |
| [Face Alignment](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/face_alignment_landmark.safetensors) | [1adrianb/face-alignment](https://github.com/1adrianb/face-alignment) | ![license](https://img.shields.io/badge/license-BSD_3-green.svg) |
| [Face Parsing Pytorch](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/bisenet_79999_iter.safetensors) | [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) | ![license](https://img.shields.io/badge/license-MIT-green.svg) |
| [Live Portrait Appearance Feature Extractor](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/live_portrait_appearance_feature_extractor.safetensors) | [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait) | ![license](https://img.shields.io/badge/license-MIT-green.svg) |
| [Live Portrait Landmark](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/live_portrait_landmark.safetensors) | [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait) | ![license](https://img.shields.io/badge/license-MIT-green.svg) |
| [Live Portrait Motion Extractor](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/live_portrait_motion_extractor.safetensors) | [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait) | ![license](https://img.shields.io/badge/license-MIT-green.svg) |
| [Live Portrait Spade Generator](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/live_portrait_spade_generator.safetensors) | [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait) | ![license](https://img.shields.io/badge/license-MIT-green.svg) |
| [Live Portrait Stitching Retargeting Module](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/live_portrait_stitching_retargeting_module.safetensors) | [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait) | ![license](https://img.shields.io/badge/license-MIT-green.svg) |
| [Live Portrait Warping Module](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/live_portrait_warping_module.safetensors) | [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait) | ![license](https://img.shields.io/badge/license-MIT-green.svg) |
| [GFPGANv1.2](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/GFPGANCleanv1-NoCE-C2.safetensors) | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) | ![license](https://img.shields.io/badge/license-Apache_2.0-green.svg) |
| [GFPGANv1.4](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/GFPGANv1.4.safetensors) | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) | ![license](https://img.shields.io/badge/license-Apache_2.0-green.svg) |
| [NSFW Filter](https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/model.safetensors) | [AdamCodd/vit-base-nsfw-detector](https://huggingface.co/AdamCodd/vit-base-nsfw-detector) | ![license](https://img.shields.io/badge/license-Apache_2.0-green.svg) |
| [SDXL Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors) | [diffusers/stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) | ![license](https://img.shields.io/badge/license-Open_RAIL++-green.svg) |
| [InsightFace](https://github.com/deepinsight/insightface/tree/master/python-package) (code) | [deepinsight/insightface](https://github.com/deepinsight/insightface) | ![license](https://img.shields.io/badge/license-MIT-green.svg) |
| [ArcFace](https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/arcface_backbone.safetensors) (optional) | [deepinsight/insightface](https://github.com/deepinsight/insightface) | ![license](https://img.shields.io/badge/license-non_commercial-red.svg) |

The datasets used in this project are:

|dataset|source|license|
|----|------|-------|
| LAION-Face | [FacePerceiver/LAION-Face](https://github.com/FacePerceiver/LAION-Face) | ![license](https://img.shields.io/badge/license-CC_BY_4.0-green.svg) |
| Lagenda | [WildChlamydia/Lagenda](https://wildchlamydia.github.io/lagenda/) | ![license](https://img.shields.io/badge/license-CC_2.0-green.svg) |

Thanks to everyone who makes this project possible!