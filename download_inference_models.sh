#!/usr/bin/env bash

mkdir -p ./weights/CVLFace
cd ./weights/CVLFace
wget https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/cvlface_adaface_vit_base_webface4m.safetensors
wget https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/cvlface_DFA_mobilenet.safetensors
cd ../..

mkdir -p ./weights/GhostV2
cd ./weights/GhostV2
wget https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/GhostV2_G_unet_2blocks.safetensors
cd ../..

mkdir -p ./weights/GFPGAN
cd ./weights/GFPGAN
wget https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/GFPGANv1.4.safetensors
cd ../..

mkdir -p ./weights/RetinaFace
cd ./weights/RetinaFace
wget -O Resnet50_Final.safetensors https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/RetinaFace_Resnet50_Final.safetensors
cd ../..

mkdir -p ./weights/FaceAlignment
cd ./weights/FaceAlignment
wget -O landmark.safetensors https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/face_alignment_landmark.safetensors
cd ../..

mkdir -p ./weights/BiSeNet
cd ./weights/BiSeNet
wget -O 79999_iter.safetensors https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/bisenet_79999_iter.safetensors
cd ../..

python3 download_hf_models.py