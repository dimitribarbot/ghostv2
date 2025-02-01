#!/usr/bin/env bash

mkdir -p ./weights/ArcFace
cd ./weights/ArcFace
wget -O backbone.safetensors https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/arcface_backbone.safetensors
cd ../..

mkdir -p ./weights/AdaFace
cd ./weights/AdaFace
wget https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/adaface_ir101_webface12m.safetensors
cd ../..

mkdir -p ./weights/CVLFace
cd ./weights/CVLFace
wget https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/cvlface_arcface_ir101_webface4m.safetensors
wget https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/cvlface_adaface_ir101_webface12m.safetensors
wget https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/cvlface_adaface_vit_base_webface4m.safetensors
wget https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/cvlface_DFA_mobilenet.safetensors
cd ../..

mkdir -p ./weights/Facenet
cd ./weights/Facenet
wget https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/facenet_pytorch.safetensors
cd ../..

mkdir -p ./weights/AdaptiveWingLoss
cd ./weights/AdaptiveWingLoss
wget https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/WFLW_4HG.safetensors
cd ../..

mkdir -p ./weights/GhostV1
cd ./weights/GhostV1
wget https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/GhostV1_G_unet_2blocks.safetensors
cd ../..

mkdir -p ./weights/GhostV2
cd ./weights/GhostV2
wget https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/GhostV2_D_unet_2blocks.safetensors
wget https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/GhostV2_G_unet_2blocks.safetensors
cd ../..

mkdir -p ./weights/GFPGAN
cd ./weights/GFPGAN
wget https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/GFPGANCleanv1-NoCE-C2.safetensors
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

mkdir -p ./weights/LivePortrait
cd ./weights/LivePortrait
wget -O appearance_feature_extractor.safetensors https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/live_portrait_appearance_feature_extractor.safetensors
wget -O landmark.safetensors https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/live_portrait_landmark.safetensors
wget -O motion_extractor.safetensors https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/live_portrait_motion_extractor.safetensors
wget -O spade_generator.safetensors https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/live_portrait_spade_generator.safetensors
wget -O stitching_retargeting_module.safetensors https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/live_portrait_stitching_retargeting_module.safetensors
wget -O warping_module.safetensors https://github.com/dimitribarbot/ghostv2/releases/download/v1.0.0/live_portrait_warping_module.safetensors
cd ../..