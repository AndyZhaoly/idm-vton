#!/bin/bash
set -e

if [ ! -f "gradio_demo/app.py" ]; then
    echo "Error: Run this script from the IDM-VTON directory"
    exit 1
fi

echo "=========================================="
echo "IDM-VTON Weight Download (~32GB total)"
echo "=========================================="

# 1. Main model via HuggingFace (~28GB)
echo ""
echo "[1/2] Main model (yisol/IDM-VTON, ~28GB)..."
if [ ! -d "checkpoints/unet" ]; then
    pip install -q huggingface_hub
    huggingface-cli download yisol/IDM-VTON --local-dir checkpoints
else
    echo "      checkpoints/ already exists, skipping."
fi

# 2. Auxiliary ckpt files (~3.4GB)
echo ""
echo "[2/2] Auxiliary ckpt files (~3.4GB)..."
mkdir -p ckpt/densepose ckpt/humanparsing ckpt/image_encoder ckpt/ip_adapter ckpt/openpose/ckpts
pip install -q huggingface_hub

# DensePose
if [ ! -f "ckpt/densepose/model_final_162be9.pkl" ]; then
    echo "  -> densepose..."
    wget -q --show-progress -O ckpt/densepose/model_final_162be9.pkl \
        https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl
fi

# Human Parsing
if [ ! -f "ckpt/humanparsing/parsing_atr.onnx" ]; then
    echo "  -> humanparsing..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('yisol/IDM-VTON-DC', 'ckpt/humanparsing/parsing_atr.onnx', local_dir='.')
hf_hub_download('yisol/IDM-VTON-DC', 'ckpt/humanparsing/parsing_lip.onnx', local_dir='.')
"
fi

# Image Encoder
if [ ! -f "ckpt/image_encoder/model.safetensors" ]; then
    echo "  -> image encoder (~2.4GB)..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('h94/IP-Adapter', 'models/image_encoder/model.safetensors', local_dir='/tmp/ipa_tmp')
import shutil, os
os.makedirs('ckpt/image_encoder', exist_ok=True)
shutil.copy('/tmp/ipa_tmp/models/image_encoder/model.safetensors', 'ckpt/image_encoder/model.safetensors')
"
fi

# IP-Adapter
if [ ! -f "ckpt/ip_adapter/ip-adapter-plus_sdxl_vit-h.bin" ]; then
    echo "  -> IP-Adapter..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('h94/IP-Adapter', 'sdxl_models/ip-adapter-plus_sdxl_vit-h.bin', local_dir='/tmp/ipa_tmp')
import shutil, os
shutil.copy('/tmp/ipa_tmp/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin', 'ckpt/ip_adapter/')
"
fi

# OpenPose
if [ ! -f "ckpt/openpose/ckpts/body_pose_model.pth" ]; then
    echo "  -> OpenPose (~200MB)..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('yisol/IDM-VTON-DC', 'ckpt/openpose/ckpts/body_pose_model.pth', local_dir='.')
"
fi

echo ""
echo "Download complete!"
echo ""
echo "Start the service:"
echo "  conda activate idm && python idm_vton_service.py"
