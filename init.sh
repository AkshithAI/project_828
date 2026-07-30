# Uninstall older versions
pip uninstall -y torch torchvision torchaudio flash-attn

# install torch 2.8 or above (CUDA 128)
pip install --pre "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/cu128

# install requirements
pip install -r requirements.txt

pip install triton>=3.0.0

# Build flash attention wheel from source
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.2/flash_attn-2.7.4+cu128torch2.8-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl