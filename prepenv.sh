#!/bin/bash

pip install -r requirements.txt

# Download the weights for the Florence model
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done
mv weights/icon_caption weights/icon_caption_florence

