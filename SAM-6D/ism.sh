#!/bin/bash

export CAD_PATH=$PWD/Data/Example/obj_000005.ply    # path to a given cad model(mm)
export RGB_PATH=$PWD/Data/Example/rgb.png           # path to a given RGB image
export DEPTH_PATH=$PWD/Data/Example/depth.png       # path to a given depth map(mm)
export CAMERA_PATH=$PWD/Data/Example/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=$PWD/Data/Example/outputs         # path to a pre-defined file for saving results

echo "Run ISM..."
# Run instance segmentation model
export SEGMENTOR_MODEL=sam

cd Instance_Segmentation_Model

if [ ! -f checkpoints/segment-anything/sam_vit_h_4b8939.pth ]; then
    python download_sam.py
fi

if [ ! -f checkpoints/FastSAM/FastSAM-x.pt ]; then
    python download_fastsam.py
fi

if [ ! -f checkpoints/dinov2/dinov2_vitl14_pretrain.pth ]; then
    python download_dinov2.py
fi

python run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH
