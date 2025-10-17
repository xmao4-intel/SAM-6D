#!/bin/bash

export CAD_PATH=$PWD/Data/Example/obj_000005.ply    # path to a given cad model(mm)
export RGB_PATH=$PWD/Data/Example/rgb.png           # path to a given RGB image
export DEPTH_PATH=$PWD/Data/Example/depth.png       # path to a given depth map(mm)
export CAMERA_PATH=$PWD/Data/Example/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=$PWD/Data/Example/outputs         # path to a pre-defined file for saving results

echo "Run PEM..."
# Run pose estimation model
export SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json

cd Pose_Estimation_Model

if [ ! -f checkpoints/sam-6d-pem-base.pth ]; then
    python download_sam6d-pem.py
fi

python run_inference_custom.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH # --gpus=cpu

