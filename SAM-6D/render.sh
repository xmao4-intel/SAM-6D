#!/bin/bash

export CAD_PATH=$PWD/Data/Example/obj_000005.ply    # path to a given cad model(mm)
export RGB_PATH=$PWD/Data/Example/rgb.png           # path to a given RGB image
export DEPTH_PATH=$PWD/Data/Example/depth.png       # path to a given depth map(mm)
export CAMERA_PATH=$PWD/Data/Example/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=$PWD/Data/Example/outputs         # path to a pre-defined file for saving results

# Render CAD templates
cd Render
blenderproc run render_custom_templates.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH #--colorize True 
