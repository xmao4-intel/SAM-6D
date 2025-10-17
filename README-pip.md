# SAM-6D Python venv Setup (pip)

This guide explains how to set up a Python virtual environment for SAM-6D using `requirements.txt` and pip. It is verified in ubuntu22.04 and python3.11.

## 1. Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
souce /opt/intel/oneapi/setvars.sh # for xpu
```

## 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Download Model Weights

- Instance Segmentation:
  ```bash
  cd SAM-6D/Instance_Segmentation_Model
  python download_sam.py
  python download_fastsam.py
  ```
- Pose Estimation:
  ```bash
  cd SAM-6D/Pose_Estimation_Model
  python download_sam6d-pem.py
  ```

## 4. Run Demo (Custom Data)

```bash
export CAD_PATH=Data/Example/obj_000005.ply
export RGB_PATH=Data/Example/rgb.png
export DEPTH_PATH=Data/Example/depth.png
export CAMERA_PATH=Data/Example/camera.json
export OUTPUT_DIR=Data/Example/outputs
sh demo.sh
```

### run the demo with seprate steps:
```bash
sh render.sh
sh ism.sh
sh pem.sh
```

## Notes
- If you encounter issues, check the `requirements.txt` for package versions.
- For advanced usage (training, evaluation), see the main `README.md` and submodule READMEs.
