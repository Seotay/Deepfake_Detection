# Deepfake Detection

## Purpose
This repository performs **inference** for deepfake detection and produces a submission output CSV file.

## Model Summary
- **Detector**: `EffortDetector` (single model inference)
- **Backbone**: CLIP ViT-L/14 (`clip-vit-large-patch14`)
- **Face detector**: Ultralytics YOLO
- **Final weights (used at inference)**: `./model/model.pt`
- **Pre-trained weights (used at training & evaluation)**: `./model/effort_clip_L14_trainOn_FaceForensic.pth`
- **YOLO weights**: `./model/yolo_model.pt`

---


**Inference pipeline**
  1) Load files from `./test_data` (images/videos)  
  2) Detect/crop face using **YOLO** (`./model/yolo_model.pt`)
    - Fallback: best-candidate crop → full-frame padded fallback
  3) Resize to `224×224` and apply **CLIP normalization** (mean/std from config)  
  4) Predict with **EffortDetector** loaded from `./model/model.pt`
  5) Save results to `./submission/submission_finetuned.csv`


---



## How to Inference

### Step 1. Build Docker image
Run at project root: `docker build -f env\Dockerfile -t submit_test:latest .`

### Step 2. Ensure finetuned Effort model weights and test data are prepared
Before running inference, make sure that the following files and directories exist:

  - Finetuned Effort model weights:
    - `./model/model.pt`
  - YOLO face detector weights:
    - `./model/yolo_model.pt`
  - CLIP pretrained backbone (local directory):
    - `./model/clip-vit-large-patch14/`
  - Test data directory:
    - `./test_data/` (images and/or videos)

### Step 3. Run inference (offline, GPU enabled)
```powershell
docker run --rm --network none --gpus all -it `
  -v "${PWD}\model:/workspace/model" `
  -v "${PWD}\config:/workspace/config" `
  -v "${PWD}\test_data:/workspace/test_data" `
  -v "${PWD}\submission:/workspace/submission" `
  submit_test:latest python /workspace/inference.py
```

## Default Inference Arguments
inference.py defaults:

--detector_path: `./config/config.yaml`

--weights_path: `./model/model.pt`

--test_folder: `./test_data`

--baseline_csv: `./submission/baseline_submission.csv`

--output: `./submission/submission_finetuned.csv`

## Output
The output file is written inside the container to: `./submission/submission_finetuned.csv`

Because `submission/` is mounted, it will be created/updated on the host at: `submission\submission_finetuned.csv`

---


## Download Dataset & Weights (Google Drive)
Download the required datasets and pretrained weights from the Google Drive link below:

https://drive.google.com/file/d/1glFBFe3RL0ATxY4ve558ai8ZfWDxLxN2/view

After downloading, extract and place the highlighted folders and files in the following directory structure:


```
  project/
      ├── config/
      │   └── config.yaml  # Training & Inference configuration
      │
      ├── model/
      │   ├── effort_clip_L14_trainOn_FaceForensic.pth  # pre-trained-EffortDetector
      │   ├── model.pt                  # finetuned-EffortDetector 
      │   ├── yolo_model.pt             # YOLO face detector weights
      │   └── clip-vit-large-patch14/   # CLIP pretrained backbone
      │       ├── config.json
      │       ├── model.safetensors
      │       └── preprocessor_config.json
      │       └── ...
      │
      ├── env/
      │   ├── Dockerfile              
      │   ├── requirements.txt
      │   └── environment.yml
      ├── src/
      │   ├── models.py        # EffortDetector definition
      │   ├── dataset.py             
      │   └── utils.py               
      │
      ├── train_data/             
      │   ├── csv/
      │   ├── Deepfakes/
      │   ├── Face2Face/
      │   ├── FaceShifter/
      │   ├── FaceSwap/
      │   ├── NeuralTextures/
      │   ├── original/
      │   └── preprocessing/  # Preprocessed data(folders/files) for training
      │
      ├── test_data/          # Input data for inference
      │   ├── TEST_000.mp4
      │   ├── TEST_001.jpg
      │   ├── ...
      │
      ├── submission/           # Output directory (generated)
      │   └── baseline_submission.csv
      │   └── submission_finetuned.csv
      │
      ├── train_eval.py         # Training / evaluation script
      ├── inference.py          # Main inference entry point
      └── README.md                     
