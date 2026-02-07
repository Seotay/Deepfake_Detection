# Deepfake Detection Inference (Effort)


## Purpose
This repository performs **offline inference** for deepfake detection and produces a submission CSV.

**Inference pipeline**
1) Load files from `./test_data` (images/videos)  
2) Detect/crop face using **YOLO** (`./model/yolo_model.pt`)  
   - Fallback: best-candidate crop → full-frame padded fallback  
3) Resize to `224×224` and apply **CLIP normalization** (mean/std from config)  
4) Predict with **EffortDetector** loaded from `./model/model.pt`  
5) Save predictions to `./submission/submission_finetuned.csv`

---

## Model Summary
- **Detector**: `EffortDetector` (single model inference)
- **Backbone**: CLIP ViT-L/14 (`clip-vit-large-patch14`)
- **Face detector**: Ultralytics YOLO
- **Final weights (used at inference)**: `./model/model.pt`
- **YOLO weights**: `./model/yolo_model.pt`

---

## Offline Execution
Inference must run with **no internet**.

### 1) Run container with network disabled
Use `--network none` at runtime.

### 2) CLIP must be loaded locally (no HuggingFace download)
Set in `config/config.yaml`:
```yaml
clip_pretrained_path: ./model/clip-vit-large-patch14
```
The local folder `./model/clip-vit-large-patch14` must contain HuggingFace save_pretrained() outputs such as:

config.json

model.safetensors (or pytorch_model.bin)

preprocessor_config.json

If clip_pretrained_path is not a local directory, inference will fail offline due to Hub download attempts.

---

## How to Run

### Step 1) Build Docker image
Run at project root: `docker build -f env\Dockerfile -t submit_test:latest .`

### Step 2) Prepare host output directory
`mkdir submission -Force`

### Step 3) Run inference (offline, GPU enabled)
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

## Download Dataset & Weights at Google Drive !!

