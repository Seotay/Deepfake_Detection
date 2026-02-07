import argparse
import os
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
#from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
from src.dataset import sample_frames, get_full_frame_padded, get_iou
from src.utils import set_seed, load_cfg


def extract_faces_directly(model, cfg, data_root, dataset_names, out_root, videos_per_dataset=None, frames_per_video=4):  
    for dataset in dataset_names:
        dataset_dir = data_root / dataset
        if not dataset_dir.exists(): continue

        out_dir = out_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        videos = sorted([p for p in dataset_dir.iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}])
        
        limit = videos_per_dataset if videos_per_dataset else len(videos)
        print(f"\n[INFO] Processing {dataset}")

        for vp in tqdm(videos[:limit]):
            frames = sample_frames(vp, frames_per_video)
            if not frames: continue

            # 4 frmaes
            results = model(frames, imgsz=cfg['yolo_config']['image_size'], 
                            conf=cfg['yolo_config']['conf_thres'], 
                            iou=cfg['yolo_config']['iou_thres'], 
                            verbose=False)
            target_box = None
            
            for i, (frame, res) in enumerate(zip(frames, results)):
                boxes = res.boxes.xyxy.cpu().numpy()
                h, w, _ = frame.shape
                chosen_face = None

                if len(boxes) > 0:
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) # box area
                    
                    if target_box is None:
                        best_idx = np.argmax(areas)
                        target_box = boxes[best_idx]
                        chosen_face = target_box
                    else:
                        ious = [get_iou(target_box, b) for b in boxes]
                        best_iou_idx = np.argmax(ious)
                        
                        if ious[best_iou_idx] > 0.2:
                            target_box = boxes[best_iou_idx]
                            chosen_face = target_box
                        else:
                            best_idx = np.argmax(areas)
                            target_box = boxes[best_idx]
                            chosen_face = target_box
                
                # 2. save image crop or padded full frame
                if chosen_face is not None:
                    x1, y1, x2, y2 = map(int, chosen_face)
                    
                    # margin
                    margin_w = int((x2 - x1) * cfg['yolo_config']['margin_ratio'])
                    margin_h = int((y2 - y1) * cfg['yolo_config']['margin_ratio'])
                    
                    x1_m, y1_m = max(0, x1 - margin_w), max(0, y1 - margin_h)
                    x2_m, y2_m = min(w, x2 + margin_w), min(h, y2 + margin_h)
                    
                    face_img = frame[y1_m:y2_m, x1_m:x2_m]
                    if face_img.size == 0: 
                        out_img = cv2.resize(get_full_frame_padded(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                                                   (cfg['resolution'], cfg['resolution']))
                                                                   )
                    else:
                        out_img = cv2.resize(face_img, (cfg['resolution'], cfg['resolution']))
                else:
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    padded_pil = get_full_frame_padded(pil_frame, (cfg['resolution'], cfg['resolution']))
                    out_img = cv2.cvtColor(np.array(padded_pil), cv2.COLOR_RGB2BGR) if not isinstance(padded_pil, np.ndarray) else padded_pil

                save_path = out_dir / f"{vp.stem}_f{i:03d}.jpg"
                cv2.imwrite(str(save_path), out_img)
        print(f"[DONE] {dataset}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector_path", type=str, default="./config/config.yaml")
    parser.add_argument("--train_data_path", type=str, default="./train_data")
    parser.add_argument("--output_path", type=str, default="./train_data/preprocessing")
    return parser.parse_args()

if __name__ == "__main__":
    set_seed(42)
    args = parse_args()
    cfg = load_cfg(args.detector_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.train_data_path)
    output_path =  Path(args.output_path)
    os.makedirs(output_path, exist_ok=True)
    datasets = ["Deepfakes", "Face2Face", "FaceShifter","FaceSwap", "NeuralTextures", "original"]
    

    # YOLO init
    #yolo_face_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    #model = YOLO(yolo_face_path).to(device)
    yolo_face_model = YOLO('./model/yolo_model.pt')

    extract_faces_directly(model, cfg=cfg, 
                           data_root = data_path, dataset_names=datasets, 
                           out_root=output_path, 
                           videos_per_dataset=None, 
                           frames_per_video=cfg['frame_num']['train'])


