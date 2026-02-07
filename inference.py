import os, glob, cv2, argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from src.dataset import is_image, is_video,  sample_frames, get_full_frame_padded, crop_faces_batch
from src.utils import set_seed, load_cfg, load_detector
from src.models import EffortDetector
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


class Inference:
    def __init__(self, model, yolo_face_model, cfg, test_folder, baseline_csv, output, device):  
        self.device = device
        self.model = model.to(self.device)
        self.yolo_face_model = yolo_face_model.to(self.device)

        # YOLO configs
        self.IMG_SIZE = cfg['yolo_config']['image_size']
        self.CONF_THRES = cfg['yolo_config']['conf_thres']
        self.IOU_THRES = cfg['yolo_config']['iou_thres']
        self.MIN_FACE_RATIO = cfg['yolo_config']['min_face_ratio']
        self.MARGIN_RATIO = cfg['yolo_config']['margin_ratio']
        self.USE_FP16 = cfg['yolo_config']['use_fp16']
        
        # inference configs
        self.resolution = cfg['resolution']
        self.num_frames = cfg['frame_num']['test']
    
        # outputs
        self.test_folder = test_folder
        self.baseline_csv = baseline_csv
        self.output = output

        # preprocess (CLIP normalize)
        self.clip_mean = cfg['mean']
        self.clip_std  = cfg['std']
        self.preprocess = T.Compose([T.ToTensor(),
                                     T.Normalize(self.clip_mean, self.clip_std)])


    def inferencing(self):
        files = sorted(glob.glob(os.path.join(self.test_folder, "*")))
        if len(files) == 0:
            raise RuntimeError(f"No files found in test_folder: {self.test_folder}")

        pred_dict = {}
        self.model.eval()

        with torch.no_grad():
            for f in tqdm(files, desc="Inference"):
                try:
                    probs = self.run_inference_on_file(f)
                    pred_dict[os.path.basename(f)] = float(np.mean(probs)) if probs else 0.5
                except Exception as e:
                    print(f"[ERROR] {f}: {e}")
                    pred_dict[os.path.basename(f)] = 0.5

        df = pd.read_csv(self.baseline_csv)
        df["prob"] = df["filename"].map(pred_dict).fillna(0.5)

        out_dir = os.path.dirname(self.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        df.to_csv(self.output, index=False)
        print(f"Saved: {self.output}")

    def get_best_face_candidate(self, frame):
        results = self.yolo_face_model.predict(frame,
                                               imgsz=self.IMG_SIZE,
                                               conf=0.01,
                                               device=self.device,
                                               verbose=False)
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            best_idx = torch.argmax(boxes.conf)
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()

            w, h = x2 - x1, y2 - y1
            margin_w, margin_h = w * self.MARGIN_RATIO, h * self.MARGIN_RATIO

            img_h, img_w = frame.shape[:2]
            nx1, ny1 = max(0, int(x1 - margin_w)), max(0, int(y1 - margin_h))
            nx2, ny2 = min(img_w, int(x2 + margin_w)), min(img_h, int(y2 + margin_h))

            return frame[ny1:ny2, nx1:nx2]
        return None

    def infer_faces(self, faces):
        probs = []
        for face in faces:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (self.resolution, self.resolution))
            x = self.preprocess(Image.fromarray(face)).unsqueeze(0).to(self.device)
            out = self.model({"image": x}, inference=True)
            probs.append(float(out["prob"].item()))
        return probs


    def infer_image(self, img_bgr):
        # normal face crop
        faces = crop_faces_batch([img_bgr],
                                 self.yolo_face_model,
                                 self.IMG_SIZE,
                                 self.CONF_THRES,
                                 self.IOU_THRES,
                                 self.device,
                                 self.USE_FP16,
                                 self.MIN_FACE_RATIO,
                                 self.MARGIN_RATIO)
        if faces:
            return self.infer_faces(faces)

        # best candidate
        best_face = self.get_best_face_candidate(img_bgr)
        if best_face is not None:
            return self.infer_faces([best_face])

        # full frame fallback
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ff = get_full_frame_padded(pil, (self.resolution, self.resolution))
        x = self.preprocess(ff).unsqueeze(0).to(self.device)
        out = self.model({"image": x}, inference=True)
        return [float(out["prob"].item())]


    def infer_video(self, video_path):
        frames = sample_frames(video_path, self.num_frames)
        if not frames:
            return []

        # 1) normal face crop
        faces = crop_faces_batch(frames,
                                 self.yolo_face_model,
                                 self.IMG_SIZE,
                                 self.CONF_THRES,
                                 self.IOU_THRES,
                                 self.device,
                                 self.USE_FP16,
                                 self.MIN_FACE_RATIO,
                                 self.MARGIN_RATIO)
        if faces:
            return self.infer_faces(faces)

        # best candidate from mid frame
        mid_frame = frames[len(frames) // 2]
        best_face = self.get_best_face_candidate(mid_frame)
        if best_face is not None:
            return self.infer_faces([best_face])

        # middle frame padded fallback
        pil = Image.fromarray(cv2.cvtColor(mid_frame, cv2.COLOR_BGR2RGB))
        ff = get_full_frame_padded(pil, (self.resolution, self.resolution))
        x = self.preprocess(ff).unsqueeze(0).to(self.device)
        out = self.model({"image": x}, inference=True)
        return [float(out["prob"].item())]

    def run_inference_on_file(self, fpath):
        if is_image(fpath):
            img = cv2.imread(fpath)
            if img is None:
                return []
            return self.infer_image(img)
        elif is_video(fpath):
            return self.infer_video(fpath)
        return []




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector_path", type=str, default="./config/config.yaml")
    parser.add_argument("--weights_path", type=str, default="./model/model.pt")
    parser.add_argument("--test_folder", type=str, default="./test_data")
    parser.add_argument("--baseline_csv", type=str, default="./submission/baseline_submission.csv")
    parser.add_argument("--output", type=str, default="./submission/submission_finetuned.csv")
    return parser.parse_args()


def main():
    set_seed(42)
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # YOLO init
    #yolo_face_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    #yolo_face_model = YOLO(yolo_face_path)
    yolo_face_model = YOLO('./model/yolo_model.pt')

    # detector init + weight inject
    cfg = load_cfg(args.detector_path)
    model = EffortDetector(cfg).to(device)
    model = load_detector(model=model, weights_path=args.weights_path, device=device)
    #device = device if torch.device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference = Inference(model, yolo_face_model, cfg, 
                        args.test_folder, 
                        args.baseline_csv, 
                        args.output, 
                        device)
    inference.inferencing()

if __name__ == "__main__":
    main()
