import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

def is_image(p: str) -> bool:
    return p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif'))

def is_video(p):
    return p.lower().endswith(('.mp4','.avi','.mov','.mkv','.webm'))

def person_id_from_path(p: str) -> str:
    # 000_003_f000.jpg -> 000, 01_02_xxx_f001.jpg -> 01, 000_f000.jpg -> 000
    return Path(p).stem.split('_')[0]

def scan_image_paths(data_root, include_folders, exclude_folders):
    items = []
    include_set = set(include_folders) if include_folders else None
    exclude_set = set(exclude_folders) if exclude_folders else set()

    for sub in sorted(data_root.iterdir()):
        if not sub.is_dir():
            continue

        name = sub.name
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue

        for p in sorted(sub.glob("*")):
            if p.is_file() and is_image(str(p)):
                items.append((str(p), name))
    return items


def build_person_split(samples, seed, val_ratio):
    persons = sorted({person_id_from_path(p) for p, _ in samples})
    rng = random.Random(seed)
    rng.shuffle(persons)

    n_val = max(1, int(len(persons) * val_ratio))
    val_persons = set(persons[:n_val])
    train_persons = set(persons[n_val:])
    return train_persons, val_persons


def filter_by_persons(samples, allowed_persons: set):
    return [(p, folder) for p, folder in samples if person_id_from_path(p) in allowed_persons]

def uniform_frame_indices(total, k):
    if total <= k:
        return np.arange(total)
    return np.linspace(0, total - 1, k, dtype=int)


def sample_frames(video_path, k):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    frames = []
    for i in uniform_frame_indices(total, k):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames

def get_full_frame_padded(pil_img, size):
    img = pil_img.convert("RGB")
    img.thumbnail(size, Image.BICUBIC)
    canvas = Image.new("RGB", size, (0, 0, 0))
    canvas.paste(
        img,
        ((size[0] - img.size[0]) // 2, (size[1] - img.size[1]) // 2)
    )
    return canvas

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def select_face_from_boxes(boxes, img_shape, min_face_ratio):
    h, w = img_shape[:2]
    valid = []

    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
        score = float(b.conf.cpu().numpy())

        if (x2 - x1) * (y2 - y1) / (w * h) < min_face_ratio:
            continue

        valid.append((x1, y1, x2, y2, score))

    return (
        max(valid, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]) * f[4])
        if valid else None
    )

def crop_with_margin(img, bbox, margin_ratio):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    bw = x2 - x1
    bh = y2 - y1

    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    xx1 = max(0, x1 - mx)
    yy1 = max(0, y1 - my)
    xx2 = min(w, x2 + mx)
    yy2 = min(h, y2 + my)

    if xx2 <= xx1 or yy2 <= yy1:
        return None

    return img[yy1:yy2, xx1:xx2]


def crop_faces_batch(frames, yolo_model, imgsz, conf_thres, iou_thres, device,
                      use_fp16, min_face_ratio, margin_ratio):
    """
    frames: List[np.ndarray] (BGR)
    yolo_model: initialized YOLO model
    """
    results = yolo_model.predict(
        frames,
        imgsz=imgsz,
        conf=conf_thres,
        iou=iou_thres,
        device=device,
        half=bool(use_fp16),
        verbose=False
    )


    faces = []
    for frame, res in zip(frames, results):
        if res.boxes is None:
            continue

        sel = select_face_from_boxes(
            res.boxes,
            frame.shape,
            min_face_ratio
        )
        if sel is None:
            continue

        face = crop_with_margin(frame, sel[:4], margin_ratio)
        if face is not None and face.size > 0:
            faces.append(face)

    return faces



class FaceFolderDataset(Dataset):
    def __init__(self, samples, real_folders: set, transform):
        self.samples = samples
        self.real_folders = real_folders
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, folder = self.samples[idx]
        y = 0 if folder in self.real_folders else 1  # real=0, fake=1
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, torch.tensor(y, dtype=torch.long), path
    

