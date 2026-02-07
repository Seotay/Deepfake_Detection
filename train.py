import os
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from torch import amp
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.utils import set_seed, load_cfg, load_detector, EarlyStopping, compute_metrics
from src.dataset import scan_image_paths, build_person_split, filter_by_persons, FaceFolderDataset
from torchvision import transforms as T
from src.models import EffortDetector


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg, optimizer, scaler, early_stopping, device):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.optimizer = optimizer
        self.scaler = scaler
        self.early_stopping = early_stopping
    
    def forward_get_loss_and_prob(self, x, y):
        data_dict = {'image': x, 'label': y.long()}
        pred_dict = self.model(data_dict)

        # CrossEntropy + Orthogonal + Weight Loss
        loss_dict = self.model.get_losses(data_dict, pred_dict)
        loss = loss_dict['overall']

        # probability of positive class
        prob = pred_dict['prob']
        return loss, prob.detach()
    
    def training(self):
        for epoch in range(1, self.cfg['nEpochs'] + 1):
            train_loss, train_metrics = self.train_one_epoch()
            val_loss, val_metrics =  self.evaluate(data_loader=self.val_loader, loader_name='Validation Evaluating...')

            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, " f"Acc: {train_metrics['accuracy']:.4f}, " f"Prec: {train_metrics['precision']:.4f}, " f"Rec: {train_metrics['recall']:.4f}, "f"F1: {train_metrics['f1']:.4f}, "f"AUC: {train_metrics['auc']:.4f}")
            print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}, " f"Acc: {val_metrics['accuracy']:.4f}, "f"Prec: {val_metrics['precision']:.4f}, " f"Rec: {val_metrics['recall']:.4f}, "f"F1: {val_metrics['f1']:.4f}, "f"AUC: {val_metrics['auc']:.4f}")
            
            self.early_stopping(val_metrics['auc'], self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered...")
                break

        # load best model
        self.early_stopping.load_best_model(self.model)
        print("Loaded best model.")

        _, best_val_metrics = self.evaluate(data_loader=self.val_loader, loader_name = "Validation Evaluating...")
        print(f"[Best Val] Acc: {best_val_metrics['accuracy']:.4f}, "f"Prec: {best_val_metrics['precision']:.4f}, "f"Rec: {best_val_metrics['recall']:.4f}, "f"F1: {best_val_metrics['f1']:.4f}, "f"AUC: {best_val_metrics['auc']:.4f}")
        print("[Best Model Validation] Classification Report:\n", best_val_metrics["classification_report"])
        print("[Best Model Validation] Confusion Matrix:\n", best_val_metrics["confusion_matrix"])
        print('\n')

    @torch.no_grad()
    def evaluate(self, data_loader, loader_name='Evaluating...'):
        self.model.eval()
        total_loss = []
        all_labels = []
        all_probs = []

        for image, label, _ in tqdm(data_loader, desc=loader_name, leave=False):
            
            x = image.to(self.device)
            label = label.to(self.device).long()

            loss, prob = self.forward_get_loss_and_prob(x, label)
            total_loss.append(float(loss.item()))

            all_probs.append(prob.detach().cpu().numpy().reshape(-1))
            all_labels.append(label.detach().cpu().numpy().reshape(-1))

        y_prob = np.concatenate(all_probs) if all_probs else np.array([])
        y_true = np.concatenate(all_labels) if all_labels else np.array([])

        metrics = compute_metrics(y_true, y_prob, thr=0.5)
        val_loss = float(np.mean(total_loss)) if total_loss else 0.0
        return val_loss, metrics

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        all_probs, all_labels = [], []

        for image, label, _ in tqdm(self.train_loader, desc="Training", leave=False):
            x = image.to(self.device)
            label = label.to(self.device).long()

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                use_amp = (self.device.type == "cuda") and self.cfg["use_fp16"]
                with amp.autocast("cuda", enabled=use_amp):
                    loss, prob = self.forward_get_loss_and_prob(x, label)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, prob = self.forward_get_loss_and_prob(x, label)
                loss.backward()
                self.optimizer.step()

            total_loss += float(loss.item())
            all_probs.append(prob.detach().cpu().numpy())
            all_labels.append(label.detach().cpu().numpy())

        y_prob = np.concatenate(all_probs) if all_probs else np.array([])
        y_true = np.concatenate(all_labels) if all_labels else np.array([])
        train_metrics = compute_metrics(y_true, y_prob)
        return total_loss / len(self.train_loader), train_metrics

# model/config
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector_path', type=str, default='./config/config.yaml')
    parser.add_argument('--weights_path', type=str, default='./model/effort_clip_L14_trainOn_FaceForensic.pth',
                        help='download pre-trained Effort-AIGI-Detection Model at https://github.com/YZY-stack/Effort-AIGI-Detection')
    parser.add_argument('--save_path', type=str, default='./model/model.pt')
    # data
    parser.add_argument('--data_root', type=str, default='./train_data/preprocessing',
                        help='root that contains subfolders (Deepfakes, Face2Face, FaceSwap, NeuralTextures, original, ...)')
    parser.add_argument('--real_folders', nargs='+', default=['original'],
                        help='subfolder names treated as real (label=0). Others -> fake (label=1)')
    parser.add_argument('--include_folders', nargs='+', default=[],
                        help='if set, only these subfolders will be used')
    parser.add_argument('--exclude_folders', nargs='+', default=[],
                        help='subfolders to exclude')
    return parser.parse_args()


if __name__ == "__main__":
    seed = 42
    args = parse_args()
    cfg = load_cfg(args.detector_path)
    set_seed(seed=seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_root)
    real_folders = set(args.real_folders)


    if not data_root.exists():
        raise RuntimeError(f"data_root not found: {data_root}")

    samples = scan_image_paths(data_root=data_root, include_folders=args.include_folders, exclude_folders=args.exclude_folders)
    if len(samples) == 0:
        raise RuntimeError("No images found. Check --data_root / folder structure.")

    
    # person-level split
    train_persons, val_persons = build_person_split(samples, seed=seed, val_ratio=cfg['val_ratio'])
    train_samples = filter_by_persons(samples, train_persons)
    val_samples = filter_by_persons(samples, val_persons)

    print(f"[Split] total_imgs={len(samples)} | train_imgs={len(train_samples)} | val_imgs={len(val_samples)}")
    print(f"[Split] train_persons={len(train_persons)} | val_persons={len(val_persons)}")
    assert train_persons.isdisjoint(val_persons), "Person leakage: train/val persons overlap!"

    # datasets / loaders
    train_tf = T.Compose([T.Resize((cfg['resolution'], cfg['resolution'])),
                          T.RandomHorizontalFlip(p=0.5),
                          T.ToTensor(),
                          T.Normalize(cfg['mean'], cfg['std'])])

    val_tf = T.Compose([T.Resize((cfg['resolution'], cfg['resolution'])),
                        T.ToTensor(),
                        T.Normalize(cfg['mean'], cfg['std'])])

    train_ds = FaceFolderDataset(train_samples, real_folders=real_folders, transform=train_tf)
    val_ds = FaceFolderDataset(val_samples, real_folders=real_folders, transform=val_tf)

    # optional: balance classes via sampler
    labels = np.array([0 if folder in real_folders else 1 for _, folder in train_samples], dtype=np.int64)
    if len(labels) > 0:
        class_count = np.bincount(labels, minlength=2)
        class_count = np.maximum(class_count, 1)
        class_weight = 1.0 / class_count
        sample_weights = class_weight[labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(train_ds,batch_size=cfg['train_batchSize'], shuffle=shuffle if sampler is None else False, 
                              sampler=sampler, num_workers=cfg['workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds,batch_size=cfg['val_batchSize'], shuffle=False,
                             num_workers=cfg['workers'], pin_memory=True, drop_last=False)

    # Training setup
    model = EffortDetector(cfg).to(device)
    model = load_detector(model=model, weights_path=args.weights_path, device=device)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=cfg['optimizer']['adam']['lr'], 
                                  weight_decay=cfg['optimizer']['adam']['weight_decay'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = amp.GradScaler("cuda") if (cfg["use_fp16"] and device.type == "cuda") else None
    early_stopping = EarlyStopping(patience=cfg['patience'], verbose=True, delta=cfg['delta'], path=args.save_path)

    # Trainer init
    trainer = Trainer(model=model, train_loader=train_loader, val_loader = val_loader,
                      optimizer=optimizer,scaler=scaler, cfg=cfg,
                      early_stopping=early_stopping, device=device)
    trainer.training() # Start fine-tuning