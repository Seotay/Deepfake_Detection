import numpy as np
import torch
from sklearn import metrics
import random
import yaml
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cfg(detector_path):
    with open(detector_path, "r") as f:
        cfg = yaml.safe_load(f)
        # cfg['backbone_config']['rank'] = 1
        # cfg['rank'] = 1
        # print(f"[Training] 모델 생성 규격 강제 설정: Rank = {cfg['rank']}")
    return cfg


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        return ckpt
    else:
        return ckpt


def inject_state_dict(model, state_dict, device):
    matched_count = 0
    for m_key, m_param in model.named_parameters():
        f_key = m_key.replace('.S_r', '.S_residual').replace('.U_r', '.U_residual').replace('.V_r', '.V_residual')
        
        candidates = [
            m_key, f_key, 
            f"module.{m_key}", f"module.{f_key}", 
            f"model.{m_key}", f"model.{f_key}",
            f"module.model.{m_key}", f"module.model.{f_key}"
        ]
        
        for cand in candidates:
            if cand in state_dict:
                ckpt_data = state_dict[cand]
                
                if m_param.shape != ckpt_data.shape:
                    new_param = torch.nn.Parameter(ckpt_data.clone().to(device))
                    
                    attrs = m_key.split('.')
                    submod = model
                    for attr in attrs[:-1]:
                        submod = getattr(submod, attr)
                    setattr(submod, attrs[-1], new_param)
                else:
                    m_param.data.copy_(ckpt_data)
                
                matched_count += 1
                break
    return model, matched_count

def load_detector(model, weights_path, device):
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict = extract_state_dict(ckpt)
    model, _ = inject_state_dict(model, state_dict, device)
    return model

def get_person_id(video_path: Path, dataset_name: str):
    stem = video_path.stem
    if dataset_name == "original":
        # 000.mp4 → 000
        return stem
    else:
        # 000_123.mp4 → 000
        return stem.split("_")[0]


def calculate_metrics_for_train(label, output):
    if output.size(1) == 2:
        prob = torch.softmax(output, dim=1)[:, 1]
    else:
        prob = output

    # Accuracy
    _, prediction = torch.max(output, 1)
    correct = (prediction == label).sum().item()
    accuracy = correct / prediction.size(0)

    # Average Precision
    y_true = label.cpu().detach().numpy()
    y_pred = prob.cpu().detach().numpy()
    ap = metrics.average_precision_score(y_true, y_pred)

    # AUC and EER
    try:
        fpr, tpr, thresholds = metrics.roc_curve(label.squeeze().cpu().numpy(),
                                                 prob.squeeze().cpu().numpy(),
                                                 pos_label=1)
    except:
        # for the case when we only have one sample
        return None, None, accuracy, ap

    if np.isnan(fpr[0]) or np.isnan(tpr[0]):
        # for the case when all the samples within a batch is fake/real
        auc, eer = None, None
    else:
        auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return auc, eer, accuracy, ap



class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path="best_model.pt"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_val_auc = None
        self.early_stop = False

    def __call__(self, current_val_auc, model):
        if self.best_val_auc is None:
            self.best_val_auc = current_val_auc
            self.save_checkpoint(model)

        elif current_val_auc >= self.best_val_auc + self.delta:
            self.save_checkpoint(model)
            self.best_val_auc = current_val_auc
            self.counter = 0

        else:
            self.counter += 1
            if self.verbose:
                print(f"\tEarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        if self.verbose and self.best_val_auc is not None:
            print(f"\tValidation AUC improved → Saving model: {self.path}")
        torch.save(model.state_dict(), self.path)


    def load_best_model(self, model):
        state_dict = torch.load(self.path, map_location="cpu")
        model.load_state_dict(state_dict)


def compute_metrics(y_true, y_prob, thr = 0.5):
    y_true = np.asarray(y_true).astype(np.int64)
    y_prob = np.asarray(y_prob).astype(np.float64)
    y_pred = (y_prob >= thr).astype(np.int64)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')

    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "auc": float(auc),
            "classification_report": report, "confusion_matrix": cm}