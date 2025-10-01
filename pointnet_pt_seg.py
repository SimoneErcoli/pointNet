# pointnet_pt_seg.py
import os
import random
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import DBSCAN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.simplefilter(action="ignore", category=pd.errors.DtypeWarning)

# =========================================================
# Utils base
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(force_cpu: bool = False):
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def pc_normalize(pc: np.ndarray) -> np.ndarray:
    c = np.mean(pc, axis=0)
    pc = pc - c
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if m > 0:
        pc = pc / m
    return pc

def has_csv(root_dir: str) -> bool:
    try:
        return any(Path(root_dir).glob("*.csv"))
    except Exception:
        return False

# =========================================================
# Letture CSV robuste (solo colonne utili)
# =========================================================
def read_csv_train(path: str, label_col: str = "classification") -> pd.DataFrame:
    usecols = ["x", "y", "z", label_col]
    dtype = {"x": "float64", "y": "float64", "z": "float64", label_col: "string"}
    return pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=False)

def read_csv_infer(path: str) -> pd.DataFrame:
    usecols = ["x", "y", "z"]
    dtype = {"x": "float64", "y": "float64", "z": "float64"}
    return pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=False)

# =========================================================
# Augmentations (weak/medium/strong)
# =========================================================
def aug_random_rotate_z(pc):
    theta = np.random.uniform(0, 2*np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return pc @ R.T

def aug_random_flip_xy(pc, p=0.5):
    if np.random.rand() < p:
        pc[:, 0] = -pc[:, 0]
    if np.random.rand() < p:
        pc[:, 1] = -pc[:, 1]
    return pc

def aug_random_scale(pc, low=0.8, high=1.25):
    s = np.random.uniform(low, high)
    return pc * s

def aug_random_anisotropic_scale(pc, low=0.9, high=1.1):
    sx, sy, sz = np.random.uniform(low, high, size=3)
    return pc * np.array([sx, sy, sz], dtype=np.float32)

def aug_jitter(pc, sigma=0.01, clip=0.05):
    noise = np.clip(np.random.randn(*pc.shape) * sigma, -clip, clip)
    return pc + noise

def aug_point_dropout(pc, drop_max=0.2):
    n = len(pc)
    drop_ratio = np.random.uniform(0.0, drop_max)
    if drop_ratio <= 1e-6:
        return pc
    m = int(n * drop_ratio)
    drop_idx = np.random.choice(n, m, replace=False)
    keep_idx = np.setdiff1d(np.arange(n), drop_idx)
    if len(keep_idx) == 0:
        return pc
    src_idx = np.random.choice(keep_idx, m, replace=True)
    pc[drop_idx] = pc[src_idx]
    return pc

def aug_height_shift(pc, max_shift=0.1):
    dz = np.random.uniform(-max_shift, max_shift)
    pc[:, 2] += dz
    return pc

def aug_elastic_distortion(pc, granularity=0.8, magnitude=0.15):
    if len(pc) == 0:
        return pc
    mins = pc.min(axis=0)
    maxs = pc.max(axis=0)
    dims = np.maximum(maxs - mins, 1e-6)
    grid_size = np.maximum((dims / granularity).astype(int), 1) + 3
    noise = [np.random.randn(*grid_size) for _ in range(3)]
    def _smooth(x):
        for _ in range(2):
            x = (x[:-1, :, :] + x[1:, :, :]) / 2
            x = (x[:, :-1, :] + x[:, 1:, :]) / 2
            x = (x[:, :, :-1] + x[:, :, 1:]) / 2
        return x
    noise = [_smooth(n) for n in noise]
    coords = (pc - mins) / dims
    coords = np.clip(coords, 0, 0.999)
    def _trilinear_interp(noi):
        gx, gy, gz = noi.shape
        ix = coords[:, 0] * (gx - 1)
        iy = coords[:, 1] * (gy - 1)
        iz = coords[:, 2] * (gz - 1)
        i0x = np.floor(ix).astype(int); i1x = np.minimum(i0x + 1, gx - 1)
        i0y = np.floor(iy).astype(int); i1y = np.minimum(i0y + 1, gy - 1)
        i0z = np.floor(iz).astype(int); i1z = np.minimum(i0z + 1, gz - 1)
        dx, dy, dz = ix - i0x, iy - i0y, iz - i0z
        c000 = noi[i0x, i0y, i0z]; c001 = noi[i0x, i0y, i1z]
        c010 = noi[i0x, i1y, i0z]; c011 = noi[i0x, i1y, i1z]
        c100 = noi[i1x, i0y, i0z]; c101 = noi[i1x, i0y, i1z]
        c110 = noi[i1x, i1y, i0z]; c111 = noi[i1x, i1y, i1z]
        c00 = c000 * (1 - dz) + c001 * dz
        c01 = c010 * (1 - dz) + c011 * dz
        c10 = c100 * (1 - dz) + c101 * dz
        c11 = c110 * (1 - dz) + c111 * dz
        c0 = c00 * (1 - dy) + c01 * dy
        c1 = c10 * (1 - dy) + c11 * dy
        c = c0 * (1 - dx) + c1 * dx
        return c
    disp = np.stack([_trilinear_interp(noise[d]) for d in range(3)], axis=1)
    pc = pc + disp * magnitude
    return pc

def augment_pointcloud(pc, strength="strong"):
    if strength == "weak":
        pc = aug_random_rotate_z(pc)
        pc = aug_jitter(pc, sigma=0.005, clip=0.02)
    elif strength == "medium":
        pc = aug_random_rotate_z(pc)
        pc = aug_random_flip_xy(pc, p=0.5)
        pc = aug_random_scale(pc, 0.9, 1.1)
        pc = aug_jitter(pc, sigma=0.01, clip=0.03)
        pc = aug_point_dropout(pc, drop_max=0.1)
    else:  # strong
        pc = aug_random_rotate_z(pc)
        pc = aug_random_flip_xy(pc, p=0.5)
        pc = aug_random_scale(pc, 0.85, 1.2)
        pc = aug_random_anisotropic_scale(pc, 0.9, 1.1)
        pc = aug_height_shift(pc, max_shift=0.1)
        pc = aug_elastic_distortion(pc, granularity=0.8, magnitude=0.15)
        pc = aug_jitter(pc, sigma=0.01, clip=0.03)
        pc = aug_point_dropout(pc, drop_max=0.2)
    return pc

# =========================================================
# Dataset (baseline & stratified)
# =========================================================
def list_csv_files(root_dir: str):
    files = sorted([str(p) for p in Path(root_dir).glob("*.csv")])
    if not files:
        raise FileNotFoundError(f"Nessun CSV in {root_dir}")
    return files

def sample_points(pts: np.ndarray, labels: np.ndarray, n_points: int):
    if len(pts) >= n_points:
        idx = np.random.choice(len(pts), n_points, replace=False)
    else:
        idx = np.random.choice(len(pts), n_points, replace=True)
    return pts[idx], labels[idx]

def _stratified_indices(labels, n_points, target_ratios, classes=(0,1,2)):
    desired = {c: int(round(target_ratios.get(c, 0) * n_points)) for c in classes}
    delta = n_points - sum(desired.values())
    if delta != 0:
        maxc = max(target_ratios, key=target_ratios.get)
        desired[maxc] += delta
    idx_by_c = {c: np.where(labels == c)[0] for c in classes}
    out_idx = []
    for c in classes:
        need = desired[c]
        pool = idx_by_c.get(c, np.array([], dtype=int))
        if len(pool) == 0:
            continue
        choose = np.random.choice(pool, size=need, replace=(len(pool) < need))
        out_idx.append(choose)
    out_idx = np.concatenate(out_idx) if out_idx else np.arange(len(labels))
    if len(out_idx) < n_points:
        extra = np.random.choice(np.arange(len(labels)), size=(n_points - len(out_idx)), replace=True)
        out_idx = np.concatenate([out_idx, extra])
    elif len(out_idx) > n_points:
        out_idx = np.random.choice(out_idx, size=n_points, replace=False)
    np.random.shuffle(out_idx)
    return out_idx

class PointCloudCSVDataset(Dataset):
    def __init__(self, root_dir, n_points=4096, augment_strength="medium", label_col="classification"):
        self.files = list_csv_files(root_dir)
        self.n_points = n_points
        self.augment_strength = augment_strength
        self.label_col = label_col
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        df = read_csv_train(self.files[idx], label_col=self.label_col)
        pts = df[["x", "y", "z"]].values.astype(np.float32)
        labels = pd.to_numeric(df[self.label_col], errors="coerce").fillna(0).astype(np.int64).values
        pts, labels = sample_points(pts, labels, self.n_points)
        pts = pc_normalize(pts)
        if self.augment_strength:
            pts = augment_pointcloud(pts, strength=self.augment_strength)
        pts = torch.from_numpy(pts.T).float()       # (3,N)
        labels = torch.from_numpy(labels).long()    # (N,)
        return pts, labels

class BalancedPointCloudCSVDataset(Dataset):
    def __init__(self, root_dir, n_points=4096, augment_strength="strong",
                 label_col="classification", target_ratios=None):
        self.files = list_csv_files(root_dir)
        self.n_points = n_points
        self.augment_strength = augment_strength
        self.label_col = label_col
        self.target_ratios = target_ratios or {0:1/3, 1:1/3, 2:1/3}
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        df = read_csv_train(self.files[idx], label_col=self.label_col)
        pts = df[["x", "y", "z"]].values.astype(np.float32)
        labels = pd.to_numeric(df[self.label_col], errors="coerce").fillna(0).astype(np.int64).values
        idxs = _stratified_indices(labels, self.n_points, self.target_ratios, classes=tuple(self.target_ratios.keys()))
        pts, labels = pts[idxs], labels[idxs]
        pts = pc_normalize(pts)
        if self.augment_strength:
            pts = augment_pointcloud(pts, strength=self.augment_strength)
        pts = torch.from_numpy(pts.T).float()       # (3,N)
        labels = torch.from_numpy(labels).long()    # (N,)
        return pts, labels

def collate_points(batch):
    pts = torch.stack([b[0] for b in batch], dim=0)      # (B,3,N)
    labels = torch.stack([b[1] for b in batch], dim=0)   # (B,N)
    return pts, labels

def build_dataloader(root_dir, n_points, batch_size, augment_strength, label_col,
                     balance="stratified", target_ratios=None, shuffle=True):
    if balance == "stratified":
        ds = BalancedPointCloudCSVDataset(root_dir, n_points, augment_strength, label_col, target_ratios)
    else:
        ds = PointCloudCSVDataset(root_dir, n_points, augment_strength if augment_strength!="weak" else "weak", label_col)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                      num_workers=0, collate_fn=collate_points)

# =========================================================
# PointNet (PyTorch)
# =========================================================
class STN3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    def forward(self, x):  # (B,3,N)
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))   # (B,1024,N)
        x = torch.max(x, 2)[0]                # (B,1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)                       # (B,9)
        iden = torch.eye(3, device=x.device).view(1, 9).repeat(B, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class PointNetSeg(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.conv4 = nn.Conv1d(1088, 512, 1)   # 64 (pointfeat) + 1024 (global)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, num_classes, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)
    def forward(self, x):  # x: (B,3,N)
        B, _, N = x.size()
        trans = self.stn(x)
        x = torch.bmm(trans, x)                    # (B,3,N)
        x = F.relu(self.bn1(self.conv1(x)))        # (B, 64, N)
        pointfeat = x                               # usa 64 canali locali
        x = F.relu(self.bn2(self.conv2(x)))        # (B,128, N)
        x = F.relu(self.bn3(self.conv3(x)))        # (B,1024,N)
        global_feat = torch.max(x, 2, keepdim=True)[0]  # (B,1024,1)
        global_feat = global_feat.repeat(1, 1, N)  # (B,1024,N)
        x = torch.cat([pointfeat, global_feat], dim=1)  # (B,1088,N)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.conv7(x)                           # (B,C,N)
        return x

# =========================================================
# Training / Validation
# =========================================================
def train(args):
    set_seed(args.seed)
    device = get_device(force_cpu=args.cpu)
    print("Using device:", device)

    # Dataloaders
    ratios = None
    if args.target_ratios:
        r0, r1, r2 = [float(x) for x in args.target_ratios.split(",")]
        ratios = {0: r0, 1: r1, 2: r2}

    train_loader = build_dataloader(
        args.train_dir, args.n_points, args.batch_size, args.augment,
        args.label_col, balance=args.balance, target_ratios=ratios, shuffle=True
    )

    if len(train_loader) == 0:
        n_files = len(train_loader.dataset)
        raise RuntimeError(
            f"Nessun batch creato: train files={n_files}, batch_size={args.batch_size}. "
            f"Riduci --batch-size (<= {n_files}) oppure usa più CSV; drop_last è False di default."
        )

    val_loader = None
    if args.val_dir and has_csv(args.val_dir):
        val_loader = build_dataloader(
            args.val_dir, args.n_points, args.batch_size, "weak",
            args.label_col, balance=args.balance, target_ratios=ratios, shuffle=False
        )
        if len(val_loader) == 0:
            print("⚠ Nessun batch di validazione: controlla --batch-size e numero di CSV in val.")
    elif args.val_dir:
        print(f"⚠ Nessun CSV trovati in '{args.val_dir}': validazione saltata.")

    # Modello
    num_classes = 3
    model = PointNetSeg(num_classes=num_classes).to(device)

    # BN più stabile con batch piccoli
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.momentum = 0.01

    class_weights = torch.tensor([1.0, 1.0, args.tree_weight], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        # warmup lineare primi 5 epoch
        if epoch <= 5:
            for g in optimizer.param_groups:
                g["lr"] = args.lr * epoch / 5

        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for points, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            points = points.to(device)   # (B,3,N)
            labels = labels.to(device)   # (B,N)

            optimizer.zero_grad()
            logits = model(points)       # (B,C,N)
            loss = criterion(logits, labels)
            loss.backward()
            if device.type == "mps":
                torch.mps.synchronize()
            optimizer.step()

            epoch_loss += loss.item() * points.size(0)
            preds = torch.argmax(logits, dim=1)    # (B,N)
            correct += (preds == labels).sum().item()
            total += labels.numel()

        train_acc = (correct / total) if total > 0 else 0.0
        train_loss = epoch_loss / max(len(train_loader.dataset), 1)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

        # Validazione
        if val_loader:
            model.eval()
            correct = 0
            total = 0
            iou_sum = 0.0
            iou_class_accum = np.zeros(3, dtype=np.float64)
            iou_class_counts = np.zeros(3, dtype=np.int64)

            with torch.no_grad():
                for points, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                    points = points.to(device)
                    labels = labels.to(device)
                    logits = model(points)
                    preds = torch.argmax(logits, dim=1)

                    correct += (preds == labels).sum().item()
                    total += labels.numel()

                    # IoU per classe
                    batch_ious = []
                    for c in [0,1,2]:
                        inter = torch.logical_and(preds == c, labels == c).sum().item()
                        union = torch.logical_or(preds == c, labels == c).sum().item()
                        if union > 0:
                            iou = inter/union
                            batch_ious.append(iou)
                            iou_class_accum[c] += iou
                            iou_class_counts[c] += 1
                    if batch_ious:
                        iou_sum += sum(batch_ious)/len(batch_ious)

            val_acc = (correct / total) if total > 0 else 0.0
            val_iou = iou_sum / max(len(val_loader), 1)
            per_class_iou = [
                (iou_class_accum[c]/iou_class_counts[c]) if iou_class_counts[c] > 0 else float("nan")
                for c in [0,1,2]
            ]
            print(f"Epoch {epoch}: val_acc={val_acc:.4f} val_meanIoU={val_iou:.4f}  IoU[other,ground,tree]={per_class_iou}")

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(model.state_dict(), args.out_model)
                print(f"✓ Saved best model to {args.out_model} (meanIoU={best_val_iou:.4f})")
        else:
            if epoch % args.save_every == 0 or epoch == args.epochs:
                torch.save(model.state_dict(), args.out_model)
                print(f"✓ Saved model to {args.out_model}")

        scheduler.step()

# =========================================================
# Inferenza + TTA + clustering alberi
# =========================================================
def rotate_z_torch(x, theta):
    # x: (B,3,N), theta: tensor shape () or (B,)
    c = torch.cos(theta); s = torch.sin(theta)
    zero = torch.zeros_like(c); one = torch.ones_like(c)
    R = torch.stack([
        torch.stack([c, -s, zero], dim=-1),
        torch.stack([s,  c, zero], dim=-1),
        torch.stack([zero, zero, one], dim=-1)
    ], dim=-2)  # (B,3,3)
    if x.size(0) != R.size(0):
        R = R.expand(x.size(0), 3, 3)
    return torch.bmm(R, x)

def infer(args):
    set_seed(args.seed)
    device = get_device(force_cpu=args.cpu)
    print("Using device:", device)

    num_classes = 3
    model = PointNetSeg(num_classes=num_classes).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("✓ Modello caricato")

    df = read_csv_infer(args.input_csv)
    pts_orig = df[["x", "y", "z"]].values.astype(np.float32)
    N = len(pts_orig)

    pts_norm = pc_normalize(pts_orig.copy())
    preds_all = np.zeros(N, dtype=np.int64)
    CHUNK = args.chunk_points

    # parse TTA rotazioni (gradi)
    thetas_deg = [t for t in str(args.tta_rot).split(",") if t.strip()!=""]
    thetas = [float(t)/180.0*np.pi for t in thetas_deg] if thetas_deg else [0.0]

    with torch.no_grad():
        for start in tqdm(range(0, N, CHUNK), desc="Infer"):
            end = min(start + CHUNK, N)
            chunk = pts_norm[start:end]           # (M,3)
            M = len(chunk)
            inp = torch.from_numpy(chunk.T).unsqueeze(0).to(device).float()  # (1,3,M)

            logits_acc = None
            for t in thetas:
                th = torch.tensor([t], device=device)
                inp_rot = rotate_z_torch(inp, th)
                logits_t = model(inp_rot)         # (1,C,M)
                logits_acc = logits_t if logits_acc is None else (logits_acc + logits_t)
            logits = logits_acc / len(thetas)

            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (M,)
            preds_all[start:end] = pred

    out_df = pd.DataFrame(pts_orig, columns=["x","y","z"])
    out_df["pred_label"] = preds_all  # 0=other, 1=ground, 2=tree

    tree_mask = out_df["pred_label"].values == 2
    tree_points = pts_orig[tree_mask]
    tree_ids = np.full(N, -1, dtype=int)

    if len(tree_points) > 0:
        clustering = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples).fit(tree_points)
        labels = clustering.labels_
        labels = np.where(labels >= 0, labels + 1, 0)  # 1..K, 0=rumore
        tree_ids[tree_mask] = labels

    out_df["tree_instance_id"] = tree_ids
    out_csv = args.output_csv or (Path(args.input_csv).with_suffix("").as_posix() + "_pred.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"✓ Inferenza completata. Output: {out_csv}")
    print("Legenda: pred_label => 0=other, 1=ground, 2=tree; tree_instance_id: 0=rumore, 1..K=alberi")

# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="PointNet (PyTorch) segmentazione per-punto su CSV x,y,z con etichette")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_tr = sub.add_parser("train")
    p_tr.add_argument("--train-dir", type=str, required=True)
    p_tr.add_argument("--val-dir", type=str, default=None)
    p_tr.add_argument("--label-col", type=str, default="classification",
                      help="Colonna etichetta (default: classification)")
    p_tr.add_argument("--n-points", type=int, default=4096)
    p_tr.add_argument("--batch-size", type=int, default=8)
    p_tr.add_argument("--epochs", type=int, default=60)
    p_tr.add_argument("--lr", type=float, default=1e-3)
    p_tr.add_argument("--tree-weight", type=float, default=2.0)
    p_tr.add_argument("--label-smoothing", type=float, default=0.05)
    p_tr.add_argument("--out-model", type=str, default="pointnet_pt_seg.pth")
    p_tr.add_argument("--save-every", type=int, default=10)
    p_tr.add_argument("--augment", type=str, default="strong",
                      choices=["weak","medium","strong"], help="Forza data augmentation")
    p_tr.add_argument("--balance", type=str, default="stratified",
                      choices=["none","stratified"], help="Bilanciamento per-batch")
    p_tr.add_argument("--target-ratios", type=str, default="0.34,0.33,0.33",
                      help="Quote target per classi 0,1,2 (es. '0.3,0.3,0.4')")
    p_tr.add_argument("--cpu", action="store_true", help="Forza CPU (ignora MPS/CUDA)")
    p_tr.add_argument("--seed", type=int, default=42)
    p_tr.set_defaults(func=train)

    # infer
    p_inf = sub.add_parser("infer")
    p_inf.add_argument("--model", type=str, required=True)
    p_inf.add_argument("--input-csv", type=str, required=True)
    p_inf.add_argument("--output-csv", type=str, default=None)
    p_inf.add_argument("--chunk-points", type=int, default=200000, help="punti per chunk d’inferenza")
    p_inf.add_argument("--dbscan-eps", type=float, default=0.6)
    p_inf.add_argument("--dbscan-min-samples", type=int, default=40)
    p_inf.add_argument("--tta-rot", type=str, default="0,90,180,270",
                       help="rotazioni Z (gradi) per TTA, es. '0,90,180,270'; vuoto per disattivare")
    p_inf.add_argument("--cpu", action="store_true", help="Forza CPU (ignora MPS/CUDA)")
    p_inf.add_argument("--seed", type=int, default=42)
    p_inf.set_defaults(func=infer)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
