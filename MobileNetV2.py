# =========================================================
# LeJEPA — MobileNetV2 (Tile-Based)
# PAPER-MATCHED LeJEPA LOSS:
#   LeJEPA = (1 - λ) * L_pred + λ * SIGReg
#
# Geometric augmentations ONLY.
#
# Saves:
#   - 3 interactive PCA HTML plots
#   - 1 interactive training loss HTML plot
# =========================================================

import os
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import tqdm
from torchvision.transforms import v2
from torchvision.ops import MLP
from torchvision import models

from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# CONFIG
# =========================================================
JPEG_ROOT = r"C:\Users\akintano\OneDrive - Michigan State University\Documents\Master thesis work\data_jpeg_clean\alfafa_all_jpg\Pretraining like alfafa"
CHECKPOINT_DIR = r"C:\Users\akintano\OneDrive - Michigan State University\Documents\Master thesis work\Forageonlycheckpoints"

IMAGE_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 120
V = 2

PROJ_DIM = 128
LR = 0.001788039218013555
LR_MIN = 1e-4
WEIGHT_DECAY = 0.005291842753172914
LAMBDA = 0.5

NUM_WORKERS = 0
PIN_MEMORY = False

PCA_EPOCHS = {0, EPOCHS // 2, EPOCHS - 1}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUTOCAST_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"


# =========================================================
# SIGREG
# =========================================================
class SIGReg(nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)

        Vv, B, D = proj.shape
        A = torch.randn(D, 256, device=proj.device)
        A = A / (A.norm(dim=0, keepdim=True) + 1e-12)

        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * B
        return statistic.mean()


# =========================================================
# TILE FUNCTION
# =========================================================
def tile_image(img: Image.Image):
    w, h = img.size
    return [
        img.crop((0, 0, w // 2, h // 2)),
        img.crop((w // 2, 0, w, h // 2)),
        img.crop((0, h // 2, w // 2, h)),
        img.crop((w // 2, h // 2, w, h)),
    ]


# =========================================================
# DATASET (Tile-Based)
# =========================================================
class ForageTileDataset(Dataset):
    def __init__(self, root):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"{root} not found")

        self.paths = [
            str(root / f)
            for f in os.listdir(root)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        self.aug = v2.Compose([
            v2.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0), ratio=(0.9, 1.1)),
            v2.RandomHorizontalFlip(0.5),
            v2.RandomVerticalFlip(0.1),
            v2.RandomRotation(25),
            v2.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.85, 1.15)
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        tiles = tile_image(img)

        selected = np.random.choice(4, size=V, replace=False)
        views = [self.aug(tiles[i]) for i in selected]
        return torch.stack(views, dim=0)


# =========================================================
# ENCODER — MobileNetV2
# =========================================================
class MobileNetV2Encoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()

        mobilenet = models.mobilenet_v2(weights=None)

        # Use feature extractor only
        self.backbone = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        feat_dim = mobilenet.last_channel  # 1280

        self.proj = MLP(
            feat_dim,
            [2048, 2048, proj_dim],
            norm_layer=nn.BatchNorm1d
        )

    def forward(self, x):
        B, Vv = x.shape[:2]
        x = x.flatten(0, 1)

        z = self.backbone(x)
        z = self.pool(z).flatten(1)

        p = self.proj(z)

        z = z.view(B, Vv, -1)
        p = p.view(B, Vv, -1)
        return z, p


# =========================================================
# LeJEPA Prediction Loss
# =========================================================
def lejepa_prediction_loss(proj):
    mu = proj.mean(dim=1, keepdim=True)
    dif = mu - proj
    return dif.square().mean()


# =========================================================
# INTERACTIVE PCA
# =========================================================
@torch.no_grad()
def save_interactive_pca(net, loader, epoch, out_dir, max_batches=60):
    net.eval()
    feats = []
    count = 0

    for vs in loader:
        vs = vs.to(DEVICE)
        emb, _ = net(vs)
        emb = emb.reshape(-1, emb.shape[-1])
        feats.append(emb.cpu().numpy())

        count += 1
        if count >= max_batches:
            break

    X = np.concatenate(feats, axis=0)
    Z = PCA(n_components=3, random_state=0).fit_transform(X)

    fig = px.scatter_3d(
        x=Z[:, 0],
        y=Z[:, 1],
        z=Z[:, 2],
        opacity=0.7,
        title=f"MobileNetV2 Embeddings — Epoch {epoch+1}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_dir / f"pca_epoch_{epoch+1:03d}.html"))
    print("Saved PCA plot.")


def save_training_curves(history, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["lejepa_total"]) + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["lejepa_total"], name="LeJEPA total"))
    fig.add_trace(go.Scatter(x=epochs, y=history["pred"], name="Prediction (L2)"))
    fig.add_trace(go.Scatter(x=epochs, y=history["sigreg"], name="SIGReg"))

    fig.update_layout(
        title="Training Loss Curves — MobileNetV2",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified"
    )

    fig.write_html(str(out_dir / "training_losses.html"))
    print("Saved training curve plot.")


def save_loss_history(history, out_dir: Path, model_name: str):
    """
    Saves epoch-wise losses to CSV for later comparison.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "epoch": np.arange(1, len(history["lejepa_total"]) + 1),
        "prediction_loss": history["pred"],
        "sigreg_loss": history["sigreg"],
        "lejepa_total": history["lejepa_total"]
    })

    out_path = out_dir / f"loss_history_{model_name}.csv"
    df.to_csv(out_path, index=False)

    print(f"📁 Saved loss history to: {out_path}")

# =========================================================
# TRAIN
# =========================================================
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_dir = Path(CHECKPOINT_DIR)
    plot_dir = ckpt_dir / "Mobilnetv2_interactive_plots_2"
    pca_dir = plot_dir / "pca_3d"

    dataset = ForageTileDataset(JPEG_ROOT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    net = MobileNetV2Encoder(PROJ_DIM).to(DEVICE)
    sigreg = SIGReg().to(DEVICE)

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    warmup = len(loader)
    total = len(loader) * EPOCHS
    sched = SequentialLR(
        opt,
        schedulers=[
            LinearLR(opt, start_factor=0.01, total_iters=warmup),
            CosineAnnealingLR(opt, T_max=max(1, total - warmup), eta_min=LR_MIN),
        ],
        milestones=[warmup]
    )

    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    history = {"lejepa_total": [], "pred": [], "sigreg": []}

    for epoch in range(EPOCHS):
        net.train()

        ep_total = ep_pred = ep_sig = 0.0
        n_batches = 0

        for vs in tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            vs = vs.to(DEVICE)

            with autocast(device_type=AUTOCAST_DEVICE, enabled=(DEVICE == "cuda")):
                _, proj = net(vs)
                pred_loss = lejepa_prediction_loss(proj)
                sig_loss = sigreg(proj.transpose(0, 1))
                loss = (1 - LAMBDA) * pred_loss + LAMBDA * sig_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            ep_total += loss.item()
            ep_pred += pred_loss.item()
            ep_sig += sig_loss.item()
            n_batches += 1

        ep_total /= n_batches
        ep_pred /= n_batches
        ep_sig /= n_batches

        history["lejepa_total"].append(ep_total)
        history["pred"].append(ep_pred)
        history["sigreg"].append(ep_sig)

        print(f"Epoch {epoch+1:03d} | LeJEPA {ep_total:.6f}")

        if epoch in PCA_EPOCHS:
            save_interactive_pca(net, loader, epoch, pca_dir)

    torch.save(net.state_dict(),
               ckpt_dir / "lejepa_mobilenetv2_tile_L2pred.pth")

    save_training_curves(history, plot_dir)
    save_loss_history(history, plot_dir, model_name="resnet18")


if __name__ == "__main__":
    main()