# =========================================================
# LeJEPA — ResNet18 (Tile-Based)
# PAPER-MATCHED LeJEPA LOSS:
#   LeJEPA = (1 - λ) * L_pred + λ * SIGReg
# where L_pred is the squared ℓ2 prediction loss across views
# (global_views = all_views for ResNet-style encoders).
#
# Saves:
#   - 3 interactive 3D PCA plots (HTML) at epochs: {1, mid, last}
#   - 1 interactive training-loss plot (HTML) with 3 curves:
#       pred_loss, sigreg_loss, lejepa_total
# =========================================================

import os
from pathlib import Path
from PIL import Image
import numpy as np

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
import pandas as pd
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
V = 2  # number of views per sample (keep 2 as in your code)

PROJ_DIM = 128
LR = 0.001788039218013555
LR_MIN = 1e-4
WEIGHT_DECAY = 0.005291842753172914

# LeJEPA uses a single trade-off parameter λ between SIGReg and prediction loss
LAMBDA = 0.5  # paper recommends 0.05 as a robust default in many settings; tune as needed

NUM_WORKERS = 0
PIN_MEMORY = False

PCA_EPOCHS = {0, EPOCHS // 2, EPOCHS - 1}  # save PCA at: epoch1, mid, last

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUTOCAST_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"


# =========================================================
# SIGREG (kept from your template)
# =========================================================
class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization (SIGReg)
    Implementation consistent with your current template.
    """
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
        # proj: (V, B, D) or (B, D) depending on caller; we handle (V,B,D)
        if proj.dim() == 2:
            proj = proj.unsqueeze(0)

        Vv, B, D = proj.shape
        # Random directions
        A = torch.randn(D, 256, device=proj.device)
        A = A / (A.norm(dim=0, keepdim=True) + 1e-12)

        # characteristic function slices
        x_t = (proj @ A).unsqueeze(-1) * self.t  # (V,B,256,knots)

        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * B
        return statistic.mean()


# =========================================================
# TILE HELPERS
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
# DATASET (Tile-Based) — geometric augmentations ONLY
# =========================================================
class ForageTileDataset(Dataset):
    def __init__(self, root):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"JPEG_ROOT not found: {root}")

        self.paths = [
            str(root / f)
            for f in os.listdir(root)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root}")

        # Geometric-only augmentations:
        # - crop/resize (includes zoom-in/out)
        # - flips
        # - rotations
        # - affine (translate + mild scaling)
        self.aug = v2.Compose([
            v2.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0), ratio=(0.9, 1.1)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.1),
            v2.RandomRotation(degrees=25),
            v2.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.85, 1.15),
                shear=None
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

        # Choose two different tiles for the two views (as in your template)
        selected = np.random.choice(4, size=V, replace=False)
        views = [self.aug(tiles[selected[v]]) for v in range(V)]
        return torch.stack(views, dim=0)  # (V, 3, H, W)


# =========================================================
# ENCODER — ResNet18
# =========================================================
class ResNet18Encoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()

        resnet = models.resnet18(weights=None)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = 512

        self.proj = MLP(
            feat_dim,
            [2048, 2048, proj_dim],
            norm_layer=nn.BatchNorm1d
        )

    def forward(self, x):
        """
        x: (B, V, 3, H, W)
        returns:
          emb:  (B, V, feat_dim)
          proj: (B, V, proj_dim)
        """
        B, Vv = x.shape[:2]
        x = x.flatten(0, 1)  # (B*V, 3, H, W)

        z = self.backbone(x)
        z = self.pool(z).flatten(1)  # (B*V, feat_dim)

        p = self.proj(z)  # (B*V, proj_dim)

        z = z.view(B, Vv, -1)
        p = p.view(B, Vv, -1)
        return z, p


# =========================================================
# PAPER-MATCHED LeJEPA PREDICTION LOSS (ℓ2)
# For ResNet: global_views = all_views, so:
#   mu_n = mean_{v in views} z_{n,v}
#   L_pred = mean_{n,v} || mu_n - z_{n,v} ||^2
# =========================================================
def lejepa_prediction_loss(proj: torch.Tensor) -> torch.Tensor:
    """
    proj: (B, V, D)
    Returns scalar.
    """
    mu = proj.mean(dim=1, keepdim=True)           # (B, 1, D)
    dif = mu - proj                               # (B, V, D)
    return (dif.square().mean())                  # mean over B,V,D


# =========================================================
# INTERACTIVE PCA (saved as HTML locally)
# =========================================================
@torch.no_grad()
def save_interactive_pca(net, loader, epoch, out_dir: Path, max_batches=60):
    """
    Saves a 3D PCA plot as an interactive HTML file.
    We sample up to `max_batches` batches to keep runtime manageable.
    PCA is computed on backbone embeddings (not projection head).
    """
    net.eval()
    feats = []

    batches_seen = 0
    for vs in loader:
        vs = vs.to(DEVICE)  # (B, V, 3, H, W)
        emb, _ = net(vs)    # emb: (B, V, 512)
        # Flatten across views so you can see view-wise structure if any:
        emb = emb.reshape(-1, emb.shape[-1])  # (B*V, 512)
        feats.append(emb.detach().cpu().numpy())

        batches_seen += 1
        if batches_seen >= max_batches:
            break

    X = np.concatenate(feats, axis=0)
    Z = PCA(n_components=3, random_state=0).fit_transform(X)

    fig = px.scatter_3d(
        x=Z[:, 0],
        y=Z[:, 1],
        z=Z[:, 2],
        opacity=0.7,
        title=f"ResNet18 Tile Embeddings — Epoch {epoch + 1}",
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pca_epoch_{epoch+1:03d}.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"🧩 Saved interactive PCA: {out_path}")


def save_training_curves(history, out_dir: Path):
    """
    history: dict with keys: 'lejepa_total', 'pred', 'sigreg'
    Saves an interactive HTML plot.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["lejepa_total"]) + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["lejepa_total"], mode="lines", name="LeJEPA total"))
    fig.add_trace(go.Scatter(x=epochs, y=history["pred"], mode="lines", name="Prediction / invariance (L2)"))
    fig.add_trace(go.Scatter(x=epochs, y=history["sigreg"], mode="lines", name="SIGReg"))

    fig.update_layout(
        title="Training Loss Curves",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    out_path = out_dir / "training_losses.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"📉 Saved interactive training loss plot: {out_path}")
import pandas as pd

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
    plots_dir = ckpt_dir / "ResNet18_interactive_plots_2"
    pca_dir = plots_dir / "pca_3d"
    torch.manual_seed(0)

    dataset = ForageTileDataset(JPEG_ROOT)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    net = ResNet18Encoder(PROJ_DIM).to(DEVICE)
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

        ep_total = 0.0
        ep_pred = 0.0
        ep_sig = 0.0
        n_batches = 0

        for vs in tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            vs = vs.to(DEVICE)  # (B, V, 3, H, W)

            with autocast(device_type=AUTOCAST_DEVICE, enabled=(DEVICE == "cuda")):
                _, proj = net(vs)                # (B, V, D)
                pred_loss = lejepa_prediction_loss(proj)

                # SIGReg expects (V,B,D) in your earlier template; we pass that shape
                sig_loss = sigreg(proj.transpose(0, 1))  # (V, B, D)

                # Paper-matched LeJEPA total
                loss = (1.0 - LAMBDA) * pred_loss + LAMBDA * sig_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            ep_total += float(loss.detach().cpu())
            ep_pred += float(pred_loss.detach().cpu())
            ep_sig += float(sig_loss.detach().cpu())
            n_batches += 1

        ep_total /= max(1, n_batches)
        ep_pred /= max(1, n_batches)
        ep_sig /= max(1, n_batches)

        history["lejepa_total"].append(ep_total)
        history["pred"].append(ep_pred)
        history["sigreg"].append(ep_sig)

        print(
            f"Epoch {epoch+1:03d} | "
            f"LeJEPA: {ep_total:.6f} | "
            f"Pred(L2): {ep_pred:.6f} | "
            f"SIGReg: {ep_sig:.6f}"
        )

        # Save PCA at 3 epochs (interactive HTML)
        if epoch in PCA_EPOCHS:
            save_interactive_pca(net, loader, epoch, pca_dir)

    # Save model
    ckpt_path = ckpt_dir / "lejepa_resnet18_tile_L2pred.pth"
    torch.save(net.state_dict(), ckpt_path)
    print(f"🖤 Model saved to {ckpt_path}")

    # Save interactive training curves
    save_training_curves(history, plots_dir)
    save_loss_history(history, plots_dir, model_name="resnet18")


if __name__ == "__main__":
    main()