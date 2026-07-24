# =========================================================
# LeJEPA with ConvNet Encoder (TILED + MULTI-VIEW)
# FINAL VERSION — TRAIN FIRST, VISUALIZE ONCE (3D PCA)
# =========================================================

import os
from pathlib import Path
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import tqdm
import wandb
from torchvision.transforms import v2
from torchvision.ops import MLP

from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =========================================================
# ======================= CONFIG ==========================
# =========================================================
JPEG_ROOT = r"C:\Users\akintano\OneDrive - Michigan State University\Documents\Master thesis work\data_jpeg_clean\alfafa_all_jpg\Pretraining like alfafa"
CHECKPOINT_DIR = r"C:\Users\akintano\OneDrive - Michigan State University\Documents\Master thesis work\Forageonlycheckpoints"



NUM_WORKERS = 0
PIN_MEMORY = False

IMAGE_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 120
V = 2                     # JEPA views per tile

PROJ_DIM = 128
LR = 0.001788039218013555
LR_MIN = 1e-4
WEIGHT_DECAY = 0.005291842753172914
LAMBDA = 0.5

NUM_WORKERS = 0
PIN_MEMORY = False
PCA_EPOCHS = {0, EPOCHS // 2, EPOCHS - 1}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# ======================= SIGREG ==========================
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
        # proj: (V, N, D)
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A / (A.norm(dim=0, keepdim=True) + 1e-12)

        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

# =========================================================
# ======================= DATASET =========================
# =========================================================
def tile_image(img: Image.Image):
    w, h = img.size
    return [
        img.crop((0, 0, w//2, h//2)),
        img.crop((w//2, 0, w, h//2)),
        img.crop((0, h//2, w//2, h)),
        img.crop((w//2, h//2, w, h)),
    ]

class ForageTileDataset(Dataset):
    """
    One dataset item = ONE IMAGE.
    Returns TWO randomly selected tiles.
    No view duplication.
    """
    def __init__(self, root):
        self.paths = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".jpg", ".png"))
        ]

        self.aug = v2.Compose([
            v2.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(0.2, 0.2, 0.2, 0.05),
            v2.RandomApply([v2.GaussianBlur(5, (0.1, 1.0))], p=0.2),
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

        selected = np.random.choice(4, size=2, replace=False)

        t1 = self.aug(tiles[selected[0]])
        t2 = self.aug(tiles[selected[1]])

        return torch.stack([t1, t2])  # (2, 3, H, W)

# =========================================================
# ======================= ENCODER =========================
# =========================================================
class ConvNetEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.proj = MLP(
            512,
            [2048, 2048, proj_dim],
            norm_layer=nn.BatchNorm1d
        )

    def forward(self, x):
        # x: (N, V, 3, H, W)
        N, V = x.shape[:2]
        x = x.flatten(0, 1)
        emb = self.backbone(x).flatten(1)       # (N*V, 512)
        proj = self.proj(emb).reshape(N, V, -1).transpose(0, 1)
        return emb, proj
def run_pca(net, loader, title):
    net.eval()
    feats = []

    with torch.no_grad():
        for vs in loader:
            vs = vs.to(DEVICE)
            emb, _ = net(vs)
            feats.append(emb.cpu().numpy())

    X = np.concatenate(feats, axis=0)
    Z = PCA(n_components=3, random_state=0).fit_transform(X)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=6, alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    plt.tight_layout()
    plt.show()

# =========================================================
# ======================= TRAIN ===========================
# =========================================================
def main():
    
    wandb.init(
        project="LeJEPA-ConvNet-Tiled",
        config=dict(
            epochs=EPOCHS,
            lr=LR,
            proj_dim=PROJ_DIM,
            tile_consistency=True
        )
    )
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
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

    net = ConvNetEncoder(PROJ_DIM).to(DEVICE)
    sigreg = SIGReg().to(DEVICE)

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    warmup = len(loader)
    total = len(loader) * EPOCHS
    sched = SequentialLR(
        opt,
        schedulers=[
            LinearLR(opt, start_factor=0.01, total_iters=warmup),
            CosineAnnealingLR(opt, T_max=total - warmup, eta_min=LR_MIN),
        ],
        milestones=[warmup]
    )

    scaler = GradScaler(enabled=(DEVICE == "cuda"))
    # =========================================================
    # LOSS HISTORY (per epoch)
    # =========================================================
    history = {
        "lejepa": [],
        "inv": [],
        "sigreg": []
    }


    for epoch in range(EPOCHS):
        net.train()

        epoch_lejepa = 0.0
        epoch_inv = 0.0
        epoch_sig = 0.0
        n_batches = 0

        for vs in tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            vs = vs.to(DEVICE)

            with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                emb, proj = net(vs)
                # proj shape: (V, N, D) from original forward
                # but now V = 2 tiles

                tile_a = proj[0]   # (N, D)
                tile_b = proj[1]   # (N, D)

                tile_a = F.normalize(tile_a, dim=1)
                tile_b = F.normalize(tile_b, dim=1)

                inv = (tile_a - tile_b).square().mean()
                sig = sigreg(proj)
                loss = sig * LAMBDA + inv * (1 - LAMBDA)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            # =======================
            # ACCUMULATE LOSSES
            # =======================
            epoch_lejepa += loss.item()
            epoch_inv += inv.item()
            epoch_sig += sig.item()
            n_batches += 1

            wandb.log({
                "train/lejepa": loss.item(),
                "train/inv": inv.item(),
                "train/sigreg": sig.item(),
            })

        # ===========================
        # EPOCH AVERAGES
        # ===========================
        history["lejepa"].append(epoch_lejepa / n_batches)
        history["inv"].append(epoch_inv / n_batches)
        history["sigreg"].append(epoch_sig / n_batches)

        print(
            f"Epoch {epoch+1:03d} | "
            f"LeJEPA: {history['lejepa'][-1]:.4f} | "
            f"INV: {history['inv'][-1]:.4f} | "
            f"SIGREG: {history['sigreg'][-1]:.4f}"
        )
        if epoch in PCA_EPOCHS:
            print(f"📊 Running PCA at epoch {epoch+1}")
            run_pca(
                net,
                loader,
                title=f"3D PCA of Tile Embeddings — Epoch {epoch+1}"
            )



    # =====================================================
    # SAVE MODEL
    # =====================================================
    ckpt_path = Path(CHECKPOINT_DIR) / "lejepa_convnet_encoder_tile_loss.pth"
    torch.save(net.state_dict(), ckpt_path)
    print(f"✅ Encoder saved to {ckpt_path}")

    # =====================================================
    # FINAL EMBEDDING VISUALIZATION (3D PCA)
    # =====================================================
    print("📊 Extracting embeddings for final 3D PCA...")

    net.eval()
    feats = []

    with torch.no_grad():
        for vs in loader:
            vs = vs.to(DEVICE)
            emb, _ = net(vs)     # (N*V, D)

            N = vs.shape[0]
            V = vs.shape[1]
            emb = emb.reshape(N, V, -1).mean(dim=1)  # average tiles

            feats.append(emb.cpu().numpy())

    X = np.concatenate(feats, axis=0)

    Z = PCA(n_components=10, random_state=0).fit_transform(X)

   

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    r = np.linalg.norm(Z, axis=1)

    sc = ax.scatter(
        Z[:, 0], Z[:, 1], Z[:, 2],
        c=r,
        cmap="viridis",
        s=10,
        alpha=0.85,
        depthshade=True
    )

    ax.set_box_aspect([1, 1, 1])
    ax.set_title("3D PCA of LeJEPA ConvNet Tile Embeddings")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    plt.colorbar(sc, shrink=0.6)
    plt.tight_layout()
    plt.show()


    wandb.finish()


if __name__ == "__main__":
    main()
