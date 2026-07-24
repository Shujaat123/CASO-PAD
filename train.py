import argparse
import json
import os
import time
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import (
    mobilenet_v3_large, mobilenet_v3_small,
    MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights,
)
from tqdm import tqdm

from build_data import build_train_val_datasets, build_test_dataset


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class AdaptiveCenterCropAndResize:
    def __init__(self, output_size):
        """
        Args:
            output_size (tuple or int): The desired output size after resizing (e.g., (32, 32)).
        """
        self.output_size = output_size
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        # Convert tensor to PIL image if necessary
        if isinstance(img, torch.Tensor):
            img = self.to_pil(img)

        # Get image size (width, height)
        width, height = img.size

        # Find the minimum dimension to create the largest possible square
        crop_size = min(width, height)

        # Calculate the coordinates to center-crop the square
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = (width + crop_size) // 2
        bottom = (height + crop_size) // 2

        # Crop the image to the largest square
        img = img.crop((left, top, right, bottom))

        # Resize the cropped square to the desired output size
        img = img.resize(self.output_size, Image.Resampling.LANCZOS)

        # Convert the resized image back to a tensor
        img = self.to_tensor(img)

        return img


def get_transforms():
    tf = transforms.Compose([
        AdaptiveCenterCropAndResize((256, 256)),
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return tf, tf   # same for train and eval (augmentation handled inside dataset)


# ---------------------------------------------------------------------------
# Model: Involution + MobileNetV3
# ---------------------------------------------------------------------------


class Involution(nn.Module):
    def __init__(self, channels, kernel_size=7, stride=1, reduction=4,
                 kernel_norm: str = "l2", softmax_temp: float = 1.0,
                 groups: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction = max(1, reduction)
        self.kernel_norm = kernel_norm
        self.softmax_temp = softmax_temp

        # groups: number of channel groups for dynamic kernels
        self.groups = channels if (groups is None) else int(groups)
        assert self.groups >= 1 and channels % self.groups == 0, \
            f"'groups' must divide channels: got C={channels}, groups={self.groups}"

        hidden = max(1, channels // self.reduction)

        # C -> hidden -> (k^2 * groups)
        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.bn     = nn.BatchNorm2d(hidden)
        self.act    = nn.ReLU(inplace=True)
        self.kproj  = nn.Conv2d(hidden, (kernel_size * kernel_size) * self.groups,
                                kernel_size=1, bias=True)

        self.pool_for_k = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()

    def _normalize_kernel(self, ker: torch.Tensor) -> torch.Tensor:
        if self.kernel_norm == "softmax":
            B, G, k, _, H, W = ker.shape
            ker = F.softmax(ker.view(B, G, k*k, H, W) / self.softmax_temp, dim=2)
            return ker.view(B, G, k, k, H, W)
        if self.kernel_norm == "l2":
            ker = ker - ker.mean(dim=(2, 3), keepdim=True)
            denom = ker.norm(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            ker = ker / denom
        return ker

    @torch.no_grad()
    def get_kernels(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.kernel_size
        xk = self.pool_for_k(x)
        K  = self.kproj(self.act(self.bn(self.reduce(xk))))            # [B, k^2*G, H', W']
        if K.shape[-2:] != (H, W):
            K = F.interpolate(K, size=(H, W), mode="bilinear", align_corners=True)
        K = K.view(B, self.groups, k, k, H, W)                         # [B,G,k,k,H,W]
        return self._normalize_kernel(K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k, G = self.kernel_size, self.groups
        assert C % G == 0
        groupC = C // G

        K = self.get_kernels(x)                                        # [B,G,k,k,H,W]
        x_unfold = F.unfold(x, kernel_size=k, padding=k//2)            # [B,C*k*k,H*W]
        x_unfold = x_unfold.view(B, C, k, k, H, W).view(B, G, groupC, k, k, H, W)
        out = (x_unfold * K.unsqueeze(2)).sum(dim=(3,4)).view(B, C, H, W)
        return out


class InvHead(nn.Module):
    def __init__(self, channels, reduce=4, k=9, inv_reduction=4,
                 kernel_norm="l2", softmax_temp=1.0, inv_groups: Optional[int] = None):
        super().__init__()
        hidden = max(8, channels // reduce)
        self.hidden = hidden

        self.reduce = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1    = nn.BatchNorm2d(hidden)
        self.act    = nn.ReLU(inplace=True)

        # 'inv_groups' applies over HIDDEN channels
        if inv_groups is None:
            inv_groups = hidden  # depthwise by default
        else:
            inv_groups = int(inv_groups)
        assert hidden % inv_groups == 0, \
            f"'inv_groups' must divide hidden={hidden}; got {inv_groups}"

        self.inv = Involution(
            channels=hidden, kernel_size=k, stride=1,
            reduction=inv_reduction, kernel_norm=kernel_norm,
            softmax_temp=softmax_temp, groups=inv_groups
        )

        self.expand = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn2    = nn.BatchNorm2d(channels)
        self.gamma  = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        y = self.act(self.bn1(self.reduce(x)))
        y = self.inv(y)
        y = self.bn2(self.expand(y))
        return self.act(x + self.gamma * y)


class MobileNetV3_INV(nn.Module):
    def __init__(self, num_classes, variant: str = "large",
                 weights=None, k=9, reduce=4, dropout=0.2,
                 inv_reduction=4, kernel_norm="l2", softmax_temp=1.0,
                 inv_groups: Optional[int] = None):
        super().__init__()
        variant = variant.lower()
        if variant == "large":
            base = mobilenet_v3_large(weights=(weights or MobileNet_V3_Large_Weights.IMAGENET1K_V1))
        elif variant == "small":
            base = mobilenet_v3_small(weights=(weights or MobileNet_V3_Small_Weights.IMAGENET1K_V1))
        else:
            raise ValueError("variant must be 'large' or 'small'")

        self.features = base.features
        self.feat_dim = base.classifier[0].in_features     # 960 (large), 576 (small)

        self.inv_head = InvHead(
            channels=self.feat_dim, reduce=reduce, k=k,
            inv_reduction=inv_reduction, kernel_norm=kernel_norm,
            softmax_temp=softmax_temp, inv_groups=inv_groups
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.feat_dim, num_classes)

    # ... (forward_features / extract_intermediate_features / forward same as before)
    # single image
    def _forward_features_2d(self, x):
        x = self.features(x)         # [B, feat_dim, 7, 7] at 224×224
        x = self.inv_head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)      # [B, feat_dim]
        return x

    # image or video
    def forward_features(self, x):
        if x.dim() == 4:
            return self._forward_features_2d(x)
        elif x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B*T, C, H, W)
            x = self.features(x)
            x = self.inv_head(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = x.view(B, T, self.feat_dim).mean(dim=1)
            return x
        else:
            raise ValueError(f"Expected 4D or 5D input, got {tuple(x.shape)}")

    def extract_intermediate_features(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.features(x)
        x = self.inv_head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        per_frame = x.view(B, T, self.feat_dim)   # [B,T,D]
        avg_feats = per_frame.mean(dim=1)         # [B,D]
        return per_frame, avg_feats

    def forward(self, x):
        feats = self.forward_features(x)
        logits = self.fc(self.dropout(feats))
        return logits

# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """
    Stable Supervised Contrastive Loss (handles no-positive anchors, avoids -inf).
    features: [B, D] or [B, V, D]; labels: [B]
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, features, labels):
        if features.dim() == 2:  # [B, D] -> [B, 1, D]
            features = features.unsqueeze(1)
        # L2 normalize
        features = F.normalize(features, dim=-1)

        B, V, D = features.shape
        feats = features.reshape(B * V, D)                     # [BV, D]

        # Cosine sim logits
        logits = torch.matmul(feats, feats.t()) / self.t       # [BV, BV]

        # Mask self-contrast with a large negative (not -inf) for log_softmax stability
        self_mask = torch.eye(B * V, dtype=torch.bool, device=feats.device)
        logits = logits.masked_fill(self_mask, -1e9)

        # Log-softmax over keys
        log_prob = F.log_softmax(logits, dim=1)                # [BV, BV]

        # Supervised positives: same label
        labels = labels.view(B, 1)
        pos_mask = (labels == labels.t()).float().to(feats.device)  # [B, B]
        pos_mask = pos_mask.repeat_interleave(V, 0).repeat_interleave(V, 1)
        pos_mask = pos_mask.masked_fill(self_mask, 0.0)

        # Count positives per anchor
        pos_count = pos_mask.sum(dim=1)                        # [BV]
        valid = pos_count > 0                                  # anchors with >=1 positive
        if not valid.any():
            # no positives in the whole batch -> return 0 to avoid NaN and let CE drive
            return logits.new_zeros(())

        # Average log-prob over positives only
        mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1) / pos_count.clamp(min=1)
        loss = -mean_log_prob_pos[valid].mean()
        return loss

# ---------------- Projection Head (for SupCon) ----------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=True),
        )
    def forward(self, x):  # x: [B, in_dim]
        return self.net(x)

# ---------------------------------------------------------------------------
# Collate for dict-returning VideoDataset_Oulu
# ---------------------------------------------------------------------------

def collate_fn(batch):
    imgs   = torch.stack([b['img'] for b in batch])
    labels = torch.tensor([b['label'] for b in batch])
    return imgs, labels


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="OULU-NPU Face PAD Trainer — MobileNetV3_INV")

    p.add_argument("--protocol",     type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--fold",         type=int, default=None,
                   help="Fold index 1-6 for protocols 3/4 (ignored for 1/2)")
    p.add_argument("--all_folds",    action="store_true",
                   help="Loop all 6 folds for protocols 3/4")

    # paths
    p.add_argument("--dataset_path", default="/media/oem/storage01/Shakeel/facePAD_datasets/datasets/OULU_videos",
                   help="Root of the OULU-NPU dataset (contains Baseline/, Train_files/, etc.)")
    p.add_argument("--ckpt_dir",     default="checkpoints",
                   help="Root directory for saved checkpoints and results")

    # model
    p.add_argument("--variant",      default="large", choices=["small", "large"])
    p.add_argument("--num_frames",   type=int, default=1,
                   help="Frames per clip for training")
    p.add_argument("--num_frames_val", type=int, default=1,
                   help="Frames per clip for dev/test")
    p.add_argument("--k",            type=int, default=5,
                   help="Involution kernel size")
    p.add_argument("--reduce",       type=int, default=4,
                   help="Channel reduction factor in InvHead")
    p.add_argument("--inv_groups",   type=int, default=120,
                   help="Involution groups (None = depthwise)")
    p.add_argument("--dropout",      type=float, default=0.2)

    # training
    p.add_argument("--epochs",       type=int, default=100)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience",     type=int, default=40,
                   help="Early stopping patience (epochs)")
    p.add_argument("--supcon_weight", type=float, default=0.0,
                   help="Weight for SupCon loss (0 = pure CE)")
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--resume",       default=None,
                   help="Path to checkpoint to resume from")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(args, device):
    weights = (MobileNet_V3_Large_Weights.IMAGENET1K_V1
               if args.variant == "large"
               else MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model = MobileNetV3_INV(
        num_classes=2,
        variant=args.variant,
        weights=weights,
        k=args.k,
        reduce=args.reduce,
        dropout=args.dropout,
        inv_groups=args.inv_groups,
    ).to(device)

    proj_head = ProjectionHead(
        in_dim=model.feat_dim,
        hidden=256,
        out_dim=128,
    ).to(device)

    return model, proj_head


# ---------------------------------------------------------------------------
# Training / validation steps
# ---------------------------------------------------------------------------

def train_epoch(model, proj_head, loader, optimizer,
                ce_criterion, supcon_criterion, supcon_weight, device, epoch):
    model.train()
    proj_head.train()
    total_loss, correct, total = 0.0, 0, 0

    with tqdm(loader, desc=f"Train E{epoch}", leave=False) as bar:
        for inputs, labels in bar:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.dim() == 4:
                inputs = inputs.unsqueeze(1)

            optimizer.zero_grad()
            _, feats = model.extract_intermediate_features(inputs)

            loss = torch.zeros(1, device=device)
            if supcon_weight > 0:
                z = proj_head(feats)
                loss = loss + supcon_weight * supcon_criterion(z, labels)

            logits = model.fc(feats)
            loss = loss + ce_criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            bar.set_postfix(loss=total_loss / total, acc=100 * correct / total)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate_epoch(model, loader, ce_criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(1)
        logits = model(inputs)
        loss = ce_criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Full evaluation with FacePAD metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    correct, total = 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.dim() == 4:
            inputs = inputs.unsqueeze(1)
        logits = model(inputs)
        probs  = torch.softmax(logits, 1)[:, 1]
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    auc_roc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]
    best_idx  = np.argmax(tpr - fpr)
    opt_thresh = thresholds[best_idx]
    far  = fpr[best_idx]
    frr  = fnr[best_idx]
    hter = (far + frr) / 2.0

    return {
        "acc":    100.0 * correct / max(total, 1),
        "auc":    float(auc_roc),
        "eer":    float(eer),
        "hter":   float(hter),
        "far":    float(far),
        "frr":    float(frr),
        "thresh": float(opt_thresh),
    }


def print_metrics(tag, m):
    print(f"  [{tag}] Acc={m['acc']:.2f}%  AUC={m['auc']:.4f}  "
          f"EER={m['eer']:.4f}  HTER={m['hter']:.4f}  "
          f"FAR={m['far']:.4f}  FRR={m['frr']:.4f}")


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------

def train_fold(args, protocol, fold, device):
    tag = f"p{protocol}" + (f"_f{fold}" if fold is not None else "")
    ckpt_root = os.path.join(args.ckpt_dir, f"Protocol_{protocol}")
    os.makedirs(ckpt_root, exist_ok=True)

    # Build a namespace that build_data.py functions expect
    data_args = SimpleNamespace(
        dataset='OULU',
        protocol=str(protocol),
        orig_dataset_path=args.dataset_path,
        num_frames=args.num_frames,
        num_frames_val=args.num_frames_val,
        n_split=fold,
    )

    train_tf, eval_tf = get_transforms()

    train_ds, dev_ds = build_train_val_datasets(data_args, train_tf)
    test_ds          = build_test_dataset(data_args, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True)

    print(f"\n{'='*65}")
    print(f" Protocol {protocol}  |  Fold: {fold if fold is not None else 'N/A'}  |  Tag: {tag}")
    print(f" Train: {len(train_ds)}  Dev: {len(dev_ds)}  Test: {len(test_ds)}")
    print(f"{'='*65}")

    model, proj_head = build_model(args, device)
    ce_criterion     = nn.CrossEntropyLoss()
    supcon_criterion = SupConLoss(temperature=0.07)

    optimizer = optim.Adam(
        list(model.parameters()) + list(proj_head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    start_epoch    = 0
    best_dev_loss  = float("inf")
    patience_count = 0
    best_ckpt_path = os.path.join(ckpt_root, f"best_{tag}.pth")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        proj_head.load_state_dict(ckpt["proj_head_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch   = ckpt.get("epoch", 0)
        best_dev_loss = ckpt.get("best_dev_loss", float("inf"))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    history = []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, proj_head, train_loader, optimizer,
            ce_criterion, supcon_criterion, args.supcon_weight, device, epoch + 1,
        )
        dev_loss, dev_acc = validate_epoch(model, dev_loader, ce_criterion, device)
        elapsed = time.time() - t0

        print(f"Ep {epoch+1:3d}/{args.epochs} | "
              f"Tr {tr_loss:.4f}/{tr_acc:.2f}% | "
              f"Dev {dev_loss:.4f}/{dev_acc:.2f}% | "
              f"{elapsed:.1f}s")

        history.append({"epoch": epoch + 1,
                         "tr_loss": tr_loss, "tr_acc": tr_acc,
                         "dev_loss": dev_loss, "dev_acc": dev_acc})

        if dev_loss < best_dev_loss:
            best_dev_loss  = dev_loss
            patience_count = 0
            torch.save({
                "epoch":              epoch + 1,
                "model_state_dict":   model.state_dict(),
                "proj_head_state_dict": proj_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dev_loss":      best_dev_loss,
                "dev_acc":            dev_acc,
                "args":               vars(args),
            }, best_ckpt_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch+1} (patience={args.patience})")
                break

    with open(os.path.join(ckpt_root, f"history_{tag}.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nLoading best checkpoint: {best_ckpt_path}")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    dev_metrics  = evaluate(model, dev_loader,  device)
    test_metrics = evaluate(model, test_loader, device)

    print(f"\n--- Final results [{tag}] ---")
    print_metrics("Dev ", dev_metrics)
    print_metrics("Test", test_metrics)

    results = {"tag": tag, "protocol": protocol, "fold": fold,
               "dev": dev_metrics, "test": test_metrics}
    with open(os.path.join(ckpt_root, f"results_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    protocol     = args.protocol
    needs_folds  = protocol in (3, 4)

    if needs_folds:
        if args.all_folds:
            folds = list(range(1, 7))
        elif args.fold is not None:
            folds = [args.fold]
        else:
            print("Protocol 3/4 requires --fold N (1-6) or --all_folds. Defaulting to fold 1.")
            folds = [1]
    else:
        folds = [None]

    all_results = []
    for fold in folds:
        result = train_fold(args, protocol, fold, device)
        all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*65}")
        print(f" Protocol {protocol} — Aggregate over {len(all_results)} folds")
        print(f"{'='*65}")
        for split in ("dev", "test"):
            for metric in ("acc", "auc", "eer", "hter", "far", "frr"):
                vals = [r[split][metric] for r in all_results]
                print(f"  {split} {metric:6s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        ckpt_root = os.path.join(args.ckpt_dir, f"Protocol_{protocol}")
        with open(os.path.join(ckpt_root, "all_fold_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
