import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
)

import metrics as temp_module
from build_data import build_test_dataset
from metrics import evaluate_all, generate_evaluation_summary
from train import MobileNetV3_INV, AdaptiveCenterCropAndResize


def get_transform():
    return transforms.Compose([
        AdaptiveCenterCropAndResize((256, 256)),
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])


def test_collate_fn(batch):
    imgs         = torch.stack([b['img'] for b in batch])
    labels       = torch.tensor([b['label'] for b in batch])
    access_types = torch.tensor([b['access_type'] for b in batch])
    return imgs, labels, access_types


def build_model(args, device):
    weights = (MobileNet_V3_Large_Weights.IMAGENET1K_V1
               if args.variant == 'large'
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
    return model


def run_test(args, protocol, fold, device):
    tag       = f"p{protocol}" + (f"_f{fold}" if fold is not None else "")
    ckpt_path = os.path.join(args.ckpt_dir, f"Protocol_{protocol}", f"best_{tag}.pth")

    if not os.path.exists(ckpt_path):
        print(f"[SKIP] Checkpoint not found: {ckpt_path}")
        return None

    data_args = SimpleNamespace(
        dataset='OULU',
        protocol=str(protocol),
        orig_dataset_path=args.dataset_path,
        num_frames=args.num_frames,
        num_frames_val=args.num_frames,
        n_split=fold,
    )

    transform   = get_transform()
    test_ds     = build_test_dataset(data_args, transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_collate_fn,
        pin_memory=True,
    )

    print(f"\n{'='*60}")
    print(f" Protocol {protocol}  |  Fold: {fold if fold is not None else 'N/A'}  |  Tag: {tag}")
    print(f" Test samples: {len(test_ds)}")
    print(f" Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    model = build_model(args, device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # evaluate_all references `device` as a free variable in temp.py's module scope
    temp_module.device = device

    results = evaluate_all(model, test_loader)
    generate_evaluation_summary(results)

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Test OULU-NPU checkpoints for all protocols")
    p.add_argument("--protocol",     type=int, default=None, choices=[1, 2, 3, 4],
                   help="Run a single protocol (default: all four)")
    p.add_argument("--dataset_path", default="/media/oem/storage01/Shakeel/facePAD_datasets/datasets/OULU_videos")
    p.add_argument("--ckpt_dir",     default="checkpoints")
    p.add_argument("--variant",      default="large", choices=["small", "large"])
    p.add_argument("--num_frames",   type=int, default=1)
    p.add_argument("--k",            type=int, default=5)
    p.add_argument("--reduce",       type=int, default=4)
    p.add_argument("--inv_groups",   type=int, default=120)
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--num_workers",  type=int, default=4)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    protocols   = [args.protocol] if args.protocol else [1, 2, 3, 4]
    all_results = {}

    for protocol in protocols:
        folds = list(range(1, 7)) if protocol in (3, 4) else [None]

        fold_results = []
        for fold in folds:
            result = run_test(args, protocol, fold, device)
            if result is not None:
                fold_results.append(result)

        if len(fold_results) > 1:
            print(f"\n{'='*60}")
            print(f" Protocol {protocol} — Mean over {len(fold_results)} folds")
            print(f"{'='*60}")
            for metric in ('test_acc', 'auc_roc', 'eer', 'hter', 'far', 'frr', 'apcer', 'bpcer', 'acer'):
                vals = [r[metric] for r in fold_results]
                print(f"  {metric:15s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        all_results[f"protocol_{protocol}"] = fold_results


if __name__ == "__main__":
    main()
