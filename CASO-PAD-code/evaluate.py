import os
import json
import argparse
from types import SimpleNamespace
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from build_data import build_test_dataset
from evaluate_fn import evaluate_all

from build_model import build_model

from utils import (
    AdaptiveCenterCropAndResize,
    collate_fn,
    load_config,
    update_config,
    load_checkpoint,
)


###############################################################
######################## Evaluation ###########################
###############################################################

def evaluate_from_config(log_dir, device=None, override_dataset=None):

    cfg = load_config(log_dir)

    ###########################################################
    # Device
    ###########################################################

    if device is None:

        gpu = cfg.get("gpu", 0)

        device = torch.device(
            f"cuda:{gpu}"
            if torch.cuda.is_available()
            else "cpu"
        )

    ###########################################################
    # Build args namespace
    ###########################################################

    args = SimpleNamespace(**cfg)

    args.device = device

    if override_dataset is not None:
        args.dataset = override_dataset
    elif args.dataset is None:
        if args.training_mode == "single":
            args.dataset = args.datasets[0]
        elif args.training_mode == "loo":
            args.dataset = args.leave_out

    print(f"[Evaluation] Dataset: {args.dataset}")

    ###########################################################
    # Build transform
    ###########################################################

    transform = transforms.Compose([
        AdaptiveCenterCropAndResize(
            (
                args.img_size,
                args.img_size,
            )
        ),
    ])

    ###########################################################
    # Dataset
    ###########################################################

    test_dataset = build_test_dataset(
        args,
        transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"\nTest samples : {len(test_dataset)}")

    ###########################################################
    # Model
    ###########################################################

    model = build_model(args)

    checkpoint = os.path.join(
        log_dir,
        "checkpoints",
        "best_model.pth",
    )

    model, _, epoch = load_checkpoint(
        model,
        checkpoint,
        optimizer=None,
        map_location=device,
    )

    model.to(device)

    ###########################################################
    # Criterion
    ###########################################################

    criterion = nn.CrossEntropyLoss()

    ###########################################################
    # Evaluate
    ###########################################################

    results = evaluate_all(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        dataset=args.dataset,
    )

    ###########################################################
    # Logging
    ###########################################################

    log_file = os.path.join(
        log_dir,
        "evaluation_log.txt",
    )

    with open(log_file, "a") as f:

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"Evaluation : {datetime.now().isoformat(timespec='seconds')}\n"
        )
        f.write(f"Dataset    : {args.dataset}\n")
        f.write(f"Checkpoint : best_model.pth\n")
        f.write(f"Epoch      : {epoch}\n")
        f.write("-" * 80 + "\n")

        f.write(
            f"Loss                : {results['test_loss']:.6f}\n"
        )

        f.write(
            f"Accuracy (%)        : {results['test_acc']:.4f}\n"
        )

        f.write(
            f"AUC                : {results['auc_roc']:.6f}\n"
        )

        f.write(
            f"EER                : {results['eer']:.6f}\n"
        )

        f.write(
            f"HTER               : {results['hter']:.6f}\n"
        )

        f.write(
            f"FAR                : {results['far']:.6f}\n"
        )

        f.write(
            f"FRR                : {results['frr']:.6f}\n"
        )

        f.write(
            f"Threshold          : {results['optimal_threshold']:.6f}\n"
        )

        if "apcer" in results:
            f.write(f"APCER             : {results['apcer']:.6f}\n")
            f.write(f"BPCER             : {results['bpcer']:.6f}\n")
            f.write(f"ACER              : {results['acer']:.6f}\n")

        f.write(
            f"Avg inference (s) : {results['avg_inference_time']:.6f}\n"
        )

        #######################################################
        # SIW detailed protocol metrics
        #######################################################

        if args.dataset == "SiW" and "siw_protocol" in results:

            f.write("\n")
            f.write("SIW Protocol-1 Results\n")
            f.write("-" * 80 + "\n")

            for spoof_type, metrics in results["siw_protocol"].items():

                f.write(f"{spoof_type}\n")
                f.write(
                    f"    APCER : {metrics['APCER']:.6f}\n"
                )
                f.write(
                    f"    BPCER : {metrics['BPCER']:.6f}\n"
                )
                f.write(
                    f"    ACER  : {metrics['ACER']:.6f}\n"
                )
                f.write(
                    f"    EER   : {metrics['EER']:.6f}\n"
                )

        f.write("=" * 80 + "\n")

    ###########################################################
    # Update config
    ###########################################################

    update_config(
        log_dir,
        last_evaluated_at=datetime.now().isoformat(
            timespec="seconds"
        ),
        last_evaluation_dataset=args.dataset,
        test_size=len(test_dataset),
        last_results=results,
    )

    ###########################################################
    # Console summary
    ###########################################################

    print("\n" + "=" * 70)
    print(f"Dataset           : {args.dataset}")
    print(f"Test Loss         : {results['test_loss']:.6f}")
    print(f"Accuracy (%)      : {results['test_acc']:.4f}")
    print(f"AUC              : {results['auc_roc']:.6f}")
    print(f"EER              : {results['eer']:.6f}")
    print(f"HTER             : {results['hter']:.6f}")
    print(f"FAR              : {results['far']:.6f}")
    print(f"FRR              : {results['frr']:.6f}")
    print(f"Threshold        : {results['optimal_threshold']:.6f}")
    if "apcer" in results:
        print(f"APCER           : {results['apcer']:.6f}")
        print(f"BPCER           : {results['bpcer']:.6f}")
        print(f"ACER            : {results['acer']:.6f}")
    print(
        f"Avg inference(s): {results['avg_inference_time']:.6f}"
    )

    if args.dataset == "SiW" and "siw_protocol" in results:

        print("\nSIW Protocol-1 Results")

        for spoof_type, metrics in results["siw_protocol"].items():

            print(
                f"{spoof_type:20}"
                f" APCER={metrics['APCER']:.4f}"
                f" BPCER={metrics['BPCER']:.4f}"
                f" ACER={metrics['ACER']:.4f}"
                f" EER={metrics['EER']:.4f}"
            )

    print("=" * 70 + "\n")

    return results


###############################################################
############################ CLI ##############################
###############################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log_dir",
        required=True,
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional evaluation dataset override.",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if args.gpu is None:

        device = None

    else:

        device = torch.device(
            f"cuda:{args.gpu}"
            if torch.cuda.is_available()
            else "cpu"
        )

    evaluate_from_config(
        log_dir=args.log_dir,
        device=device,
        override_dataset=args.dataset,
    )

