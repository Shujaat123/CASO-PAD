import json
import os
import random
import time
from datetime import datetime

import numbers
import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
)

from torch.utils.data import ConcatDataset


###############################################################
################ Dataset registry #############################
###############################################################

DATASET_JSON = "dataset_paths.json"


def load_dataset_paths(json_path=DATASET_JSON):

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"{json_path} not found."
        )

    with open(json_path, "r") as f:
        return json.load(f)


def get_dataset_path(dataset_name):

    dataset_paths = load_dataset_paths()

    if dataset_name not in dataset_paths:
        raise ValueError(
            f"Dataset '{dataset_name}' not present in {DATASET_JSON}"
        )

    return dataset_paths[dataset_name]


###############################################################
################ Experiment directories ########################
###############################################################

def create_log_directory(
    base_logs_dir,
    training_mode,
    experiment_name,
):
    """
    Creates

    logs/

        single/
            OULU/
                log_001_xxxxx

        joint/
            RA_RY_MSU/
                log_001_xxxxx

        loo/
            CASIA/
                log_001_xxxxx
    """

    training_mode = training_mode.lower()

    base_dir = os.path.join(
        base_logs_dir,
        training_mode,
        experiment_name,
    )

    os.makedirs(base_dir, exist_ok=True)

    existing = []

    for d in os.listdir(base_dir):

        if d.startswith("log_"):
            existing.append(d)

    nums = []

    for d in existing:

        try:
            nums.append(
                int(
                    d.split("_")[1]
                )
            )
        except:
            pass

    next_num = 1 if len(nums) == 0 else max(nums) + 1

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_dir = os.path.join(
        base_dir,
        f"log_{next_num:03d}_{timestamp}",
    )

    os.makedirs(log_dir)

    os.makedirs(
        os.path.join(log_dir, "checkpoints"),
        exist_ok=True,
    )

    os.makedirs(
        os.path.join(log_dir, "tensorboard"),
        exist_ok=True,
    )

    return log_dir



def _jsonify(obj):
    """
    Recursively convert objects into JSON-serializable types.
    """

    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, torch.device):
        return str(obj)

    if isinstance(obj, numbers.Number):
        return obj

    return obj

###############################################################
################ Config helpers ###############################
###############################################################

def save_config(config, log_dir):

    config = _jsonify(config)

    with open(
        os.path.join(log_dir, "config.json"),
        "w",
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
            sort_keys=True,
        )


def load_config(log_dir):

    cfg_file = os.path.join(
        log_dir,
        "config.json",
    )

    if not os.path.exists(cfg_file):
        raise FileNotFoundError(cfg_file)

    with open(cfg_file, "r") as f:
        return json.load(f)


def update_config(log_dir, **kwargs):

    cfg = load_config(log_dir)

    cfg.update(kwargs)

    save_config(cfg, log_dir)


###############################################################
################ Logging helpers ###############################
###############################################################

def log_message(log_file, message, print_msg=True):

    if print_msg:
        print(message)

    with open(log_file, "a") as f:
        f.write(message + "\n")


def log_separator(log_file):

    with open(log_file, "a") as f:
        f.write(
            "\n"
            + "=" * 70
            + "\n"
        )


###############################################################
################ Checkpoint helpers ############################
###############################################################

def save_checkpoint(
    model,
    optimizer,
    epoch,
    log_dir,
    best=False,
):

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    ckpt_dir = os.path.join(
        log_dir,
        "checkpoints",
    )

    latest_path = os.path.join(
        ckpt_dir,
        "latest_model.pth",
    )

    torch.save(
        state,
        latest_path,
    )

    if best:

        best_path = os.path.join(
            ckpt_dir,
            "best_model.pth",
        )

        torch.save(
            state,
            best_path,
        )


def load_checkpoint(
    model,
    checkpoint_path,
    optimizer=None,
    map_location="cpu",
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    epoch = checkpoint.get(
        "epoch",
        0,
    )

    return model, optimizer, epoch


###############################################################
################ Reproducibility ###############################
###############################################################

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


###############################################################
################ Dataset helpers ###############################
###############################################################

def get_dataset_samples(dataset):

    """
    Works for

        Dataset

    and

        ConcatDataset
    """

    if isinstance(dataset, ConcatDataset):

        samples = []

        for ds in dataset.datasets:

            samples.extend(ds.samples)

        return samples

    return dataset.samples


###############################################################
################ Image Transform ##############################
###############################################################

class AdaptiveCenterCropAndResize:
    """
    Center-crop the largest square and resize.

    Input:
        Tensor or PIL Image

    Output:
        Tensor
    """

    def __init__(self, output_size):

        self.output_size = output_size

        self.to_pil = transforms.ToPILImage()

        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):

        if isinstance(img, torch.Tensor):
            img = self.to_pil(img)

        width, height = img.size

        crop = min(width, height)

        left = (width - crop) // 2
        top = (height - crop) // 2

        img = img.crop(
            (
                left,
                top,
                left + crop,
                top + crop,
            )
        )

        img = img.resize(
            self.output_size,
            Image.Resampling.LANCZOS,
        )

        return self.to_tensor(img)


###############################################################
######################## Collate ###############################
###############################################################

def collate_fn(batch):
    """
    Dataset returns dictionaries.

    Training:

        {
            "img": Tensor(T,C,H,W),
            "label": int
        }

    Testing:

        {
            "img": Tensor(T,C,H,W),
            "label": int,
            "access_type": ...
        }
    """

    imgs = [sample["img"] for sample in batch]

    labels = torch.tensor(
        [sample["label"] for sample in batch],
        dtype=torch.long,
    )

    max_frames = max(
        img.shape[0]
        for img in imgs
    )

    padded_imgs = []

    for img in imgs:

        if img.shape[0] < max_frames:

            pad = torch.zeros(
                (
                    max_frames - img.shape[0],
                    *img.shape[1:],
                ),
                dtype=img.dtype,
            )

            img = torch.cat(
                (img, pad),
                dim=0,
            )

        padded_imgs.append(img)

    output = {
        "img": torch.stack(padded_imgs),
        "label": labels,
    }

    if "access_type" in batch[0]:
        output["access_type"] = [sample["access_type"] for sample in batch]

    if "dataset" in batch[0]:
        output["dataset"] = torch.tensor([sample["dataset"] for sample in batch], dtype=torch.long)

    return output


###############################################################
######################## Timer ################################
###############################################################

class AverageMeter:

    def __init__(self):

        self.reset()

    def reset(self):

        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):

        self.sum += value * n
        self.count += n

    @property
    def avg(self):

        if self.count == 0:
            return 0

        return self.sum / self.count


class InferenceTimer:

    def __init__(self):

        self.total_time = 0.0

        self.total_samples = 0

    def update(self, elapsed_time, batch_size):

        self.total_time += elapsed_time

        self.total_samples += batch_size

    @property
    def avg_time(self):

        if self.total_samples == 0:
            return 0

        return self.total_time / self.total_samples


###############################################################
################ Binary Metrics ###############################
###############################################################

def compute_auc(labels, scores):

    return roc_auc_score(labels, scores)


def compute_roc(labels, scores):

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
    )

    return fpr, tpr, thresholds


def compute_eer(labels, scores):

    fpr, tpr, thresholds = compute_roc(
        labels,
        scores,
    )

    fnr = 1.0 - tpr

    idx = np.nanargmin(
        np.absolute(fnr - fpr)
    )

    # eer = (fnr[idx] + fpr[idx]) / 2.0
    eer = fpr[idx]

    threshold = thresholds[idx]

    return eer, threshold


def compute_youden_threshold(labels, scores):

    fpr, tpr, thresholds = compute_roc(
        labels,
        scores,
    )

    youden = tpr - fpr

    idx = np.argmax(youden)

    return (
        thresholds[idx],
        youden[idx],
    )


###############################################################
################ Confusion Matrix Metrics ######################
###############################################################

def compute_confusion_metrics(labels, preds):
    """
    Labels:
        0 -> Attack
        1 -> Real
    """

    cm = confusion_matrix(
        labels,
        preds,
        labels=[0, 1],
    )

    if cm.size == 1:
        # Handle degenerate cases
        if labels[0] == 0:
            tn = cm[0, 0]
            fp = fn = tp = 0
        else:
            tp = cm[0, 0]
            tn = fp = fn = 0
    else:
        tn, fp, fn, tp = cm.ravel()

    accuracy = (
        (tp + tn) /
        max(tp + tn + fp + fn, 1)
    )

    precision = tp / max(tp + fp, 1)

    recall = tp / max(tp + fn, 1)

    specificity = tn / max(tn + fp, 1)

    f1 = (
        2 * precision * recall /
        max(precision + recall, 1e-8)
    )

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


###############################################################
################ FAR / FRR ####################################
###############################################################

def compute_far_frr(labels, scores, threshold):

    labels = np.asarray(labels)

    scores = np.asarray(scores)

    preds = (scores >= threshold).astype(np.int32)

    attack_mask = labels == 0
    real_mask = labels == 1

    num_attack = np.sum(attack_mask)
    num_real = np.sum(real_mask)

    false_accepts = np.sum(
        preds[attack_mask] == 1
    )

    false_rejects = np.sum(
        preds[real_mask] == 0
    )

    far = false_accepts / max(num_attack, 1)

    frr = false_rejects / max(num_real, 1)

    return far, frr


###############################################################
######################## HTER #################################
###############################################################

def compute_hter(labels, scores, threshold):

    far, frr = compute_far_frr(
        labels,
        scores,
        threshold,
    )

    return (far + frr) / 2.0


###############################################################
################ APCER / BPCER / ACER ##########################
###############################################################

def compute_apcer_bpcer_acer(
    labels,
    scores,
    threshold,
):
    """
    Attack -> 0
    Bona fide -> 1
    """

    labels = np.asarray(labels)

    scores = np.asarray(scores)

    preds = (
        scores >= threshold
    ).astype(np.int32)

    attack_mask = labels == 0

    bona_mask = labels == 1

    attack_total = np.sum(attack_mask)

    bona_total = np.sum(bona_mask)

    attack_as_real = np.sum(
        preds[attack_mask] == 1
    )

    real_as_attack = np.sum(
        preds[bona_mask] == 0
    )

    apcer = attack_as_real / max(
        attack_total,
        1,
    )

    bpcer = real_as_attack / max(
        bona_total,
        1,
    )

    acer = (apcer + bpcer) / 2.0

    return apcer, bpcer, acer


###############################################################
################ Parameter Helpers ############################
###############################################################

def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


###############################################################
################ Experiment Naming ############################
###############################################################

def get_experiment_name(args):

    if args.training_mode == "single":
        return args.datasets[0]

    elif args.training_mode == "joint":
        return "_".join(sorted(args.datasets))

    elif args.training_mode == "loo":
        return args.leave_out

    else:
        raise ValueError(f"Unknown training mode: {args.training_mode}")


###############################################################
################ Config Builder ###############################
###############################################################

def build_config_from_args(args):

    cfg = vars(args).copy()

    if "device" in cfg:
        cfg["device"] = str(cfg["device"])

    cfg["created_at"] = datetime.now().isoformat(
        timespec="seconds"
    )

    return cfg


###############################################################
################ Device Helper ###############################
###############################################################

def get_device(gpu):

    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")

    return torch.device("cpu")


###############################################################
################ Timing #######################################
###############################################################

def start_timer():

    return time.perf_counter()


def stop_timer(start):

    return time.perf_counter() - start