import time
import numpy as np
import torch
from tqdm import tqdm

from utils import (
    compute_auc,
    compute_roc,
    compute_eer,
    compute_far_frr,
    compute_hter,
    compute_apcer_bpcer_acer,
)


###############################################################
######################## Helpers ##############################
###############################################################

def _ensure_video_shape(x):

    if x.dim() == 4:
        x = x.unsqueeze(1)

    return x


def _move_to_device(batch, device):

    imgs = batch["img"].to(device, non_blocking=True)

    labels = batch["label"].to(device, non_blocking=True)

    imgs = _ensure_video_shape(imgs)

    return imgs, labels


###############################################################
######################## Evaluation ###########################
###############################################################

@torch.no_grad()
def evaluate_all(
    model,
    loader,
    criterion,
    device,
    dataset=None,
):

    model.eval()

    running_loss = 0.0

    running_correct = 0

    running_samples = 0

    all_labels = []

    all_probs = []

    inference_times = []

    access_types = []

    progress = tqdm(
        loader,
        desc="Evaluation",
        leave=False,
    )

    for batch in progress:

        imgs, labels = _move_to_device(
            batch,
            device,
        )

        batch_size = labels.size(0)

        if "access_type" in batch:
            access_types.extend(batch["access_type"])

        ########################################################
        # inference timing
        ########################################################

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        logits = model(imgs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        inference_times.append(
            elapsed / batch_size
        )

        ########################################################

        loss = criterion(
            logits,
            labels,
        )

        probs = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        preds = logits.argmax(dim=1)

        running_loss += (
            loss.item() *
            batch_size
        )

        running_correct += (
            preds == labels
        ).sum().item()

        running_samples += batch_size

        ########################################################

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_probs.extend(
            probs.cpu().numpy()
        )

        ########################################################

        progress.set_postfix(
            loss=f"{running_loss/max(running_samples,1):.4f}",
            acc=f"{100*running_correct/max(running_samples,1):.2f}",
        )

    ############################################################
    ##################### Final metrics ########################
    ############################################################

    test_loss = (
        running_loss /
        max(running_samples, 1)
    )

    test_acc = (
        100.0 *
        running_correct /
        max(running_samples, 1)
    )

    # ROC / AUC
    fpr, tpr, thresholds = compute_roc(
        all_labels,
        all_probs,
    )

    auc = compute_auc(
        all_labels,
        all_probs,
    )

    # EER
    eer, eer_threshold = compute_eer(
        all_labels,
        all_probs,
    )

    # Youden threshold
    # optimal_idx = np.argmax(tpr - fpr)
    optimal_idx = np.nanargmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    youdens_index = tpr[optimal_idx] - fpr[optimal_idx]

    # FAR / FRR / HTER
    far, frr = compute_far_frr(
        all_labels,
        all_probs,
        optimal_threshold,
    )

    hter = compute_hter(
        all_labels,
        all_probs,
        optimal_threshold,
    )

    avg_time = float(
        np.mean(inference_times)
    )

    ############################################################
    ################ Dataset-specific metrics ##################
    ############################################################

    results = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "auc_roc": auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "hter": hter,
        "far": far,
        "frr": frr,
        "youdens_index": youdens_index,
        "optimal_threshold": optimal_threshold,
        "avg_inference_time": avg_time,

        # "fpr": fpr.tolist(),
        # "tpr": tpr.tolist(),
        # "labels": all_labels,
        # "probs": all_probs,        
    }

    ############################################################
    ################ Common FAS metrics ########################
    ############################################################

    if dataset.upper() in ["OULU-NPU", "OULU"]:
        apcer, bpcer, acer = compute_apcer_bpcer_acer(
            all_labels,
            all_probs,
            optimal_threshold,
        )

        results["apcer"] = apcer
        results["bpcer"] = bpcer
        results["acer"] = acer

    ############################################################
    ###################### SiW #################################
    ############################################################

    if dataset.upper() == "SIW":

        protocol_results = {}

        spoof_types = {}

        for label, prob, access in zip(
            all_labels,
            all_probs,
            access_types,
        ):

            if isinstance(access, tuple):
                _, spoof = access
            else:
                spoof = access

            if spoof not in spoof_types:
                spoof_types[spoof] = {
                    "labels": [],
                    "scores": [],
                }

            spoof_types[spoof]["labels"].append(label)
            spoof_types[spoof]["scores"].append(prob)

        for spoof_type, data in spoof_types.items():

            fpr_p, tpr_p, thresholds_p = compute_roc(
                data["labels"],
                data["scores"],
            )

            eer_p, eer_thr_p = compute_eer(
                data["labels"],
                data["scores"],
            )

            idx = np.argmax(tpr_p - fpr_p)

            thr_p = thresholds_p[idx]

            apcer_p, bpcer_p, acer_p = compute_apcer_bpcer_acer(
                data["labels"],
                data["scores"],
                thr_p,
            )

            protocol_results[spoof_type] = {
                "EER": eer_p,
                "APCER": apcer_p,
                "BPCER": bpcer_p,
                "ACER": acer_p,
            }

        results["siw_protocol"] = protocol_results

    return results