import time
import torch
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
import oulumetrics

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np


def evaluate_all(model, loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    all_times = []
    all_access_types = []
    
    with torch.no_grad():
        for inputs, labels, access_types  in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            inputs = inputs.squeeze() # FOR NON SPATIO
            
            # Start timing for prediction
            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            all_times.append(inference_time)
            
            # Loss calculation
            # loss = criterion(outputs, labels)
            # running_loss += loss.item() * inputs.size(0)
            
            # For AUC, EER, etc.
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Assuming class 1 is the target class
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_access_types.extend(access_types.cpu().numpy().tolist())
            
            # Accuracy calculation
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate metrics like AUC-ROC, EER, etc. based on collected labels and probabilities
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    auc_roc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    # Calculate FAR, FRR, HTER, and Youden's Index
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    youdens_index = tpr[optimal_idx] - fpr[optimal_idx]
    far = fpr[optimal_idx]
    frr = fnr[optimal_idx]
    hter = (far + frr) / 2

    # Average inference time
    avg_inference_time = np.mean(all_times)

    # Test loss and accuracy
    # test_loss = running_loss / len(loader.dataset)
    test_acc = 100. * correct / total

    
    all_probs_np = np.array(all_probs, dtype=np.float32)
    y_attack_types = np.array(all_access_types, dtype=np.int32)
    y_pred_scores = all_probs_np  # same as for ROC, prob of class 1 (live)

    # threshold optional; default is 0.5
    apcer, bpcer, acer = oulumetrics.calculate_metrics(
        y_attack_types.tolist(),
        y_pred_scores.tolist()
    )
    
    # Return dictionary with all results
    return {
        # 'test_loss': test_loss,
        'test_acc': test_acc,
        'auc_roc': auc_roc,
        'eer': eer,
        'hter': hter,
        'far': far,
        'frr': frr,
        'youdens_index': youdens_index,
        'optimal_threshold': optimal_threshold,
        'avg_inference_time': avg_inference_time,
        'fpr': fpr,
        'tpr': tpr,
        'labels': all_labels,
        'probs': all_probs,
        'apcer': float(apcer),
        'bpcer': float(bpcer),
        'acer': float(acer)}


def generate_evaluation_summary(results):
    # Extract metrics from the results dictionary
    # test_loss = results['test_loss']
    test_acc = results['test_acc']
    auc_roc = results['auc_roc']
    eer = results['eer']
    hter = results['hter']
    far = results['far']
    frr = results['frr']
    youdens_index = results['youdens_index']
    optimal_threshold = results['optimal_threshold']
    avg_inference_time = results['avg_inference_time']

    apcer = results['apcer']
    bpcer = results['bpcer']
    acer = results['acer']

    
    # Print summary
    print("\n--- Evaluation Summary ---")
    # print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"Equal Error Rate (EER): {eer:.4f}")
    print(f"Half Total Error Rate (HTER): {hter:.8f}")
    print(f"False Acceptance Rate (FAR): {far:.4f}")
    print(f"False Rejection Rate (FRR): {frr:.4f}")
    print(f"Youden's Index (Max): {youdens_index:.4f}")
    print(f"Optimal Threshold (Youden's Index): {optimal_threshold:.4f}")
    print(f"Average inference time per sample: {avg_inference_time:.6f} seconds")

    print(f"apcer: {apcer:.8f}")
    print(f"bpcer: {bpcer:.8f}")
    print(f"acer: {acer:.8f}")