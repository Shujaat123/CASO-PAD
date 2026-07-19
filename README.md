# CASO-PAD

## Face Presentation Attack Detection with Content-Adaptive Spatial Operators

CASO-PAD is a lightweight, RGB-only, single-frame framework for face presentation attack detection (FacePAD). It augments a MobileNetV3-Large backbone with a grouped involution module that generates location-specific spatial kernels conditioned on the input feature map.

Unlike conventional convolution, which applies the same spatial kernel at every location, CASO-PAD adapts its spatial filtering to local image content. This helps the model capture localized spoofing cues such as display artifacts, printing patterns, boundaries, and illumination inconsistencies while retaining mobile-class computational complexity.

The implementation accompanies the paper:

> **Face Presentation Attack Detection via Content-Adaptive Spatial Operators**  
> Shujaat Khan

## Highlights

- RGB-only face presentation attack detection
- Single-frame inference without temporal aggregation
- MobileNetV3-Large feature extractor
- Content-adaptive grouped involution near the network head
- Binary classification of attack and bona fide samples
- Support for the datasets and protocols evaluated in the paper
- Reporting of standard FacePAD metrics, including AUC, EER, HTER, FAR, FRR, APCER, BPCER, and ACER

## Model Overview

The main CASO-PAD pipeline is:

```text
RGB frame
   │
   ▼
MobileNetV3-Large feature extractor
   │
   ▼
Grouped content-adaptive involution head
   │
   ▼
Global average pooling
   │
   ▼
Dropout and binary classifier
   │
   ▼
Attack / Bona fide
```

The grouped involution operator generates a spatial kernel for each feature-map location and channel group. Channels belonging to the same group share the generated kernel, providing spatial adaptivity with limited computational overhead.

## Default Paper Configuration

| Setting | Value |
|---|---:|
| Backbone | MobileNetV3-Large |
| Input modality | RGB |
| Input frames | 1 frame per video |
| Input resolution | 256 × 256 |
| Involution kernel size | 5 × 5 |
| Number of groups | 120 |
| Reduction factor | 4 |
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁴ |
| Batch size | 32 |
| Maximum epochs | 100 |
| Early-stopping patience | 5 |
| Number of classes | 2 |

The model uses the label convention:

```text
0 = attack
1 = bona fide
```

## Reported Results

The following results are reported in the paper as mean ± standard deviation over three runs where applicable.

| Dataset | Accuracy (%) | AUC-ROC | EER (%) | HTER (%) |
|---|---:|---:|---:|---:|
| Replay-Attack | 100.00 ± 0.00 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Replay-Mobile | 100.00 ± 0.00 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| OULU-NPU | 99.68 ± 0.13 | 0.9999 ± 0.0000 | 0.44 ± 0.11 | 0.44 ± 0.04 |
| ROSE-Youtu | 98.90 ± 0.35 | 0.9900 ± 0.0000 | 0.82 ± 0.09 | 0.82 ± 0.21 |
| SiW-Mv2 Protocol 1 | 95.45 ± 1.63 | 0.9906 ± 0.0022 | 3.13 ± 0.70 | 3.11 ± 1.02 |

Results can vary with dataset preparation, video decoding, random frame selection, software versions, and hardware.

## Repository Structure

The implementation is located in the `CASO-PAD-code` directory.

```text
CASO-PAD-code/
├── train.py             # Training entry point
├── evaluate.py          # Checkpoint evaluation
├── build_model.py       # CASO-PAD and grouped involution implementation
├── build_data.py        # Dataset loading and frame sampling
├── augmentations.py     # Training and testing transformations
├── train_fn.py          # Training and validation loops
├── evaluate_fn.py       # FacePAD evaluation metrics
└── utils.py             # Configuration, checkpoint, and metric utilities
```

## Installation

Clone the repository and enter the code directory:

```bash
git clone https://github.com/Shujaat123/CASO-PAD.git
cd CASO-PAD/CASO-PAD-code
```

Create and activate a virtual environment:

```bash
python -m venv caso_pad_env
```

Linux or macOS:

```bash
source caso_pad_env/bin/activate
```

Windows:

```bash
caso_pad_env\Scripts\activate
```

Install the required packages:

```bash
pip install torch torchvision numpy opencv-python scikit-learn pillow tqdm tensorboard
```

For GPU execution, install the PyTorch build compatible with the CUDA version available on your system.

## Dataset Preparation

The datasets are not distributed with this repository. They must be obtained from their respective owners and used according to their licenses and access conditions.

The paper evaluates CASO-PAD on:

- Replay-Attack (`RA`)
- Replay-Mobile (`RM`)
- ROSE-Youtu (`RY`)
- OULU-NPU (`OULU`)
- SiW-Mv2 Protocol 1 (`SiW`)

Create a file named `dataset_paths.json` inside `CASO-PAD-code`:

```json
{
  "RA": "/absolute/path/to/Replay-Attack",
  "RM": "/absolute/path/to/Replay-Mobile",
  "RY": "/absolute/path/to/ROSE-Youtu",
  "OULU": "/absolute/path/to/OULU-NPU",
  "SiW": "/absolute/path/to/SiW-Mv2"
}
```

Use absolute paths where possible.

### Replay-Attack, Replay-Mobile, and ROSE-Youtu

For these loaders, arrange the processed videos as follows:

```text
dataset_root/
├── train/
│   ├── attack/
│   └── real/
├── devel/
│   ├── attack/
│   └── real/
└── test/
    ├── attack/
    └── real/
```

Video files may be stored in nested directories under `attack` and `real`.

### OULU-NPU

The OULU-NPU loader expects the official video partitions and protocol files:

```text
OULU-NPU/
├── Train_files/
├── Dev_files/
├── Test_files/
└── Baseline/
    ├── Protocol_1/
    ├── Protocol_2/
    ├── Protocol_3/
    └── Protocol_4/
```

For Protocols 3 and 4, specify the required split using `--oulu_n_split`.

### SiW-Mv2 Protocol 1

The loader expects:

```text
SiW-Mv2/
├── Spoof/
├── Live/
└── protocol_files/
    ├── trainlist_all.txt
    ├── trainlist_live.txt
    ├── testlist_all.txt
    └── testlist_live.txt
```

## Training

Run commands from the `CASO-PAD-code` directory.

### Example: ROSE-Youtu

```bash
python train.py \
  --paper_method caso_pad \
  --training_mode single \
  --datasets RY \
  --backbone mobilenet_v3_large \
  --img_size 256 \
  --num_frames 1 \
  --num_frames_val 1 \
  --batch_size 32 \
  --num_epochs 100 \
  --lr 1e-4 \
  --optimizer adam \
  --scheduler none \
  --early_stop_patience 5 \
  --inv_kernel 5 \
  --inv_reduce 4 \
  --inv_reduction 4 \
  --inv_groups 120
```

Replace `RY` with `RA`, `RM`, `OULU`, or `SiW` to train on another dataset.

### Example: OULU-NPU Protocol 1

```bash
python train.py \
  --paper_method caso_pad \
  --training_mode single \
  --datasets OULU \
  --oulu_protocol 1 \
  --backbone mobilenet_v3_large \
  --img_size 256 \
  --num_frames 1 \
  --num_frames_val 1 \
  --batch_size 32 \
  --num_epochs 100 \
  --lr 1e-4 \
  --optimizer adam \
  --scheduler none \
  --early_stop_patience 5 \
  --inv_kernel 5 \
  --inv_reduce 4 \
  --inv_reduction 4 \
  --inv_groups 120
```

Training automatically creates an experiment directory under:

```text
logs/caso_pad/single/<DATASET>/log_<number>_<timestamp>/
```

The directory contains:

```text
checkpoints/best_model.pth
checkpoints/latest_model.pth
config.json
training_log.txt
evaluation_log.txt
tensorboard/
```

By default, evaluation is performed after training using the best validation checkpoint.

## Evaluation

Evaluate a completed experiment using its log directory:

```bash
python evaluate.py \
  --log_dir logs/caso_pad/single/RY/log_001_YYYYMMDD_HHMMSS
```

Select a GPU explicitly:

```bash
python evaluate.py \
  --log_dir logs/caso_pad/single/RY/log_001_YYYYMMDD_HHMMSS \
  --gpu 0
```

The evaluation script loads:

```text
<log_dir>/checkpoints/best_model.pth
```

and records the results in:

```text
<log_dir>/evaluation_log.txt
```

The reported outputs include:

- Test loss and accuracy
- AUC-ROC
- EER
- HTER
- FAR and FRR
- Youden index and operating threshold
- Average inference time
- APCER, BPCER, and ACER for OULU-NPU
- Protocol-specific attack metrics for SiW-Mv2

## TensorBoard

Training statistics can be visualized with:

```bash
tensorboard --logdir logs/caso_pad
```

Open the local address displayed by TensorBoard in a web browser.

## Reproducibility Notes

- Use `--num_frames 1 --num_frames_val 1` for the single-frame setting reported in the paper.
- The training loader randomly selects the frame used from each video.
- Use `--img_size 256`, `--inv_kernel 5`, `--inv_groups 120`, and `--inv_reduce 4` for the default CASO-PAD configuration.
- Set the random seed with `--seed`; the default value is `1234`.
- Dataset organization and protocol files must match the expected loader structure.
- Repeated independent runs are required to obtain mean and standard-deviation results comparable to those reported in the paper.

## Citation

Citation information will be updated after publication. Until then, please cite the manuscript as:

```bibtex
@article{khan2026casopad,
  title   = {Face Presentation Attack Detection via Content-Adaptive Spatial Operators},
  author  = {Khan, Shujaat},
  year    = {2026},
  note    = {Manuscript}
}
```

## Acknowledgment

This work was supported by King Fahd University of Petroleum & Minerals under Early Career Research Grant **EC241027**.
