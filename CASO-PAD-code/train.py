from email import parser
import os
import json
import argparse
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from augmentations import (
    build_train_transform,
    build_test_transform,
)

from build_data import (
    build_train_val_datasets,
)

from build_model import build_model

from train_fn import (
    train_one_epoch,
    validate_one_epoch,
)

from utils import (
    AdaptiveCenterCropAndResize,
    collate_fn,
    create_log_directory,
    save_checkpoint,
    save_config,
    update_config,
    load_config,
    set_seed,
    get_dataset_samples,
    count_parameters,
    get_experiment_name,
)

# from losses import TotalLoss
from losses import (
    ProjectionHead,
    SupConLoss,
    CenterLoss,
)

from samplers import BalancedBatchSampler

###############################################################
######################## Arguments ############################
###############################################################

def parse_args():

    parser = argparse.ArgumentParser(description="FacePAD Training")

    ################ Experiment ###############################
    parser.add_argument("--exp_description", type=str, default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval", type=bool, default=True, help="Evaluate automatically after training.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_amp", type=bool, default=False, help="Use automatic mixed precision.")

    ################ Dataset ##################################
    parser.add_argument("--training_mode", default="single", choices=["single", "joint", "loo"])
    # parser.add_argument("--datasets",nargs="+", default=["OULU", "RA", "CASIA", "MSU"], help="Datasets to use for training.")
    parser.add_argument("--datasets",nargs="+", default=["RY", ], help="Datasets to use for training.")
    parser.add_argument("--leave_out",default="CASIA", help="Leave out dataset from traing for LOO training mode. (None otherwise)")
    parser.add_argument("--loo_val_source", type=str, default="leave_out", choices=["train_datasets", "leave_out"],
        help=("Validation source for LOO training.\n"
            "train_datasets : concatenate validation splits of training datasets.\n"
            "leave_out      : use leave-out dataset validation split (or train split if no validation exists)."))
    parser.add_argument("--dataset", default=None, help="Evaluation dataset override. (None otherwise)") # eval_dataset

    parser.add_argument("--oulu_protocol", default="all")
    parser.add_argument("--oulu_n_split",default="NA")
    parser.add_argument("--siw_protocol", default="1")
    parser.add_argument("--casia_split", default="train", choices=["train", "train80"])
    parser.add_argument("--msu_protocol", default="grand")
    parser.add_argument("--msu_split", default="train", choices=["train", "train80"])

    ################ Training #################################
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop_patience", type=int, default=30)

    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor for CrossEntropyLoss. 0.0 means no label smoothing.") # 0.0 original

    parser.add_argument("--coral_weight", type=float, default=0.0) # 0.0 originally

    parser.add_argument("--sampler", type=str, default="random", choices=["random", "domain_balanced"]) # random originally, not working with domain_balanced sampler, need to fix it

    parser.add_argument("--mixstyle_p", type=float, default=0.0) # 0.0 originally
    parser.add_argument("--mixstyle_alpha", type=float, default=0.1)

    parser.add_argument("--center_loss_weight", type=float, default=0.0)
    # parser.add_argument("--center_loss_lr", type=float, default=0.5)

    parser.add_argument("--supcon_weight", type=float, default=0.0)
    parser.add_argument("--supcon_temperature", type=float, default=0.07)
    parser.add_argument("--supcon_proj_dim", type=int, default=128)
    parser.add_argument("--supcon_hidden_dim", type=int, default=256)

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"]) # adam originally
    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "cosine_restart", "plateau"]) # none originally
    parser.add_argument("--cosine_tmax", type=int, default=100)
    parser.add_argument("--cosine_eta_min", type=float, default=1e-6)
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--plateau_patience", type=int, default=5)

    ################ Images ###################################
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_frames", type=int, default=3)
    parser.add_argument("--num_frames_val", type=int, default=3)


    ################ Data Augmentations ####################################
    parser.add_argument("--aug_hflip", type=float, default=0.0)
    parser.add_argument("--aug_vflip", type=float, default=0.0)
    parser.add_argument("--aug_brightness", type=float, default=0.0)
    parser.add_argument("--aug_contrast", type=float, default=0.0)
    parser.add_argument("--aug_saturation", type=float, default=0.0)
    parser.add_argument("--aug_hue", type=float, default=0.00)
    parser.add_argument("--aug_blur_prob", type=float, default=0.0)
    parser.add_argument("--aug_blur_kernel", type=int, default=5) # default 5 originally
    parser.add_argument("--aug_random_erasing", type=float, default=0.0)

    ################ Model ####################################
    parser.add_argument("--paper_method", default="caso_pad", choices=["caso_pad", "deformable", "SK_spatio_temporal", "SE_spatio_temporal", "spatio_temporal"])

    # parser.add_argument("--variant", default="large", choices=["large", "small",])
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_large",
        choices=["mobilenet_v3_large", "mobilenet_v3_small",
                "resnet18", "resnet34", "resnet50",
                "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
                "convnext_tiny", "convnext_small",
                "vgg16",
                "shufflenet_v2_x1_0",
                "mobilenet_v2",
                ])

    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--inv_kernel", type=int, default=7)
    parser.add_argument("--inv_reduce", type=int, default=4)
    parser.add_argument("--inv_reduction", type=int, default=4)
    parser.add_argument("--inv_groups", type=int, default=120)
    parser.add_argument("--kernel_norm", default="l2")
    parser.add_argument("--softmax_temp", type=float, default=1.0)
    parser.add_argument("--inv_gamma", type=float, default=0.05) # 0.05 original

    return parser.parse_args()
###############################################################

args = parse_args()

set_seed(args.seed)

device = torch.device(
    f"cuda:{args.gpu}"
    if torch.cuda.is_available()
    else "cpu"
)
args.device = device

###############################################################
#################### Experiment folder ########################
###############################################################

experiment_name = get_experiment_name(args)
log_dir = create_log_directory(
    base_logs_dir=f"logs/{args.paper_method}",
    training_mode=args.training_mode,
    experiment_name=experiment_name,
)
args.log_dir = log_dir

###############################################################
#################### Save config ##############################
###############################################################

cfg = vars(args).copy()

cfg["device"] = str(device)

cfg["created_at"] = datetime.now().isoformat(
    timespec="seconds"
)

save_config(
    cfg,
    log_dir,
)

###############################################################
#################### Logging #################################
###############################################################

print()

for k, v in vars(args).items():

    print(f"{k}: {v}")

with open(
    os.path.join(log_dir, "training_log.txt"),
    "w",
) as f:

    for k, v in vars(args).items():
        f.write(f"{k}: {v}\n")


###############################################################
###################### Transforms #############################
###############################################################

train_transform = build_train_transform(args)
val_transform = build_test_transform(args)

###############################################################
###################### Datasets ###############################
###############################################################

train_dataset, val_dataset = build_train_val_datasets(args, train_transform, val_transform)

print(f"\nTraining samples   : {len(train_dataset)}")
print(f"Validation samples : {len(val_dataset)}")

# print(f">>>>>>>>>>>>> [DEBUG] Train dataset shape: {train_dataset[0]['img'].shape}")
# print(f">>>>>>>>>>>>> [DEBUG] Validation dataset shape: {val_dataset[0]['img'].shape}")

###############################################################
###################### Dataloaders ############################
###############################################################

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=args.batch_size,
#     shuffle=True,
#     collate_fn=collate_fn,
#     pin_memory=True,
# )

if (args.sampler == "domain_balanced" and args.training_mode != "single"):
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BalancedBatchSampler(train_dataset, batch_size=args.batch_size),
        collate_fn=collate_fn,
        pin_memory=True,
    )
else:
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=True,
)

print(f"Train Loader : {len(train_loader)} batches")
print(f"Val Loader   : {len(val_loader)} batches\n")

###############################################################
###################### Model ##################################
###############################################################

model = build_model(args)

projection_head = None
supcon_loss = None
if args.supcon_weight > 0:
    projection_head = ProjectionHead(in_dim=model.embedding_dim, hidden_dim=args.supcon_hidden_dim, proj_dim=args.supcon_proj_dim).to(device)
    supcon_loss = SupConLoss(temperature=args.supcon_temperature)

center_loss = None
if args.center_loss_weight > 0:
    center_loss = CenterLoss(num_classes=2, feat_dim=model.embedding_dim).to(device)

    # print()
    # print(f"~~~~~~~~~~~~~~~~~[DEBUG] Center Loss: {center_loss}")
    # print(f"~~~~~~~~~~~~~~~~~[DEBUG] Center Loss Centers Shape: {center_loss.centers.shape}")

total_params, trainable_params = count_parameters(model)

print(f"Total Parameters      : {total_params:,}")
print(f"Trainable Parameters  : {trainable_params:,}")

with open(
    os.path.join(log_dir, "training_log.txt"),
    "a",
) as f:

    f.write("\n")
    f.write(f"Total Parameters     : {total_params:,}\n")
    f.write(f"Trainable Parameters : {trainable_params:,}\n\n")

update_config(
    log_dir,
    num_parameters=total_params,
    trainable_parameters=trainable_params,
)

###############################################################
################ Criterion ###############################
###############################################################

# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

###############################################################
###################### Optimizer ##############################
###############################################################

params = list(model.parameters())
if projection_head is not None:
    params += list(projection_head.parameters())
if center_loss is not None:
    params += list(center_loss.parameters())

if args.optimizer == "adam":
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == "adamw":
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
else:
    raise ValueError(f"Unknown optimizer: {args.optimizer}")

# ###############################################################
# ###################### Scheduler ##############################
# ###############################################################

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode="min",
#     factor=0.5,
#     patience=5,
#     verbose=True,
# )

scheduler = None

if args.scheduler == "cosine":
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.cosine_tmax,
        eta_min=args.cosine_eta_min,
    )
elif args.scheduler == "cosine_restart":
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.cosine_tmax,
        eta_min=args.cosine_eta_min,
    )
elif args.scheduler == "plateau":
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.plateau_factor,
        patience=args.plateau_patience,
    )

# ###############################################################
# ###################### AMP ####################################
# ###############################################################

scaler = None

if args.use_amp and torch.cuda.is_available():
    scaler = torch.cuda.amp.GradScaler()

###############################################################
###################### TensorBoard ############################
###############################################################

writer = SummaryWriter(
    log_dir=os.path.join(
        log_dir,
        "tensorboard",
    )
)

###############################################################
###################### Training ###############################
###############################################################

best_val_loss = float("inf")

best_epoch = 0

early_stop_counter = 0

for epoch in range(args.num_epochs):

    train_loss, train_acc = train_one_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epoch=epoch + 1,
        scaler=scaler,
        projection_head=projection_head,
        supcon_loss=supcon_loss,
        supcon_weight=args.supcon_weight,
        coral_weight=args.coral_weight,
        center_loss=center_loss,
        center_loss_weight=args.center_loss_weight,
    )

    val_loss, val_acc = validate_one_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    if scheduler is not None:
        if args.scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

    # scheduler.step(val_loss)

    ###########################################################
    # TensorBoard
    ###########################################################

    writer.add_scalar(
        "Loss/Train",
        train_loss,
        epoch,
    )

    writer.add_scalar(
        "Loss/Validation",
        val_loss,
        epoch,
    )

    writer.add_scalar(
        "Accuracy/Train",
        train_acc,
        epoch,
    )

    writer.add_scalar(
        "Accuracy/Validation",
        val_acc,
        epoch,
    )

    ###########################################################
    # Console
    ###########################################################

    msg = (
        f"Epoch [{epoch+1:03d}/{args.num_epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    print(msg)

    with open(
        os.path.join(log_dir, "training_log.txt"),
        "a",
    ) as f:

        f.write(msg + "\n")

    ###########################################################
    # Checkpoints
    ###########################################################

    is_best = False

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_epoch = epoch + 1

        early_stop_counter = 0

        is_best = True

    else:

        early_stop_counter += 1

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch + 1,
        log_dir=log_dir,
        best=is_best,
    )

    ###########################################################
    # Early stopping
    ###########################################################

    if early_stop_counter >= args.early_stop_patience:

        print("\nEarly stopping triggered.\n")

        break

###############################################################
###################### Finish #################################
###############################################################

writer.close()

update_config(
    log_dir,
    best_epoch=best_epoch,
    best_val_loss=best_val_loss,
    final_epoch=epoch + 1,
)

###############################################################
###################### Evaluation #############################
###############################################################

if args.eval:

    print("\nRunning evaluation...\n")

    from evaluate import evaluate_from_config

    evaluate_from_config(
        log_dir,
        device=device,
    )
