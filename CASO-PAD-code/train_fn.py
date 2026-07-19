import torch
from tqdm import tqdm

from losses import multi_domain_coral

###############################################################
######################## Helpers ##############################
###############################################################

def _ensure_video_shape(x):
    """
    Model expects:
        [B, T, C, H, W]

    If a single frame is given:
        [B, C, H, W]

    convert to

        [B, 1, C, H, W]
    """

    if x.dim() == 4:
        x = x.unsqueeze(1)

    return x


def _move_to_device(batch, device):

    imgs = batch["img"].to(device, non_blocking=True)

    labels = batch["label"].to(device, non_blocking=True)

    imgs = _ensure_video_shape(imgs)

    return imgs, labels


###############################################################
#################### Training Epoch ###########################
###############################################################

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch,
    scaler=None,
    projection_head=None,
    supcon_loss=None,
    supcon_weight=0.0,
    coral_weight=0.0,
    center_loss=None,
    center_loss_weight=0.0,    
):

    model.train()

    running_loss = 0.0
    running_correct = 0
    running_samples = 0
    progress = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)

    for batch in progress:

        imgs, labels = _move_to_device(batch, device)

        domains = batch["dataset"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        #######################################################
        # Forward
        #######################################################

        if model.__class__.__name__ == "MobileNetV3_INV":

            if scaler is None:

                # logits = model(imgs)
                logits, feats = model.forward_with_features(imgs)
                loss = criterion(logits, labels)

                # Optional SupCon
                if (supcon_weight > 0 and projection_head is not None):
                    # feats = model.forward_features(imgs)
                    proj = projection_head(feats)
                    loss = (loss + supcon_weight * supcon_loss(proj, labels))

                # Optional CORAL
                if coral_weight > 0:
                    coral = multi_domain_coral(feats, domains)
                    loss = (loss + coral_weight * coral)

                # Optional Center Loss
                if (center_loss_weight > 0 and center_loss is not None):
                    center = center_loss(feats, labels)
                    loss = (loss + center_loss_weight * center)

                loss.backward()
                optimizer.step()

            else:

                with torch.cuda.amp.autocast():

                    # logits = model(imgs)
                    logits, feats = model.forward_with_features(imgs)
                    loss = criterion(logits, labels)

                    # Optional SupCon
                    if (supcon_weight > 0 and projection_head is not None):
                        # feats = model.forward_features(imgs)
                        proj = projection_head(feats)
                        loss = (loss + supcon_weight * supcon_loss(proj, labels))

                    # Optional CORAL
                    if coral_weight > 0:
                        coral = multi_domain_coral(feats, domains)
                        loss = (loss + coral_weight * coral)

                    # Optional Center Loss
                    if (center_loss_weight > 0 and center_loss is not None):
                        center = center_loss(feats, labels)
                        loss = (loss + center_loss_weight * center)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        else:
            if scaler is None:
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            else:
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        #######################################################
        # Statistics
        #######################################################

        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size

        running_correct += (
            preds == labels
        ).sum().item()

        running_samples += batch_size

        avg_loss = (
            running_loss /
            running_samples
        )

        avg_acc = (
            100.0 *
            running_correct /
            running_samples
        )

        progress.set_postfix(
            loss=f"{avg_loss:.4f}",
            acc=f"{avg_acc:.2f}",
        )

    epoch_loss = (
        running_loss /
        max(running_samples, 1)
    )

    epoch_acc = (
        100.0 *
        running_correct /
        max(running_samples, 1)
    )

    return epoch_loss, epoch_acc


###############################################################
#################### Validation Epoch #########################
###############################################################

@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    running_correct = 0

    running_samples = 0

    progress = tqdm(
        loader,
        desc="Validation",
        leave=False,
    )

    for batch in progress:

        imgs, labels = _move_to_device(
            batch,
            device,
        )

        logits = model(imgs)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        running_loss += (
            loss.item() *
            batch_size
        )

        running_correct += (
            preds == labels
        ).sum().item()

        running_samples += batch_size

        avg_loss = (
            running_loss /
            running_samples
        )

        avg_acc = (
            100.0 *
            running_correct /
            running_samples
        )

        progress.set_postfix(
            loss=f"{avg_loss:.4f}",
            acc=f"{avg_acc:.2f}",
        )

    epoch_loss = (
        running_loss /
        max(running_samples, 1)
    )

    epoch_acc = (
        100.0 *
        running_correct /
        max(running_samples, 1)
    )

    return epoch_loss, epoch_acc



