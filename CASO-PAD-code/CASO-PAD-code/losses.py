import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################
################ Projection Head ##############################
###############################################################

class ProjectionHead(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim=256,
        proj_dim=128,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):

        x = self.net(x)

        return F.normalize(
            x,
            dim=1,
        )


###############################################################
################ Supervised Contrastive ########################
###############################################################

class SupConLoss(nn.Module):

    def __init__(
        self,
        temperature=0.07,
    ):
        super().__init__()

        self.temperature = temperature

    def forward(
        self,
        features,
        labels,
    ):

        features = F.normalize(
            features,
            dim=1,
        )

        similarity = torch.matmul(
            features,
            features.T,
        )

        similarity = similarity / self.temperature

        labels = labels.contiguous().view(-1, 1)

        positive_mask = torch.eq(
            labels,
            labels.T,
        ).float()

        logits_mask = (
            torch.ones_like(positive_mask)
            - torch.eye(
                positive_mask.size(0),
                device=positive_mask.device,
            )
        )

        positive_mask = positive_mask * logits_mask

        exp_logits = torch.exp(similarity) * logits_mask

        log_prob = similarity - torch.log(
            exp_logits.sum(
                dim=1,
                keepdim=True,
            ) + 1e-12
        )

        positives = positive_mask.sum(dim=1)

        valid = positives > 0

        mean_log_prob = torch.zeros_like(positives)

        mean_log_prob[valid] = (
            (
                positive_mask[valid]
                * log_prob[valid]
            ).sum(dim=1)
            / positives[valid]
        )

        loss = -mean_log_prob[valid].mean()

        return loss
    

###############################################################
######################## CORAL ################################
###############################################################

def coral_loss(source, target):

    d = source.size(1)

    source = source - source.mean(
        dim=0,
        keepdim=True,
    )

    target = target - target.mean(
        dim=0,
        keepdim=True,
    )

    cov_source = (
        source.T @ source
    ) / max(source.size(0) - 1, 1)

    cov_target = (
        target.T @ target
    ) / max(target.size(0) - 1, 1)

    return (
        (cov_source - cov_target)
        .pow(2)
        .mean()
    ) / (4 * d * d)


def multi_domain_coral(
    feats,
    domains,
):

    unique_domains = torch.unique(domains)

    if len(unique_domains) < 2:
        return feats.new_tensor(0.)

    losses = []

    for i in range(len(unique_domains)):

        for j in range(i + 1, len(unique_domains)):

            fi = feats[
                domains == unique_domains[i]
            ]

            fj = feats[
                domains == unique_domains[j]
            ]

            if (
                fi.size(0) < 2
                or
                fj.size(0) < 2
            ):
                continue

            losses.append(
                coral_loss(fi, fj)
            )

    if len(losses) == 0:
        return feats.new_tensor(0.)

    return torch.stack(losses).mean()


###############################################################
######################## Center Loss ##########################
###############################################################

class CenterLoss(nn.Module):

    def __init__(
        self,
        num_classes,
        feat_dim,
    ):
        super().__init__()

        self.centers = nn.Parameter(
            torch.randn(
                num_classes,
                feat_dim,
            )
        )

    def forward(
        self,
        features,
        labels,
    ):

        centers_batch = self.centers[
            labels
        ]

        return (
            (features - centers_batch)
            .pow(2)
            .sum(dim=1)
            .mean()
        )