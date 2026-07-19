import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
import torchvision.models as tv_models

import numpy as np

from torchvision.models import (
    mobilenet_v3_large, mobilenet_v3_small,
    MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
)
from typing import Optional
import types

from mixstyle import MixStyle

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
                 kernel_norm="l2", softmax_temp=1.0, inv_gamma=0.05, inv_groups: Optional[int] = None):
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
            if hidden % inv_groups != 0:
                # 'inv_groups' is a single value coming from args, often tuned
                # for one specific backbone's hidden size (e.g. mobilenet_v3
                # large: hidden=240 -> inv_groups=120). Other backbones have a
                # different channels -> hidden mapping, so that same value
                # won't necessarily divide evenly. Rather than hard-crash,
                # fall back to the closest divisor of hidden <= requested.
                resolved = next((g for g in range(inv_groups, 0, -1) if hidden % g == 0), 1)
                print(
                    f"[WARN] InvHead: inv_groups={inv_groups} does not divide "
                    f"hidden={hidden} (channels={channels}, reduce={reduce}); "
                    f"falling back to inv_groups={resolved}."
                )
                inv_groups = resolved
        assert hidden % inv_groups == 0, \
            f"'inv_groups' must divide hidden={hidden}; got {inv_groups}"

        self.inv = Involution(
            channels=hidden, kernel_size=k, stride=1,
            reduction=inv_reduction, kernel_norm=kernel_norm,
            softmax_temp=softmax_temp, groups=inv_groups
        )

        self.expand = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn2    = nn.BatchNorm2d(channels)
        self.gamma  = nn.Parameter(torch.tensor(inv_gamma))

    def forward(self, x):
        y = self.act(self.bn1(self.reduce(x)))
        y = self.inv(y)
        y = self.bn2(self.expand(y))
        return self.act(x + self.gamma * y)


# class ProjectionHead(nn.Module):

#     def __init__(
#         self,
#         in_dim,
#         proj_dim=128,
#         hidden_dim=512,
#     ):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, proj_dim),
#         )

#     def forward(self, x):

#         x = self.net(x)

#         return F.normalize(
#             x,
#             dim=1,
#         )


def _get_backbone_spatial(backbone_name: str, pretrained: bool = True):
    """
    Generic backbone loader for CASO-PAD.

    Unlike BaseBackboneMixin._make_backbone (used by the spatio-temporal
    models below), this must NOT global-average-pool the backbone's output:
    InvHead needs the spatial feature map [B, C, H, W] so its involution can
    look at spatial neighborhoods. Pooling happens afterwards, in
    InvolutionPAD, once InvHead is done.

    Returns:
        features:  nn.Module mapping (B, 3, H, W) -> (B, feat_dim, h, w)
        feat_dim:  int, number of channels C in that feature map
    """
    backbone_name = backbone_name.lower()
    weights = "DEFAULT" if pretrained else None

    if backbone_name in ("mobilenet_v3_large", "mobilenet_v3_small"):
        base = getattr(tv_models, backbone_name)(weights=weights)
        features = base.features
        feat_dim = base.classifier[0].in_features           # 960 / 576

    elif backbone_name == "mobilenet_v2":
        base = tv_models.mobilenet_v2(weights=weights)
        features = base.features
        feat_dim = base.classifier[-1].in_features           # 1280

    elif backbone_name in ("resnet18", "resnet34", "resnet50"):
        base = getattr(tv_models, backbone_name)(weights=weights)
        feat_dim = base.fc.in_features                       # 512 / 512 / 2048
        features = nn.Sequential(*list(base.children())[:-2])  # drop avgpool + fc

    elif backbone_name in ("efficientnet_b0", "efficientnet_b1", "efficientnet_b2"):
        base = getattr(tv_models, backbone_name)(weights=weights)
        features = base.features
        feat_dim = base.classifier[1].in_features             # 1280 / 1280 / 1408

    elif backbone_name in ("convnext_tiny", "convnext_small"):
        base = getattr(tv_models, backbone_name)(weights=weights)
        features = base.features
        feat_dim = base.classifier[-1].in_features            # 768 / 768

    elif backbone_name == "vgg16":
        base = tv_models.vgg16(weights=weights)
        features = base.features
        last_conv = [m for m in features if isinstance(m, nn.Conv2d)][-1]
        feat_dim = last_conv.out_channels                     # 512

    elif backbone_name == "shufflenet_v2_x1_0":
        base = tv_models.shufflenet_v2_x1_0(weights=weights)
        features = nn.Sequential(
            base.conv1, base.maxpool,
            base.stage2, base.stage3, base.stage4,
            base.conv5,
        )
        feat_dim = base.fc.in_features                        # 1024

    else:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}' for CASO-PAD. Supported: "
            f"mobilenet_v3_large, mobilenet_v3_small, mobilenet_v2, "
            f"resnet18, resnet34, resnet50, "
            f"efficientnet_b0, efficientnet_b1, efficientnet_b2, "
            f"convnext_tiny, convnext_small, vgg16, shufflenet_v2_x1_0."
        )

    return features, feat_dim


# 'inv_groups' configs already logged/completed were tuned as an ABSOLUTE
# group count against mobilenet_v3_large's hidden size. Keep that exact
# reference point fixed so old configs keep reproducing identically.
_REFERENCE_BACKBONE = "mobilenet_v3_large"
_REFERENCE_FEAT_DIM = 960  # mobilenet_v3_large's feat_dim (base.classifier[0].in_features)


def _scale_inv_groups(inv_groups: Optional[int], reduce: int, inv_reduction: int,
                       feat_dim: int, backbone_name: str) -> Optional[int]:
    """
    Backward-compatible, backbone-aware rescaling of 'inv_groups'.

    Historically 'inv_groups' was stored in configs as an absolute group
    count (e.g. 120), tuned against mobilenet_v3_large's hidden size
    (hidden = feat_dim // reduce = 960 // 4 = 240 by default). That number is
    only meaningful as an absolute count for THAT backbone; for any other
    backbone it needs to be re-expressed as the same ratio of hidden
    channels, otherwise it silently under/over-groups.

    For backbone == mobilenet_v3_large this is an exact no-op: it returns the
    same value that was passed in, so every already-completed
    experiment/config reproduces byte-for-byte identically, regardless of
    'reduce'. For any other backbone, it rescales the group count
    proportionally to that backbone's hidden size.

    If that ratio-based rescaling doesn't land on an exact divisor of
    'hidden' (rounding), the fallback is deterministic rather than a search:
    inv_groups = hidden // inv_reduction. This reuses an arg you already set
    per-run instead of silently picking "whatever nearby number happens to
    divide evenly."
    """
    if inv_groups is None:
        return None

    inv_groups = int(inv_groups)

    if backbone_name.lower() == _REFERENCE_BACKBONE:
        return inv_groups  # exact backward-compatible path, untouched

    reference_hidden = max(8, _REFERENCE_FEAT_DIM // reduce)
    current_hidden = max(8, feat_dim // reduce)

    if current_hidden == reference_hidden:
        return inv_groups

    ratio = inv_groups / reference_hidden
    scaled = max(1, round(ratio * current_hidden))

    if current_hidden % scaled != 0:
        fallback = max(1, current_hidden // max(1, inv_reduction))
        # walk fallback down to the nearest actual divisor of current_hidden,
        # in case inv_reduction itself doesn't divide it either
        fallback = next((g for g in range(fallback, 0, -1) if current_hidden % g == 0), 1)
        print(
            f"[INFO] InvolutionPAD: ratio-scaled inv_groups={scaled} does not "
            f"divide hidden={current_hidden}; falling back to "
            f"hidden // inv_reduction -> inv_groups={fallback}."
        )
        scaled = fallback
    else:
        print(
            f"[INFO] InvolutionPAD: rescaling inv_groups={inv_groups} (tuned for "
            f"{_REFERENCE_BACKBONE}, hidden={reference_hidden}) -> {scaled} for "
            f"backbone='{backbone_name}' (hidden={current_hidden})."
        )

    return scaled


class InvolutionPAD(nn.Module):
    def __init__(self, num_classes, backbone: str = "mobilenet_v3_large",
                 pretrained: bool = True, k=9, reduce=4, dropout=0.2,
                 inv_reduction=4, kernel_norm="l2", softmax_temp=1.0,
                 inv_gamma=0.05, mixstyle_p=0.0, mixstyle_alpha=0.1,
                 inv_groups: Optional[int] = None):
        super().__init__()
        self.backbone_name = backbone.lower()

        self.features, self.feat_dim = _get_backbone_spatial(
            self.backbone_name, pretrained=pretrained
        )

        inv_groups = _scale_inv_groups(inv_groups, reduce, inv_reduction, self.feat_dim, self.backbone_name)

        self.inv_head = InvHead(
            channels=self.feat_dim, reduce=reduce, k=k,
            inv_reduction=inv_reduction, kernel_norm=kernel_norm,
            softmax_temp=softmax_temp, inv_gamma=inv_gamma, inv_groups=inv_groups
        )

        self.mixstyle = (MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if mixstyle_p > 0 else nn.Identity())

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.feat_dim, num_classes)

    # ... (forward_features / extract_intermediate_features / forward same as before)
    # single image
    def _forward_features_2d(self, x):
        x = self.features(x)         # [B, feat_dim, 7, 7] at 224×224
        x = self.mixstyle(x)
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
            x = self.mixstyle(x)
            x = self.inv_head(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = x.view(B, T, self.feat_dim).mean(dim=1)
            return x
        else:
            raise ValueError(f"Expected 4D or 5D input, got {tuple(x.shape)}")

    def forward_with_features(self, x):
        feats = self.forward_features(x)
        logits = self.fc(self.dropout(feats))
        return logits, feats

    def extract_intermediate_features(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.features(x)
        x = self.mixstyle(x)
        x = self.inv_head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        per_frame = x.view(B, T, self.feat_dim)   # [B,T,D]
        avg_feats = per_frame.mean(dim=1)         # [B,D]
        return per_frame, avg_feats

    def classify(self, feats):
        return self.fc(self.dropout(feats))

    @property
    def embedding_dim(self):
        return self.feat_dim

    def forward(self, x):
        feats = self.forward_features(x)
        return self.classify(feats)


####################################################################
######################## deformable_facePAD #############################
####################################################################

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        # op = (n - (k * d - 1) + 2p / s)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x


def mobilenet_forward(self, x):
    # Accept [B, T, C, H, W]
    if x.ndim == 5:
        if x.shape[1] != 1:
            raise ValueError(
                f"MobileNetV2 expects num_frames=1, got {x.shape}"
            )
        x = x.squeeze(1)
    return self._forward_impl(x)

####################################################################
######################## Bahdanau spatio_temporal #############################
####################################################################


class BaseBackboneMixin:
    """
    Mixin that provides a _make_backbone(backbone_name, pretrained) method.

    Returns:
        feature_extractor: nn.Module mapping (N, 3, H, W) -> (N, D)
        feature_dim: int, D
    """
    def _make_backbone(self, backbone_name: str, pretrained: bool):
        backbone_name = backbone_name.lower()

        # -------------------------
        # MobileNet family
        # -------------------------
        if backbone_name == 'mobilenet_v3_large':
            model = tv_models.mobilenet_v3_large(pretrained=pretrained)
            feature_dim = model.features[-1].out_channels  # 960
            feature_extractor = nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1)
            )

        elif backbone_name == 'mobilenet_v2':
            model = tv_models.mobilenet_v2(pretrained=pretrained)
            feature_dim = model.classifier[-1].in_features  # 1280
            feature_extractor = nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1)
            )

        # -------------------------
        # ResNet family
        # -------------------------
        elif backbone_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            ctor = getattr(models, backbone_name)
            model = ctor(pretrained=pretrained)
            feature_dim = model.fc.in_features
            feature_extractor = nn.Sequential(
                *list(model.children())[:-1],  # -> (N, D, 1, 1)
                nn.Flatten(1)
            )

        # -------------------------
        # EfficientNet family
        # -------------------------
        elif backbone_name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']:
            ctor = getattr(models, backbone_name)
            model = ctor(pretrained=pretrained)
            feature_dim = model.classifier[1].in_features
            feature_extractor = nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1)
            )

        # -------------------------
        # ConvNeXt family
        # -------------------------
        elif backbone_name in ['convnext_tiny', 'convnext_small', 'convnext_base']:
            ctor = getattr(models, backbone_name)
            model = ctor(pretrained=pretrained)
            feature_dim = model.classifier[-1].in_features
            feature_extractor = nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1)
            )

        # -------------------------
        # Vision Transformer family
        # -------------------------
        elif backbone_name in ['vit_b_16', 'vit_b_32']:
            ctor = getattr(models, backbone_name)
            model = ctor(pretrained=pretrained)
            feature_dim = model.heads.head.in_features
            model.heads = nn.Identity()

            class ViTFeatureExtractor(nn.Module):
                def __init__(self, vit_model):
                    super().__init__()
                    self.vit = vit_model

                def forward(self, x):
                    # torchvision ViT returns [N, D] once heads are Identity
                    return self.vit(x)

            feature_extractor = ViTFeatureExtractor(model)

        # -------------------------
        # Swin Transformer family
        # -------------------------
        elif backbone_name in ['swin_t', 'swin_s']:
            ctor = getattr(models, backbone_name)
            model = ctor(pretrained=pretrained)
            feature_dim = model.head.in_features
            model.head = nn.Identity()

            class SwinFeatureExtractor(nn.Module):
                def __init__(self, swin_model):
                    super().__init__()
                    self.swin = swin_model

                def forward(self, x):
                    # torchvision Swin returns [N, D] once head is Identity
                    return self.swin(x)

            feature_extractor = SwinFeatureExtractor(model)

        else:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                f"Supported: "
                f"'mobilenet_v3_large', 'mobilenet_v2', "
                f"'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', "
                f"'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', "
                f"'convnext_tiny', 'convnext_small', 'convnext_base', "
                f"'vit_b_16', 'vit_b_32', 'swin_t', 'swin_s'."
            )

        return feature_extractor, feature_dim


class CNNTemporalBahdanauAttention(nn.Module, BaseBackboneMixin):
    """
    CNN backbone + Bahdanau-style additive temporal attention.

    Input:  x ∈ R^{B, T, 3, H, W}
    Steps:
      - Extract per-frame CNN features: h_t ∈ R^D.
      - Compute attention scores:
            e_t = v^T tanh(W h_t), implemented as
            Linear(D -> att_hidden_dim) + Tanh + Linear(att_hidden_dim -> 1)
      - Get weights α_t = softmax(e_t) along time.
      - Aggregate: h = Σ_t α_t h_t.
      - Classify: h -> logits ∈ R^{num_classes}.
    """
    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "mobilenet_v3_large",
        pretrained: bool = True,
        att_hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.pretrained = pretrained

        # -------- Backbone (shared for all frames) --------
        self.cnn, self.feature_dim = self._make_backbone(self.backbone_name, self.pretrained)

        # -------- Bahdanau-style temporal attention --------
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, att_hidden_dim),
            nn.Tanh(),
            nn.Linear(att_hidden_dim, 1)   # scalar score per frame
        )

        # -------- Classifier head --------
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 3, H, W]
        """
        B, T, C, H, W = x.shape

        # Merge batch & time: [B*T, C, H, W]
        x_flat = x.view(B * T, C, H, W)

        # CNN feature extraction: [B*T, D]
        frame_feats_flat = self.cnn(x_flat)

        # Back to [B, T, D]
        frame_feats = frame_feats_flat.view(B, T, self.feature_dim)  # h_t

        # --- temporal additive attention ---
        att_scores = self.attention(frame_feats)        # [B, T, 1]
        att_weights = torch.softmax(att_scores, dim=1)  # [B, T, 1]

        # Weighted sum h = Σ_t α_t h_t
        agg_feats = (frame_feats * att_weights).sum(dim=1)  # [B, D]

        # Classifier
        out = self.classifier(agg_feats)                # [B, num_classes]
        return out

    def extract_intermediate_features(
        self,
        x: torch.Tensor,
    ):
        """
        For analysis / visualization:

        Args:
            x: [B, T, 3, H, W]

        Returns:
            frame_feats: [B, T, D]   per-frame features
            agg_feats:   [B, D]      attention-aggregated features
            att_weights: [B, T]      attention weights per frame
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        frame_feats_flat = self.cnn(x_flat)                  # [B*T, D]
        frame_feats = frame_feats_flat.view(B, T, self.feature_dim)

        att_scores = self.attention(frame_feats)             # [B, T, 1]
        att_weights = torch.softmax(att_scores, dim=1)       # [B, T, 1]
        agg_feats = (frame_feats * att_weights).sum(dim=1)   # [B, D]

        return frame_feats, agg_feats, att_weights.squeeze(-1)  # [B,T]


# ---------------------------------------------------------------------
# SE-style temporal attention model
# ---------------------------------------------------------------------
class CNNTemporalAttPooling(nn.Module, BaseBackboneMixin):
    """
    OLD model: CNN backbone + SE-style temporal attention over frame features.

    Expects input of shape: (B, T, 3, H, W)
    """
    def __init__(self, num_classes=2, backbone='mobilenet_v3_large',
                 pretrained=True, attn_reduction=16):
        super().__init__()

        self.backbone_name = backbone
        self.pretrained = pretrained

        # Build backbone and get feature dimension D
        self.cnn, self.feature_dim = self._make_backbone(self.backbone_name, self.pretrained)

        # SE-style temporal attention (per-frame weights)
        hidden_dim = max(self.feature_dim // attn_reduction, 1)
        self.temporal_attn = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)   # scalar score per frame
        )

        # Final classifier on aggregated feature
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        """
        x: (B, T, 3, H, W)
        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = x.shape

        x_flat = x.view(B * T, C, H, W)
        feats_flat = self.cnn(x_flat)              # (B*T, D)
        feats = feats_flat.view(B, T, self.feature_dim)

        attn_scores = self.temporal_attn(feats)    # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)

        context = (attn_weights * feats).sum(dim=1)        # (B, D)
        logits = self.fc(context)                          # (B, num_classes)
        return logits

    def extract_intermediate_features(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats_flat = self.cnn(x_flat)                      # (B*T, D)
        feats = feats_flat.view(B, T, self.feature_dim)

        attn_scores = self.temporal_attn(feats)            # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)   # (B, T, 1)
        context = (attn_weights * feats).sum(dim=1)        # (B, D)

        return feats, context, attn_weights.squeeze(-1)


# ---------------------------------------------------------------------
# SK 2025 spatio-temporal model
# ---------------------------------------------------------------------

class CNNTemporalAvgPooling(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNTemporalAvgPooling, self).__init__()
        # Use MobileNetV3 as the CNN backbone
        self.cnn = tv_models.mobilenet_v3_large(pretrained=True)
        self.cnn.classifier = nn.Identity()  # Remove the classifier to use the feature extractor

        # Adjust the final fully connected layer to match MobileNetV3's feature size (960 instead of 1280)
        self.fc = nn.Linear(960, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()  # Expect input as [batch_size, seq_len, channels, height, width]
        out_decision_t = []
        cnn_features = []
        
        for t in range(seq_len):
            # with torch.no_grad():
            feature = self.cnn(x[:, t, :, :, :])  # Extract CNN features for each frame
            cnn_features.append(feature)
            # out_decision_t.append(self.fc(feature))
        
        # Stack features from all frames and average over the temporal dimension (seq_len)
        # out_decision_t = torch.stack(out_decision_t, dim=1)  # Shape: [batch_size, seq_len, 960]
        # out = out_decision_t.mean(dim=1)  # Temporal average pooling: Shape: [batch_size, 960]

        cnn_features = torch.stack(cnn_features, dim=1)  # Shape: [batch_size, seq_len, 960]
        temporal_avg_features = cnn_features.mean(dim=1)  # Temporal average pooling: Shape: [batch_size, 960]

        # Pass through the final fully connected layer
        out = self.fc(temporal_avg_features)  # Shape: [batch_size, num_classes]
        return out

    def extract_intermediate_features(self, x):
        """Extract features before and after LSTM."""
        batch_size, seq_len, C, H, W = x.size()
        cnn_features = []
        for t in range(seq_len):
            # with torch.no_grad():
            feature = self.cnn(x[:, t, :, :, :])  # CNN output (before LSTM)
            cnn_features.append(feature)
        cnn_features = torch.stack(cnn_features, dim=1)  # Shape: [batch_size, seq_len, cnn_output_dim]
        temporal_avg_features = cnn_features.mean(dim=1)  # Temporal average pooling: Shape: [batch_size, 960]
        return cnn_features, temporal_avg_features


####################################################################
######################## Model Builder #############################
####################################################################

def build_model(args):


    if getattr(args, "paper_method", "caso_pad").lower() == "deformable":
        model = tv_models.mobilenet_v2(pretrained=True) # Load pre-trained MobileNetV2
        model.features[-1] = nn.Sequential(
            DeformableConv2d(320, 1280, 3, 2),
            nn.BatchNorm2d(1280),
            nn.ReLU6()
        )
        model.classifier[1] = nn.Linear(in_features=1280, out_features=2) #default in_features =1280, out_features = 1000

        # Overriding the MobileNetV2 forward for one additional frame dimension (T=1) to support video frameinput
        model.forward = types.MethodType(mobilenet_forward, model)
        
        print(">>>>>>>>> [DEBUG] Using Deformable Conv2D with MobileNetV2 backbone for facePAD.")


    elif getattr(args, "paper_method", "caso_pad").lower() == "spatio_temporal":
        model = CNNTemporalBahdanauAttention(num_classes=2, backbone="mobilenet_v3_large", pretrained=args.pretrained, att_hidden_dim=128, dropout=getattr(args, "dropout", 0.2))
        print(">>>>>>>>> [DEBUG] Using Spatio-Temporal Bahnadou Attention with MobileNetV3 Large backbone for facePAD.")

    elif getattr(args, "paper_method", "caso_pad").lower() == "se_spatio_temporal":
        model = CNNTemporalAttPooling(num_classes=2, backbone="mobilenet_v3_large", pretrained=args.pretrained, attn_reduction=16)
        print(">>>>>>>>> [DEBUG] Using SE-style Spatio-Temporal Attention with MobileNetV3 Large backbone for facePAD.")

    elif getattr(args, "paper_method", "caso_pad").lower() == "sk_spatio_temporal":
        model = CNNTemporalAvgPooling(num_classes=2)
        print(">>>>>>>>> [DEBUG] Using STDL-FacePAD 2025 (Spatio-Temporal Attention) with MobileNetV3 Large backbone for facePAD.")

    else:

        backbone_name = getattr(args, "backbone", "mobilenet_v3_large")

        model = InvolutionPAD(
            num_classes=2,
            backbone=backbone_name,
            pretrained=args.pretrained,
            k=getattr(args, "inv_kernel", 9),
            reduce=getattr(args, "inv_reduce", 4),
            dropout=getattr(args, "dropout", 0.2),
            inv_reduction=getattr(args, "inv_reduction", 4),
            kernel_norm=getattr(args, "kernel_norm", "l2"),
            softmax_temp=getattr(args, "softmax_temp", 1.0),
            inv_gamma=getattr(args, "inv_gamma", 0.05),
            mixstyle_p=getattr(args, "mixstyle_p", 0.0),
            mixstyle_alpha=getattr(args, "mixstyle_alpha", 0.1),
            inv_groups=getattr(args, "inv_groups", None),
        )
        print(f">>>>>>>>> [DEBUG] Using Involution with {backbone_name} backbone for facePAD (CASO-PAD).")

    return model.to(args.device)