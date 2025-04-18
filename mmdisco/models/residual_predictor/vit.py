"""this code is copied from https://github.com/openai/CLIP/blob/main/clip/model.py and modified."""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32, embed_dim, heads, output_dim
        )

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        # currently double backward of sdp is not supported. Avoid using efficient algorithm as a workaround.
        from torch.backends.cuda import enable_flash_sdp, enable_mem_efficient_sdp

        enable_flash_sdp(False)
        enable_mem_efficient_sdp(False)

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformerND(nn.Module):
    @property
    def out_temporal_dim(self):
        if self.use_class_emb:
            return 1

        return self.input_size[0] // self.patch_size[0]

    def __init__(
        self,
        input_dim: int,
        input_ndim: int,
        input_size: tuple[int],
        patch_size: tuple[int],
        time_embed_dim: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        output_dim: int,
        use_class_emb: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.input_ndim = input_ndim
        self.input_size = input_size
        self.patch_size = patch_size
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.use_class_emb = use_class_emb

        if input_ndim == 2:
            conv1_cls = nn.Conv2d
        elif input_ndim == 3:
            conv1_cls = nn.Conv3d
        else:
            raise ValueError(f"input_ndim = {input_ndim} is not supported")

        self.conv1 = conv1_cls(
            in_channels=input_dim,
            out_channels=model_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = model_dim**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(model_dim))

        self.time_embed_in = nn.Linear(time_embed_dim, model_dim)

        num_tokens = (input_size[-2] // patch_size[-2]) * (
            input_size[-1] // patch_size[-1]
        )
        init_pemb: torch.Tensor = scale * torch.randn(num_tokens, model_dim)
        if len(input_size) == 3:
            assert len(patch_size) == 3
            n_repeat = input_size[0] // patch_size[0]
            init_pemb = init_pemb.repeat((n_repeat, 1))

        init_pemb_cls_temb: torch.Tensor = scale * torch.randn(2, model_dim)
        init_pemb = torch.cat([init_pemb_cls_temb, init_pemb], dim=0)

        self.positional_embedding = nn.Parameter(init_pemb)

        self.ln_pre = LayerNorm(model_dim)

        self.transformer = Transformer(model_dim, num_layers, num_heads)

        self.ln_post = LayerNorm(model_dim)
        self.proj = nn.Parameter(scale * torch.randn(model_dim, output_dim))

    def forward(self, x: torch.Tensor, temb: torch.Tensor = None):
        # x: (B, C, H, W)
        # temb: (B, C)

        # embed input
        x = self.conv1(x)  # shape = [B, C, Hg, Wg]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [B, C, L]
        x = x.permute(0, 2, 1)  # shape = [B, L, C]

        # cat input embs, cls emb, and emb
        cls_emb = (
            self.class_embedding.to(dtype=x.dtype, device=x.device)
            .reshape(1, 1, x.shape[-1])
            .repeat(x.shape[0], 1, 1)
        )
        temb = self.time_embed_in(temb).unsqueeze(1)
        x = torch.cat([cls_emb, temb, x], dim=1)  # shape = [B, L + 2, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)
        x = self.transformer(x)

        if self.use_class_emb:
            x = x[:, 0:1, :]  # discard all except class emb
        else:
            x = x[:, 2:, :]  # discard class emb and temb

        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj

        return x
