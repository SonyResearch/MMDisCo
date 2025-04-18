"""
This code is copied from https://github.com/researchmm/MM-Diffusion/blob/main/mm_diffusion/multimodal_unet.py.
"""

import math
import random
from dataclasses import asdict, dataclass

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from mmdisco.utils.logger import RankedLogger

from .base import (
    FeatureExtractorBase,
    VAEmbs,
    VAResidualPredictorBase,
    VAResidualPredictorOutputs,
    VATimestepBlock,
    VATimestepEmbedSequential,
)
from .nn import GroupNorm, avg_pool_nd, conv_nd, normalization, zero_module
from .timestep import (
    TimestepEmbedSequential,
    TimestepEncoder,
    TimestepModulatedNormalization,
)
from .x_formers import TransformerWrapperNoTokenEmb

logger = RankedLogger(__name__, rank_zero_only=True)


class VideoConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding="same",
        dilation=1,
        conv_type="2d+1d",
    ):
        super().__init__()
        self.conv_type = conv_type
        self.padding = padding

        if conv_type == "2d+1d":
            self.video_conv_spatial = conv_nd(
                2, in_channels, out_channels, kernel_size, stride, padding, dilation
            )
            self.video_conv_temporal = conv_nd(
                1, out_channels, out_channels, kernel_size, stride, padding, dilation
            )
        elif conv_type == "3d":
            self.video_conv = conv_nd(
                3,
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        else:
            raise NotImplementedError

    def forward(self, video):
        if self.conv_type == "2d+1d":
            b, f, c, h, w = video.shape
            video = rearrange(video, "b f c h w -> (b f) c h w")
            video = self.video_conv_spatial(video)
            video = rearrange(video, "(b f) c h w -> (b h w) c f", b=b)
            video = self.video_conv_temporal(video)
            video = rearrange(video, "(b h w) c f -> b f c h w", b=b, h=h)

        elif self.conv_type == "3d":
            video = rearrange(video, "b f c h w -> b c f h w")
            video = self.video_conv(video)
            video = rearrange(video, "b c f h w -> b f c h w")

        return video


class AudioConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding="same",
        dilation=1,
        conv_type="2d",
    ):
        super().__init__()

        if conv_type == "1d":
            self.audio_conv = conv_nd(
                1, in_channels, out_channels, kernel_size, stride, padding, dilation
            )
        elif conv_type == "linear":
            self.audio_conv = conv_nd(
                1, in_channels, out_channels, kernel_size, stride, padding, dilation
            )
        elif conv_type == "2d":
            self.audio_conv = conv_nd(
                2, in_channels, out_channels, kernel_size, stride, padding, dilation
            )
        else:
            raise NotImplementedError(f"AudioConv doesn't support mode = {conv_type}")

    def forward(self, audio):
        audio = self.audio_conv(audio)
        return audio


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if dims == 3:
            # for video
            stride = (1, 2, 2)  # (2,2,2)
        elif dims == 1:
            # for audio
            stride = 4
        else:
            # for image
            # stride = 2

            # for melspec or latent audio
            stride = (1, 2)  # (2, 2)

        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x: Tensor, is_video: bool):
        if is_video:
            assert x.ndim == 5
            x = rearrange(x, "b f c h w -> b c f h w")

        x = self.op(x)

        if is_video:
            assert x.ndim == 5
            x = rearrange(x, "b c f h w -> b f c h w")

        return x


class SingleModalQKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class SingleModalAtten(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = SingleModalQKVAttention(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape

        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)


class ResBlock(VATimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        video_type="2d+1d",
        audio_type="1d",
        audio_dilation=1,
        use_scale_shift_norm=False,
        use_checkpoint=False,
        down=False,
        use_conv=False,
        video_attention=False,
        audio_attention=False,
        num_heads=4,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.video_in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            VideoConv(channels, self.out_channels, 3, conv_type=video_type),
        )
        self.audio_in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            AudioConv(
                channels,
                self.out_channels,
                3,
                conv_type=audio_type,
                dilation=audio_dilation,
            ),
        )

        self.down = down
        self.video_attention = video_attention
        self.audio_attention = audio_attention

        if down:
            self.vh_upd = Downsample(channels, False, 3)
            self.vx_upd = Downsample(channels, False, 3)
            self.ah_upd = Downsample(channels, False, 2 if audio_type == "2d" else 1)
            self.ax_upd = Downsample(channels, False, 2 if audio_type == "2d" else 1)
        else:
            self.ah_upd = self.ax_upd = self.vh_upd = self.vx_upd = nn.Identity()

        self.video_out_layers = TimestepEmbedSequential(
            TimestepModulatedNormalization(
                emb_channels,
                self.out_channels,
                GroupNorm(32, self.out_channels, affine=not use_scale_shift_norm),
                channel_dim=2,
                mode="scale_shift" if use_scale_shift_norm else "add",
            ),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                VideoConv(self.out_channels, self.out_channels, 1, conv_type="3d")
            ),
        )
        self.audio_out_layers = TimestepEmbedSequential(
            TimestepModulatedNormalization(
                emb_channels,
                self.out_channels,
                GroupNorm(32, self.out_channels, affine=not use_scale_shift_norm),
                channel_dim=1,
                mode="scale_shift" if use_scale_shift_norm else "add",
            ),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                AudioConv(self.out_channels, self.out_channels, 1, conv_type=audio_type)
            ),
        )

        if self.out_channels == channels:
            self.video_skip_connection = nn.Identity()
            self.audio_skip_connection = nn.Identity()
        elif use_conv:
            self.video_skip_connection = VideoConv(
                channels, self.out_channels, 3, conv_type="2d+1d"
            )
            self.audio_skip_connection = AudioConv(
                channels, self.out_channels, 3, conv_type=audio_type
            )
        else:
            self.video_skip_connection = VideoConv(
                channels, self.out_channels, 1, conv_type="3d"
            )
            self.audio_skip_connection = AudioConv(
                channels, self.out_channels, 1, conv_type=audio_type
            )

        if self.video_attention:
            self.spatial_attention_block = SingleModalAtten(
                channels=self.out_channels,
                num_heads=num_heads,
                num_head_channels=-1,
                use_checkpoint=use_checkpoint,
            )
            self.temporal_attention_block = SingleModalAtten(
                channels=self.out_channels,
                num_heads=num_heads,
                num_head_channels=-1,
                use_checkpoint=use_checkpoint,
            )
        if self.audio_attention:
            self.audio_attention_block = SingleModalAtten(
                channels=self.out_channels,
                num_heads=num_heads,
                num_head_channels=-1,
                use_checkpoint=use_checkpoint,
            )

    def forward(self, video, audio, emb: VAEmbs):
        """
        video:(b,f,c,h,w)
        audio:(b,c,l,m) or (b, c, l)
        emb:(b,c)
        """
        if self.down:
            video_h = self.video_in_layers(video)
            video_h = self.vh_upd(video_h, is_video=True)
            video = self.vx_upd(video, is_video=True)

            audio_h = self.audio_in_layers(audio)
            audio_h = self.ah_upd(audio_h, is_video=False)
            audio = self.ax_upd(audio, is_video=False)

        else:
            video_h = self.video_in_layers(video)
            audio_h = self.audio_in_layers(audio)

        video_out = self.video_skip_connection(video) + self.video_out_layers(
            video_h, emb.video
        )
        audio_out = self.audio_skip_connection(audio) + self.audio_out_layers(
            audio_h, emb.audio
        )

        if self.video_attention:
            b, f, c, h, w = video.shape
            video_out = rearrange(video_out, "b f c h w -> (b f) c (h w)")
            video_out = self.spatial_attention_block(video_out)
            video_out = rearrange(video_out, "(b f) c (h w) -> (b h w) c f", f=f, h=h)
            video_out = self.temporal_attention_block(video_out)
            video_out = rearrange(video_out, "(b h w) c f -> b f c h w", h=h, w=w)
        if self.audio_attention:
            if audio.ndim == 4:
                b, c, l, m = audio.shape
                audio_out = rearrange(audio_out, "b c l m -> b c (l m)")
                audio_out = self.audio_attention_block(audio_out)
                audio_out = rearrange(audio_out, "b c (l m) -> b c l m", l=l, m=m)
            else:
                assert audio.ndim == 3
                audio_out = self.audio_attention_block(audio_out)

        return video_out, audio_out


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(
        self,
        qkv,
        video_attention_index,
        audio_attention_index,
        frame_size,
        audio_per_frame,
    ):
        """
        Apply QKV attention.
        : attention_index_v:[V_len x H], V_len = f x h x w
        : attention_index_a:[A_len x H]
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """

        bs, width, _ = qkv.shape
        video_len = video_attention_index.shape[0]  # f x h x w
        audio_len = audio_attention_index.shape[0]
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))

        v_as = []
        a_as = []
        video_q = q[:, :, :video_len]  # [bsz, c*head, video length]
        audio_q = q[:, :, video_len:]

        for idx in range(0, video_len // frame_size):
            video_frame_k = th.index_select(
                k, -1, video_attention_index[idx * frame_size]
            )  # [bsz, c*head, k_num]
            video_frame_v = th.index_select(
                v, -1, video_attention_index[idx * frame_size]
            )  # [bsz, c*head, k_num]
            video_frame_q = video_q[:, :, idx * frame_size : (idx + 1) * frame_size]

            w_slice = th.einsum(
                "bct,bcs->bts",
                (video_frame_q * scale).view(bs * self.n_heads, ch, -1),
                (video_frame_k * scale).view(bs * self.n_heads, ch, -1),
            )  # More stable with f16 than dividing afterwards

            w_slice = th.softmax(w_slice, dim=-1)  # [bsz, 1, k_len]
            a = th.einsum(
                "bts,bcs->bct", w_slice, video_frame_v.view(bs * self.n_heads, ch, -1)
            ).reshape(bs * self.n_heads, ch, -1)
            v_as.append(a)

            audio_frame_k = th.index_select(
                k, -1, audio_attention_index[idx * audio_per_frame]
            )  # [bsz, c*head, k_num]
            audio_frame_v = th.index_select(
                v, -1, audio_attention_index[idx * audio_per_frame]
            )  # [bsz, c*head, k_num]
            if idx == (video_len // frame_size - 1):
                audio_frame_q = audio_q[:, :, idx * audio_per_frame :]
            else:
                audio_frame_q = audio_q[
                    :, :, idx * audio_per_frame : (idx + 1) * audio_per_frame
                ]
            w_slice = th.einsum(
                "bct,bcs->bts",
                (audio_frame_q * scale).view(bs * self.n_heads, ch, -1),
                (audio_frame_k * scale).view(bs * self.n_heads, ch, -1),
            )  # More stable with f16 than dividing afterwards

            w_slice = th.softmax(w_slice, dim=-1)  # [bsz, 1, k_len]
            a = th.einsum(
                "bts,bcs->bct", w_slice, audio_frame_v.view(bs * self.n_heads, ch, -1)
            ).reshape(bs * self.n_heads, ch, -1)
            a_as.append(a)

        v_a = th.cat(v_as, dim=2)
        a_a = th.cat(a_as, dim=2)

        return v_a.reshape(bs, -1, video_len), a_a.reshape(bs, -1, audio_len)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        local_window=-1,
        window_shift=False,
    ):
        super().__init__()
        self.channels = channels

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                self.channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"

            self.num_heads = self.channels // num_head_channels

        self.local_window = local_window
        self.window_shift = window_shift
        self.use_checkpoint = use_checkpoint
        self.v_norm = normalization(self.channels)
        self.a_norm = normalization(self.channels)
        self.v_qkv = conv_nd(1, self.channels, self.channels * 3, 1)
        self.a_qkv = conv_nd(1, self.channels, self.channels * 3, 1)
        # self.attention = QKVAttention(self.num_heads)

        self.video_proj_out = zero_module(
            VideoConv(self.channels, self.channels, 1, conv_type="3d")
        )
        self.audio_proj_out = zero_module(
            AudioConv(self.channels, self.channels, 1, conv_type="2d")
        )
        self.va_index = None
        self.av_index = None
        self.ws_video = None
        self.ws_audio = None

    def naive_cross_attention(self, qkv_v: th.Tensor, qkv_a: th.Tensor):
        """
        Apply QKV attention for each window.
        : attention_index_v:[V_len x Ha], V_len = f x h x w, Ha = ws_audio x m
        : attention_index_a:[A_len x Hv], A_len = l x m, Hv = ws_video x h x w
        :param qkv_v, qkv_a: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        B, C, video_len = qkv_v.shape
        _, _, audio_len = qkv_a.shape
        assert (B, C) == qkv_a.shape[:2]
        assert C % (3 * self.num_heads) == 0
        ch = C // (3 * self.num_heads)
        q_v, k_v, v_v = qkv_v.chunk(3, dim=1)
        q_a, k_a, v_a = qkv_a.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))

        # attention for video
        w = th.einsum(
            "bct,bcs->bts",
            (q_v * scale).view(B * self.num_heads, ch, -1),
            (k_a * scale).view(B * self.num_heads, ch, -1),
        )  # More stable with f16 than dividing afterwards

        w = th.softmax(w, dim=-1)  # [bsz, 1, k_len]
        v_out = th.einsum(
            "bts,bcs->bct", w, v_a.reshape(B * self.num_heads, ch, -1)
        ).reshape(B * self.num_heads, ch, -1)

        # attention for audio
        w = th.einsum(
            "bct,bcs->bts",
            (q_a * scale).view(B * self.num_heads, ch, -1),
            (k_v * scale).view(B * self.num_heads, ch, -1),
        )  # More stable with f16 than dividing afterwards

        w = th.softmax(w, dim=-1)  # [bsz, 1, k_len]
        a_out = th.einsum(
            "bts,bcs->bct", w, v_v.reshape(B * self.num_heads, ch, -1)
        ).reshape(B * self.num_heads, ch, -1)

        return v_out.reshape(B, -1, video_len), a_out.reshape(B, -1, audio_len)

    def block_attention(
        self,
        qkv_v: th.Tensor,
        qkv_a: th.Tensor,
        attn_index_v: th.Tensor,
        attn_index_a: th.Tensor,
        v_dims_per_frame: int,
        a_dims_per_frame: int,
    ):
        """
        Apply QKV attention for each window.
        : attention_index_v:[V_len x Ha], V_len = f x h x w, Ha = ws_audio x m
        : attention_index_a:[A_len x Hv], A_len = l x m, Hv = ws_video x h x w
        :param qkv_v, qkv_a: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, video_len = qkv_v.shape
        _, _, audio_len = qkv_a.shape
        assert (bs, width) == qkv_a.shape[:2]
        assert width % (3 * self.num_heads) == 0
        ch = width // (3 * self.num_heads)
        q_v, k_v, v_v = qkv_v.chunk(3, dim=1)
        q_a, k_a, v_a = qkv_a.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))

        v_as = []
        a_as = []

        assert len(attn_index_a) == len(attn_index_v)

        for idx in range(len(attn_index_v)):
            video_frame_k = th.index_select(
                k_a, -1, attn_index_v[idx]
            )  # [bsz, c*head, k_num]
            video_frame_v = th.index_select(
                v_a, -1, attn_index_v[idx]
            )  # [bsz, c*head, k_num]
            ws_v = self.ws_video * v_dims_per_frame
            video_frame_q = q_v[:, :, idx * ws_v : (idx + 1) * ws_v]

            w_slice = th.einsum(
                "bct,bcs->bts",
                (video_frame_q * scale).view(bs * self.num_heads, ch, -1),
                (video_frame_k * scale).view(bs * self.num_heads, ch, -1),
            )  # More stable with f16 than dividing afterwards

            w_slice = th.softmax(w_slice, dim=-1)  # [bsz, 1, k_len]
            a = th.einsum(
                "bts,bcs->bct", w_slice, video_frame_v.view(bs * self.num_heads, ch, -1)
            ).reshape(bs * self.num_heads, ch, -1)
            v_as.append(a)

            audio_frame_k = th.index_select(
                k_v, -1, attn_index_a[idx]
            )  # [bsz, c*head, k_num]
            audio_frame_v = th.index_select(
                v_v, -1, attn_index_a[idx]
            )  # [bsz, c*head, k_num]
            ws_a = self.ws_audio * a_dims_per_frame
            audio_frame_q = q_a[:, :, idx * ws_a : (idx + 1) * ws_a]
            w_slice = th.einsum(
                "bct,bcs->bts",
                (audio_frame_q * scale).view(bs * self.num_heads, ch, -1),
                (audio_frame_k * scale).view(bs * self.num_heads, ch, -1),
            )  # More stable with f16 than dividing afterwards

            w_slice = th.softmax(w_slice, dim=-1)  # [bsz, 1, k_len]
            a = th.einsum(
                "bts,bcs->bct", w_slice, audio_frame_v.view(bs * self.num_heads, ch, -1)
            ).reshape(bs * self.num_heads, ch, -1)
            a_as.append(a)

        v_a = th.cat(v_as, dim=2)
        a_a = th.cat(a_as, dim=2)

        return v_a.reshape(bs, -1, video_len), a_a.reshape(bs, -1, audio_len)

    def attention_index(self, audio_size, video_size, device):
        f, h, w = video_size
        l, m = audio_size

        # window size for both audio and video
        ws_video = self.local_window

        assert (
            (ws_video * l) % f == 0
        ), f"ws_video: {ws_video}, l (audio length): {l}, f (video length): {f}"
        ws_audio = ws_video * l // f

        # # TODO: support arbitrary shape for audio and video by padding
        assert (
            (f % ws_video) == 0
        ), f"video length must be divisible by window size. {f}(=f) % {ws_video}(=ws) != 0."
        assert (
            (l % ws_audio) == 0
        ), f"audio length must be divisible by window size. {l}(=l) % {ws_audio}(=ws) != 0."

        # keep window size for block attention op
        self.ws_video = ws_video
        self.ws_audio = ws_audio

        video_dims_per_frame = h * w
        video_dims = f * video_dims_per_frame
        audio_dims_per_frame = m
        audio_dims = l * audio_dims_per_frame
        if self.window_shift:
            window_shift = random.randint(0, f - ws_video)
        else:
            window_shift = 0

        if self.va_index is None:
            # calculate base index mapper for video -> audio (audio index to be attended for each video frame)
            va_index_x = th.arange(0, ws_audio * audio_dims_per_frame).view(1, -1)
            va_index_y = th.arange(0, f // ws_video).unsqueeze(-1).view(-1, 1)
            va_index_y = va_index_y * ws_audio * audio_dims_per_frame
            self.va_index = (va_index_y + va_index_x).to(device)

        va_index = (self.va_index + window_shift * audio_dims_per_frame) % audio_dims

        if self.av_index is None:
            av_index_x = th.arange(0, ws_video * video_dims_per_frame).view(1, -1)
            av_index_y = th.arange(0, l // ws_audio).unsqueeze(-1).view(-1, 1)
            av_index_y = av_index_y * ws_video * video_dims_per_frame
            self.av_index = (av_index_y + av_index_x).to(device)

        av_index = (self.av_index + window_shift * video_dims_per_frame) % video_dims

        return va_index, av_index

    def forward(self, video, audio):
        b, f, c, h, w = video.shape
        b, c, l, m = audio.shape

        video_token = rearrange(video, "b f c h w -> b c (f h w)")
        # audio_token = audio
        audio_token = rearrange(audio, "b c l m -> b c (l m)")

        v_qkv = self.v_qkv(self.v_norm(video_token))  # [bsz, 3c, f*h*w]
        a_qkv = self.a_qkv(self.a_norm(audio_token))  # [bsz, 3c, l*m]

        if self.local_window > 0:
            # apply block attention
            attention_index_v, attention_index_a = self.attention_index(
                (l, m), (f, h, w), video.device
            )
            video_h, audio_h = self.block_attention(
                v_qkv, a_qkv, attention_index_v, attention_index_a, h * w, m
            )
        else:
            # apply cross attention
            video_h, audio_h = self.naive_cross_attention(v_qkv, a_qkv)

        video_h = rearrange(video_h, "b c (f h w)-> b f c h w ", f=f, h=h)
        video_h = self.video_proj_out(video_h)
        video_h = video + video_h

        audio_h = rearrange(audio_h, "b c (l m)-> b c l m ", l=l, m=m)
        audio_h = self.audio_proj_out(audio_h)
        audio_h = audio + audio_h

        return video_h, audio_h


class InitialBlock(VATimestepBlock):
    def __init__(
        self,
        video_in_channels,
        audio_in_channels,
        video_out_channels,
        audio_out_channels,
        kernel_size=3,
        input_norm_type=None,
        emb_channels=None,
        use_scale_shift_norm=False,
        audio_type="2d",
    ):
        super().__init__()
        self.video_conv = VideoConv(
            video_in_channels, video_out_channels, kernel_size, conv_type="2d+1d"
        )
        self.audio_conv = AudioConv(
            audio_in_channels, audio_out_channels, kernel_size, conv_type=audio_type
        )

        self.input_norm_type = input_norm_type
        if input_norm_type == "tmod_norm":
            self.video_norm = TimestepModulatedNormalization(
                emb_channels,
                video_in_channels,
                GroupNorm(
                    video_in_channels,
                    video_in_channels,
                    affine=not use_scale_shift_norm,
                ),  # equivalent to LayerNorm
                channel_dim=2,
                mode="scale_shift" if use_scale_shift_norm else "add",
            )

            self.audio_norm = TimestepModulatedNormalization(
                emb_channels,
                audio_in_channels,
                GroupNorm(
                    audio_in_channels,
                    audio_in_channels,
                    affine=not use_scale_shift_norm,
                ),  # equivalent to LayerNorm
                channel_dim=1,
                mode="scale_shift" if use_scale_shift_norm else "add",
            )
        elif input_norm_type == "norm":
            self.video_norm = GroupNorm(
                video_in_channels, video_in_channels
            )  # equivalent to LayerNorm
            self.audio_norm = GroupNorm(
                audio_in_channels, audio_in_channels
            )  # equivalent to LayerNorm
        else:
            assert input_norm_type is None

    def forward(self, video, audio, emb):
        if self.input_norm_type == "tmod_norm":
            video = self.video_norm(video, emb.video)
            audio = self.audio_norm(audio, emb.audio)
        elif self.input_norm_type == "norm":
            video = self.video_norm(video)
            audio = self.audio_norm(audio)

        return self.video_conv(video), self.audio_conv(audio)


class LastBlock(nn.Module):
    def __init__(
        self,
        model_channels,
        video_out_channels,
        audio_out_channels,
        kernel_size=3,
        audio_type="2d",
    ):
        super().__init__()
        self.video_block = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(
                VideoConv(
                    model_channels, video_out_channels, kernel_size, conv_type="3d"
                )
            ),
        )
        self.audio_block = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            AudioConv(
                model_channels, audio_out_channels, kernel_size, conv_type=audio_type
            ),
        )

    def forward(self, video, audio):
        return self.video_block(video), self.audio_block(audio)


class VAResidualBlocks(FeatureExtractorBase):
    def __init__(
        self,
        video_cdim,
        audio_cdim,
        model_channels,
        time_embed_dims,
        num_res_blocks,
        cross_attention_resolutions,
        cross_attention_windows,
        cross_attention_shift,
        video_attention_resolutions,
        audio_attention_resolutions,
        video_conv_type,  # ="2d+1d" originally
        audio_conv_type,  # ="1d" originally
        dropout=0,
        channel_mult=(1, 2, 3, 4),
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        first_norm_type=None,
        max_dila=10,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.video_cdim = video_cdim
        self.audio_cdim = audio_cdim
        self.model_channels = model_channels
        self.time_embed_dims = time_embed_dims
        self.num_res_blocks = num_res_blocks
        self.cross_attention_resolutions = cross_attention_resolutions
        self.cross_attention_windows = cross_attention_windows
        self.cross_attention_shift = cross_attention_shift
        self.video_attention_resolutions = video_attention_resolutions
        self.audio_attention_resolutions = audio_attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.first_norm_type = first_norm_type

        ch = int(channel_mult[0] * model_channels)

        self._feature_size = ch
        input_block_chans = [ch]

        # define input block
        # video_in_dim = self.video_cdim * self.video_input_channels_factor
        # audio_in_dim = self.audio_cdim * self.audio_input_channels_factor
        video_in_dim = self.video_cdim
        audio_in_dim = self.audio_cdim
        self.input_blocks = nn.ModuleList(
            [
                VATimestepEmbedSequential(
                    InitialBlock(
                        video_in_dim,
                        audio_in_dim,
                        video_out_channels=ch,
                        audio_out_channels=ch,
                        input_norm_type=first_norm_type,
                        emb_channels=time_embed_dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        audio_type=audio_conv_type,
                    )
                )
            ]
        )

        resblock_common_args = {
            "emb_channels": time_embed_dims,
            "dropout": dropout,
            "video_type": video_conv_type,
            "audio_type": audio_conv_type,
            "use_checkpoint": use_checkpoint,
            "use_scale_shift_norm": use_scale_shift_norm,
        }

        attn_common_args = {
            "num_heads": num_heads,
            "num_head_channels": num_head_channels,
            "use_checkpoint": use_checkpoint,
        }

        # define encoder part
        len_audio_conv = 1

        ds = 1
        bid = 1
        dilation = 1

        for level, mult in enumerate(channel_mult):
            for block_id in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        out_channels=int(mult * model_channels),
                        audio_dilation=2 ** (dilation % max_dila),
                        video_attention=ds in self.video_attention_resolutions,
                        audio_attention=ds in self.audio_attention_resolutions,
                        num_heads=num_heads,
                        **resblock_common_args,
                    )
                ]

                dilation += len_audio_conv
                ch = int(mult * model_channels)

                if ds in cross_attention_resolutions:
                    ds_i = cross_attention_resolutions.index(ds)
                    layers.append(
                        CrossAttentionBlock(
                            ch,
                            local_window=self.cross_attention_windows[ds_i],
                            window_shift=self.cross_attention_shift,
                            **attn_common_args,
                        )
                    )

                self.input_blocks.append(VATimestepEmbedSequential(*layers))
                bid += 1
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    VATimestepEmbedSequential(
                        ResBlock(
                            ch,
                            out_channels=out_ch,
                            audio_dilation=2 ** (dilation % max_dila),
                            down=True,
                            **resblock_common_args,
                        )
                    )
                )

                dilation += len_audio_conv
                bid += 1
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.out_channels = ch

    def forward(self, v, a, emb: VAEmbs):
        for module in self.input_blocks:
            v, a = module(v, a, emb)

        # apply avg pooling to create a (B, T, C) features for both modalities
        v_feats = v.mean(dim=(3, 4))

        a_feats = a
        if a.ndim == 4:
            a_feats = a_feats.mean(dim=(3,))

        a_feats = a_feats.permute((0, 2, 1))  # (B, C, L) -> (B, L, C)

        return v_feats, a_feats


class VAVitBlocks(FeatureExtractorBase):
    def __init__(
        self, video_cdim, audio_cdim, output_dim, audio_extractor, video_extractor
    ):
        super().__init__()

        self.out_channels = output_dim

        self.audio_extractor = audio_extractor
        self.video_extractor = video_extractor

    def forward(self, v: Tensor, a: Tensor, emb: VAEmbs):
        # a: (B, C, L, M)
        # v: (B, F, C, H, W)

        v = v.permute(0, 2, 1, 3, 4)  # (B, C, F, H, W)
        v = self.video_extractor(v, emb.video)
        v = v.reshape(
            v.shape[0], self.video_extractor.out_temporal_dim, -1, v.shape[-1]
        ).mean(2)

        a = self.audio_extractor(a, emb.audio)
        a = a.reshape(
            a.shape[0], self.audio_extractor.out_temporal_dim, -1, a.shape[-1]
        ).mean(2)

        return v, a


class FusionModel(nn.Module):
    @dataclass
    class ModelConfig:
        fusion_type: str
        num_attn_layers: int
        model_channels: int
        out_channels: int
        use_class_token: bool

    def __init__(
        self,
        fusion_type: str,
        input_channels: int,
        time_embed_dims: int,
        num_attn_layers: int,
        model_channels: int,
        out_channels: int,
        use_class_token: bool,
    ):
        super().__init__()

        # register config
        self.fusion_type = fusion_type
        self.input_channels = input_channels
        self.time_embed_dims = time_embed_dims
        self.num_attn_layers = num_attn_layers
        self.model_channels = model_channels
        # self.out_channels = out_channels -> will be determined later
        self.use_class_token = use_class_token

        # build model
        if fusion_type == "aggregate_time_and_cat_ch":
            self.out_channels = input_channels * 2
        elif self.fusion_type == "interp_time_and_cat_ch_and_apply_transformer":
            assert not (model_channels & 1)
            # input transform
            self.fm_video_in = nn.Linear(
                in_features=input_channels, out_features=model_channels // 2
            )
            self.fm_audio_in = nn.Linear(
                in_features=input_channels, out_features=model_channels // 2
            )
            self.fm_time_in = nn.Linear(
                in_features=time_embed_dims, out_features=model_channels
            )

            # feature matching module
            self.fm = TransformerWrapperNoTokenEmb(
                model_dims=model_channels,
                n_attn_layers=num_attn_layers,
                out_dims=out_channels,
                max_seq_len=128,  # TODO: this may cause an error in some case
                use_pos_emb=True,
                use_class_token=use_class_token,
            )
            self.out_channels = out_channels
        elif self.fusion_type == "cat_time_and_apply_transformer":
            # input transform
            self.fm_video_in = nn.Linear(
                in_features=input_channels, out_features=model_channels
            )
            self.fm_audio_in = nn.Linear(
                in_features=input_channels, out_features=model_channels
            )
            self.fm_time_in = nn.Linear(
                in_features=time_embed_dims, out_features=model_channels
            )

            self.fm = TransformerWrapperNoTokenEmb(
                model_dims=model_channels,
                n_attn_layers=num_attn_layers,
                out_dims=out_channels,
                max_seq_len=128,  # TODO: this may cause an error in some case
                use_pos_emb=True,
                use_class_token=use_class_token,
            )
            self.out_channels = out_channels
        else:
            raise ValueError(f"fusion type '{self.fusion_type}' is not supported.")

    def forward(
        self,
        v_feats: Tensor,  # (B, F, C)
        a_feats: Tensor,  # (B, L, C)
        temb: Tensor,  # (B, Ct)
    ):
        if self.fusion_type == "aggregate_time_and_cat_ch":
            v_feats = v_feats.mean(dim=(1,))  # V: (B, F, C) -> (B, C)
            a_feats = a_feats.mean(dim=(1,))  # A: (B, L, C) -> (B, C)
            last_feat = th.cat([v_feats, a_feats], dim=1)  # (B, 2C)

        elif self.fusion_type == "interp_time_and_cat_ch_and_apply_transformer":
            # apply interp to align the temporal length between audio and video
            max_length = max(v_feats.shape[1], a_feats.shape[1])

            if v_feats.shape[1] < max_length:
                # v_feats = v_feats.repeat_interleave(max_length // v_feats.shape[1], dim=1)

                # interpolate is performed at the last axis. need transpose.
                v_feats = F.interpolate(v_feats.permute(0, 2, 1), max_length).permute(
                    0, 2, 1
                )

            if a_feats.shape[1] < max_length:
                # a_feats = a_feats.repeat_interleave(max_length // a_feats.shape[1], dim=1)

                # interpolate is performed at the last axis. need transpose.
                a_feats = F.interpolate(a_feats.permute(0, 2, 1), max_length).permute(
                    0, 2, 1
                )

            # apply fm input layers to align their channel size
            v_feats = self.fm_video_in(v_feats)
            a_feats = self.fm_audio_in(a_feats)
            fm_temb = self.fm_time_in(temb).unsqueeze(1)

            assert v_feats.shape[1] == a_feats.shape[1]
            av_feats = th.cat([v_feats, a_feats], dim=2)

            # concat time emb at the beginning as done in Frieren
            av_feats = th.cat([fm_temb, av_feats], dim=1)

            last_feat: Tensor = self.fm(av_feats)

            # TODO: improve aggregation method for time axis
            if self.use_class_token:
                last_feat = last_feat[:, 0, :]
            else:
                last_feat = last_feat.mean(dim=(1,))

        elif self.fusion_type == "cat_time_and_apply_transformer":
            # apply fm input layers to align their channel size
            v_feats = self.fm_video_in(v_feats)
            a_feats = self.fm_audio_in(a_feats)
            fm_temb = self.fm_time_in(temb).unsqueeze(1)

            # concat time emb at the beginning as done in Frieren
            av_feats = th.cat([fm_temb, v_feats, a_feats], dim=1)

            last_feat: Tensor = self.fm(av_feats)

            # TODO: improve aggregation method for time axis
            if self.use_class_token:
                last_feat = last_feat[:, 0, :]
            else:
                last_feat = last_feat.mean(dim=(1,))
        else:
            raise ValueError(f"fusion type '{self.fusion_type}' is not supported.")

        return last_feat


class MultimodalDiscriminator(VAResidualPredictorBase):
    """
    The Multimodal Encoder like Discriminator with attention and timestep embedding.

    """

    def __init__(
        self,
        feature_extractor: FeatureExtractorBase,
        fusion_model_conf: FusionModel.ModelConfig,
        video_input_type: str,
        audio_input_type: str,
        time_embed_dims: int,
        text_cond: bool,
        video_text_emb_dims: int,
        audio_text_emb_dims: int,
        last_sigmoid=True,
        ckpt_path: str = None,
    ):
        super().__init__()

        self.video_input_type = video_input_type
        self.audio_input_type = audio_input_type
        self.time_embed_dims = time_embed_dims
        self.text_cond = text_cond
        self.video_text_emb_dim = video_text_emb_dims
        self.audio_text_emb_dim = audio_text_emb_dims
        self.last_sigmoid = last_sigmoid

        assert (
            self.video_input_type == self.audio_input_type == "sample_only"
        ), "Currently only support 'sample_only' for both inputs."

        # embedding handlers
        self.time_embed = TimestepEncoder(
            timestep_dims=time_embed_dims, model_dims=time_embed_dims
        )
        if self.text_cond:
            self.video_text_embed = nn.Linear(video_text_emb_dims, time_embed_dims)
            self.audio_text_embed = nn.Linear(audio_text_emb_dims, time_embed_dims)

        # feature extractor
        self.feature_extractor = feature_extractor

        # fusion model
        self.fusion_model = FusionModel(
            input_channels=self.feature_extractor.out_channels,
            time_embed_dims=time_embed_dims,
            **asdict(fusion_model_conf),
        )

        # output layers
        self.last_linear = nn.Linear(
            in_features=self.fusion_model.out_channels, out_features=1
        )
        # self.last_linear = nn.Sequential(
        #     nn.Linear(in_features=self.fusion_model.out_channels, out_features=64),
        #     nn.Linear(in_features=64, out_features=1)
        # )

        if ckpt_path is not None:
            logger.info("Loading pretrained weights from {}".format(ckpt_path))
            state_dict = th.load(ckpt_path, map_location="cpu", weights_only=True)
            self.load_state_dict(state_dict, strict=True)

    def forward(
        self,
        video: Tensor,
        pred_video_noise,
        audio: Tensor,
        pred_audio_noise,
        timesteps,
        video_text_emb,
        audio_text_emb,
        score_factor,
        video_channel_time_transpose,
        audio_spatial_transpose,
        prob_only=False,
        second=False,
        inference=False,
    ) -> VAResidualPredictorOutputs:
        """
        Apply the model to an input batch.
        :param video: an [N x C x F x H x W] Tensor of inputs.
        :param audio: an [N x C x L x M] or [N x C x L] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: a video output of [N x F x C x H x W] Tensor, an audio output of [N x C x L]
        """
        if not prob_only and not second:
            # this case we need to compute gradient
            with th.enable_grad():
                return self.forward(
                    video,
                    pred_video_noise,
                    audio,
                    pred_audio_noise,
                    timesteps,
                    video_text_emb,
                    audio_text_emb,
                    score_factor=score_factor,
                    video_channel_time_transpose=video_channel_time_transpose,
                    audio_spatial_transpose=audio_spatial_transpose,
                    prob_only=prob_only,
                    second=True,
                    inference=inference,
                )

        if not prob_only:
            video.requires_grad = True
            audio.requires_grad = True

        # preprocess inputs
        v, a = self.setup_inputs(
            video,
            pred_video_noise,
            audio,
            pred_audio_noise,
            timesteps,
            video_channel_time_transpose,
            audio_spatial_transpose,
        )

        # FNO, assume v is [B, F, C, H, W] and a is [B, C, L, M]

        # embedding
        temb = self.time_embed(timesteps)
        if self.text_cond:
            if video_text_emb.ndim == 3:
                video_text_emb = video_text_emb.mean(dim=1)
            if audio_text_emb.ndim == 3:
                audio_text_emb = audio_text_emb.mean(dim=1)

            video_text_emb = self.video_text_embed(video_text_emb.to(dtype=video.dtype))
            audio_text_emb = self.audio_text_embed(audio_text_emb.to(dtype=audio.dtype))
        else:
            video_text_emb = 0
            audio_text_emb = 0

        # shape of each emb is (B, C)
        emb = VAEmbs(video=temb + video_text_emb, audio=temb + audio_text_emb)

        # apply feature extraction blocks; (B, T, C) for both feats.
        v_feats, a_feats = self.feature_extractor(v, a, emb)

        # modality fusion
        fused_feats = self.fusion_model(v_feats, a_feats, temb)

        # prob
        h = self.last_linear(fused_feats).squeeze(1)
        if self.last_sigmoid:
            prob = th.sigmoid(h)
        else:
            prob = h

        logit_exp = th.log(prob + 1e-5) - th.log1p(-prob + 1e-5)
        # logit = th.special.logit(prob, eps=1e-5) -> not working. this provides NaN.

        outs = VAResidualPredictorOutputs(
            pred_video=None,
            pred_audio=None,
            logit_exp=logit_exp,
            logit=h,
            prob=prob,
            pred_grad_scale_audio=1.0,
            pred_grad_scale_video=1.0,
        )

        if prob_only:
            return outs

        # residual prediction can be done by taking gradient
        g_video, g_audio = th.autograd.grad(
            logit_exp,
            (video, audio),
            grad_outputs=th.ones(logit_exp.shape, device=logit_exp.device),
            create_graph=not inference,
        )

        outs.pred_video = g_video * score_factor["video"]
        outs.pred_audio = g_audio * score_factor["audio"]

        return outs
