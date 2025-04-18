from abc import abstractmethod

import torch
from diffusers.models.embeddings import TimestepEmbedding, get_timestep_embedding
from torch import Tensor, nn


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):  #
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):  #
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class TimestepEncoder(nn.Module):
    def __init__(self, timestep_dims, model_dims=None):
        super().__init__()

        self.timestep_dims = timestep_dims
        self.model_dims = timestep_dims if model_dims is None else model_dims
        self.projection = TimestepEmbedding(
            self.timestep_dims, self.model_dims, post_act_fn=None
        )

    def forward(self, timesteps: Tensor):
        t_emb = get_timestep_embedding(timesteps, self.timestep_dims)
        t_proj = self.projection(t_emb)
        return t_proj


class TimestepModulatedNormalization(TimestepBlock):
    def __init__(
        self,
        t_emb_dims: int,
        model_dims: int,
        norm_module: nn.Module,
        channel_dim: int,
        mode="scale_shift",
    ):
        super().__init__()

        assert mode in ["scale_shift", "add"]
        self.mode = mode
        self.norm = norm_module
        self.channel_dim = channel_dim
        proj_out_dims = model_dims * 2 if mode == "scale_shift" else model_dims
        self.projection = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dims, proj_out_dims))

    def bc_temb(self, t_emb: Tensor, target: Tensor):
        # broadcast temb so that it can be added to the target tensor.
        if len(target.shape) == 2:
            return t_emb

        B, C = t_emb.shape
        s = [1 for _ in range(target.ndim)]
        s[0] = B
        s[self.channel_dim] = C

        return t_emb.reshape(s)

    def forward(self, h: Tensor, t_emb: Tensor):
        # project t_emb
        t_emb = self.projection(t_emb)

        # modulate hidden
        if self.mode == "scale_shift":
            h = self.norm(h)
            scale, shift = torch.chunk(t_emb, 2, dim=1)
            h = h * (1 + self.bc_temb(scale, h)) + self.bc_temb(shift, h)
        elif self.mode == "add":
            h += self.bc_temb(t_emb, h)
            h = self.norm(h)

        return h
