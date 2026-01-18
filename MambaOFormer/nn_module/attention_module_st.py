import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, orthogonal_
# from torch_cluster import fps
# helpers


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1/64, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]


class CrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,  # ['fourier', 'galerkin']
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 init_params=True,
                 relative_emb=False,
                 scale=1.,
                 init_method='orthogonal',  # ['xavier', 'orthogonal']
                 init_gain=None,
                 relative_emb_dim=2,
                 min_freq=1 / 64,
                 cat_pos=False,
                 pos_dim=2,
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.attn_type = attn_type

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        if not cat_pos:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim + pos_dim*heads, dim),
                nn.Dropout(dropout)
            )

        if init_gain is None:
            self.init_gain = 1. / dim_head
            self.diagonal_weight = 1. / dim_head
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain
        self.init_method = init_method
        if init_params:
            self._init_params()

        self.cat_pos = cat_pos
        self.pos_dim = pos_dim

        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        if relative_emb:
            self.emb_module = RotaryEmbedding(dim_head // self.relative_emb_dim, min_freq=min_freq, scale=scale)

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise Exception('Unknown initialization')

        for layer in [self.to_q, self.to_k, self.to_v]:
            for param in layer.parameters():
                if param.ndim > 1:
                    init_fn(param, gain=1. / self.dim_head)

    def forward(self, x, z, x_pos=None, z_pos=None):
        q = self.to_q(x)
        k = self.to_k(z)
        v = self.to_v(z)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        if self.relative_emb:
            if self.relative_emb_dim == 2:

                x_freqs_x = self.emb_module.forward(x_pos[..., 0], x.device)
                x_freqs_y = self.emb_module.forward(x_pos[..., 1], x.device)
                x_freqs_x = repeat(x_freqs_x, 'b n d -> b h n d', h=q.shape[1])
                x_freqs_y = repeat(x_freqs_y, 'b n d -> b h n d', h=q.shape[1])

                z_freqs_x = self.emb_module.forward(z_pos[..., 0], z.device)
                z_freqs_y = self.emb_module.forward(z_pos[..., 1], z.device)
                z_freqs_x = repeat(z_freqs_x, 'b n d -> b h n d', h=q.shape[1])
                z_freqs_y = repeat(z_freqs_y, 'b n d -> b h n d', h=q.shape[1])

                q = apply_2d_rotary_pos_emb(q, x_freqs_x, x_freqs_y)
                k = apply_2d_rotary_pos_emb(k, z_freqs_x, z_freqs_y)

            elif self.relative_emb_dim == 1:
                assert x_pos.shape[-1] == 1 and z_pos.shape[-1] == 1
                x_freqs = self.emb_module.forward(x_pos[..., 0], x.device)
                x_freqs = repeat(x_freqs, 'b n d -> b h n d', h=q.shape[1])

                z_freqs = self.emb_module.forward(z_pos[..., 0], x.device)
                z_freqs = repeat(z_freqs, 'b n d -> b h n d', h=q.shape[1])

                q = apply_rotary_pos_emb(q, x_freqs)  # query from x domain
                k = apply_rotary_pos_emb(k, z_freqs)  # key from z domain
            else:
                raise Exception('Currently doesnt support relative embedding > 2 dimensions')
        elif self.cat_pos:
            assert x_pos.size(-1) == self.pos_dim and z_pos.size(-1) == self.pos_dim
            x_pos = x_pos.unsqueeze(1)
            x_pos = x_pos.repeat([1, self.heads, 1, 1])
            q = torch.cat([x_pos, q], dim=-1)

            z_pos = z_pos.unsqueeze(1)
            z_pos = z_pos.repeat([1, self.heads, 1, 1])
            k = torch.cat([z_pos, k], dim=-1)
            v = torch.cat([z_pos, v], dim=-1)

        dots = torch.matmul(q, k.transpose(-1, -2)) / (self.dim_head ** 0.5)
        attn = F.softmax(dots, dim=-1)

        weighted_values = torch.matmul(attn, v)
        out = rearrange(weighted_values, 'b h n d -> b n (h d)')

        return self.to_out(out)


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    # split t into first half and second half
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y = t[..., :d//2], t[..., d//2:]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)
