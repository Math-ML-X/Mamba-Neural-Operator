# try:
#     from layers import *
#     from utils_ft import *
# except:
#     from galerkin_transformer.layers import *
#     from galerkin_transformer.utils_ft import *
from .layers import *
from .utils_ft import *
import copy
import os
import sys
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import MultiheadAttention, TransformerEncoderLayer
from torch.nn.init import constant_, xavier_uniform_
from torchinfo import summary

# current_path = os.path.dirname(os.path.abspath(__file__))
# SRC_ROOT = os.path.dirname(current_path)
# sys.path.append(SRC_ROOT)

ADDITIONAL_ATTR = ['normalizer', 'raw_laplacian', 'return_latent',
                   'residual_type', 'norm_type', 'norm_eps', 'boundary_condition',
                   'upscaler_size', 'downscaler_size', 'spacial_dim', 'spacial_fc',
                   'regressor_activation', 'attn_activation', 
                   'downscaler_activation', 'upscaler_activation',
                   'encoder_dropout', 'decoder_dropout', 'ffn_dropout']

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

class SimpleMambaEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=96,
                 pos_dim=1,
                 n_head=2,
                 dim_feedforward=512,
                 attention_type='fourier',
                 pos_emb=False,
                 layer_norm=True,
                 attn_norm=None,
                 norm_type='layer',
                 norm_eps=None,
                 batch_norm=False,
                 attn_weight=False,
                 xavier_init: float=1e-2,
                 diagonal_weight: float=1e-2,
                 symmetric_init=False,
                 residual_type='add',
                 activation_type='relu',
                 dropout=0.1,
                 ffn_dropout=None,
                 ssm_d_state=16,
                 ssm_ratio=2.0,
                 ssm_rank_ratio=2.0,
                 forward_type="v2",
                 debug=False,
                 ):
        super(SimpleMambaEncoderLayer, self).__init__()

        dropout = default(dropout, 0.05)
        if attention_type in ['linear', 'softmax']:
            dropout = 0.1
        ffn_dropout = default(ffn_dropout, dropout)
        norm_eps = default(norm_eps, 1e-5)
        attn_norm = default(attn_norm, not layer_norm)
        if (not layer_norm) and (not attn_norm):
            attn_norm = True
        norm_type = default(norm_type, 'layer')

        from .mamba_ops import SS2D
        mamba_config = {}
        mamba_config = DictToObject(mamba_config)
        mamba_config.ssm_d_model = d_model
        mamba_config.ssm_d_state = ssm_d_state
        mamba_config.ssm_ratio = ssm_ratio
        mamba_config.ssm_rank_ratio = ssm_rank_ratio
        mamba_config.ssm_dt_rank = "auto"
        mamba_config.ssm_act_layer = "silu"
        mamba_config.ssm_conv = 0
        mamba_config.ssm_conv_bias = False
        mamba_config.ssm_drop_rate = 0.0
        mamba_config.drop_path_rate = 0.0
        mamba_config.forward_type = forward_type
        mamba_config.downsample_version = "none"
        mamba_config.ssm_simple_init = False
        mamba_config.softmax_version = False
        mamba_config.share_proj = False

        self.attn = SS2D(mamba_config)
        # self.attn = SimpleAttention(n_head=n_head,
        #                             d_model=d_model,
        #                             attention_type=attention_type,
        #                             diagonal_weight=diagonal_weight,
        #                             xavier_init=xavier_init,
        #                             symmetric_init=symmetric_init,
        #                             pos_dim=pos_dim,
        #                             norm=attn_norm,
        #                             norm_type=norm_type,
        #                             eps=norm_eps,
        #                             dropout=dropout)
        self.d_model = d_model
        # self.n_head = n_head
        self.pos_dim = pos_dim
        self.add_layer_norm = layer_norm
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(d_model, eps=norm_eps)
            self.layer_norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        dim_feedforward = default(dim_feedforward, 2 * d_model)
        self.ff = FeedForward(in_dim=d_model,
                              dim_feedforward=dim_feedforward,
                              batch_norm=batch_norm,
                              activation=activation_type,
                              dropout=ffn_dropout,
                              )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual_type = residual_type  # plus or minus
        self.add_pos_emb = pos_emb
        if self.add_pos_emb:
            self.pos_emb = PositionalEncoding(d_model)

        self.debug = debug
        self.attn_weight = attn_weight
        self.__name__ = attention_type.capitalize() + 'MambaEncoderLayer'

    def forward(self, x, pos=None, weight=None):
        '''
        - x: node feature, (batch_size, seq_len, n_feats)
        - pos: position coords, needed in every head

        Remark:
            - for n_head=1, no need to encode positional 
            information if coords are in features
        '''
        if self.add_pos_emb:
            x = x.permute((1, 0, 2))
            x = self.pos_emb(x)
            x = x.permute((1, 0, 2))

        if pos is not None and self.pos_dim > 0:
            att_output = self.attn(x)  # encoder no mask
        else:
            att_output = self.attn(x)
        attn_weight = None

        if self.residual_type in ['add', 'plus'] or self.residual_type is None:
            x = x + self.dropout1(att_output)
        else:
            x = x - self.dropout1(att_output)
        if self.add_layer_norm:
            x = self.layer_norm1(x)

        x1 = self.ff(x)
        x = x + self.dropout2(x1)

        if self.add_layer_norm:
            x = self.layer_norm2(x)

        if self.attn_weight:
            return x, attn_weight
        else:
            return x


class SimpleMambaC4EncoderLayer(nn.Module):
    def __init__(self,
                 d_model=96,
                 pos_dim=1,
                 n_head=2,
                 dim_feedforward=512,
                 attention_type='fourier',
                 pos_emb=False,
                 layer_norm=True,
                 attn_norm=None,
                 norm_type='layer',
                 norm_eps=None,
                 batch_norm=False,
                 attn_weight=False,
                 xavier_init: float=1e-2,
                 diagonal_weight: float=1e-2,
                 symmetric_init=False,
                 residual_type='add',
                 activation_type='relu',
                 dropout=0.1,
                 ffn_dropout=None,
                 ssm_d_state=16,
                 ssm_ratio=2.0,
                 ssm_rank_ratio=2.0,
                 forward_type="v2",
                 debug=False,
                 ):
        super(SimpleMambaC4EncoderLayer, self).__init__()

        dropout = default(dropout, 0.05)
        if attention_type in ['linear', 'softmax']:
            dropout = 0.1
        ffn_dropout = default(ffn_dropout, dropout)
        norm_eps = default(norm_eps, 1e-5)
        attn_norm = default(attn_norm, not layer_norm)
        if (not layer_norm) and (not attn_norm):
            attn_norm = True
        norm_type = default(norm_type, 'layer')

        from .mamba_ops_4 import SS2D
        mamba_config = {}
        mamba_config = DictToObject(mamba_config)
        mamba_config.ssm_d_model = d_model
        mamba_config.ssm_d_state = ssm_d_state
        mamba_config.ssm_ratio = ssm_ratio
        mamba_config.ssm_rank_ratio = ssm_rank_ratio
        mamba_config.ssm_dt_rank = "auto"
        mamba_config.ssm_act_layer = "silu"
        mamba_config.ssm_conv = 0
        mamba_config.ssm_conv_bias = False
        mamba_config.ssm_drop_rate = 0.0
        mamba_config.drop_path_rate = 0.0
        mamba_config.forward_type = forward_type
        mamba_config.downsample_version = "none"
        mamba_config.ssm_simple_init = False
        mamba_config.softmax_version = False
        mamba_config.share_proj = False

        self.attn = SS2D(mamba_config)
        # self.attn = SimpleAttention(n_head=n_head,
        #                             d_model=d_model,
        #                             attention_type=attention_type,
        #                             diagonal_weight=diagonal_weight,
        #                             xavier_init=xavier_init,
        #                             symmetric_init=symmetric_init,
        #                             pos_dim=pos_dim,
        #                             norm=attn_norm,
        #                             norm_type=norm_type,
        #                             eps=norm_eps,
        #                             dropout=dropout)
        self.d_model = d_model
        # self.n_head = n_head
        self.pos_dim = pos_dim
        self.add_layer_norm = layer_norm
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(d_model, eps=norm_eps)
            self.layer_norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        dim_feedforward = default(dim_feedforward, 2 * d_model)
        self.ff = FeedForward(in_dim=d_model,
                              dim_feedforward=dim_feedforward,
                              batch_norm=batch_norm,
                              activation=activation_type,
                              dropout=ffn_dropout,
                              )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual_type = residual_type  # plus or minus
        self.add_pos_emb = pos_emb
        if self.add_pos_emb:
            self.pos_emb = PositionalEncoding(d_model)

        self.debug = debug
        self.attn_weight = attn_weight
        self.__name__ = attention_type.capitalize() + 'MambaEncoderLayer'

    def forward(self, x, pos=None, weight=None):
        '''
        - x: node feature, (batch_size, seq_len, n_feats)
        - pos: position coords, needed in every head

        Remark:
            - for n_head=1, no need to encode positional
            information if coords are in features
        '''
        if self.add_pos_emb:
            x = x.permute((1, 0, 2))
            x = self.pos_emb(x)
            x = x.permute((1, 0, 2))

        if pos is not None and self.pos_dim > 0:
            att_output = self.attn(x)  # encoder no mask
        else:
            att_output = self.attn(x)
        attn_weight = None

        if self.residual_type in ['add', 'plus'] or self.residual_type is None:
            x = x + self.dropout1(att_output)
        else:
            x = x - self.dropout1(att_output)
        if self.add_layer_norm:
            x = self.layer_norm1(x)

        x1 = self.ff(x)
        x = x + self.dropout2(x1)

        if self.add_layer_norm:
            x = self.layer_norm2(x)

        if self.attn_weight:
            return x, attn_weight
        else:
            return x


class SimpleTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=96,
                 pos_dim=1,
                 n_head=2,
                 dim_feedforward=512,
                 attention_type='fourier',
                 pos_emb=False,
                 layer_norm=True,
                 attn_norm=None,
                 norm_type='layer',
                 norm_eps=None,
                 batch_norm=False,
                 attn_weight=False,
                 xavier_init: float = 1e-2,
                 diagonal_weight: float = 1e-2,
                 symmetric_init=False,
                 residual_type='add',
                 activation_type='relu',
                 dropout=0.1,
                 ffn_dropout=None,
                 debug=False,
                 ):
        super(SimpleTransformerEncoderLayer, self).__init__()

        dropout = default(dropout, 0.05)
        if attention_type in ['linear', 'softmax']:
            dropout = 0.1
        ffn_dropout = default(ffn_dropout, dropout)
        norm_eps = default(norm_eps, 1e-5)
        attn_norm = default(attn_norm, not layer_norm)
        if (not layer_norm) and (not attn_norm):
            attn_norm = True
        norm_type = default(norm_type, 'layer')

        self.attn = SimpleAttention(n_head=n_head,
                                    d_model=d_model,
                                    attention_type=attention_type,
                                    diagonal_weight=diagonal_weight,
                                    xavier_init=xavier_init,
                                    symmetric_init=symmetric_init,
                                    pos_dim=pos_dim,
                                    norm=attn_norm,
                                    norm_type=norm_type,
                                    eps=norm_eps,
                                    dropout=dropout)
        self.d_model = d_model
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.add_layer_norm = layer_norm
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(d_model, eps=norm_eps)
            self.layer_norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        dim_feedforward = default(dim_feedforward, 2 * d_model)
        self.ff = FeedForward(in_dim=d_model,
                              dim_feedforward=dim_feedforward,
                              batch_norm=batch_norm,
                              activation=activation_type,
                              dropout=ffn_dropout,
                              )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual_type = residual_type  # plus or minus
        self.add_pos_emb = pos_emb
        if self.add_pos_emb:
            self.pos_emb = PositionalEncoding(d_model)

        self.debug = debug
        self.attn_weight = attn_weight
        self.__name__ = attention_type.capitalize() + 'TransformerEncoderLayer'

    def forward(self, x, pos=None, weight=None):
        '''
        - x: node feature, (batch_size, seq_len, n_feats)
        - pos: position coords, needed in every head

        Remark:
            - for n_head=1, no need to encode positional
            information if coords are in features
        '''
        if self.add_pos_emb:
            x = x.permute((1, 0, 2))
            x = self.pos_emb(x)
            x = x.permute((1, 0, 2))

        if pos is not None and self.pos_dim > 0:
            att_output, attn_weight = self.attn(
                x, x, x, pos=pos, weight=weight)  # encoder no mask
        else:
            att_output, attn_weight = self.attn(x, x, x, weight=weight)

        if self.residual_type in ['add', 'plus'] or self.residual_type is None:
            x = x + self.dropout1(att_output)
        else:
            x = x - self.dropout1(att_output)
        if self.add_layer_norm:
            x = self.layer_norm1(x)

        x1 = self.ff(x)
        x = x + self.dropout2(x1)

        if self.add_layer_norm:
            x = self.layer_norm2(x)

        if self.attn_weight:
            return x, attn_weight
        else:
            return x


class DownScaler(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 dropout=0.1,
                 padding=5,
                 downsample_mode='conv',
                 activation_type='silu',
                 interp_size=None,
                 debug=False):
        super(DownScaler, self).__init__()
        '''
        A wrapper for conv2d/interp downscaler
        '''
        if downsample_mode == 'conv':
            self.downsample = nn.Sequential(Conv2dEncoder(in_dim=in_dim,
                                                          out_dim=out_dim,
                                                          activation_type=activation_type,
                                                          debug=debug),
                                            Conv2dEncoder(in_dim=out_dim,
                                                          out_dim=out_dim,
                                                          padding=padding,
                                                          activation_type=activation_type,
                                                          debug=debug))
        elif downsample_mode == 'interp':
            self.downsample = Interp2dEncoder(in_dim=in_dim,
                                              out_dim=out_dim,
                                              interp_size=interp_size,
                                              activation_type=activation_type,
                                              dropout=dropout,
                                              debug=debug)
        else:
            raise NotImplementedError("downsample mode not implemented.")
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        '''
        2D:
            Input: (-1, n, n, in_dim)
            Output: (-1, n_s, n_s, out_dim)
        '''
        n_grid = x.size(1)
        bsz = x.size(0)
        x = x.view(bsz, n_grid, n_grid, self.in_dim)
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x)
        x = x.permute(0, 2, 3, 1)
        return x


class UpScaler(nn.Module):
    def __init__(self, in_dim: int,
                 out_dim: int,
                 hidden_dim=None,
                 padding=2,
                 output_padding=0,
                 dropout=0.1,
                 upsample_mode='conv',
                 activation_type='silu',
                 interp_mode='bilinear',
                 interp_size=None,
                 debug=False):
        super(UpScaler, self).__init__()
        '''
        A wrapper for DeConv2d upscaler or interpolation upscaler
        Deconv: Conv1dTranspose
        Interp: interp->conv->interp
        '''
        hidden_dim = default(hidden_dim, in_dim)
        if upsample_mode in ['conv', 'deconv']:
            self.upsample = nn.Sequential(
                DeConv2dBlock(in_dim=in_dim,
                              out_dim=out_dim,
                              hidden_dim=hidden_dim,
                              padding=padding,
                              output_padding=output_padding,
                              dropout=dropout,
                              activation_type=activation_type,
                              debug=debug),
                DeConv2dBlock(in_dim=in_dim,
                              out_dim=out_dim,
                              hidden_dim=hidden_dim,
                              padding=padding * 2,
                              output_padding=output_padding,
                              dropout=dropout,
                              activation_type=activation_type,
                              debug=debug))
        elif upsample_mode == 'interp':
            self.upsample = Interp2dUpsample(in_dim=in_dim,
                                             out_dim=out_dim,
                                             interp_mode=interp_mode,
                                             interp_size=interp_size,
                                             dropout=dropout,
                                             activation_type=activation_type,
                                             debug=debug)
        else:
            raise NotImplementedError("upsample mode not implemented.")
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        '''
        2D:
            Input: (-1, n_s, n_s, in_dim)
            Output: (-1, n, n, out_dim)
        '''
        x = x.permute(0, 3, 1, 2)
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1)
        return x


class FourierTransformerNew(nn.Module):
    def __init__(self, **kwargs):
        super(FourierTransformerNew, self).__init__()
        self.config = defaultdict(lambda: None, **kwargs)
        self._get_setting()
        self._initialize()
        self.__name__ = self.attention_type.capitalize() + 'Transformer2D'

    def forward(self, node, pos):

        if not self.downscaler_size:
            node = torch.cat([node, pos], dim=-1)
        x = self.downscaler(node)  # ()

        x = self.dpo(x)

        for encoder in self.encoder_layers:
            x = encoder(x, pos)  # x (B, N, C); pos (B, N, 2)

        x = self.upscaler(x)

        x = self.dpo(x)

        x = self.regressor(x, pos=pos)

        return x

    def _initialize(self):
        self._get_scaler()
        self._get_encoder()
        self._get_regressor()
        self.config = dict(self.config)

    def cuda(self, device=None):
        self = super().cuda(device)
        if self.normalizer:
            self.normalizer = self.normalizer.cuda(device)
        return self

    def cpu(self):
        self = super().cpu()
        if self.normalizer:
            self.normalizer = self.normalizer.cpu()
        return self

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if self.normalizer:
            self.normalizer = self.normalizer.to(*args, **kwargs)
        return self

    def print_config(self):
        for a in self.config.keys():
            if not a.startswith('__'):
                print(f"{a}: \t", getattr(self, a))

    @staticmethod
    def _initialize_layer(layer, gain=1e-2):
        for param in layer.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=gain)
            else:
                constant_(param, 0)

    def _get_setting(self):
        all_attr = list(self.config.keys()) + ADDITIONAL_ATTR
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.dim_feedforward = default(self.dim_feedforward, 2 * self.n_hidden)
        self.dropout = default(self.dropout, 0.05)
        self.dpo = nn.Dropout(self.dropout)
        self.attention_types = ['fourier', 'integral', 'local', 'global',
                                'cosine', 'galerkin', 'linear', 'softmax',
                                'standardattention'
                                ]

    def _get_scaler(self):

        self.downscaler = Identity(in_features=self.node_feats + self.spacial_dim,
                                   out_features=self.n_hidden)

        self.upscaler = Identity()

    def _get_encoder(self):
        if self.attention_type in self.attention_types:
            encoder_layer = SimpleTransformerEncoderLayer(d_model=self.n_hidden,
                                                          n_head=self.n_head,
                                                          attention_type=self.attention_type,
                                                          dim_feedforward=self.dim_feedforward,
                                                          layer_norm=self.layer_norm,
                                                          attn_norm=self.attn_norm,
                                                          batch_norm=self.batch_norm,
                                                          pos_dim=self.pos_dim,
                                                          xavier_init=self.xavier_init,
                                                          diagonal_weight=self.diagonal_weight,
                                                          symmetric_init=self.symmetric_init,
                                                          attn_weight=self.return_attn_weight,
                                                          dropout=self.encoder_dropout,
                                                          ffn_dropout=self.ffn_dropout,
                                                          norm_eps=self.norm_eps,
                                                          debug=self.debug)
        elif self.attention_type == 'mamba':
            # from .model_mamba import MambaTransformerEncoderLayer
            encoder_layer = SimpleMambaEncoderLayer(d_model=self.n_hidden,
                                                    n_head=self.n_head,
                                                    attention_type=self.attention_type,
                                                    dim_feedforward=self.dim_feedforward,
                                                    layer_norm=self.layer_norm,
                                                    attn_norm=self.attn_norm,
                                                    batch_norm=self.batch_norm,
                                                    pos_dim=self.pos_dim,
                                                    xavier_init=self.xavier_init,
                                                    diagonal_weight=self.diagonal_weight,
                                                    symmetric_init=self.symmetric_init,
                                                    attn_weight=self.return_attn_weight,
                                                    dropout=self.encoder_dropout,
                                                    ffn_dropout=self.ffn_dropout,
                                                    norm_eps=self.norm_eps,
                                                    debug=self.debug,
                                                    ssm_d_state=self.ssm_d_state,
                                                    ssm_ratio=self.ssm_ratio,
                                                    ssm_rank_ratio=self.ssm_rank_ratio,
                                                    forward_type=self.forward_type,
                                                    )
        elif self.attention_type == 'mamba_c4':
            # from .model_mamba import MambaTransformerEncoderLayer
            encoder_layer = SimpleMambaC4EncoderLayer(d_model=self.n_hidden,
                                                    n_head=self.n_head,
                                                    attention_type=self.attention_type,
                                                    dim_feedforward=self.dim_feedforward,
                                                    layer_norm=self.layer_norm,
                                                    attn_norm=self.attn_norm,
                                                    batch_norm=self.batch_norm,
                                                    pos_dim=self.pos_dim,
                                                    xavier_init=self.xavier_init,
                                                    diagonal_weight=self.diagonal_weight,
                                                    symmetric_init=self.symmetric_init,
                                                    attn_weight=self.return_attn_weight,
                                                    dropout=self.encoder_dropout,
                                                    ffn_dropout=self.ffn_dropout,
                                                    norm_eps=self.norm_eps,
                                                    debug=self.debug,
                                                    ssm_d_state=self.ssm_d_state,
                                                    ssm_ratio=self.ssm_ratio,
                                                    ssm_rank_ratio=self.ssm_rank_ratio,
                                                    forward_type=self.forward_type,
                                                    )
        elif self.attention_type == 'official':
            encoder_layer = TransformerEncoderLayer(d_model=self.n_hidden + self.pos_dim * self.n_head,
                                                    nhead=self.n_head,
                                                    dim_feedforward=self.dim_feedforward,
                                                    dropout=self.encoder_dropout,
                                                    batch_first=True,
                                                    layer_norm_eps=self.norm_eps,
                                                    )
        else:
            raise NotImplementedError("encoder type not implemented.")
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])

    def _get_regressor(self):
        if self.decoder_type == 'pointwise':
            raise NotImplementedError("pointwise regressor not implemented.")
        elif self.decoder_type == 'ifft2':
            self.regressor = SpectralRegressor(in_dim=self.n_hidden,
                                               n_hidden=self.freq_dim,
                                               freq_dim=self.freq_dim,
                                               out_dim=self.n_targets,
                                               num_spectral_layers=self.num_regressor_layers,
                                               modes=self.fourier_modes,
                                               spacial_dim=self.spacial_dim,
                                               spacial_fc=self.spacial_fc,
                                               activation=self.regressor_activation,
                                               last_activation=self.last_activation,
                                               dropout=self.decoder_dropout,
                                               return_latent=self.return_latent,
                                               debug=self.debug
                                               )

class SpectralRegressor(nn.Module):
    def __init__(self, in_dim,
                 n_hidden,
                 freq_dim,
                 out_dim,
                 modes: int,
                 num_spectral_layers: int = 2,
                 n_grid=None,
                 dim_feedforward=None,
                 spacial_fc=False,
                 spacial_dim=2,
                 return_freq=False,
                 return_latent=False,
                 normalizer=None,
                 activation='silu',
                 last_activation=True,
                 dropout=0.1,
                 debug=False):
        super(SpectralRegressor, self).__init__()
        '''
        A wrapper for both SpectralConv1d and SpectralConv2d
        Ref: Li et 2020 FNO paper
        https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
        A new implementation incoporating all spacial-based FNO
        in_dim: input dimension, (either n_hidden or spacial dim)
        n_hidden: number of hidden features out from attention to the fourier conv
        '''
        if spacial_dim == 2:  # 2d, function + (x,y)
            spectral_conv = SpectralConv2d
        elif spacial_dim == 1:  # 1d, function + x
            spectral_conv = SpectralConv1d
        else:
            raise NotImplementedError("3D not implemented.")
        activation = default(activation, 'silu')
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        dropout = default(dropout, 0.1)
        self.spacial_fc = spacial_fc  # False in Transformer
        if self.spacial_fc:
            self.fc = nn.Linear(in_dim + spacial_dim, n_hidden)
        self.spectral_conv = nn.ModuleList([spectral_conv(in_dim=n_hidden,
                                                          out_dim=freq_dim,
                                                          n_grid=n_grid,
                                                          modes=modes,
                                                          dropout=dropout,
                                                          activation=activation,
                                                          return_freq=return_freq,
                                                          debug=debug)])
        for _ in range(num_spectral_layers - 1):
            self.spectral_conv.append(spectral_conv(in_dim=freq_dim,
                                                    out_dim=freq_dim,
                                                    n_grid=n_grid,
                                                    modes=modes,
                                                    dropout=dropout,
                                                    activation=activation,
                                                    return_freq=return_freq,
                                                    debug=debug))
        if not last_activation:
            self.spectral_conv[-1].activation = Identity()

        self.n_grid = n_grid  # dummy for debug
        self.dim_feedforward = default(dim_feedforward, 2*spacial_dim*freq_dim)
        self.regressor = nn.Sequential(
            nn.Linear(freq_dim, self.dim_feedforward),
            self.activation,
            nn.Linear(self.dim_feedforward, out_dim),
        )
        self.normalizer = normalizer
        self.return_freq = return_freq
        self.return_latent = return_latent
        self.debug = debug

    def forward(self, x, pos=None):
        '''
        2D:
            Input: (-1, n, n, in_features)
            Output: (-1, n, n, n_targets)
        1D:
            Input: (-1, n, in_features)
            Output: (-1, n, n_targets)
        '''

        if self.spacial_fc:
            x = torch.cat([x, pos], dim=-1)
            x = self.fc(x)

        for layer in self.spectral_conv:
            x = layer(x)

        x = self.regressor(x)

        return x


if __name__ == '__main__':
    pass
