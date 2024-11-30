import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional
from Module import TSFM
from Module.MHCB import MHCB
from Module.CBAM import CBAMBlock
from Module.SLC import SLC
from torch_mimicry.nets.infomax_gan import infomax_gan_base
from torch_mimicry.modules.layers import SNConv2d, SNLinear
from torch_mimicry.modules.resblocks import DBlockOptimized, DBlock, GBlock

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

#####
import torch
import torch.nn as nn

class fuGenerator(nn.Module):

    def __init__(self, in_chans, output_nc, patch_size=1, embed_dim=24, depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24), window_size=7, qkv_bias=True, drop_rate=0,
                 attn_drop_rate=0, drop_path_rate=0., norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, HFF_dp=0.,
                 conv_depths=(2, 2, 2, 2), conv_dims=(24, 48, 96, 192), conv_drop_path_rate=0.,
                 conv_head_init_scale: float = 1., **kwargs):
        super().__init__()

        ###### Local Branch Setting #######

        self.downsample_layers = nn.ModuleList()   # stem + 3 stage downsample
        stem = nn.Sequential(nn.Conv2d(in_chans, conv_dims[0], kernel_size=1, stride=1),
                             LayerNorm(conv_dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        # stage2-4 downsample
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(conv_dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(conv_dims[i], conv_dims[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, conv_drop_path_rate, sum(conv_depths))]
        cur = 0

        # Build stacks of blocks in each stage
        for i in range(4):
            stage = nn.Sequential(
                *[Local_block(dim=conv_dims[i], drop_rate=dp_rates[cur + j])
                  for j in range(conv_depths[i])]
            )
            self.stages.append((stage))
            cur += conv_depths[i]

        ###### Global Branch Setting ######

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        # The channels of stage4 output feature matrix
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        i_layer = 0
        self.layers1 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                  depth=depths[i_layer],
                                  num_heads=num_heads[i_layer],
                                  window_size=window_size,
                                  qkv_bias=qkv_bias,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  downsample=PatchMerging if (i_layer > 0) else None,
                                  use_checkpoint=use_checkpoint)

        i_layer = 1
        self.layers2 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                  depth=depths[i_layer],
                                  num_heads=num_heads[i_layer],
                                  window_size=window_size,
                                  qkv_bias=qkv_bias,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  downsample=PatchMerging if (i_layer > 0) else None,
                                  use_checkpoint=use_checkpoint)

        i_layer = 2
        self.layers3 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                  depth=depths[i_layer],
                                  num_heads=num_heads[i_layer],
                                  window_size=window_size,
                                  qkv_bias=qkv_bias,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  downsample=PatchMerging if (i_layer > 0) else None,
                                  use_checkpoint=use_checkpoint)

        i_layer = 3
        self.layers4 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                  depth=depths[i_layer],
                                  num_heads=num_heads[i_layer],
                                  window_size=window_size,
                                  qkv_bias=qkv_bias,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  downsample=PatchMerging if (i_layer > 0) else None,
                                  use_checkpoint=use_checkpoint)
        self.att1 = MHCB(channel=24, reduction=16, kernel_size=3)
        self.att2 = MHCB(channel=48, reduction=16, kernel_size=3)
        self.att3 = MHCB(channel=96, reduction=16, kernel_size=3)
        self.att4 = MHCB(channel=192, reduction=16, kernel_size=3)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        ###### Hierarchical Feature Fusion Block Setting #######

        self.fu1 = HFF_block(ch_1=24, ch_2=24, r_2=16, ch_int=24, ch_out=24, drop_rate=HFF_dp)
        self.fu2 = HFF_block(ch_1=48, ch_2=48, r_2=16, ch_int=48, ch_out=48, drop_rate=HFF_dp)
        self.fu3 = HFF_block(ch_1=96, ch_2=96, r_2=16, ch_int=96, ch_out=96, drop_rate=HFF_dp)
        self.fu4 = HFF_block(ch_1=192, ch_2=192, r_2=16, ch_int=192, ch_out=192, drop_rate=HFF_dp)

        self.decoder1 = nn.Sequential(nn.ConvTranspose2d(conv_dims[-1], conv_dims[-2], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(conv_dims[-2]),
            nn.ReLU(inplace=True))
        self.decoder2 = nn.Sequential(nn.ConvTranspose2d(conv_dims[-2], conv_dims[-3], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(conv_dims[-3]),
            nn.ReLU(inplace=True))
        self.decoder3 = nn.Sequential(nn.ConvTranspose2d(conv_dims[-3], 1, kernel_size=4, stride=4, padding=0),  # Set output channels to 1
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),  # Upsample spatial dimensions
            )
        self.decoder4 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=1),  # Change the number of channels to 1
            nn.Tanh())

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(conv_dims[-1], conv_dims[-2], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(conv_dims[-2]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(conv_dims[-2], conv_dims[-3], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(conv_dims[-3]),
            nn.ReLU(inplace=True),
            # Continue adding decoder layers
            nn.ConvTranspose2d(conv_dims[-3], 1, kernel_size=4, stride=4, padding=0),  # Set output channels to 1
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),  # Upsample spatial dimensions
            nn.Conv2d(1, 1, kernel_size=1),  # Change the number of channels to 1
            nn.Tanh()
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward(self, imgs):

        ###### Global Branch ######
        x_s, H, W = self.patch_embed(imgs)
        x_s = self.pos_drop(x_s)
        x_s_1, H, W = self.layers1(x_s, H, W)
        x_s_2, H, W = self.layers2(x_s_1, H, W)
        x_s_3, H, W = self.layers3(x_s_2, H, W)
        x_s_4, H, W = self.layers4(x_s_3, H, W)

        # [B,L,C] ---> [B,C,H,W]
        x_s_1 = torch.transpose(x_s_1, 1, 2)
        x_s_1 = x_s_1.view(x_s_1.shape[0], -1, 256, 256)
        x_s_2 = torch.transpose(x_s_2, 1, 2)
        x_s_2 = x_s_2.view(x_s_2.shape[0], -1, 128, 128)
        x_s_3 = torch.transpose(x_s_3, 1, 2)
        x_s_3 = x_s_3.view(x_s_3.shape[0], -1, 64, 64)
        x_s_4 = torch.transpose(x_s_4, 1, 2)
        x_s_4 = x_s_4.view(x_s_4.shape[0], -1, 32, 32)

        ###### Local Branch ######
        x_c = self.downsample_layers[0](imgs)
        x_c_a1 = self.att1(x_c)
        x_c_a1 = self.stages[0](x_c_a1)

        x_c = self.downsample_layers[1](x_c_a1)
        x_c_a2 = self.att2(x_c)
        x_c_a2 = self.stages[1](x_c_a2)

        x_c = self.downsample_layers[2](x_c_a2)
        x_c_a3 = self.att3(x_c)
        x_c_a3 = self.stages[2](x_c_a3)

        x_c = self.downsample_layers[3](x_c_a3)
        x_c_a4 = self.att4(x_c)
        x_c_a4 = self.stages[3](x_c_a4)

        ###### Hierarchical Feature Fusion Path ######
        x_f_1 = self.fu1(x_c_a1, x_s_1, None)
        # print(x_f_1.size())#[1, 24, 256, 256]
        x_f_2 = self.fu2(x_c_a2, x_s_2, x_f_1)
        # print(x_f_2.size())#[1, 48, 128, 128]
        x_f_3 = self.fu3(x_c_a3, x_s_3, x_f_2)
        x_f_sls2_3 = self.att3(x_f_3)
        # print(x_f_3.size())#[1, 96, 64, 64]
        x_f_4 = self.fu4(x_c_a4, x_s_4, x_f_3)
        # print(x_f_4.size())#[1, 192, 32, 32]

        x_dec_1 = self.decoder1(x_f_4)
        # print(x_dec_1.size())#[1, 96, 64, 64]
        x_dec_2 = self.decoder2(x_dec_1 + x_f_sls2_3)
        # print(x_dec_2.size())#[1, 48, 128, 128]
        x_dec_3 = self.decoder3(x_dec_2)
        # print(x_dec_3.size())#[1, 1, 256, 256]
        x_dec_4 = self.decoder4(x_dec_3)
        # print(x_dec_4.size())
        # x_dec = self.decoder(x_f_4)

        return x_dec_4


##### Local Feature Block Component #####

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Local_block(nn.Module):
    r""" Local Feature Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_rate=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv = nn.Linear(dim, dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        x = shortcut + self.drop_path(x)
        return x

# Hierachical Feature Fusion Block
class HFF_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(HFF_block, self).__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(ch_2, ch_2 // r_2, 1,bias=False),
            nn.ReLU(),
            nn.Conv2d(ch_2 // r_2, ch_2, 1,bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.W_l = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.Updim = Conv(ch_int//2, ch_int, 1, bn=True, relu=True)
        self.norm1 = LayerNorm(ch_int * 3, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first")
        self.W3 = Conv(ch_int * 3, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int * 2, ch_int, 1, bn=True, relu=False)

        self.gelu = nn.GELU()

        self.residual = IRMLP(ch_1 + ch_2 + ch_int, ch_out)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, l, g, f):

        W_local = self.W_l(l)   # local feature from Local Feature Block
        W_global = self.W_g(g)   # global feature from Global Feature Block
        if f is not None:
            W_f = self.Updim(f)
            W_f = self.Avg(W_f)
            shortcut = W_f
            X_f = torch.cat([W_f, W_local, W_global], 1)
            X_f = self.norm1(X_f)
            X_f = self.W3(X_f)
            X_f = self.gelu(X_f)
        else:
            shortcut = 0
            X_f = torch.cat([W_local, W_global], 1)
            X_f = self.norm2(X_f)
            X_f = self.W(X_f)
            X_f = self.gelu(X_f)

        # spatial attention for ConvNeXt branch
        l_jump = l
        max_result, _ = torch.max(l, dim=1, keepdim=True)
        avg_result = torch.mean(l, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        l = self.spatial(result)
        l = self.sigmoid(l) * l_jump

        # channel attetion for transformer branch
        g_jump = g
        max_result=self.maxpool(g)
        avg_result=self.avgpool(g)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        g = self.sigmoid(output) * g_jump

        fuse = torch.cat([g, l, X_f], 1)
        fuse = self.norm3(fuse)
        fuse = self.residual(fuse)
        fuse = shortcut + self.drop_path(fuse)
        return fuse

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

#### Inverted Residual MLP
class IRMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(IRMLP, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(inp_dim)

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out

####### Shift Window MSA #############

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

### Global Feature Block
class Global_block(nn.Module):
    r""" Global Feature Block from modified Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = act_layer()

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:

            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = self.fc1(x)
        x = self.act(x)
        x = shortcut + self.drop_path(x)

        return x

class BasicLayer(nn.Module):
    """
    Downsampling and Global Feature Block for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            Global_block(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size

        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):

        if self.downsample is not None:
            x = self.downsample(x, H, W)         #patch merging stage2 in [6,3136,96] out [6,784,192]
            H, W = (H + 1) // 2, (W + 1) // 2

        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:                  # global block
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        return x, H, W

def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # downsample patch_size times
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        dim = dim//2
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x

def HiFuse_Tiny(num_classes: int):
    model = main_model(depths=(2, 2, 2, 2),
                     conv_depths=(2, 2, 2, 2),
                     num_classes=num_classes)
    return model


def HiFuse_Small(num_classes: int):
    model = main_model(depths=(2, 2, 6, 2),
                     conv_depths=(2, 2, 6, 2),
                     num_classes=num_classes)
    return model

def HiFuse_Base(num_classes: int):
    model = main_model(depths=(2, 2, 18, 2),
                     conv_depths=(2, 2, 18, 2),
                     num_classes=num_classes)
    return model

class Discriminator(infomax_gan_base.BaseDiscriminator):
    def __init__(self, input_nc, nrkhs=1024, ndf=1024, **kwargs):
        super().__init__(nrkhs=nrkhs, ndf=ndf, **kwargs)
        self.input_nc = input_nc  # 新增的输入通道数参数

        # Decide activation used
        self.activation = nn.ReLU(True)

        self.att1 = TSFM(channel=128, d_model=64)
        self.att2 = TSFM(channel=256, d_model=32)
        self.att3 = TSFM(channel=512, d_model=16)
        self.att4 = TSFM(channel=1024, d_model=8)

        # ----------------
        #   GAN Layers
        # ----------------
        self.local_feat_block1 = nn.Sequential(DBlockOptimized(self.input_nc, self.ndf >> 4),
                                               DBlock(self.ndf >> 4, self.ndf >> 3, downsample=True))
        self.local_feat_block2 = nn.Sequential(DBlock(self.ndf >> 3, self.ndf >> 2, downsample=True))
        self.local_feat_block3 = nn.Sequential(DBlock(self.ndf >> 2, self.ndf >> 1, downsample=True))
        self.local_feat_block4 = nn.Sequential(DBlock(self.ndf >> 1, self.ndf, downsample=True))
        self.local_feat_blocks = nn.Sequential(
            DBlockOptimized(self.input_nc, self.ndf >> 4),  # 使用 input_nc 来初始化第一个卷积层
            DBlock(self.ndf >> 4, self.ndf >> 3, downsample=True),
            DBlock(self.ndf >> 3, self.ndf >> 2, downsample=True),
            DBlock(self.ndf >> 2, self.ndf >> 1, downsample=True),
            DBlock(self.ndf >> 1, self.ndf, downsample=True))

        self.global_feat_blocks = nn.Sequential(
            DBlock(self.ndf, self.ndf, downsample=False))

        self.linear = SNLinear(self.ndf, 1)
        nn.init.xavier_uniform_(self.linear.weight.data, 1.0)

        # --------------------
        #   InfoMax Layers
        # --------------------
        # Critic network layers for local features
        self.local_nrkhs_a = SNConv2d(self.ndf, self.ndf, 1, 1, 0)
        self.local_nrkhs_b = SNConv2d(self.ndf, self.nrkhs, 1, 1, 0)
        self.local_nrkhs_sc = SNConv2d(self.ndf, self.nrkhs, 1, 1, 0)

        nn.init.xavier_uniform_(self.local_nrkhs_a.weight.data, 1.0)
        nn.init.xavier_uniform_(self.local_nrkhs_b.weight.data, 1.0)
        nn.init.xavier_uniform_(self.local_nrkhs_sc.weight.data, 1.0)

        # Critic network layers for global features
        self.global_nrkhs_a = SNLinear(self.ndf, self.ndf)
        self.global_nrkhs_b = SNLinear(self.ndf, self.nrkhs)
        self.global_nrkhs_sc = SNLinear(self.ndf, self.nrkhs)

        nn.init.xavier_uniform_(self.global_nrkhs_a.weight.data, 1.0)
        nn.init.xavier_uniform_(self.global_nrkhs_b.weight.data, 1.0)
        nn.init.xavier_uniform_(self.global_nrkhs_sc.weight.data, 1.0)

    def forward(self, x):
        h = x

        # Get features
        local_feat1 = self.local_feat_block1(h)  # (N, C, H, W)
        local_feat_a1 = self.att1(local_feat1)
        # print(local_feat1.size())
        local_feat2 = self.local_feat_block2(local_feat_a1)
        # print(local_feat2.size())
        local_feat_a2 = self.att2(local_feat2)
        local_feat3 = self.local_feat_block3(local_feat_a2)
        # print(local_feat3.size())
        local_feat_a3 = self.att3(local_feat3)
        local_feat = self.local_feat_block4(local_feat_a3)
        # print(local_feat4.size())
        # local_feat = self.att4(local_feat4)

        global_feat = self.global_feat_blocks(local_feat)
        global_feat = self.activation(global_feat)
        global_feat = torch.sum(global_feat, dim=(2, 3))

        # GAN task output
        output = self.linear(global_feat)

        return output, local_feat, global_feat

class Compute_InfoLoss():
    def __init__(self, nrkhs = 1024):
        self.nrkhs = nrkhs

    def infonce_loss(self, l, m):
        N, units, n_locals = l.size()
        _, _, n_multis = m.size()

        # First we make the input tensors the right shape.
        l_p = l.permute(0, 2, 1)
        m_p = m.permute(0, 2, 1)

        l_n = l_p.reshape(-1, units)
        m_n = m_p.reshape(-1, units)

        # Inner product for positive samples. Outer product for negative.
        u_p = torch.matmul(l_p, m).unsqueeze(2)
        u_n = torch.mm(m_n, l_n.t())
        u_n = u_n.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

        # Mask the diagonal part of the negative tensor.
        mask = torch.eye(N)[:, :, None, None].to(l.device)
        n_mask = 1 - mask

        # Masking is done by shifting the diagonal before exp.
        u_n = (n_mask * u_n) - (10. * (1 - n_mask))  # mask out "self" examples
        u_n = u_n.reshape(N, N * n_locals, n_multis).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

        # Concatenate the positive along the class dimension before performing log softmax.
        pred_lgt = torch.cat([u_p, u_n], dim=2)
        pred_log = F.log_softmax(pred_lgt, dim=2)

        # The positive score is the first element of the log softmax.
        loss = -pred_log[:, :, 0].mean()

        return loss

    def compute_infomax_loss(self, local_feat, global_feat):
        if local_feat.shape[1] != self.nrkhs:
            raise ValueError(
                "Features have not been projected. Expected {} dim but got {} instead"
                .format(self.nrkhs, local_feat.shape[1]))

        # Prepare shapes for local and global features.
        local_feat = torch.flatten(local_feat, start_dim=2, end_dim=3)  # (N, C, H, W) --> (N, C, H*W)
        global_feat = torch.unsqueeze(global_feat, 2)  # (N, C) --> (N, C, 1)

        # Compute infomax loss and scale
        loss = self.infonce_loss(l=local_feat, m=global_feat)

        # loss = scale * loss

        return loss