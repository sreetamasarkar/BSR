###################################

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
# from timm.models.vision_transformer import PatchEmbed, HybridEmbed, Block, DropPath
from timm.models.vision_transformer import PatchEmbed, Block, DropPath
import math
from .helpers import complement_idx
from .layers import DropPath, to_2tuple, trunc_normal_
from os.path import join as pjoin
from scipy import ndimage

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def topkmask(feature, sparsity=0.5):
    # import pdb; pdb.set_trace()
    B, N, C = feature.shape
    feature_flat = feature.view(B, -1)
    value, _ = torch.topk(feature_flat, int(sparsity * feature_flat.shape[1]), dim=1)
    min_value = value[:,-1].unsqueeze(-1).unsqueeze(-1).expand(-1, N, C) # min value to keep for each feature map
    return (feature > min_value) + 0

class GELUSparse(nn.GELU):
    def forward(self, x, sparsity=0.5):
        mask = topkmask(x.abs(), sparsity=sparsity) # Top-K abs values
        mask_inv = torch.ones_like(mask) - mask # Inverse mask
        out = x * mask
        out = super().forward(out) + x * mask_inv # Apply GELU to large values only
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MainAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., keep_rate=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.keep_rate = keep_rate
        # assert 0 < keep_rate <= 1, "keep_rate must > 0 and <= 1, got {0}".format(keep_rate)

    def forward(self, x, keep_rate=0.7, main_branch_ratio=1.,  return_idx=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Compute top tokens by class attention
        idx = None
        if (keep_rate < 1. or main_branch_ratio < 1.) and return_idx:
            left_tokens = math.ceil(keep_rate * (N - 1))
            cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]
            return x, idx, index, cls_attn
        return x, None, None, None
    
class MainBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, fuse_token=False, main_branch_ratio=1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MainAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.fuse_token = fuse_token
        # self.main_branch_ratio = main_branch_ratio

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            hidden_size  = self.attn.qkv.weight.shape[-1]
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(hidden_size, hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(hidden_size, hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(hidden_size, hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(hidden_size, hidden_size).t()
            qkv_weight = torch.cat([query_weight, key_weight, value_weight], dim=0)
    
            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)
            qkv_bias = torch.cat([query_bias, key_bias, value_bias], dim=0)

            # self.attn.query.weight.copy_(query_weight)
            # self.attn.key.weight.copy_(key_weight)
            # self.attn.value.weight.copy_(value_weight)
            self.attn.qkv.weight.copy_(qkv_weight)
            self.attn.proj.weight.copy_(out_weight)
            # self.attn.query.bias.copy_(query_bias)
            # self.attn.key.bias.copy_(key_bias)
            # self.attn.value.bias.copy_(value_bias)
            self.attn.qkv.bias.copy_(qkv_bias)
            self.attn.proj.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.mlp.fc1.weight.copy_(mlp_weight_0)
            self.mlp.fc2.weight.copy_(mlp_weight_1)
            self.mlp.fc1.bias.copy_(mlp_bias_0)
            self.mlp.fc2.bias.copy_(mlp_bias_1)

            self.norm1.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.norm1.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.norm2.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.norm2.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

    def forward(self, x, keep_rate=0.7, main_branch_ratio=1.0, return_idx=False):
        tmp, idx, index, cls_attn = self.attn(self.norm1(x), keep_rate=keep_rate, main_branch_ratio=main_branch_ratio, return_idx=return_idx)
        x = x + self.drop_path(tmp)

        main_idx, side_idx, main_index, side_index = idx, idx, index, index
        if index is not None:
            if main_branch_ratio < 1.0:
                # randomly divide tokens between main and side branches
                main_branch_tokens = math.ceil(main_branch_ratio * idx.shape[-1])
                side_branch_tokens = idx.shape[-1] - main_branch_tokens
                shuffle_idx = torch.randperm(idx.shape[-1]) # shuffle ids of left tokens
                main_idx = idx[:, shuffle_idx][:, :main_branch_tokens]
                side_idx = idx[:, shuffle_idx][:, main_branch_tokens:]
                if main_branch_tokens != side_branch_tokens: # if idx has odd number of tokens
                    side_idx = idx[:, shuffle_idx][:, main_branch_tokens-1:] # one token overlap
                main_index = main_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                side_index = side_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])

            B, N, C = x.shape
            non_cls = x[:, 1:]
            x_others = torch.gather(non_cls, dim=1, index=main_index)  # [B, main_tokens, C]

            if self.fuse_token:
                compl = complement_idx(main_idx, N - 1)  # [B, N-1-left_tokens]
                non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]

                non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
                extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
                x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
            else:
                x = torch.cat([x[:, 0:1], x_others], dim=1)
    
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, side_idx, side_index, cls_attn
 
   

class ReprogramViT(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, reprogram_index=None,
                 reprogram_network_params=None, base_keep_rate=None, drop_loc=None, global_pool=False, fuse_token=True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # if hybrid_backbone is not None:
        #     self.patch_embed = HybridEmbed(
        #         hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        # else:
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # base_keep_rate = reprogram_network_params['keep_rate'] if reprogram_network_params is not None else 1.
       
        self.reprogram_index = reprogram_index   
        # Reprogramming network parameters
        # if reprogram_index is not None:
        #     self.reprogram_index = reprogram_index
        # else:
        #     self.reprogram_index = list(range(depth))
        self.reprogram_network_params = reprogram_network_params
        self.fuse_token = fuse_token
        if self.reprogram_network_params is not None:
            reprog_embed_dim = embed_dim
            reprog_num_heads = reprogram_network_params['num_heads']
            reprog_mlp_ratio = reprogram_network_params['mlp_ratio']
            reprog_act_layer = reprogram_network_params['act_layer']
            main_branch_ratio = reprogram_network_params['main_branch_ratio']
            if 'patch_size' in reprogram_network_params:
                reprogram_patch_size = reprogram_network_params['patch_size']
                self.reprogram_patch_embed = PatchEmbed(
                    img_size=img_size, patch_size=reprogram_patch_size, in_chans=in_chans, embed_dim=reprog_embed_dim)
                num_patches = self.reprogram_patch_embed.num_patches

                self.reprogram_cls_token = nn.Parameter(torch.zeros(1, 1, reprog_embed_dim))
                self.reprogram_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, reprog_embed_dim))
        else:
            reprog_embed_dim = embed_dim
            reprog_num_heads = num_heads
            reprog_mlp_ratio = mlp_ratio

        # act_layer = []
        # for i in range(depth):
        #     if i < 3:
        #         act_layer.append(GELUSparse)
        #     else:
        #         act_layer.append(nn.GELU)
        self.blocks = nn.ModuleList([
            MainBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, fuse_token=fuse_token, main_branch_ratio=main_branch_ratio)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        if self.reprogram_index is not None:
            self.reprogram_patch_embed = PatchEmbed(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            self.reprogram_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.reprogram_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

            self.reprogram_blocks = nn.ModuleList([
                Block(reprog_embed_dim, reprog_num_heads, reprog_mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, act_layer=reprog_act_layer)
                for i in range(len(self.reprogram_index))])
            # self.reprog_embed_dim = reprog_embed_dim
        # else:
        #     assert False, "reprogram_index or reprogram_network_params must be provided"
        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
     # Main Network

        self.drop_loc = drop_loc
        self.keep_rate = [1.0] * 12
        self.main_branch_ratio = [1] * 12
        if self.drop_loc is not None:
            for loc in self.drop_loc:
                if loc < depth - 1:
                    self.keep_rate[loc] = base_keep_rate
        # self.keep_rate = [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0]
        print('Layerwise keep rate:', self.keep_rate)
        # for loc in self.reprogram_index:
        #     self.main_branch_ratio[loc] = 0.5
        
        self.global_pool = global_pool
        if self.global_pool:
            # norm_layer = kwargs['norm_layer']
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
    

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def load_from(self, weights, zero_head=False):
        with torch.no_grad():
            if zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.patch_embed.proj.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.patch_embed.proj.bias.copy_(np2th(weights["embedding/bias"]))
            self.cls_token.copy_(np2th(weights["cls"]))
            self.norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.pos_embed
            if posemb.size() == posemb_new.size():
                self.pos_embed.copy_(posemb)
            else:
                print("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.pos_embed.copy_(np2th(posemb))

            for bname, block in self.blocks.named_children():
                # for uname, unit in block.named_children():
                    block.load_from(weights, n_block=bname)
    
    def forward_features(self, x, keep_rate=None):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if keep_rate is None:
            keep_rate = self.keep_rate
        if self.reprogram_index is not None and len(self.reprogram_index) == 1:
            reprog_cls_tokens = self.reprogram_cls_token.expand(B, -1, -1)
            side_x = self.reprogram_patch_embed(x)
            side_x = torch.cat((reprog_cls_tokens, side_x), dim=1)
            side_x = side_x + self.reprogram_pos_embed
            side_x = self.pos_drop(side_x)

        x = self.patch_embed(x)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if self.reprogram_index is not None and i in self.reprogram_index:
                # with torch.no_grad():
                main_x, idx, index, cls_attn = blk(x, return_idx=True, keep_rate=keep_rate[i], main_branch_ratio=self.main_branch_ratio[i])

                if index is not None:
                    B, N, C = x.shape
                    non_cls = x[:, 1:]
                    x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]

                    if self.fuse_token:
                        compl = complement_idx(idx, N - 1)  # [B, N-1-left_tokens]
                        non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]

                        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
                        extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
                        x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1) # [B, N-left_tokens, C]
                    else:
                        x = torch.cat([x[:, 0:1], x_others], dim=1) # [B, N-1-left_tokens, C]
                if len(self.reprogram_index) == 1:
                    side_x = self.reprogram_blocks[self.reprogram_index.index(i)](side_x)
                else:
                    side_x = self.reprogram_blocks[self.reprogram_index.index(i)](x)

                # update tokens in main_x for positions in index
                # if index is not None:
                #     x = main_x.clone()
                #     x[:, 0, :] = side_x[:, 0, :] # cls token
                #     x[:,1:,:].scatter(1, index, side_x[:,1:-1,:])
                # else:
                x = main_x + side_x
                # x = torch.cat((main_x, side_x), dim=1)
            # elif self.reprogram_network_params is not None:
            #     with torch.no_grad():
            #         main_x = self.blocks[idx](x)
            #     side_ip = nn.functional.interpolate(x, size=self.reprog_embed_dim)
            #     side_x = self.reprogram_blocks[idx](side_ip)
            #     side_x = nn.functional.interpolate(side_x, size=main_x.shape[-1])
            #     x = main_x + side_x
            else:
                # with torch.no_grad():
                # x, _, _, _ = blk(x, return_idx=False)
                x, idx, index, cls_attn = blk(x, return_idx=True, keep_rate=keep_rate[i], main_branch_ratio=self.main_branch_ratio[i])


        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0] # just take embedding of cls token

        return outcome
    
    def forward(self, x, base_keep_rate=None, depth=12):
        # drop_loc = [3, 6, 9]
        keep_rate = None
        if base_keep_rate is not None:
            keep_rate = [1.0] * 12
            for loc in self.reprogram_index:
                if loc < depth - 1:
                    keep_rate[loc] = base_keep_rate
        x = self.forward_features(x, keep_rate=keep_rate)
        x = self.head(x)
        return x


def deit_tiny_patch16(pretrained=None, skip_block=1, **kwargs):
    depth = 12
    reprogram_index = list(range(depth))[::skip_block]
    model = ReprogramViT(
                patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), reprogram_index=reprogram_index, **kwargs)
    # if pretrained: 
    #     # checkpoint = torch.load('deit_tiny_patch16_224-a1311bcf.pth', map_location="cpu")
    #     # checkpoint = torch.hub.load_state_dict_from_url(
    #     #     url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
    #     #     map_location="cpu", check_hash=True
    #     # )
    #     checkpoint = torch.load(pretrained, map_location="cpu")
    #     model.load_state_dict(checkpoint["model"], strict=False)
    return model

    
def deit_small_patch16(pretrained=None, reprogram_index=None, reprogram_network=None, **kwargs):
    # depth = 12
    # reprogram_index = None
    # reprogram_network_params = None
    # if reprogram_index is not None:
    #     reprogram_index = list(range(depth))[::skip_block]
    reprogram_network_params = {'num_heads':6, 'mlp_ratio':4, 'qkv_bias':True, 'act_layer':nn.GELU, 'main_branch_ratio':1.0}
    print('reprogram_network_params', reprogram_network_params)
    model = ReprogramViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), reprogram_index=reprogram_index, reprogram_network_params=reprogram_network_params, **kwargs)
    setattr(model, "num_attention_heads", 6)
    setattr(model, "all_head_size", 384)
    # if pretrained:
    #     # checkpoint = torch.load('deit_small_patch16_224-cd65a155.pth', map_location="cpu")
    #     # checkpoint = torch.hub.load_state_dict_from_url(
    #     #     url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
    #     #     map_location="cpu", check_hash=True
    #     # )
    #     checkpoint = torch.load(pretrained, map_location="cpu")
    #     msg = model.load_state_dict(checkpoint["model"], strict=False)
    #     print(msg)
    return model


def small_dWr(pretrained=None, **kwargs):
    model = ReprogramViT(
        img_size=240, 
        patch_size=16, embed_dim=330, depth=14, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        # checkpoint = torch.load('fa_deit_ldr_14_330_240.pth', map_location="cpu")
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def deit_base_patch16(reprogram_index=None, **kwargs):
    reprogram_network_params = {'num_heads':12, 'mlp_ratio':4, 'qkv_bias':True, 'act_layer':nn.GELU, 'main_branch_ratio':1.0}
    print('reprogram_network_params', reprogram_network_params)
    model = ReprogramViT(
        # img_size=384, 
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), reprogram_index=reprogram_index, reprogram_network_params=reprogram_network_params, **kwargs)    
    setattr(model, "num_attention_heads", 12)
    setattr(model, "all_head_size", 768)
    # model = ReprogramViT(
    #     img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6),is_distill=True, **kwargs)
    # if pretrained:
    #     # checkpoint = torch.load('deit_base_distilled_patch16_384-d0272ac0.pth', map_location="cpu")
    #     # checkpoint = torch.hub.load_state_dict_from_url(
    #     #     url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
    #     #     map_location="cpu", check_hash=True
    #     # )
    #     checkpoint = torch.load(pretrained, map_location="cpu")
    #     model.load_state_dict(checkpoint["model"], strict=False)
    return model

def vit_base_patch16(reprogram_index=None, **kwargs):
    reprogram_network_params = {'num_heads':12, 'mlp_ratio':4, 'qkv_bias':True, 'act_layer':nn.GELU, 'main_branch_ratio':1.0}
    print('reprogram_network_params', reprogram_network_params)
    model = ReprogramViT(
        # img_size=384, 
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), reprogram_index=reprogram_index, reprogram_network_params=reprogram_network_params, **kwargs)    
    setattr(model, "num_attention_heads", 12)
    setattr(model, "all_head_size", 768)
    return model


def vit_large_patch16(**kwargs):
    model = ReprogramViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = ReprogramViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model