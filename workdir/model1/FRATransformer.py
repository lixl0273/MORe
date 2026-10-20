import numpy as np
from timm.models.layers import trunc_normal_
from functools import partial
from collections import OrderedDict
import torch
from model1.theory.frasa_mfr import FRASA, membership_or_graph_to_graph
from model1.theory.motion_calibrated_prior import build_motion_calibrated_priors
import torch.nn as nn
from itertools import repeat
import collections.abc
import random
def drop_path(x, drop_prob: float = 0., training: bool = False):
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
        return drop_path(x, self.drop_prob, self.training)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


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


class HeadDropOut(nn.Module):
    """
    1. choose samples in probability p
    2. random set k head to zero in selected samples
    """

    def __init__(self, p: float=0.5, head_drop_upper: int=1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                'dropout probability has to be between 0 and 1, but got {}'
                .format(p))
        self.p = p
        self.upper = head_drop_upper

    def forward(self, x):
        B, N, c, num, dim = x.shape
        if self.training and self.p > 0 and self.upper > 0:
            # batch_rand = paddle.rand((B,))
            batch_rand = np.random.rand(B)
            batch_rand = torch.from_numpy(batch_rand)
            batch_mask = (batch_rand <= self.p).numpy()
            all_indexs = [i for i in range(num)]
            weights = [None] * B
            for i in range(B):
                if batch_mask[i]:
                    k = random.randint(1, self.upper)
                    zero_index = random.choices(all_indexs, k=k, weights=\
                        weights[i])
                    x[i, zero_index, ...] = 0
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        # 计算方差
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # varq = torch.var(q, dim=2).sum(dim=2).sum() / B / N
        # vark = torch.var(k, dim=2).sum(dim=2).sum() / B / N
        # varv = torch.var(v, dim=2).sum(dim=2).sum() / B / N

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) 

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,use_frasa=False, M_R=4, N=196):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # 根据 use_frasa 选择 Attention 机制
        if use_frasa:
            self.attn = FRASA(dim, num_heads=num_heads,qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  attn_drop=attn_drop, proj_drop=drop, M_R=M_R, N=N)
        else:
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  attn_drop=attn_drop, proj_drop=drop)



        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # 记录是否使用 FRASA
        self.use_frasa = use_frasa
        self.M_R = M_R
        self.N = N

    def forward(self, x,A_Component=None, A_Region=None, e_com=None, e_reg=None):
        if self.use_frasa:
            x = x + self.drop_path(self.attn(self.norm1(x), A_Component, A_Region, e_com=e_com, e_reg=e_reg))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class FRAT(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(self, img_size, patch_size,use_frasa, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                drop_rate=0., attn_drop_rate=0.,drop_path_rate=0.15, hybrid_backbone=None, norm_layer=None, M_R=4, N=196):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_tokens = 1

        self.use_frasa = use_frasa  # 是否使用 FRASA
        self.M_R = M_R
        self.N = N

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, embed_dim))
        # self.pos_embed = nn.Parameter(torch.randn(1, num_patches + self.num_tokens, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        # 这里传递 use_frasa 控制是否使用 FRASA
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,use_frasa=self.use_frasa, M_R=self.M_R, N=self.N)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.to_Mask = nn.Sequential(nn.Conv2d(in_channels=self.num_features,out_channels=1,kernel_size=3,padding=1),
        #                              nn.Hardsigmoid(),
        #                              )
        # self.to_Mask = nn.Linear(self.num_features,1)
        self.to_Mask = nn.Sequential(nn.Linear(self.num_features, 1),
                                     nn.Sigmoid(),
                                     )
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

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

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, A_Component=None, A_Region=None, raw_flow=None, return_aux=False):
        theory_aux = {
            "P0_list": [],
            "Pcom_list": [],
            "Preg_list": [],
            "gamma_com_list": [],
            "gamma_reg_list": [],
        }

        e_com = None
        e_reg = None
        prior_type = getattr(self, "prior_type", "nonparam")

        # Original strict path: patch embedding -> pos embedding -> transformer blocks -> norm -> pre_logits.
        # Do not pool or classify inside FRATransformer.
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Motion-calibrated priors are injected only into FRASA attention.
        # The token-feature forward path remains unchanged.
        if self.use_frasa and prior_type == "nonparam" and raw_flow is not None and A_Component is not None and A_Region is not None:
            try:
                A_com_base = membership_or_graph_to_graph(A_Component)
                A_reg_base = membership_or_graph_to_graph(A_Region)

                A_Component, A_Region, prior_aux = build_motion_calibrated_priors(
                    raw_flow,
                    A_com_base,
                    A_reg_base,
                    rho_com=getattr(self, "rho_com", 0.5),
                    rho_reg=getattr(self, "rho_reg", 0.5),
                    tau=getattr(self, "motion_tau", 0.7),
                )

                e_com = prior_aux.get("e_com", None)
                e_reg = prior_aux.get("e_reg", None)

                theory_aux.update({
                    "motion_desc": prior_aux.get("motion_desc", None),
                    "D_com": prior_aux.get("D_com", None),
                    "D_reg": prior_aux.get("D_reg", None),
                    "e_com": e_com,
                    "e_reg": e_reg,
                })
            except Exception as exc:
                print("[WARN] motion-calibrated prior failed; fallback to static priors:", repr(exc))

        for blk in self.blocks:
            if self.use_frasa:
                if hasattr(blk, "attn"):
                    if hasattr(blk.attn, "prior_type"):
                        blk.attn.prior_type = prior_type
                    if hasattr(blk.attn, "rho_com"):
                        blk.attn.rho_com = getattr(self, "rho_com", 0.5)
                    if hasattr(blk.attn, "rho_reg"):
                        blk.attn.rho_reg = getattr(self, "rho_reg", 0.5)
                    if hasattr(blk.attn, "gate_type"):
                        blk.attn.gate_type = getattr(self, "gate_type", "sigmoid")
                    if hasattr(blk.attn, "branch_mode"):
                        blk.attn.branch_mode = getattr(self, "branch_mode", "both")
                    if hasattr(blk.attn, "gate_mode"):
                        blk.attn.gate_mode = getattr(self, "gate_mode", "learned")
                    if hasattr(blk.attn, "fixed_gamma_com"):
                        blk.attn.fixed_gamma_com = getattr(self, "fixed_gamma_com", 1.0)
                    if hasattr(blk.attn, "fixed_gamma_reg"):
                        blk.attn.fixed_gamma_reg = getattr(self, "fixed_gamma_reg", 1.0)

                branch_mode = getattr(self, "branch_mode", "both")
                A_Component_blk = A_Component
                A_Region_blk = A_Region

                if A_Component is not None and A_Region is not None:
                    if branch_mode == "none":
                        A_Component_blk = torch.zeros_like(A_Component)
                        A_Region_blk = torch.zeros_like(A_Region)
                    elif branch_mode == "component":
                        A_Region_blk = torch.zeros_like(A_Region)
                    elif branch_mode == "region":
                        A_Component_blk = torch.zeros_like(A_Component)

                x = blk(x, A_Component_blk, A_Region_blk, e_com=e_com, e_reg=e_reg)

                if hasattr(blk.attn, "last_aux") and blk.attn.last_aux:
                    aux_i = blk.attn.last_aux

                    if "P0" in aux_i:
                        theory_aux["P0_list"].append(aux_i["P0"])
                    if "Pcom" in aux_i:
                        theory_aux["Pcom_list"].append(aux_i["Pcom"])
                    if "Preg" in aux_i:
                        theory_aux["Preg_list"].append(aux_i["Preg"])
                    if "gamma_com" in aux_i:
                        theory_aux["gamma_com_list"].append(aux_i["gamma_com"])
                    if "gamma_reg" in aux_i:
                        theory_aux["gamma_reg_list"].append(aux_i["gamma_reg"])
            else:
                x = blk(x)

        x = self.norm(x)
        x = self.pre_logits(x)

        if return_aux:
            return x, theory_aux
        return x

    def forward(self, x, A_Component=None, A_Region=None, return_aux=False):
        if return_aux:
            return self.forward_features(
                x,
                A_Component,
                A_Region,
                raw_flow=x,
                return_aux=True,
            )

        x = self.forward_features(x, A_Component, A_Region)
        return x
