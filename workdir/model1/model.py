import torch
from model1.theory.mpt import MotionPreservingTokenization
from torch import nn
from torchvision.transforms import Resize
from functools import partial
import torchvision.models as models

from model1.FRATransformer import FRAT

class MyNet_FRAT(nn.Module):
    def __init__(self, in_channels=3, num_classes=3,depth=2,num_heads=4,img_size=224,patch_size=16,use_frasa=False,embed_dim=512, use_mpt=False, mpt_variant='none', mpt_residual_scale=0.75):
        super(MyNet_FRAT, self).__init__()
        self.use_mpt = bool(use_mpt)
        self.mpt_variant = mpt_variant
        self.mpt = MotionPreservingTokenization(
            in_channels=in_channels,
            hidden_channels=24,
            unshuffle_factor=2,
            variant=mpt_variant if use_mpt else 'none',
            residual_scale=mpt_residual_scale,
        )

        self.feature_size=img_size//patch_size
        
        self.vit_pos = FRAT(img_size=img_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim, depth=depth,
                                             num_heads=num_heads, mlp_ratio=2, qkv_bias=True,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),drop_path_rate=0.3,use_frasa=use_frasa)

        self.conv1x1_pc = nn.Conv2d(in_channels=embed_dim, out_channels=256, kernel_size=1)

        fc_input_features=self.feature_size*self.feature_size*256
        
        # fc
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=fc_input_features, out_features=1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=1024, out_features=num_classes),
        )

    def forward(self, x, A_Component=None, A_Region=None, return_aux=False):
        if self.use_mpt:
            x = self.mpt(x)
        """
        Restore the original Model1 classification path.

        FRAT returns token-level features:
            token_feat: [B, N, D], where N=14*14=196 and D=embed_dim.

        Original classifier path:
            [B, N, D] -> [B, D, 14, 14] -> conv1x1_pc -> flatten -> fc.
        """
        if return_aux:
            token_feat, aux = self.vit_pos(x, A_Component, A_Region, return_aux=True)
        else:
            token_feat = self.vit_pos(x, A_Component, A_Region)
            aux = None

        if token_feat.dim() != 3:
            raise RuntimeError(f"Expected token_feat [B,N,D], got shape={tuple(token_feat.shape)}")

        B, N, D = token_feat.shape
        H = W = self.feature_size

        if N != H * W:
            raise RuntimeError(
                f"Token number mismatch: N={N}, expected H*W={H*W}. "
                f"feature_size={self.feature_size}"
            )

        feat_map = token_feat.transpose(1, 2).contiguous().view(B, D, H, W)
        feat_map = self.conv1x1_pc(feat_map)
        feat_flat = feat_map.flatten(1)
        logits = self.fc(feat_flat)

        if return_aux:
            aux["logits"] = logits
            return logits, aux

        return logits
