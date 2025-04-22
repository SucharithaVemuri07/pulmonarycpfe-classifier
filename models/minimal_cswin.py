import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx=0, heads=4):
        super().__init__()
        self.dim, self.res = dim, resolution
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.lepe_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.idx = idx
        self.heads = heads

    def forward(self, x, H, W):
        B, N, C = x.size()          # x: B×(H·W)×C
        x_img = x.view(B, H, W, C)  # B×H×W×C

        # split along H or W
        if self.idx == 0:  # split rows
            x_split = x_img.permute(0,2,1,3).reshape(-1, H, C)
            out_H, out_W = W, H
        else:              # split cols
            x_split = x_img.reshape(-1, W, C)
            out_H, out_W = H, W

        qkv = self.qkv(x_split).reshape(x_split.shape[0], x_split.shape[1], 3, self.heads, C//self.heads)
        q, k, v = qkv.permute(2,0,3,1,4)  # 3×(B*... )×heads×len×dim
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1,2).reshape(x_split.shape[0], x_split.shape[1], C)

        # fold back
        out = out.view(B, out_H, out_W, C).contiguous()
        out_flat = out.view(B, H*W, C)

        # LePE on original image
        lepe = self.lepe_conv(x_img.permute(0,3,1,2))  # B×C×H×W
        lepe = lepe.permute(0,2,3,1).reshape(B, H*W, C)

        return self.proj(out_flat + lepe)

class CSWinBlock(nn.Module):
    def __init__(self, dim, resolution, heads=4, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn_h = LePEAttention(dim, resolution, idx=0, heads=heads)
        self.attn_w = LePEAttention(dim, resolution, idx=1, heads=heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, int(dim*mlp_ratio), dropout)
        self.drop_path = nn.Identity()

    def forward(self, x, H, W):
        res = x
        x = self.norm1(x)
        xh = self.attn_h(x, H, W)
        xw = self.attn_w(x, H, W)
        x = res + self.drop_path(xh + xw)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CSWinMinimal(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dims=[64,128,256], depths=[2,2,6], heads=[2,4,8], num_classes=5):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dims[0], patch_size, patch_size)
        self.pos_drop    = nn.Dropout(0.)
        self.stages      = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i, dim in enumerate(embed_dims):
            # build a list of CSWinBlocks for this stage
            blocks = nn.ModuleList([
                CSWinBlock(dim, resolution=img_size//(patch_size*(2**i)), heads=heads[i])
                for _ in range(depths[i])
            ])
            self.stages.append(blocks)
            self.norms.append(nn.LayerNorm(dim))
            # prepare downsample to next stage (except last)
            if i < len(embed_dims)-1:
                conv = nn.Conv2d(dim, embed_dims[i+1], 2, 2)
                self.downsamples.append(conv)
            else:
                self.downsamples.append(None)

        self.head = nn.Linear(embed_dims[-1], num_classes)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        B,C,H,W = x.shape
        for blocks, norm, down in zip(self.stages, self.norms, self.downsamples):
            x = x.flatten(2).transpose(1,2)
            for blk in blocks:
                x = blk(x, H, W)
            x = norm(x)
            x = x.transpose(1,2).view(B,C,H,W)
            if down is not None:
                x = down(x)
                _,C,H,W = x.shape
        return x.mean(dim=[2,3])     # [B, embed_dims[-1]]

    def forward(self, x):
        feats = self.forward_features(x)
        return self.head(feats)      # [B, num_classes]
