# modified from: https://github.com/thstkdgus35/EDSR-PyTorch
from argparse import Namespace
import torch.nn as nn
# from .NLSA import NonLocalSparseAttention

def conv_3d(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def conv_2d(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
        
class ResBlock(nn.Module):
    def __init__(self, conv=conv_3d, n_feats=64, kernel_size=3,bias=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        # m = []
        # for i in range(2):
        #     m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
        #     if bn:
        #         m.append(nn.BatchNorm2d(n_feats))
        #     if i == 0:
        #         m.append(act)

        self.body = nn.Sequential(
            conv(n_feats, n_feats, kernel_size, bias=bias),
            nn.ReLU(),
            conv(n_feats, n_feats, kernel_size, bias=bias)
        )
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class EDSR_2d(nn.Module):
    def __init__(self, lr_slice_patch=1, n_feats=64, kernel_size=3, 
                 n_resblocks=16, res_scale=1, conv=conv_3d, act=nn.ReLU(),
                 feat_up=False):
        super(EDSR_2d, self).__init__()
        self.feat_up = feat_up
        self.lr_slice_patch = lr_slice_patch
        # define head module
        m_head = [conv(lr_slice_patch, n_feats, kernel_size)]
        # define body module
        m_body = [ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
                  for _ in range(n_resblocks)]
        # m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        # self.body = nn.Sequential(*m_body)
        self.body = nn.ModuleList(m_body)
        
        self.convup = nn.Conv2d(n_feats,n_feats*lr_slice_patch,3,1,1)

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.squeeze(1)
        # b s h w
        x = self.head(x)
        res = x.clone()
        # res = self.body(x)
        res_list = []
        for i,layer in enumerate(self.body):
            x = layer(x)
            if (i+1) % 4 == 0:
                res_list.append(x)
        
        res += x

        if self.feat_up:
            res = self.convup(res)
            b, _, h, w = res.shape
            res = res.view(b,self.lr_slice_patch,-1,h,w).transpose(1,2).contiguous()
        else:
            pass
        return res, res_list


class EDSR_3d(nn.Module):
    def __init__(self, in_channel=1, n_feats=64, kernel_size=3, 
                 n_resblocks=16, res_scale=1, conv=conv_3d, act=nn.ReLU()):
        super(EDSR_3d, self).__init__()
        
        # define head module
        m_head = [conv(in_channel, n_feats, kernel_size)]
        # define body module
        m_body = [ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
                  for _ in range(n_resblocks)]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        # b 1 s h w
        x = self.head(x)
        res = self.body(x)
        res += x
        return res, None

def build_edsr2d(args):
    """Build EDSR model based on the provided arguments."""
    # args = Namespace(**args)  # Convert dict to Namespace for easier attribute access
    return EDSR_2d(
        lr_slice_patch=getattr(args, 'lr_slice_patch', 1),
        n_feats=getattr(args, 'feature_dim', 64),
        kernel_size=getattr(args, 'kernel_size', 3),
        n_resblocks=getattr(args, 'n_resblocks', 16),
        res_scale=getattr(args, 'res_scale', 1),
        conv=conv_2d,
        act=getattr(args, 'act', nn.ReLU()),
        feat_up=getattr(args, 'feat_up', False)
    )

def build_edsr3d(args):
    """Build EDSR model based on the provided arguments."""
    # args = Namespace(**args)  # Convert dict to Namespace for easier attribute access
    return EDSR_3d(
        in_channel=getattr(args, 'in_channel', 1),
        n_feats=getattr(args, 'feature_dim', 64),
        kernel_size=getattr(args, 'kernel_size', 3),
        n_resblocks=getattr(args, 'n_resblocks', 16),
        res_scale=getattr(args, 'res_scale', 1),
        conv=conv_3d,
        act=getattr(args, 'act', nn.ReLU())
    )
