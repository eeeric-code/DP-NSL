import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoder import build_encoder
from util import make_coord

def make_model(args):
    return DPNSL(   args=args,
                    lr_slice_patch=getattr(args,'lr_slice_patch',4),
                    upscale=getattr(args,'upscale',2),
                    n_feats=getattr(args,'n_feats',64),
                    basis_ord=getattr(args,'basis_ord',3),
                    basis_M=getattr(args,'basis_M',4),
                    ord_expert_list=getattr(args,'ord_expert_list',[2,3,4]),
                    hard_router=getattr(args,'hard_router',False),
                    expand=getattr(args,'expand',2),
                    tail_depth=getattr(args,'tail_depth',2),
                    )

#####################################################################
class SplineBasis_1st(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        t = torch.where(x <= -1.0, 0.0 * x, 1.0 + x)
        t = torch.where(x <= 0.0, t, 1.0 - x)
        w = torch.where(x >= 1.0, 0.0 * x, t)
        return w

    @staticmethod
    def backward(ctx, grad_in):
        x, = ctx.saved_tensors
        t = torch.where(x <= -1.0, 0.0 * x, 1.0)
        t = torch.where(x <= 0.0, t, -1.0)
        w = torch.where(x >= 1.0, 0.0 * x, t)
        return w * grad_in

class SplineBasis_2nd(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        t = torch.where(x <= -1.5, 0.0 * x, 0.5 * (1.5 + x)**2)
        t = torch.where(x <= -0.5, t, 0.75 - x**2)
        t = torch.where(x <= 0.5, t, 0.5 * (1.5 - x)**2)
        w = torch.where(x > 1.5, 0.0 * x, t)
        return w
    
    @staticmethod
    def backward(ctx, grad_in):
        x, = ctx.saved_tensors
        t = torch.where(x <= -1.5, 0.0 * x, 1.5 + x)
        t = torch.where(x <= -0.5, t, -2 * x)
        t = torch.where(x <= 0.5, t, -(1.5 - x))
        w = torch.where(x > 1.5, 0.0 * x, t)
        return w * grad_in

class SplineBasis_3rd(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        t = torch.where(x <= -2.0, 0.0 * x, ((2.0 + x)**3) / 6.0)
        t = torch.where(x <= -1.0, t, (4.0 - 6.0 * (x**2) - 3.0 * (x**3)) / 6.0)
        t = torch.where(x <= 0.0, t, (4.0 - 6.0 * (x**2) + 3.0 * (x**3)) / 6.0)
        t = torch.where(x <= 1.0, t, ((2.0 - x)**3) / 6.0)
        w = torch.where(x > 2.0, 0.0 * x, t)
        return w
    
    @staticmethod
    def backward(ctx, grad_in):
        x, = ctx.saved_tensors
        t = torch.where(x <= -2.0, 0.0 * x, 0.5 * ((2.0 + x)**2))
        t = torch.where(x <= -1.0, t, -2.0 * x - 1.5 * (x**2))
        t = torch.where(x <= 0.0, t, -2.0 * x + 1.5 * (x**2))
        t = torch.where(x <= 1.0, t, -0.5 * ((2.0 - x)**2))
        w = torch.where(x > 2.0, 0.0 * x, t)
        return w * grad_in

class SplineBasis_4th(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        t = torch.where(x <= -2.5, 0.0 * x, ((x**4) + 10 * (x**3) + 37.5 * (x**2) + 62.5 * x + 39.0625)/24.)
        t = torch.where(x <= -1.5, t, (-4 * (x**4) - 20 * (x**3) - 30 * (x**2) - 5 * x + 13.75)/24.)
        t = torch.where(x <= -0.5, t, (6 * (x**4) - 15 * (x**2) + 14.375)/24.)
        t = torch.where(x <= 0.5, t, (-4 * (x**4) + 20 * (x**3) - 30 * (x**2) + 5 * x + 13.75)/24.)
        t = torch.where(x <= 1.5, t, ((x**4) - 10 * (x**3) + 37.5 * (x**2) - 62.5 * x + 39.0625)/24.)
        w = torch.where(x > 2.5, 0.0 * x, t)
        return w
    
    @staticmethod
    def backward(ctx, grad_in):
        x, = ctx.saved_tensors
        t = torch.where(x <= -2.5, 0.0 * x, (4 * (x**3) + 30 * (x**2) + 75 * x + 62.5)/24.)
        t = torch.where(x <= -1.5, t, (-16 * (x**3) - 60 * (x**2) - 60 * x - 5)/24.)
        t = torch.where(x <= -0.5, t, (24 * (x**3) - 30 * x)/24.)
        t = torch.where(x <= 0.5, t, (-16 * (x**3) + 60 * (x**2) - 60 * x + 5)/24.)
        t = torch.where(x <= 1.5, t, (4 * (x**3) - 30 * (x**2) + 75 * x - 62.5)/24.)
        w = torch.where(x > 2.5, 0.0 * x, t)
        return w * grad_in
    
def spline_basis(x, basis_ord=3):
    if basis_ord==0:
        return x
    elif basis_ord==1:
        x = x.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return SplineBasis_1st.apply(x)
    elif basis_ord==2:
        x = x.clamp(-1.5 + 1e-6, 1.5 - 1e-6)
        return SplineBasis_2nd.apply(x)
    elif basis_ord==3:
        x = x.clamp(-2.0 + 1e-6, 2.0 - 1e-6)
        return SplineBasis_3rd.apply(x)
    elif basis_ord==4:
        x = x.clamp(-2.5 + 1e-6, 2.5 - 1e-6)
        return SplineBasis_4th.apply(x)
    else:
        raise ValueError
    
def make_coord(shape, ranges=None, flatten=True):
    """
    Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n)
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class BsplineBlock3D(nn.Module):
    def __init__(self, n_feat, basis_M=4, basis_ord=3, hard_router=True, gumbel_tau=1.0, gumbel_hard=True) :
        super().__init__()

        self.M = basis_M
        self.coef = nn.Conv3d(n_feat, self.M**3, 3, 1, 1)
        self.knot = nn.Conv3d(n_feat, 3 * self.M, 3, 1, 1)
        self.dilation = nn.Linear(1, self.M, bias=False)
        self.basis_ord = basis_ord
        self.final_conv = nn.Conv3d(self.M**3, n_feat, 1, 1, 0)
        self.coord_cache = {}

        self.offset = nn.Sequential(
            nn.Conv3d(n_feat, n_feat, 1,1,0),
            nn.LeakyReLU(0.1),
            nn.Conv3d(n_feat, n_feat, 3,1,1),
            nn.LeakyReLU(0.1),
            nn.Conv3d(n_feat, 3, 1,1,0)
        )
        
    def check_coord(self, shape,flatten=False):
        shape = tuple(shape)
        if shape not in self.coord_cache.keys():
            coord = make_coord(shape, flatten=flatten)
            self.coord_cache[shape] = coord
        return self.coord_cache[shape]

    def forward(self, feat, data_dict):
        B, C, S, H, W = feat.shape
        hr_slice = data_dict['hr_slice']
        
        rs = 2 / feat.shape[-3] / 2
        rh = 2 / feat.shape[-2] / 2
        rw = 2 / feat.shape[-1] / 2
        hr_coord = self.check_coord([hr_slice,H,W]) \
            .permute(3,0,1,2).unsqueeze(0)\
            .expand(B,3,hr_slice,H,W).to(feat.device)  # 1 3 S H W
        hr_cell = torch.ones_like(hr_coord)
        hr_cell[:,0] *= 2 / hr_slice
        hr_cell[:,1] *= 2 / H
        hr_cell[:,2] *= 2 / W
        lr_coord = self.check_coord(feat.shape[-3:])\
            .permute(3,0,1,2).unsqueeze(0)\
            .expand(B,3,S,H,W).to(feat.device)  # 1 3 S H W

        coef = self.coef(feat)
        knot = self.knot(feat)
        
        sampled_coef = F.interpolate(coef, size=(hr_slice,H,W), mode='nearest')
        sampled_knot = F.interpolate(knot, size=(hr_slice,H,W), mode='nearest')
        sampled_coord = F.interpolate(lr_coord, size=(hr_slice,H,W), mode='nearest')

        rel_coord = hr_coord - sampled_coord
        offset = self.offset(F.interpolate(feat, size=(hr_slice,H,W), mode='trilinear'))
        offset = F.tanh(offset/10) # B 3 1 S H W
        rel_coord += offset
        
        rel_coord[:,0] *= hr_slice
        rel_coord[:,1] *= H
        rel_coord[:,2] *= W
        rel_coord = rel_coord.unsqueeze(2)
        
        rel_cell = hr_cell.clone()
        rel_cell[:, 0] *= feat.shape[-3]
        rel_cell[:, 1] *= feat.shape[-2]
        rel_cell[:, 2] *= feat.shape[-1]
        
        dilation = self.dilation(rel_cell[:,0:1,0,0,0]).view(B,1,self.M,1,1,1)
        q_knot = sampled_knot.view(B,3,self.M,hr_slice,H,W)  
        q_knot = (rel_coord - q_knot) * dilation  # B 3 M S H W
        q_knot = spline_basis(q_knot, self.basis_ord)
        
        # build 3D separable basis by outer product: produce (M^3,) per query
        # split per dimension
        bspline_s = q_knot[:,0,:,:,:,:].unsqueeze(2).unsqueeze(3)  # B M 1 1 S H W
        bspline_h = q_knot[:,1,:,:,:,:].unsqueeze(1).unsqueeze(3)  # B 1 M 1 S H W
        bspline_w = q_knot[:,2,:,:,:,:].unsqueeze(1).unsqueeze(2)  # B 1 1 M S H W
        # compute outer product via broadcasting then flatten to M^3
        basis3d = (bspline_s * bspline_h * bspline_w).view(B, -1, hr_slice, H, W)  # bs,M^3,s,h,w
        inp_feat = sampled_coef * basis3d
        inp_feat = self.final_conv(inp_feat)
        return inp_feat

class BsplineExpert(nn.Module):
    def __init__(self, n_feat, basis_M=4, basis_ord=None, ord_expert_list=[2,3,4],
                 hard_router=True, gumbel_tau=1.0, gumbel_hard=True):
        super().__init__()
        self.ordexpert_list = ord_expert_list
        self.experts = nn.ModuleList([
            BsplineBlock3D(n_feat, basis_M, ord, hard_router, gumbel_tau, gumbel_hard) 
            for ord in ord_expert_list
        ])
        
        # Router now outputs logits; Softmax/Gumbel applied in forward.
        self.router = nn.Sequential(
            nn.Conv3d(n_feat, n_feat, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(n_feat, len(ord_expert_list), 3, 1, 1)
        )
        self.hard_router = hard_router  # if True, use discrete (hard) routing via Gumbel-Softmax
        self.gumbel_tau = gumbel_tau      # temperature for Gumbel-Softmax
        self.gumbel_hard = gumbel_hard    # use hard=True in F.gumbel_softmax for straight-through one-hot
        # self.last_router_indices = None   # for analysis
        
        self.last_conv = nn.Conv3d(n_feat, n_feat, 1,1,0)
        
    def forward(self, feat, data_dict):
        
        B, C, S, H, W = feat.shape
        hr_slice = data_dict['hr_slice']

        # apply spline basis elementwise
        router_inp = F.interpolate(feat, size=(hr_slice, H, W), mode='trilinear')
        router_logits = self.router(router_inp)  # B, num_expert, S, H, W
        if self.training:
            if self.hard_router:
                # Differentiable one-hot via gumbel_softmax (straight-through). Preserves gradient.
                router = F.gumbel_softmax(router_logits, tau=self.gumbel_tau, hard=self.gumbel_hard, dim=1)
                if self.gumbel_hard:
                    # indices from hard sample (argmax of the sampled one-hot)
                    self.last_router_indices = router.argmax(dim=1, keepdim=True)
                else:
                    self.last_router_indices = None
            else:
                # Pure soft mixture
                router = F.softmax(router_logits, dim=1)
                self.last_router_indices = None
        else:
            if self.hard_router:
                # Evaluation: deterministic choice. Use argmax for clarity.
                indices = router_logits.argmax(dim=1, keepdim=True)
                router = torch.zeros_like(router_logits).scatter_(1, indices, 1.0)
                self.last_router_indices = indices
            else:
                router = F.softmax(router_logits, dim=1)
                # self.last_router_indices = router
        feat_list = []
        for expert in self.experts:
            feat_list.append(expert(feat, data_dict))
        feat_list = torch.stack(feat_list, dim=1)  # B, num_expert, 3, M, S, H, W
        feat = (feat_list * router.unsqueeze(2)).sum(dim=1)  # B, 3, M, S, H, W
        feat = self.last_conv(feat)
        
        return feat

class InceptionDWConv3d(nn.Module):
    """ Inception depthweise convolution
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.2):
        super().__init__()
        
        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.conv = nn.Conv3d(gc, gc, 3,1,1)
        self.dwconv_1 = nn.Conv3d(gc, gc, 3, padding=1, groups=gc)
        self.dwconv_2 = nn.Conv3d(gc,gc,5, padding=2, groups=gc)
        self.dwconv_3 = nn.Conv3d(gc, gc, 7, padding=3, groups=gc)

        self.split_indexes = (in_channels - 4* gc, gc, gc, gc, gc)
        
    def forward(self, x):
        x_id, x_shw, x_1,x_2, x_3 = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.conv(x_shw), self.dwconv_1(x_1), self.dwconv_2(x_2), self.dwconv_3(x_3)), 
            dim=1,
        )

class Tail(nn.Module):
    def __init__(self, n_feat=64,expand=2,depth=1):
        super(Tail, self).__init__()
        self.first_conv = nn.Conv3d(n_feat, n_feat*expand, 1,1,0)
        self.body = nn.ModuleList(
            [InceptionDWConv3d(n_feat*expand) for _ in range(depth)]
        )
        self.last_conv = nn.Conv3d(n_feat*expand,1,3,1,1)

    def forward(self, x):
        out = self.first_conv(x)
        for layer in self.body:
            out = layer(out)
            out = F.gelu(out)
        out = self.last_conv(out)
        return out

# model
class DPNSL(nn.Module):
    def __init__(self,args,upscale,
                n_feats=64,
                lr_slice_patch=4,
                basis_ord=3,
                basis_M=4,
                ord_expert_list=[2,3,4],
                hard_router=True,
                expand=2,
                tail_depth=2,
                **kwargs
                ):
        super(DPNSL, self).__init__()
        self.args = args
        self.upscale = upscale
        self.lr_slice_patch = lr_slice_patch
        # self.hr_slice_patch = (lr_slice_patch-1)*upscale + 1

        self.encoder = build_encoder(args)

        self.up = BsplineExpert(n_feats, basis_M=basis_M, basis_ord=basis_ord, ord_expert_list=ord_expert_list, hard_router=hard_router)
            
        self.coord_emb = getattr(args,'coord_emb',False)


        # expand = getattr(args,'expand',2)
        # tail_depth = getattr(args,'tail_depth',2)
        self.tail = Tail(n_feat=n_feats,expand=expand,depth=tail_depth)


    def forward(self, data_dict):
        # data_dict: list(dict)
        x = torch.stack([data['lr_volume'] for data in data_dict])

        x = x.permute(0,3,1,2)
        x = x.contiguous()

        res, _ = self.encoder(x)
        if len (res.shape) == 4:
            b, _, h, w = res.shape
        elif len (res.shape) == 5:
            b, _, s, h, w = res.shape
        
        out_list = []
        for idx,data in enumerate(data_dict):
                
            res_up = self.up(res[idx].unsqueeze(0),data)
            
            if self.coord_emb:
                shw_rel_coord = make_coord([data['hr_slice'],h,w],flatten=False)
                shw_rel_coord = shw_rel_coord.permute(3,0,1,2).unsqueeze(0).to(res_up.device)  # 1 3 h w d
                res_up = torch.cat([res_up,shw_rel_coord],dim=1)
            res_up = self.tail(res_up).squeeze(1)

            x_up = F.interpolate(x[idx].unsqueeze(0).unsqueeze(1),size=(data['hr_slice'],h,w),mode='trilinear',align_corners=False).squeeze(1)
            out = res_up + x_up
            out[:,::data['upscale']] = x[idx].unsqueeze(0)

            out = out.permute(0,2,3,1).contiguous()
            out_list.append({'sr_volume':out.squeeze()}) # h w d
        
        if self.training:
            loss_unused = 0.
            if hasattr(self.encoder,'convup'):
                loss_unused += 0. * sum(p.sum() for p in self.encoder.convup.parameters())
            return out_list, loss_unused
        else:
            return out_list

    @torch.no_grad()
    def infer_volume(self,data_dict):
        # bs = 1
        sr = torch.zeros_like(data_dict[0]['hr_volume'])
        sr_cnt = torch.zeros_like(data_dict[0]['hr_volume'])
        lr_volume = data_dict[0]['lr_volume'].clone()
        upscale = data_dict[0]['upscale']
        for tmp_s in range(lr_volume.shape[-1]-self.lr_slice_patch+1):
            data_dict[0]['lr_volume'] = lr_volume[...,tmp_s:tmp_s+self.lr_slice_patch] # b=1,h,w,d          
            tmp_sr = self.forward(data_dict)
            tmp_sr = tmp_sr[0]['sr_volume']
            tmp_sr = torch.clamp(tmp_sr,0,1)

            sr[...,tmp_s*upscale:tmp_s*upscale+((self.lr_slice_patch-1)*upscale+1)] += tmp_sr
            sr_cnt[...,tmp_s*upscale:tmp_s*upscale+((self.lr_slice_patch-1)*upscale+1)] += 1

        sr /= sr_cnt #[h,w,s]
        return sr
