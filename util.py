import torch
from torch.autograd import Variable
import numpy as np

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx, :]

def random_crop_3d(img, output_size):
    d, h, w = img.shape
    new_d, new_h, new_w = output_size

    if d < new_d or h < new_h or w < new_w:
        raise ValueError("Output size must be smaller than the input image size.")

    d1 = np.random.randint(0, d - new_d+1)
    h1 = np.random.randint(0, h - new_h+1)
    w1 = np.random.randint(0, w - new_w+1)

    return img[d1:d1 + new_d, h1:h1 + new_h, w1:w1 + new_w]

def normalize(slice):
    ma,mi=4095 , 0
    # ma,mi=3071.0 , -1024.0
    slice = (slice - mi)/(ma - mi)
    
    return slice

class RandomCrop3d(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def _get_transform(self, x):
        if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - x.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - x.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - x.shape[2]) // 2 + 1, 0)
            x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = x.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        def do_transform(image):
            if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]:
                try:
                    image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                except Exception as e:
                    print(e)
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return image

        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples)
        # 输入不含bz,in/out[h,w,s]
        return transform(samples)
        # 输入含bz，in/out[bz,h,w,s]
        # return [transform(s) for s in samples]

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # image = sample[0]
        # image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        # sample = [image] + [*sample[1:]]
        # in/out含有bz维度
        # return [torch.from_numpy(s.astype(np.float32)) for s in sample]
        return torch.from_numpy(sample.astype(np.float32))

def prepare(args,gpu_id=0,precision='half'):
    device = torch.device(f'cuda:{gpu_id}')
    def _prepare(tensor):
        if precision == 'half': tensor = tensor.half()
        return tensor.to(device)
        
    # return [_prepare(a) for a in args]
    return _prepare(args)

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

def input_matrix_wpn(outH,outW,outD, scale,flatten=False):
    
    # projection_pixel_coordinate (H,W,2) coordinate(i,j)=[[i/r],[j/r]]
    h_p_coord = torch.arange(0, outH, 1).float().mul(1.0 / scale[0])
    h_p_coord_ = torch.floor(h_p_coord).int().view(outH,1,1)
    h_p_coord_metrix = h_p_coord_.expand(outH, outW,outD).unsqueeze(3)

    w_p_coord = torch.arange(0, outW, 1).float().mul(1.0 / scale[1])
    w_p_coord_ = torch.floor(w_p_coord).int().view(1,outW,1)
    w_p_coord_metrix = w_p_coord_.expand(outH, outW,outD).unsqueeze(3)

    d_p_coord = torch.arange(0, outD, 1).float().mul(1.0 / scale[2])
    d_p_coord_ = torch.floor(d_p_coord).int().view(1, 1, outD)
    d_p_coord_metrix = d_p_coord_.expand(outH, outW, outD).unsqueeze(3)

    projection_coord= torch.cat([h_p_coord_metrix, w_p_coord_metrix,d_p_coord_metrix], dim=-1)
    
    if flatten:
        projection_coord=projection_coord.view(-1,3)    
    return projection_coord    # HxWxD,3     

def to_pixel_samples(img_shape,scale):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img_shape,flatten=True)   # shape=(H*W*D,3)
    value = None # img.reshape(-1,1)   # shape=(H*W*D,1)
    proj_coord=input_matrix_wpn(*img_shape, scale,flatten=True)
    return coord, value,proj_coord

def is_distributed():
    if not torch.cuda.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    if not torch.distributed.is_available():
        return False
    return True

def get_rank():
    if not is_distributed():
        return 0
    return torch.distributed.get_rank()

def is_main_process():
    if not is_distributed():
        return True
    return get_rank() == 0  

def dist_barrier():
    if not is_distributed():
        return
    torch.distributed.barrier()

def dist_all_reduce_avg(n):
    try:
        if not is_distributed():
            return n
        else:
            n = torch.tensor(n).cuda()
            torch.distributed.all_reduce(n,op=torch.distributed.ReduceOp.SUM)
            cnt = torch.tensor(1).cuda()
            torch.distributed.all_reduce(cnt,op=torch.distributed.ReduceOp.SUM)
            n_avr = (n / cnt).item()
            return n_avr
    except:
        return n

def dist_all_gather_object(obj):
    if not is_distributed():
        return [obj]
    output = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(output, obj)
    return output