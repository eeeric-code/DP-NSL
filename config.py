import argparse
import os
from time import struct_time
import yaml

arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Dataset
data_arg = add_argument_group('Dataset')
data_arg.add_argument('--data_type', type=str, default='direct', help='direct/bicubic/linear')
data_arg.add_argument('--hr_slice_patch', type=int, default=7, help='每个hr样本slice个数')
data_arg.add_argument('--lr_slice_patch', type=int, default=4, help='每个lr样本的slice个数，插值为中间3个slice')
data_arg.add_argument('--load_first', action='store_true', default=False)

# Model
model_arg = add_argument_group('Model')
model_arg.add_argument('--config_yaml', type=str, default=None, help='select model')
model_arg.add_argument('--upscale', type=lambda s: [int(item) for item in s.split(',')], default=[2,3,4],help='2,3,4')
model_arg.add_argument('--upscale_test', type=lambda s: [int(item) for item in s.split(',')], default=[2,3,4],help='2,3,4')
model_arg.add_argument('--bsize', type=int, default=-1,help='batch predict') 
model_arg.add_argument('--sample_size', type=int, default=-1,help='train point') 
model_arg.add_argument("--resume", type=bool, default=False, help='run resume or not')
model_arg.add_argument('--save_ckpt_period', type=int, default=100,help='train point') 
model_arg.add_argument('--test_period', type=int, default=100,help='train point') 

# Training / test parameters
learn_arg = add_argument_group('Learning')
#### optim ####
learn_arg.add_argument('--optim', type=str, default='Adam') 
learn_arg.add_argument('--lr', type=float, default=(1e-4)) # 0.0003
learn_arg.add_argument('--weight_decay ', type=float, default=(1e-4), help='weight decay')
learn_arg.add_argument('--beta1', type=float, default=0.9, help='Adam-beta1')
learn_arg.add_argument('--beta2', type=float, default=0.999, help='Adam-beta2')
learn_arg.add_argument('--eps', type=float, default=1e-08)
#### schedule ####
learn_arg.add_argument('--schedule', type=str, default='step', help='step/cos_lr/Tmax/Tmin/MultiStepLR')
learn_arg.add_argument('--lr_decay', type=int, default=200, help='step 下降epoch')
learn_arg.add_argument('--stone', type=int, default=[750], help='scheduler')
learn_arg.add_argument('--gamma', type=float, default='0.5', help='下降速度')
# learn_arg.add_argument('--patience', type=int, default=15, help='Tmax/Tmin waiting epoch')
# learn_arg.add_argument('--cos_lr', type=bool, default=True, help='if cos_lr')
# learn_arg.add_argument('--Tmax', type=int, default=40, help='cos_lr 最大迭代次数')
# learn_arg.add_argument('--lr_gap', type=int, default=100, help='cos_lr 最小学习率eta_min=lr/lr_gap')
# learn_arg.add_argument('--cycle_r', type=bool, default=False)
# learn_arg.add_argument('--Tmin', type=bool, default=False)

learn_arg.add_argument('--batch_size', type=int, default=8)
learn_arg.add_argument('--one_batch_n_sample', type=int, default=1, help='一个iter内每个样本采样n次')
learn_arg.add_argument('--start_epoch', type=int, default=0)
learn_arg.add_argument('--max_epoch', type=int, default=1000)
learn_arg.add_argument('--warmup_epoch', type=float, default=0.01)
learn_arg.add_argument('--ckpt', type=str, default='', help='pretrained model path')

learn_arg.add_argument('--traindata_path', type=str, default='data/data_slice/Task10_Colon/imagesTr')
learn_arg.add_argument('--testdata_path', type=str, default='data/data_volume/Task10_Colon/imagesTs')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--ckpt_dir', type=str, default='debug')
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--num_workers', type=int, default=4)
misc_arg.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)

parser.add_argument('--amp',action='store_true')
parser.add_argument('-kwargs', '--kwargs',type=str,default='',
                    help='Override config values with key=value, can repeat. Example: --kwargs lr=1e-4 n_feats=256')

parser.add_argument('--model',type=str,default=None)
parser.add_argument('--save_3dnpy',action='store_true',default=False,)
parser.add_argument('--test_list',type=str,default='',help='test ckpt list: c2,c234')


def parse_kwargs(kwargs):
    """
    解析 --kwargs 传入的参数，支持两种形式：
    1) 列表形式：--kwargs k1=v1 k2=v2
    2) 字符串形式：--kwargs "k1=v1 k2=v2" 或 "k1=v1, k2=v2"
    返回: dict
    """
    if not kwargs:
        return {}

    # 统一拆分为 k=v 片段列表
    if isinstance(kwargs, (list, tuple)):
        parts = []
        for item in kwargs:
            if not item:
                continue
            if isinstance(item, str) and (' ' in item or ',' in item) and '=' in item:
                # 递归解析嵌套字符串
                nested = parse_kwargs(item)
                for nk, nv in nested.items():
                    parts.append(f"{nk}={nv}")
            else:
                parts.append(str(item))
    elif isinstance(kwargs, str):
        text = kwargs.strip()
        if not text:
            return {}
        if ',' in text and ' ' not in text:
            parts = [p.strip() for p in text.split(',') if p.strip()]
        else:
            parts = [p.strip(',') for p in text.split() if p.strip(',')]
    else:
        return {}

    kwargs_dict = {}
    for kv in parts:
        if '=' not in kv:
            continue
        key, value = kv.split('=', 1)
        kwargs_dict[key.strip()] = value.strip()
    return kwargs_dict

def auto_convert_type(value, original_value=None):
    """
    自动转换参数类型，优先根据原始值的类型进行转换
    """
    if original_value is not None:
        # 根据原始值的类型进行转换
        if isinstance(original_value,bool):
            return value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(original_value, int):
            return int(value)
        elif isinstance(original_value, float):
            return float(value)
        if isinstance(original_value, list):
            return eval(value)
        else:
            return value
    else:
        # 尝试自动推断类型
        try:
            if value.startswith('[') and value.endswith(']'):
                return eval(value)
        except Exception:
            pass
        try:
            # 尝试转换为int
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        try:
            # 尝试转换为float
            return float(value)
        except ValueError:
            pass
        
        # 处理布尔值
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # 默认返回字符串
        return value

def update_cfg(args):
    kwargs = parse_kwargs(args.kwargs)
    for k,v in kwargs.items():
        v_ori = getattr(args,k, None)
        v_update = auto_convert_type(v, v_ori)
        setattr(args, k, v_update)
    return args

def get_args():
    """Parses all of the arguments above
    """
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:   
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    
    if args.config_yaml is not None:
        with open(f'{args.config_yaml}', 'r') as f:
            default_arg = yaml.safe_load(f)
            if default_arg:
                for key,value in default_arg.items():
                    setattr(args,key,value)

    args = update_cfg(args)
    return args, unparsed

