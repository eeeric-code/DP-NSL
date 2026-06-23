import os 
from importlib import import_module
import torch
import os
import config

from val import val

args, unparsed = config.get_args()
args.ckpt_dir = 'experiments/test/'+args.model+'/'+args.ckpt_dir
os.makedirs(args.ckpt_dir,exist_ok=True)

module = import_module(f'model_zoo.{args.model}.{getattr(args, "model_version", "basic_model")}')
model = module.make_model(args)
load_ckpt = torch.load(args.ckpt,map_location=torch.device('cpu'))
model.load_state_dict(load_ckpt['state_dict'],strict=False)
model = model.cuda()

for upscale in args.upscale_test:
    result_dict = val(args=args,upscale=upscale,model=model,bsize=args.bsize)

