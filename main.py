import os
import config
args, unparsed = config.get_args()

import torch
from torch.utils.data.distributed import DistributedSampler
import random
import numpy as np

from data import trainSet,variable_size_collate_fn
from util_evaluation import calc_psnr,calc_ssim
import time
import optim
from select_loss import Select_Loss
from tqdm import tqdm

import val
from importlib import import_module
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.simplefilter('ignore')
####################################################################
# train
# root = os.getcwd()

GLOBAL_SEED = int(os.getenv('GLOBAL_SEED', '777'))
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
print(f"Using GLOBAL_SEED: {GLOBAL_SEED}")

writer = SummaryWriter(f'./log/{args.model}/{args.ckpt_dir}')

args.ckpt_dir = './experiments/'+args.model+'/'+args.ckpt_dir
os.makedirs(args.ckpt_dir,exist_ok=True)


local_rank = int(os.environ["LOCAL_RANK"])
if args.local_rank != -1:
    # 调用卡的数量>实际卡数量 配置
    num_gpus = torch.cuda.device_count()
    device_idx = args.local_rank % num_gpus
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    # torch.cuda.set_device(args.local_rank)
    # device=torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")#, init_method='env://')

trainset = trainSet(data_root=args.traindata_path,args=args)
train_sampler = DistributedSampler(trainset)
# batch_size = args.batch_size*len(device_ids)
dataloader = torch.utils.data.DataLoader(trainset, sampler=train_sampler, batch_size=args.batch_size,\
                shuffle=False,num_workers=args.num_workers, pin_memory=True,persistent_workers=True,\
                collate_fn=variable_size_collate_fn)
    
module = import_module(f'model_zoo.{args.model}.{getattr(args, "model_version", "basic_model")}')
model = module.make_model(args)


model = model.to(device)
# if num_gpus > 1:
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = torch.nn.parallel.DistributedDataParallel(model,
                                                device_ids=[args.local_rank],
                                                output_device=args.local_rank,
                                                find_unused_parameters=False,
                                                broadcast_buffers=False)


#### optim ####
# optimizer, scheduler = optim.select_optim(args,model)
optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)
scheduler = optim.select_scheduler(args,optimizer)

#### loss ####
# loss_function = nn.L1Loss()
loss_function = Select_Loss(args).to(device)

########################### train ###################################
model.train()

with open(args.ckpt_dir + '/logs.txt',mode='a+') as f:
    s = "\n\n\n\n\nSTART EXPERIMENT\n"
    f.write(s)
    f.write(f'SEED: {GLOBAL_SEED}\n')
    f.write('**********  Args  **********\n')
    for i,j in args.__dict__.items():
        f.write(str(i)+' : '+str(j)+'\n')


##########  amp #########
scaler = torch.cuda.amp.GradScaler()

#重载入模型
if args.resume:
    print('load weight')
    load_ckpt = torch.load(args.ckpt,map_location=torch.device('cpu'))
    try:
        model.module.load_state_dict(load_ckpt['state_dict'])
    except:
        model.load_state_dict(load_ckpt['state_dict'])
    if 'optimizer' in load_ckpt:
        optimizer.load_state_dict(load_ckpt['optimizer'])
    if 'scheduler' in load_ckpt:
        scheduler.load_state_dict(load_ckpt['scheduler'])
    if 'scaler' in load_ckpt:
        scaler.load_state_dict(load_ckpt['scaler'])
    print('load weight success')
    args.start_epoch = load_ckpt['epoch']



with tqdm(range(args.start_epoch, args.max_epoch), unit="epoch", position=0) as epoch_pbar:
    for epoch in epoch_pbar:
        train_sampler.set_epoch(epoch)
        loss_epoch = 0
        psnr_epoch = 0
        ssim_epoch=0 
        model.train()
        epoch_pbar.set_description(f"Epoch {epoch+1}/{args.max_epoch}")

        with tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch", position=1, leave=False) as iter_pbar:
            for iter, data_dict in enumerate(iter_pbar):
                for data in data_dict:
                    data['lr_volume'] = data['lr_volume'].cuda()
                    data['hr_volume'] = data['hr_volume'].cuda()

                optimizer.zero_grad()
                # with torch.autograd.set_detect_anomaly(True):
                with torch.autocast(enabled=args.amp,device_type='cuda',dtype=torch.float16):
                    sr_dict, loss_unused = model(data_dict)
                    loss_iter = loss_function(sr_dict,data_dict)
                    loss_iter += loss_unused

                scaler.scale(loss_iter).backward()
                scaler.step(optimizer)
                scaler.update()

                psnr_iter = 0
                ssim_iter = 0
                for bz,(gt_dict,sr_dict)in enumerate(zip(data_dict,sr_dict)):
                    psnr_iter += calc_psnr(gt_dict['hr_volume'],sr_dict['sr_volume']).item()
                psnr_iter /= (bz+1)  

                lr_tmp = optimizer.state_dict()['param_groups'][0]['lr']
                iter_log = f"batch[{iter+1}/{len(dataloader)}] loss:{loss_iter:.6f} lr:{lr_tmp:.8f}"
                iter_pbar.set_description(iter_log)

                loss_epoch += loss_iter
                psnr_epoch += psnr_iter
                ssim_epoch += ssim_iter
                # torch.cuda.empty_cache()
            loss_epoch /= (iter+1)
            psnr_epoch /= (iter+1)
            ssim_epoch /= (iter+1)

            #### lr schedule ####
            if args.schedule == 'step':
                scheduler.step()
            elif args.schedule == 'cos_lr':
                # torch.cos_lr
                # scheduler.step()
                # timm.cos_lr
                scheduler.step_update(epoch)
            elif args.schedule == 'Tmin':
                scheduler.step(loss_epoch)
            elif args.schedule == 'Tmax':
                scheduler.step(psnr_epoch)

        epoch_summary = f"Loss:{loss_epoch:.6f} PSNR:{psnr_epoch:.4f} SSIM:{ssim_epoch:.4f} LR:{lr_tmp:.8f}"
        epoch_pbar.set_postfix_str(epoch_summary)

        writer.add_scalar('train_loss', loss_epoch, epoch + 1)
        writer.add_scalar('train_psnr', psnr_epoch, epoch + 1)

        # log
        if local_rank == 0:
            log = r"epoch[{}/{}] psnrTr:{:.6f} ssim:{:.6f} lossTr:{:.12f} lr:{:.12f}"\
            .format(epoch, args.max_epoch , \
                psnr_epoch,ssim_epoch,loss_epoch,lr_tmp)
            now = time.strftime('%Y/%m/%d %H:%M:%S >> ', time.gmtime(time.time())) 
            # print(now+log)
            with open(args.ckpt_dir + '/logs.txt',mode='a+') as f:
                f.write('\n'+now+log)

            if ((epoch + 1) % args.save_ckpt_period == 0 or (epoch + 1) == args.max_epoch):
                os.makedirs(args.ckpt_dir+'/pth',exist_ok=True)
                try:
                    torch.save({'epoch': epoch + 1, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict()}, args.ckpt_dir + '/pth/' + str(epoch + 1).zfill(3) + '.pth')
                except:
                    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict()}, args.ckpt_dir + '/pth/' + str(epoch + 1).zfill(3) + '.pth')

        # torch.cuda.empty_cache()
        if ((epoch + 1) % args.test_period == 0 or (epoch+1) == args.max_epoch):
            for upscale_test in args.upscale_test:
                result_dict = val.val(args=args,upscale=upscale_test,model=model,bsize=args.bsize)
                writer.add_scalar(f'psnr_x{upscale_test}', result_dict['psnr'], epoch + 1)
                writer.add_scalar(f'x_y_ssim_x{upscale_test}', result_dict['x_y_ssim'], epoch + 1)
                writer.add_scalar(f'x_z_ssim_x{upscale_test}', result_dict['x_z_ssim'], epoch + 1)
                writer.add_scalar(f'y_z_ssim_x{upscale_test}', result_dict['y_z_ssim'], epoch + 1)
                writer.add_scalar(f'flops_x{upscale_test}', result_dict['flops'], epoch + 1)
                writer.add_scalar(f'params_x{upscale_test}', result_dict['params'], epoch + 1)
        writer.flush()
        torch.cuda.empty_cache() 
writer.close()
