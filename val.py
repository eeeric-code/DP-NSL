import torch
from data import testSet,variable_size_collate_fn
from util_evaluation import calc_psnr,calc_ssim
import thop
import copy
from torch.utils.data.distributed import DistributedSampler
import util

def val(args,upscale,model,bsize=-1):
    torch.cuda.empty_cache()
    if util.is_main_process():
        with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
            f.write("\n\n\n\n\nSTART TEST\n")
            f.write(f"!!! test_scale={upscale}")
            f.write('testdata:'+args.testdata_path+'\n')
            f.write('checkpoint:'+args.ckpt_dir+'\n')

    model.eval()

    testset = testSet(data_root=args.testdata_path,
                    upscale=upscale,data_range=args.data_range,args=args)
    if torch.distributed.is_initialized():
        sampler = DistributedSampler(testset)
    else: sampler = None
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1,
    drop_last=False, shuffle=False, num_workers=4, pin_memory=False,
    sampler=sampler,collate_fn=variable_size_collate_fn)

    average_psnr=0
    total_x_y_ssim=0
    total_x_z_ssim=0
    total_y_z_ssim=0

    for id, data_dict in enumerate(dataloader):
        for data in data_dict:
            data['lr_volume'] = data['lr_volume'].cuda()
            data['hr_volume'] = data['hr_volume'].cuda()
            data['xyz_hr'] = data['xyz_hr'].cuda()
        psnr = 0
        x_y_ssim=0 
        x_z_ssim=0
        y_z_ssim=0
        
        with torch.autocast(device_type='cuda', enabled=args.amp,dtype=torch.float16):
            if hasattr(model,'module'):
                sr = model.module.infer_volume(data_dict)
            else:
                sr = model.infer_volume(data_dict)

        gt = data_dict[0]['hr_volume']
        sr = sr[...,upscale : -1*upscale]
        gt = gt[...,upscale : -1*upscale]

        # print(sr.shape) # h w s
        sr = sr.cuda()
        gt = gt.cuda()

        psnr = calc_psnr(sr,gt).item()

        average_psnr += psnr

        # calc_ssim
        x_y_ssim, x_z_ssim, y_z_ssim = 0, 0, 0
        if len(gt.shape) == 4:
            gt = gt[0]
            sr = sr[0]
        for i in range(gt.shape[2]):
            ssim = calc_ssim(gt[:,:,i],sr[:,:,i])
            x_y_ssim += ssim
        x_y_ssim /= (i+1)
        for i in range(gt.shape[0]):
            ssim = calc_ssim(gt[i,:,:],sr[i,:,:])
            x_z_ssim += ssim
        x_z_ssim /= (i+1)
        for i in range(gt.shape[1]):
            ssim = calc_ssim(gt[:,i,:],sr[:,i,:])
            y_z_ssim += ssim
        y_z_ssim /= (i+1)
        log = r"[{} / {}] PSNR:{} x_y_ssim:{:.4f} x_z_ssim:{:.4f} y_z_ssim:{:.4f} "\
            .format(id+1,dataloader.__len__(),psnr,x_y_ssim, x_z_ssim,y_z_ssim)
        print(log)
        with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
            f.write(log+'\n') 

        total_x_y_ssim+=x_y_ssim
        total_x_z_ssim+=x_z_ssim
        total_y_z_ssim+=y_z_ssim


    average_psnr /= (id+1)
    total_x_y_ssim /= (id+1)
    total_x_z_ssim /= (id+1)
    total_y_z_ssim /= (id+1)
    
    average_psnr = util.dist_all_reduce_avg(average_psnr)
    total_x_y_ssim = util.dist_all_reduce_avg(total_x_y_ssim)
    total_x_z_ssim = util.dist_all_reduce_avg(total_x_z_ssim)
    total_y_z_ssim = util.dist_all_reduce_avg(total_y_z_ssim)
    average_ssim = (total_x_y_ssim + total_x_z_ssim + total_y_z_ssim) / 3.0

    tmp_slice = (args.lr_slice_patch - 1)*upscale + 1
    if hasattr(model,'module'):
        model_copy = copy.deepcopy(model.module).eval()
    else:
        model_copy = copy.deepcopy(model).eval()
    flops, params = thop.profile(model_copy, inputs=( [ {'lr_volume': torch.randn(256,256,4).cuda(),
                                                'xyz_hr': torch.randn(256*256*tmp_slice,3).cuda(),
                                                'hr_volume': torch.randn(256,256,tmp_slice).cuda(),
                                                'upscale': upscale,
                                                'hr_slice': tmp_slice,
                                                'lr_slice': args.lr_slice_patch,
                                                'proj_coord': torch.randn(256*256*tmp_slice,3).cuda(),
                                                'cell': torch.randn(256*256*tmp_slice,3).cuda(),    
                                                }], )
                            )


    if util.is_main_process():
        print("average_psnr:",average_psnr) 
        print("average_x_y_ssim:",total_x_y_ssim) 
        print("average_x_z_ssim:",total_x_z_ssim) 
        print("average_y_z_ssim:",total_y_z_ssim) 
        print("average_ssim:",average_ssim)

        print(f"flops:{flops:.4e}, params:{params:.4e}")
        with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
            log = r"PSNR: {} x_y_ssim: {:.6f} x_z_ssim: {:.6f} y_z_ssim: {:.6f} average_ssim: {:.6f}".format(average_psnr,total_x_y_ssim,total_x_z_ssim,total_y_z_ssim,average_ssim)
            log_cost = f"flops:{flops:.4e}\nparams:{params:.4e}"
            f.write(log+'\n'+log_cost)

    torch.cuda.empty_cache()
    return {
        "psnr": average_psnr,
        "x_y_ssim": total_x_y_ssim,
        "x_z_ssim": total_x_z_ssim,
        "y_z_ssim": total_y_z_ssim,
        "flops": flops,
        "params": params,
    }
