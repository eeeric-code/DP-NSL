import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import random 
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import util

from multiprocessing import Pool
from scipy.ndimage import zoom

def normalize(s, data_range=[0,1]):
    mi, ma = data_range
    s = (s - mi)/(ma - mi)
    return s


class trainSet(Dataset):
    def __init__(self, data_root, args=None,
                random_crop=None, resize=None, augment_s=True, augment_t=True,
                ):
        self.args = args
        self.data_type = getattr(args,'data_type','direct')
        self.upscale = args.upscale
        self.data_root = data_root
        self.folder_list = [(data_root + '/' + f) for f in os.listdir(data_root)] 
        random.shuffle(self.folder_list)
        if self.args.load_first:
            self.volume_list = self.load_img(self.folder_list)
        
        self.file_len = len(self.folder_list)

        self.random_crop = random_crop
        self.augment_s = augment_s
        self.augment_t = augment_t

        self.data_range = getattr(args,'data_range',[0,1])
        self.sample_size = getattr(args,'sample_size',-1)
    
    @staticmethod
    def process_volume(vol_file):
        if vol_file.endswith('.pt') or vol_file.endswith('.pth'):
            with open(vol_file, 'rb') as _f: volume = pickle.load(_f)['image']
            volume = np.array(volume,dtype=np.float32)
            return {'filename': vol_file, 'data': volume}
        
        else:
            slice_file_list = [(vol_file + '/' + f) for f in os.listdir(vol_file)]
            slice_file_list.sort()
            volume = {'filename': vol_file, 'data': []}
            for slice_file in slice_file_list:
                with open(slice_file, 'rb') as _f:
                    img = pickle.load(_f)
                    volume['data'].append(img['image'])
        return volume

    def load_img(self, file_list):
        with Pool(processes=16) as pool:
            vol_list = list(tqdm(pool.imap(trainSet.process_volume, file_list), total=len(file_list), desc="Loading volumes"))
        return vol_list

    def getAVol(self,index,hr_slice_patch):
        if self.args.load_first:
            volumeAll = self.volume_list[index]['data']
            id = random.randint(0,len(volumeAll) - hr_slice_patch)
            volume = volumeAll[id: id+hr_slice_patch]
        else:
            volumepath = self.folder_list[index]
            slice_list = [(volumepath+'/'+f) for f in os.listdir(volumepath)]
            slice_list.sort()

            id = random.randint(0,len(slice_list) - hr_slice_patch)
            volume = []
            for i in range(hr_slice_patch):
                with open(slice_list[id+i], 'rb') as _f: img = pickle.load(_f)
                volume.append(img['image'])

        volume = np.array(volume,dtype=np.float32).transpose(1,2,0) 
        volume = normalize(volume,data_range=self.data_range) # [0-4095]->[0-1]
        if random.random() >= 0.5:
            volume = volume[:,::-1,:].copy()
        if random.random() >= 0.5:
            volume = volume[::-1,:,:].copy()

        volume = util.crop_center(volume,256,256) #[256,256,7]

        volume=torch.from_numpy(volume)
        
        return volume

    def __getitem__(self, index):

        if len(self.upscale) > 1:
            upscale = random.choice(self.upscale)
        else:
            upscale = self.upscale[0]
        hr_slice_patch = (self.args.lr_slice_patch - 1) * upscale + 1
        hr_volume = self.getAVol(index, hr_slice_patch) # h w d
        
        ######## downsample data_type ########
        if self.data_type == 'direct':
            lr_volume = hr_volume[...,::upscale]
        elif self.data_type == 'linear':
            h,w,hr_slice_patch = hr_volume.shape
            lr_slice_patch = self.args.lr_slice_patch
            lr_volume = F.interpolate(hr_volume.unsqueeze(0).unsqueeze(0), size=(h,w,lr_slice_patch), mode='trilinear', align_corners=False).squeeze(0).squeeze(0)
        elif self.data_type == 'bicubic':
            h,w,_ = hr_volume.shape
            # scipy.ndimage.zoom with order=3 is equivalent to bicubic
            zoom_factors = (1, 1, self.args.lr_slice_patch / hr_slice_patch)
            lr_volume_np = zoom(hr_volume.numpy(), zoom_factors, order=3, mode='grid-constant', grid_mode=True)
            lr_volume = torch.from_numpy(lr_volume_np)

        xyz_hr, _, proj_coord = util.to_pixel_samples(hr_volume.shape,scale=(1,1,upscale))

        cell = torch.ones_like(xyz_hr)
        cell[:, 0] *= 2 / hr_volume.shape[-3]
        cell[:, 1] *= 2 / hr_volume.shape[-2]
        cell[:, 2] *= 2 / hr_volume.shape[-1]

        if self.sample_size == -1:
            pass
        else:
            sample_indices = np.random.choice(len(xyz_hr), self.sample_size, replace=False)
            xyz_hr = xyz_hr[sample_indices]
            proj_coord = proj_coord[sample_indices]
            hr_volume = hr_volume.reshape(-1, 1)[sample_indices]
            
        return {
            'hr_volume': hr_volume,
            'lr_volume': lr_volume,
            'upscale': upscale,
            'xyz_hr': xyz_hr,
            'proj_coord': proj_coord,
            'hr_slice': hr_slice_patch,
            "lr_slice": self.args.lr_slice_patch,
            'cell':cell,
        }

    def __len__(self):
        return self.file_len

class testSet(Dataset):
    def __init__(self, data_root,
                        upscale,
                        data_range=[0,1],
                        lr_slice_patch=4,
                        args=None,):
        self.data_root = data_root
        self.data_type = getattr(args,'data_type','direct') if args is not None else 'direct'
        self.filenames = [(data_root + '/' + f) for f in os.listdir(data_root)]
        self.upscale = upscale
        self.data_range=data_range
        self.lr_slice_patch = lr_slice_patch

        self.file_len = len(self.filenames)

    def getAVol(self,filename):
        if os.path.isfile(filename): # file
            if filename.endswith('pt') or filename.endswith('pth'):
                with open(filename, 'rb') as _f: volume = pickle.load(_f)['image']
            volume = np.array(volume,dtype=np.float32)

        else: # dir
            slice_list = [(filename+'/'+f) for f in os.listdir(filename)]
            slice_list.sort()
            volume = []
            for s in slice_list:
                if s.endswith('pth'): 
                    with open(s, 'rb') as _f: img = pickle.load(_f)['image']
                if s.endswith('npy'): 
                    img = np.load(s)
            volume.append(img)
            volume = np.array(volume,dtype=np.float32).transpose(1,2,0)

        volume = util.crop_center(volume,256,256)
        volume = normalize(volume, data_range=self.data_range)
        volume = torch.from_numpy(volume)
        return volume

    def __getitem__(self, index):
        hr_volume = self.getAVol(self.filenames[index])
        h, w, _ = hr_volume.shape
        # adjust slice
        m = (hr_volume.shape[2]-1) % self.upscale
        if m != 0:
            hr_volume = hr_volume[...,:-m]

        ######## downsample data_type ########
        if self.data_type == 'direct':
            lr_volume = hr_volume[...,::self.upscale]
        elif self.data_type == 'linear':
            h,w,hr_slice_patch = hr_volume.shape
            _,_,lr_slice_patch = hr_volume[...,::self.upscale].shape
            lr_volume = F.interpolate(hr_volume.unsqueeze(0).unsqueeze(0), size=(h,w,lr_slice_patch), mode='trilinear', align_corners=False).squeeze(0).squeeze(0)
        elif self.data_type == 'bicubic':
            h,w,hr_slice_patch = hr_volume.shape
            _,_,lr_slice_patch = hr_volume[...,::self.upscale].shape
            # scipy.ndimage.zoom with order=3 is equivalent to bicubic
            zoom_factors = (1, 1, lr_slice_patch / hr_slice_patch)
            lr_volume_np = zoom(hr_volume.numpy(), zoom_factors, order=3, mode='grid-constant', grid_mode=True)
            lr_volume = torch.from_numpy(lr_volume_np)

        hr_slice_patch = (self.lr_slice_patch-1) * self.upscale + 1
        xyz_hr, _, proj_coord = util.to_pixel_samples([h,w,hr_slice_patch],scale=(1,1,self.upscale))

        cell = torch.ones_like(xyz_hr)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        cell[:, 2] *= 2 / hr_slice_patch


        name = self.filenames[index].split('/')[-1].split('.')[0]

        return {
            "lr_volume": lr_volume,
            "hr_volume": hr_volume,
            "xyz_hr": xyz_hr,
            "proj_coord": proj_coord,
            "upscale": self.upscale,
            "hr_size": [h,w,hr_slice_patch],
            "hr_slice": hr_slice_patch,
            "lr_slice": self.lr_slice_patch,
            "filename": name,
            "cell": cell,
        }

    def __len__(self):
        return self.file_len




def variable_size_collate_fn(batch):
    return batch
