import os
from random import *

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from torch import stft, istft

import util

class VBDDataset(Dataset):
    def __init__(self, cfg, type):
        super().__init__()
        if type=='train':
            self.clean_dir = '/home/wqr/vbd/clean_trainset_28spk_wav'
            self.noisy_dir = '/home/wqr/vbd/noisy_trainset_28spk_wav'
        elif type=='val':
            self.clean_dir = '/home/wqr/vbd/clean_testset_wav'
            self.noisy_dir = '/home/wqr/vbd/noisy_testset_wav'
        self.clean_name = os.listdir(self.clean_dir)
        self.noisy_name = os.listdir(self.noisy_dir)
        self.clean_name.sort()
        self.noisy_name.sort()

    def __len__(self):
        return len(self.clean_name)

    def __getitem__(self, item):
        noisy_wav = torchaudio.load(os.path.join(self.noisy_dir, self.noisy_name[item]))[0][0]
        target_wav = torchaudio.load(os.path.join(self.clean_dir, self.clean_name[item]))[0][0]
        noisy_wav = noisy_wav - torch.mean(noisy_wav)
        target_wav = target_wav - torch.mean(target_wav)
        length = len(noisy_wav)
        if len(noisy_wav) < 63744:
            noisy_wav = F.pad(noisy_wav, (0, 63744 - length))
            target_wav = F.pad(target_wav, (0, 63744 - length))
        else:
            start = randint(0, length - 63744)
            noisy_wav = noisy_wav[start : start + 63744]
            target_wav = target_wav[start : start + 63744]
        return noisy_wav, target_wav



class OnFlyDataset(Dataset):
    def __init__(self, cfg, type):
        super(OnFlyDataset, self).__init__()
        self.length = int(cfg['data_hours'] * 3600 // 4)
        self.cfg = cfg
        self.type = type
        self.clean_dir = '/mnt/g/WQR/DNS_datasets/clean'
        self.noise_dir = '/mnt/g/WQR/DNS_datasets/noise'
        self.clean_name = os.listdir(self.clean_dir)
        self.noise_name = os.listdir(self.noise_dir)   

    def __len__(self):
        return self.length if self.type=='train' else int(0.5 * 3600 // 4)

    def _mix(self,clean,noise,snr):
        clean_energy = torch.sum(torch.square(clean))
        noise_energy = torch.sum(torch.square(noise))
        A = torch.sqrt(clean_energy / (1e-6 + noise_energy) / (10**(0.1*snr)))
        mix = clean + A * noise
        return mix, clean

    def __getitem__(self, item):
        snr = randint(0,20)
        length = 63744
        clean_id = randrange(len(self.clean_name))
        noise_id = randrange(len(self.noise_name))
        clean = torchaudio.load(os.path.join(self.clean_dir,self.clean_name[clean_id]))[0][0][:length]
        noise = torchaudio.load(os.path.join(self.noise_dir,self.noise_name[noise_id]))[0][0][:length]
        clean = clean - torch.mean(clean)
        noise = noise - torch.mean(noise)
        if not len(noise)==length:
            noise = torchaudio.load(os.path.join(self.noise_dir,self.noise_name[noise_id+1]))[0][0][:length]
        noisy,clean = self._mix(clean,noise,snr)
        return noisy, clean