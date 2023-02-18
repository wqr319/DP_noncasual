from typing import Any

import numpy as np
import pysepm
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning.loggers as loggers
import torch.nn.functional as F
import torch.optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from layers import *
from datasets import *
from torch import stft, istft

eps = 1e-6


##################################################################################
########################### Research Part ########################################
##################################################################################
class LightningNet(pl.LightningModule):
    def __init__(self, cfg, steps_per_epoch):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg['learning_rate']
        self.max_epoch = cfg['max_epoch']
        self.save_hyperparameters()
        self.model = Net(cfg=cfg)
        # self.model.load_state_dict(torch.load(''))
        self.steps_per_epoch = steps_per_epoch
        self.dataset = cfg['dataset']

        self.w = torch.hann_window(512+2)[1:-1].cuda()

    def training_step(self, batch, batch_idx):
        noisy_wav, target_wav = batch
        predict_irm, enhanced_wav = self.model(noisy_wav)
        ################## Stage I ###################
        noisy_wav_long = F.pad(noisy_wav, (256, 256))
        target_wav_long = F.pad(target_wav, (256, 256))
        noisy = torch.abs(stft(noisy_wav_long,window=self.w,n_fft=512,hop_length=256,
                            center=False,return_complex=True))
        target = torch.abs(stft(target_wav_long,window=self.w,n_fft=512,hop_length=256,
                            center=False,return_complex=True))
        target_irm = target / (noisy + eps)
        target_irm.clamp_(0, 1)
        loss_1 = F.mse_loss(predict_irm, target_irm)
        ################## Stage II ###################
        loss_2 = F.mse_loss(enhanced_wav, target_wav)

        loss = loss_1 + loss_2

        self.log('train_loss', value=loss,
                 on_step=False, on_epoch=True, batch_size=self.cfg['batch_size'])
        return loss

    def validation_step(self, batch, batch_idx):
        noisy_wav, target_wav = batch
        predict_irm, enhanced_wav = self.model(noisy_wav)
        ################## Stage I ###################
        noisy_wav_long = F.pad(noisy_wav, (256, 256))
        target_wav_long = F.pad(target_wav, (256, 256))
        noisy = torch.abs(stft(noisy_wav_long,window=self.w,n_fft=512,hop_length=256,
                            center=False,return_complex=True))
        target = torch.abs(stft(target_wav_long,window=self.w,n_fft=512,hop_length=256,
                            center=False,return_complex=True))
        target_irm = target / (noisy + eps)
        target_irm.clamp_(0, 1)
        loss_1 = F.mse_loss(predict_irm, target_irm)
        ################## Stage II ###################
        loss_2 = F.mse_loss(enhanced_wav, target_wav)

        loss = loss_1 + loss_2

        self.log('val_loss', value=loss,
                 on_step=False, on_epoch=True, batch_size=self.cfg['batch_size'])
        return loss

    def on_validation_epoch_end(self) -> None:
        if (not self.current_epoch%10==0):
            return None
        
        ###################################################
        ################# VBD #############################
        ###################################################
        csig_list, cbak_list, covl_list = [], [], []
        pesq_list = []
        clean_list = os.listdir('/home/wqr/vbd/clean_testset_wav')
        noisy_list = os.listdir('/home/wqr/vbd/noisy_testset_wav')
        clean_list.sort()
        noisy_list.sort()
        for i, (clean_name, noisy_name) in tqdm(enumerate(zip(clean_list, noisy_list))):
            noisy_wav = torchaudio.load(os.path.join('/home/wqr/vbd/noisy_testset_wav', noisy_name))[0][0].cuda()
            target_wav = torchaudio.load(os.path.join('/home/wqr/vbd/clean_testset_wav', clean_name))[0][0].cuda()
            target_wav = target_wav.detach().cpu().numpy()

            if len(noisy_wav) <= 63744:
                real_length = len(noisy_wav)
                noisy_wav = F.pad(noisy_wav, (0, 63744-len(noisy_wav)))

                irm, enhanced_wav = self.model(noisy_wav.unsqueeze(0))
                enhanced_wav = enhanced_wav.squeeze()[:real_length].detach().cpu().numpy()

            else:
                num_clips = len(noisy_wav)//63744 + 1
                noisy_wav_long = F.pad(noisy_wav, (0, (num_clips+1) * 63744 - len(noisy_wav)))
                enhanced_clips = []
                for j in range(num_clips+1):
                    noisy_clip = noisy_wav_long[j * 63744 : (j+1) * 63744]
                    irm, enhanced_clip = self.model(noisy_clip.unsqueeze(0))
                    enhanced_clip = enhanced_clip.squeeze().detach().cpu().numpy()
                    enhanced_clips.append(enhanced_clip)
                enhanced_wav = np.zeros(((num_clips+1)*63744,))
                for k, clip in enumerate(enhanced_clips):
                    enhanced_wav[k * 63744 : (k + 1) * 63744] = clip
                enhanced_wav = enhanced_wav[:len(noisy_wav)]
            if i==10:
                plt.figure()
                plt.plot(target_wav)
                plt.plot(enhanced_wav+1)
                plt.grid()
                # plt.show()
                plt.savefig('1.jpg')

            try:
                csig, cbak, covl = pysepm.composite(target_wav, enhanced_wav, 16000)
                pesq = pysepm.pesq(target_wav, enhanced_wav, 16000)[1]
            except:
                pass
            else:
                csig_list.append(csig)
                cbak_list.append(cbak)
                covl_list.append(covl)
                pesq_list.append(pesq)
        def mean(lst):
            return sum(lst) / len(lst)
        self.log('csig', mean(csig_list), batch_size=1)
        self.log('cbak', mean(cbak_list), batch_size=1)
        self.log('covl', mean(covl_list), batch_size=1)
        self.log('pesq', mean(pesq_list), batch_size=1)
        
        ###################################################
        ################# DNS #############################
        ###################################################
        pesq_list_dns = []
        def take_last(name):
            name = name.split('.')[0]
            id = name.split('_')[-1]
            return id

        clean_files = os.listdir('/mnt/g/WQR/DNS_datasets/test_set/synthetic/no_reverb/clean')
        noisy_files = os.listdir('/mnt/g/WQR/DNS_datasets/test_set/synthetic/no_reverb/noisy')
        clean_files.sort(key=take_last)
        noisy_files.sort(key=take_last)
        for i, (clean_name, noisy_name) in tqdm(enumerate(zip(clean_files, noisy_files))):
            noisy_wav = torchaudio.load(
                    os.path.join('/mnt/g/WQR/DNS_datasets/test_set/synthetic/no_reverb/noisy', noisy_name))[0][0].cuda()
            target_wav = torchaudio.load(
                    os.path.join('/mnt/g/WQR/DNS_datasets/test_set/synthetic/no_reverb/clean', clean_name))[0][0].cuda()
            target_wav = target_wav.detach().cpu().numpy()

            num_clips = len(noisy_wav)//63744 + 1
            noisy_wav_long = F.pad(noisy_wav, (0, (num_clips+1) * 63744 - len(noisy_wav)))
            enhanced_clips = []
            for j in range(num_clips+1):
                noisy_clip = noisy_wav_long[j * 63744 : (j+1) * 63744]
                irm, enhanced_clip = self.model(noisy_clip.unsqueeze(0))
                enhanced_clip = enhanced_clip.squeeze().detach().cpu().numpy()
                enhanced_clips.append(enhanced_clip)
            enhanced_wav = np.zeros(((num_clips+1)*63744,))
            for k, clip in enumerate(enhanced_clips):
                enhanced_wav[k * 63744 : (k + 1) * 63744] = clip
            enhanced_wav = enhanced_wav[:len(noisy_wav)]
        
        noisy_wav = noisy_wav.detach().cpu().numpy()
        csig, cbak, covl = pysepm.composite(target_wav, enhanced_wav, 16000)
        pesq = pysepm.pesq(target_wav, enhanced_wav, 16000)[1]
        pesq_list_dns.append(pesq)
        self.log('pesq_dns', mean(pesq_list_dns), batch_size=1)

        return None

    def forward(self, input):
        _, out = self.model(input)
        return out

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                            self.learning_rate,
                                                            steps_per_epoch=self.steps_per_epoch,
                                                            epochs=self.cfg['max_epoch'])
        return self.optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.current_epoch<0.95*self.cfg['max_epoch']:
            self.scheduler.step()
        self.log('lr',self.scheduler.get_last_lr()[0])

##################################################################################
########################### Callbacks & Trainer Part #############################
##################################################################################
def build_callbacks(cfg):
    ###################### callbacks #########################################
    my_callbacks = [
        callbacks.ModelSummary(max_depth=2),
        callbacks.ModelCheckpoint(
            monitor='pesq',
            mode='max',
            dirpath='log/{}'.format(cfg['dataset']),
            filename='{epoch}'
        )
    ]
    logger = loggers.TensorBoardLogger(
        save_dir='log',
        name='{}'.format(cfg['dataset'])
    )
    return my_callbacks, logger

def build_trainer(cfg):
    my_callbacks, logger = build_callbacks(cfg=cfg)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        logger=logger,
        callbacks=my_callbacks,
        detect_anomaly=False,
        fast_dev_run=False,
        log_every_n_steps=1,
        max_epochs=cfg['max_epoch'],
        gradient_clip_val=cfg['gradient_clip_norm'],
        # strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count()>1 else None,
        num_sanity_val_steps=0,
        precision=cfg['precision']
    )
    return trainer
