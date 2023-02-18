import librosa
import numpy as np
import yaml
import torch
from scipy.fft import irfft

eps = 1e-6

def mean(lst):
    return sum(lst)/len(lst)

def toTensor(x,mode):
    'convert to tensor'
    if mode=='float':
        x = torch.as_tensor(data=x,
                            dtype=torch.float32,)
    elif mode=='complex':
        x = torch.as_tensor(data=x,
                            dtype=torch.complex64)
    return x


def get_tf_represent(cfg, x, mode='complex',drop_highest=False):
    'return stft'

    fft_length = cfg['fft_length']
    hop_length = cfg['hop_length']

    stft = librosa.stft(x,
                        n_fft=fft_length,
                        hop_length=hop_length,
                        center=False
                        )
    if drop_highest:
        stft = stft[:-1,:]
    if mode == 'complex':
        return stft
    elif mode == 'mag':
        return np.abs(stft)
    elif mode == 'lps':
        return np.log10(np.abs(stft) + eps)
    elif mode == ('angle' or 'phase'):
        return np.angle(stft)
    else:
        raise ValueError


def istft(cfg, x):
    '''
    Using OLA method.
    x is the Complex STFT of waveform, supposed to be (f,t).
    NOTICE: Theoritically, the result should *hop/W(0) to perfectly reconstruct.
            However, when use hanning window and hop==win/2, hop/W(0)=1 exactly.
    '''
    fft_length = cfg['fft_length']
    hop_length = cfg['hop_length']

    n_frame = x.shape[-1]
    wave_length = (n_frame - 1) * hop_length + fft_length
    wave = np.zeros(shape=wave_length)

    for i in range(n_frame):
        this_frame = np.real(irfft(x[:, i],n=cfg['fft_length']))
        wave[i * hop_length: i * hop_length + fft_length] += this_frame

    # 防止wave出现爆炸脉冲的情况！！！
    # 为何出现？Todo_
    wave[wave>1] = 1
    wave[wave<-1] = -1
    return wave


def get_yaml_data(yaml_file):
    with open(yaml_file, 'r', encoding="utf-8") as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return data