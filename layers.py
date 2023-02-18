import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import stft, istft


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(x)

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding='same', bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              groups=in_channels, stride=stride, padding=padding, bias=bias)
    def forward(self, x):
        x = self.conv(x)
        return x

class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding='same', bias=True):
        super(PointwiseConv, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                              stride=stride, padding=padding, bias=bias)
    def forward(self, x):
        x = self.conv(x)
        return x


class Conv1dBlock(nn.Module):
    def __init__(self, input_dim, channels, kernal_size):
        super().__init__()
        self.point_conv1 = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            PointwiseConv(input_dim, 2 * channels, stride=1, bias=True),
            nn.GLU(dim=1)
        )
        self.depth_conv = nn.Sequential(
            DepthwiseConv(channels, channels, kernal_size, stride=1,bias=True),
            nn.BatchNorm1d(channels),
            Swish(),
        )
        self.point_conv2 = nn.Sequential(
            PointwiseConv(channels, input_dim, stride=1, bias=True),
            nn.PReLU()
        )
    def forward(self,x):
        # bc,f,t
        out = self.point_conv1(x)
        out = self.depth_conv(out)
        out = self.point_conv2(out)
        return out


class FFN(nn.Module):
    def __init__(self, input_dim, ffn_dim, gru=False):
        super().__init__()
        self.gru = gru
        self.sequential1 = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.GRU(input_dim, ffn_dim, 1) if gru else torch.nn.Linear(input_dim, ffn_dim, bias=True),
        )
        self.sequential2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(ffn_dim, input_dim, bias=True)
        )
    def forward(self,input):
        if self.gru:
            input, _ = self.sequential1(input)
        else:
            input = self.sequential1(input)
        input = self.sequential2(input)
        return input


class TCN(nn.Module):
    '''
    required:    (b,f,t)
    input_dim:  frequency dim or channel dim of input,
    '''
    def __init__(self,input_dim, dilation=1):
        super().__init__()
        self.input_dim = input_dim
        self.dilation = dilation
        self.r = 1
        self.tcn = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.PReLU(),
            PointwiseConv(self.input_dim, self.input_dim//self.r),

            nn.BatchNorm1d(input_dim//self.r),
            nn.PReLU(),
            nn.Conv1d(self.input_dim//self.r, self.input_dim//self.r, 3, padding='same',dilation=self.dilation),
            
            nn.BatchNorm1d(input_dim//self.r),
            nn.PReLU(),
            PointwiseConv(self.input_dim//self.r, self.input_dim)
        )
    
    def forward(self,x):
        # b,f,t
        residual = x
        x = self.tcn(x)
        x = x + residual
        return x


class Block(nn.Module):
    '''
    required:   (b,t,f)
    input_dim:  frequency dim or channel dim of input,
    ffn_dim:    frequency dim or channel dim inside attention,
    hidden_dim: frequency dim or channel dim inside convolution,
    num_frames: time dim of input,
    '''
    def __init__(self, input_dim, ffn_dim, hidden_dim, kernal_size, num_heads, num_frames):
        super().__init__()
        self.ffn1 = FFN(input_dim, ffn_dim, gru=False)

        self.time_attn_layer_norm = nn.LayerNorm(input_dim)
        self.time_attention = nn.MultiheadAttention(input_dim,num_heads,batch_first=True)

        self.conv_block = Conv1dBlock(input_dim, hidden_dim, kernal_size)

        self.freq_attn_layer_norm = nn.LayerNorm(num_frames)
        self.freq_attention = nn.MultiheadAttention(num_frames,num_heads,batch_first=True)

        self.ffn2 = FFN(input_dim, ffn_dim, gru=False)

    def forward(self,input):
        # ffn_1: b,t,f
        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual

        # time attention: b,t,f
        residual = x
        x = self.time_attn_layer_norm(x)
        x, _ = self.time_attention(query=x,key=x,value=x,need_weights=True)
        x = x + residual

        # conv: b,t,f
        residual = x
        x = x.transpose(-1,-2)   # b,f,t
        x = self.conv_block(x)
        x = x.transpose(-1,-2)
        x = residual + x

        # freq attention: b,t,f
        residual = x
        x = x.transpose(-1,-2)  # b,f,t
        x = self.freq_attn_layer_norm(x)
        x, _ = self.freq_attention(query=x, key=x, value=x, need_weights=True)
        x = x.transpose(-1,-2)
        x = x + residual

        # ffn_2: b,t,f
        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual
        return x


class Net(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.w = torch.hann_window(512+2)[1:-1].cuda()

        ######### STAGE I ##############
        self.B = cfg['B']
        self.H = cfg['H']
        self.L = cfg['L']
        self.ffn_dim = cfg['ffn_dim']
        self.hidden_dim = cfg['hidden_dim']
        self.kernal_size = 3
        self.num_frames = 63744 // 256 - 1 + 2
        self.encoder1 = nn.ModuleList([
            TCN(256, 1) for i in range(self.L)
        ])
        self.net1 = nn.ModuleList([
            Block(256,
                self.ffn_dim,
                self.hidden_dim,
                self.kernal_size,
                self.H,
                self.num_frames) for i in range(self.B)
        ])
        self.decoder1 = nn.ModuleList([
            TCN(256, 1) for i in range(self.L)
        ])
        self.last_linear1 = nn.Linear(256,257)

    def forward(self,input):
        '''
        input:  noisy waveform of (b,t)
        output: irm of (b,f,t),
                enhanced waveform of (b,t')
        '''
        noisy_wav = input

        ########## apply stft, return (b,f,t) ################
        noisy_cmp = stft(noisy_wav,window=self.w,n_fft=512,hop_length=256,
                            center=True,return_complex=True)
        noisy_abs, noisy_ang = torch.abs(noisy_cmp), torch.angle(noisy_cmp)

        ############################################################
        ############ stage I, return (b,f,t) ######################
        ############################################################
        mag = noisy_abs[:,1:,:]
        for layer in self.encoder1:
            mag = layer(mag)
        mag = mag.transpose(-1, -2)
        for layer in self.net1: 
            mag = layer(mag)
        mag = mag.transpose(-1, -2)     # ->(b,f',t)
        for layer in self.decoder1:
            mag = layer(mag)
        mag = self.last_linear1(mag.transpose(-1, -2)).transpose(-1, -2)     # ->(b,f,t)
        irm = torch.sigmoid(mag)

        ################ recover waveform, return (b,t')##############
        enhanced_cmp = noisy_abs * irm * torch.exp(1j * noisy_ang)
        enhanced_wav_1 = istft(enhanced_cmp,window=self.w,n_fft=512,hop_length=256,center=True)

        return irm, enhanced_wav_1