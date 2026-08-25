"""선행 다중 인코더 오토인코더 (1D) — 원본 이식.

원저작물을 그대로 이식한 파일이다 (내용 무수정).
  출처 : https://github.com/mbwebster/self-supervised-bss-via-multi-encoder-ae
  파일 : models/cnn_multi_enc_ae_1d.py
  커밋 : d0c94a9d5dec8dd5d54baebdb9963b79860cb200 (2025-12-13)
  라이선스 : MIT — Copyright (c) 2023 Matthew B. Webster

인용
  Webster MB, Lee J. Blind source separation via multi-encoder autoencoders.
    Neurocomputing 2025. doi:10.1016/j.neucom.2025.131008 (arXiv:2309.07138)
  Webster MB, Lee D, Lee J. Heart rate extraction from noisy PPG via
    multi-encoder autoencoders. Comput Biol Med 2025;199:111319 (arXiv:2504.09132)

수정하지 않는다. 우리 인터페이스(component/masked_reconstruct, 패딩·크롭)와
MSE 손실은 각각 meae.py / losses.py 에서 이 파일을 감싸 구현한다.
`torch.nn.utils.weight_norm` FutureWarning은 원본 그대로 두며 수정하지 않는다.
"""

import torch
import torch.nn as nn

from torch.nn.utils import weight_norm

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3, input_padding=0):
        super(EncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                      stride=stride, padding=padding+input_padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels, momentum=0.8),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, 
                      stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(out_channels, momentum=0.8),
        )
        
    def forward(self, x):
        return self.block(x)

class ConvolutionalEncoder(nn.Module):
    def __init__(self, input_length, channels, hidden, input_padding=0):
        super(ConvolutionalEncoder, self).__init__()
        
        self.encoder = nn.Sequential()
        self.encoder.append(EncoderBlock(channels[0], channels[1], input_padding=input_padding))
        for c_i in range(1, len(channels)-1):
            self.encoder.append(EncoderBlock(channels[c_i], channels[c_i+1]))

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(channels[-1], hidden, kernel_size=1, 
                      stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        z = self.encoder_conv(x)
        
        return z
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, c_i, 
                 norm_type, num_encoders, num_channels, input_length, input_padding=0,
                 kernel_size=7, stride=1, padding=3):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential()
        self.block.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.block.append(nn.ConvTranspose1d(in_channels=in_channels, 
                                             out_channels=out_channels, 
                                             kernel_size=kernel_size, stride=stride, 
                                             padding=padding+input_padding))
        self.block.append(nn.ReLU(inplace=True))
        if norm_type == 'batch_norm':
            self.block.append(nn.BatchNorm1d(out_channels, momentum=0.8))
        elif norm_type == 'group_norm':
            self.block.append(nn.GroupNorm(num_encoders, out_channels))
        elif norm_type == 'layer_norm':
            down_sample = (num_channels-2) - c_i
            self.block.append(nn.LayerNorm([out_channels, input_length//(2**down_sample), input_length//(2**down_sample)]))
        elif norm_type == 'instance_norm':
            self.block.append(nn.InstanceNorm1d(out_channels))

    def forward(self, x):
        return self.block(x)
    
class ConvolutionalDecoder(nn.Module):
    def __init__(self, input_length, channels, hidden, num_encoders, input_padding=0, norm_type='none'):
        super(ConvolutionalDecoder, self).__init__()
        self.input_length = input_length
        self.num_channels = len(channels)
        self.channels = channels
        
        NORM_TYPES = ['none', 'batch_norm', 'batch_group_norm', 'group_norm', 'layer_norm', 'instance_norm']
        assert norm_type in NORM_TYPES, f'Given norm type, {norm_type}, not in {NORM_TYPES}.'

        # create convolutional decoder
        self.decoder = nn.Sequential()
        self.decoder.append(nn.Conv1d(hidden, channels[-1], kernel_size=1, 
                            stride=1, padding=0))
        self.decoder.append(nn.ReLU(inplace=True))
        if norm_type == 'batch_norm':
            self.decoder.append(nn.BatchNorm1d(channels[-1]))
        elif norm_type == 'layer_norm':
            self.decoder.append(nn.LayerNorm([channels[-1]]))
        elif norm_type == 'group_norm':
            self.decoder.append(nn.GroupNorm(num_encoders, channels[-1]))
        elif norm_type == 'instance_norm':
            self.decoder.append(nn.InstanceNorm1d(channels[-1]))
        for c_i in reversed(range(2, len(channels))):
            self.decoder.append(DecoderBlock(channels[c_i], channels[c_i-1], c_i, 
                                             norm_type, num_encoders, len(channels), 
                                             input_length))
        self.decoder.append(DecoderBlock(channels[1], channels[0], 1, 
                                             norm_type, num_encoders, len(channels), 
                                             input_length, input_padding=input_padding))
    def forward(self, z):
        z = torch.concatenate(z, dim=1)
        y = self.decoder(z)
        
        return y
            

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, input_channels=3, input_length=64, 
                 channels=[32, 64, 128], hidden=512, 
                 num_encoders=4, norm_type='none',
                 use_weight_norm=True, input_padding=0):
        super(ConvolutionalAutoencoder, self).__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.channels = channels
        self.hidden = hidden
        self.num_encoders = num_encoders
        
        # encoder layers
        enc_channels = [c//num_encoders for c in channels]
        self.encoders = nn.ModuleList()
        for _ in range(num_encoders):
            self.encoders.append(ConvolutionalEncoder(input_length=input_length,
                                                      channels=[input_channels] + enc_channels,
                                                      hidden=hidden//num_encoders, input_padding=input_padding))
        # decoder layers
        self.decoder = ConvolutionalDecoder(input_length=input_length,
                                            channels=[channels[0]] + channels,
                                            hidden=hidden, num_encoders=num_encoders, 
                                            norm_type=norm_type, input_padding=input_padding)
        # output layer
        if use_weight_norm:
            self.output = nn.Sequential(
                weight_norm(nn.Conv1d(in_channels=channels[0], 
                                      out_channels=input_channels, 
                                      kernel_size=1, stride=1, 
                                      padding=0))
            )
        else:
            self.output = nn.Sequential(
                nn.Conv1d(in_channels=channels[0], 
                          out_channels=input_channels, 
                          kernel_size=1, stride=1, 
                          padding=0)
            )
    
    def encode(self, x):
        z = []
        for encoder in self.encoders:
            z.append(encoder(x))
            
        return z
    
    def decode(self, z, zeros_train=False):
        if zeros_train:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm1d)\
                    or isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
                    m.eval()

        y = self.decoder(z)
        y = self.output(y)
        
        if zeros_train:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm1d)\
                    or isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
                    m.weight.requires_grad_(True)
                    m.bias.requires_grad_(True)
                    m.train()
        
        return y
    
    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)

        return y, z