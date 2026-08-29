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

────────────────────────────────────────────────────────────────────────
[version5] 이 파일은 **수정본이다** — 원본 무수정 원칙을 여기서만 해제한다.
────────────────────────────────────────────────────────────────────────
근거: 선행 구조는 인코더 깊이 8(MaxPool 2배 × 8 = 256배)이라 3,600표본이 인코딩
시간축 14개로 눌린다. 조각 하나가 0.71초로 QRS 폭(~0.1초)의 7배다. 선행은 PPG 심박
추출이 목표라 그 해상도로 충분했으나, ECG 의 QRS 세부와 고주파 잡음은 조각 안에서
뭉개진다. 손실 설계(16런)와 용량(hidden 64→256)에서 효과가 없었으므로 남은 축이
시간 해상도다.

변경 지점 두 곳뿐이다.
  1. EncoderBlock · DecoderBlock 에 `dilation` 인자를 추가했다. 기본 1 이면 원본과
     완전히 같다. 층을 빼면 수용영역이 줄어드는데(깊이 8 → 3,316표본 = 9.21초,
     깊이 6 → 820표본 = 2.28초), bw 의 느린 출렁임을 보려면 그만큼이 필요하다.
     dilation 을 키워 수용영역을 보전한다. padding 도 dilation 배로 맞춘다.
  2. ConvolutionalEncoder · ConvolutionalDecoder · ConvolutionalAutoencoder 가
     블록별 dilation 목록을 받아 넘긴다. `dilations=None` 이면 전부 1 — 원본과 동일.

깊이 자체는 `channels` 길이로 정해지므로 원본 코드가 이미 지원한다 (수정 불필요).

우리 인터페이스(component/masked_reconstruct, 패딩·크롭)와 MSE 손실은 각각
meae.py / losses.py 에서 이 파일을 감싸 구현한다.
`torch.nn.utils.weight_norm` FutureWarning은 원본 그대로 두며 수정하지 않는다.
"""

import torch
import torch.nn as nn

from torch.nn.utils import weight_norm

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3,
                 input_padding=0, dilation=1):
        super(EncoderBlock, self).__init__()
        # [version5] dilation 추가. padding 을 dilation 배로 맞춰 길이를 보존한다.
        pad = padding * dilation
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                      stride=stride, padding=pad+input_padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels, momentum=0.8),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, 
                      stride=stride, padding=pad, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(out_channels, momentum=0.8),
        )
        
    def forward(self, x):
        return self.block(x)

class ConvolutionalEncoder(nn.Module):
    def __init__(self, input_length, channels, hidden, input_padding=0, dilations=None):
        super(ConvolutionalEncoder, self).__init__()
        # [version5] 블록별 dilation. None 이면 전부 1 — 원본과 동일하다.
        d = list(dilations) if dilations else [1] * (len(channels) - 1)
        
        self.encoder = nn.Sequential()
        self.encoder.append(EncoderBlock(channels[0], channels[1],
                                         input_padding=input_padding, dilation=d[0]))
        for c_i in range(1, len(channels)-1):
            self.encoder.append(EncoderBlock(channels[c_i], channels[c_i+1],
                                             dilation=d[c_i]))

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(channels[-1], hidden, kernel_size=1, 
                      stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x, skip_levels=None):
        """[version5] skip_levels 가 주어지면 그 블록의 출력을 함께 돌려준다.

        블록 j 의 출력은 길이 L/2^(j+1), 채널 enc_channels[j] 다. K개 인코더의 것을
        채널 축으로 이어 붙이면 디코더의 대응 지점과 모양이 정확히 맞는다.
        """
        skips = []
        want = set(skip_levels or ())
        for j, blk in enumerate(self.encoder):
            x = blk(x)
            if j in want:
                skips.append(x)
        z = self.encoder_conv(x)
        return (z, skips) if skip_levels else z
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, c_i, 
                 norm_type, num_encoders, num_channels, input_length, input_padding=0,
                 kernel_size=7, stride=1, padding=3, dilation=1):
        super(DecoderBlock, self).__init__()
        # [version5] dilation 추가. 인코더와 대칭이 되도록 같은 값을 받는다.
        pad = padding * dilation
        self.block = nn.Sequential()
        self.block.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.block.append(nn.ConvTranspose1d(in_channels=in_channels, 
                                             out_channels=out_channels, 
                                             kernel_size=kernel_size, stride=stride, 
                                             padding=pad+input_padding,
                                             dilation=dilation))
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
    def __init__(self, input_length, channels, hidden, num_encoders, input_padding=0,
                 norm_type='none', dilations=None):
        super(ConvolutionalDecoder, self).__init__()
        self.input_length = input_length
        self.num_channels = len(channels)
        self.channels = channels
        # [version5] 블록별 dilation. 인코더 블록 i 와 디코더 블록 c_i=i+1 이 짝이다.
        d = list(dilations) if dilations else [1] * (len(channels) - 1)
        
        NORM_TYPES = ['none', 'batch_norm', 'batch_group_norm', 'group_norm', 'layer_norm', 'instance_norm']
        assert norm_type in NORM_TYPES, f'Given norm type, {norm_type}, not in {NORM_TYPES}.'

        # create convolutional decoder
        # [version5] head(1x1 conv + ReLU + norm)와 블록들을 나눠 둔다 — 잔차를 중간에
        # 더하려면 주입 지점이 필요하다. `self.decoder` 는 하위호환을 위해 그대로 남긴다.
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
        self.n_head = len(self.decoder)
        for c_i in reversed(range(2, len(channels))):
            self.decoder.append(DecoderBlock(channels[c_i], channels[c_i-1], c_i, 
                                             norm_type, num_encoders, len(channels), 
                                             input_length, dilation=d[c_i-1]))
        self.decoder.append(DecoderBlock(channels[1], channels[0], 1, 
                                             norm_type, num_encoders, len(channels), 
                                             input_length, input_padding=input_padding,
                                             dilation=d[0]))

    def forward(self, z, skips=None, skip_levels=None, skip_weight=0.0):
        """[version5] skips 를 주면 대응 지점에서 더한다.

        인코더 블록 j 의 출력은 디코더 블록 c_i=j+1 **직전**에 들어간다. 블록 목록이
        [c_i=D, …, 2, 1] 순이므로 주입 위치(블록 인덱스)는 D-j-2 다음이다.
        모양은 그 지점에서 정확히 일치한다 — 길이 L/2^(j+1), 채널 channels[j].
        """
        y = torch.concatenate(z, dim=1)
        n_blocks = len(self.decoder) - self.n_head
        at = {}
        if skips and skip_weight:
            for j, sk in zip(skip_levels or (), skips):
                at[n_blocks - j - 2] = sk
        for i, layer in enumerate(self.decoder):
            y = layer(y)
            b = i - self.n_head                     # 블록 인덱스 (head 이후)
            if b in at:
                y = y + skip_weight * at[b]
        return y
            

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, input_channels=3, input_length=64, 
                 channels=[32, 64, 128], hidden=512, 
                 num_encoders=4, norm_type='none',
                 use_weight_norm=True, input_padding=0, dilations=None,
                 skip_levels=None, skip_weight=0.0):
        super(ConvolutionalAutoencoder, self).__init__()
        # [version5] 잔차 연결. skip_levels 가 비어 있거나 skip_weight=0 이면
        # 기존과 완전히 같은 경로다.
        self.skip_levels = list(skip_levels) if skip_levels else []
        self.skip_weight = float(skip_weight)
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
                                                      hidden=hidden//num_encoders, input_padding=input_padding,
                                                      dilations=dilations))
        # decoder layers
        self.decoder = ConvolutionalDecoder(input_length=input_length,
                                            channels=[channels[0]] + channels,
                                            hidden=hidden, num_encoders=num_encoders, 
                                            norm_type=norm_type, input_padding=input_padding,
                                            dilations=dilations)
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

    def encode_all(self, x):
        """[version5] (인코딩 K개, 인코더별 잔차 목록). 잔차가 없으면 두 번째가 빈 목록.

        **잔차도 인코더별로 따로 들고 있어야** 성분을 뽑을 때 그 인코더 것만 살릴 수 있다.
        하나로 합쳐 두면 마스킹해도 입력 전체가 새어 들어간다.
        """
        if not self.skip_levels:
            return self.encode(x), [[] for _ in self.encoders]
        z, sk = [], []
        for encoder in self.encoders:
            zi, si = encoder(x, self.skip_levels)
            z.append(zi)
            sk.append(si)
        return z, sk
    
    def _merge(self, skips):
        """인코더별 잔차를 준위마다 채널 축으로 이어 붙인다 — 디코더 모양과 맞춘다."""
        if not skips or not self.skip_levels or not any(skips):
            return None
        return [torch.concatenate([sk[l] for sk in skips], dim=1)
                for l in range(len(self.skip_levels))]

    def decode(self, z, zeros_train=False, skips=None):
        if zeros_train:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm1d)\
                    or isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
                    m.eval()

        y = self.decoder(z, self._merge(skips), self.skip_levels, self.skip_weight)
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
        z, skips = self.encode_all(x)
        y = self.decode(z, skips=skips)

        return y, z