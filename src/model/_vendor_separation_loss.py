"""선행 sparse mixing 손실 — 원본 이식.

원저작물을 그대로 이식한 파일이다 (내용 무수정).
  출처 : https://github.com/mbwebster/self-supervised-bss-via-multi-encoder-ae
  파일 : models/separation_loss.py
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

class WeightSeparationLoss(nn.Module):
    def __init__(self, num_splits, mode='L1'):
        super(WeightSeparationLoss, self).__init__()
        self.num_splits = num_splits
        self.mode = mode
        assert mode in ['L1', 'L2'], f'Weight separation loss has an invalid argument for the \
            normalization mode: {mode}. Must be either L1 or L2.'

    def forward(self, model_item):
        loss = 0
        for name, w in model_item.named_parameters():
            if name.split('.')[-1] == 'weight' and len(w.shape) in [2, 3, 4]:
                w_out = w.shape[0]//self.num_splits
                w_in = w.shape[1]//self.num_splits
                for i in range(self.num_splits):
                    for j in range(self.num_splits):
                        if self.mode == 'L1' and i != j:
                            loss += torch.mean(torch.abs(w[w_out*i:w_out*(i+1), w_in*j:w_in*(j+1)]))
                        elif self.mode == 'L2' and i != j:
                            loss += torch.mean(torch.abs(w[w_out*i:w_out*(i+1), w_in*j:w_in*(j+1)]**2))
                    
        return loss

class WeightSeparationLossAlternative(nn.Module):
    def __init__(self, num_splits, mode='L1'):
        super(WeightSeparationLossAlternative, self).__init__()
        self.num_splits = num_splits
        self.mode = mode
        assert mode in ['L1', 'L2'], f'Weight separation loss has an invalid argument for the \
            normalization mode: {mode}. Must be either L1 or L2.'

    def forward(self, model_item):
        loss = 0
        for name, w in model_item.named_parameters():
            if name.split('.')[-1] == 'weight' and len(w.shape) in [2, 3, 4]:
                w_l = w.shape[0]//self.num_splits
                w_w = w.shape[1]//self.num_splits
                for i in range(self.num_splits-1):
                    if self.mode == 'L1':
                        loss += torch.mean(torch.abs(w[i*w_l:(i+1)*w_l, (i+1)*w_w:]))
                        loss += torch.mean(torch.abs(w[(i+1)*w_l:(i+2)*w_l, :(i+1)*w_w]))
                    elif self.mode == 'L2':
                        loss += torch.mean(w[i*w_l:(i+1)*w_l, (i+1)*w_w:]**2)
                        loss += torch.mean(w[(i+1)*w_l:(i+2)*w_l, :(i+1)*w_w]**2)
                    
        return loss
