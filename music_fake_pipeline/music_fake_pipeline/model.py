"""
음악 REAL/FAKE 이진분류용 소형 CNN.

log-mel spectrogram(1, N_MELS, N_FRAMES)을 이미지처럼 받아
conv-bn-relu-maxpool 블록을 반복한 뒤, 전역 평균 풀링(GAP) + FC 1개로
단일 로짓(logit)을 출력합니다. DF-Arena(1.1B 파라미터)와 비교해
의도적으로 매우 가볍게 설계했습니다 — 60분/1200파일 추론 시간 제약 안에서
"기존 DF-Arena 대비 오히려 연산량을 줄이는" 교체가 되도록 하기 위함입니다.
"""

import torch
import torch.nn as nn

import config


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class MusicFakeClassifier(nn.Module):
    """
    입력 : (batch, 1, N_MELS, N_FRAMES)
    출력 : (batch,) 의 raw logit  (sigmoid 적용 전. BCEWithLogitsLoss와 함께 사용)
    """

    def __init__(self, n_mels: int = config.N_MELS):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features_map = self.blocks(x)
        pooled = self.global_pool(features_map)
        logit = self.classifier(pooled)
        return logit.squeeze(-1)  # (batch,)

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.forward(x)
        return torch.sigmoid(logit)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 빠른 shape/파라미터 수 점검용
    net = MusicFakeClassifier()
    dummy = torch.randn(2, 1, config.N_MELS, config.N_FRAMES)
    out = net(dummy)
    print(f"output shape: {tuple(out.shape)}")
    print(f"parameters  : {count_parameters(net):,}")
