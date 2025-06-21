import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """
    2D CNN 기반 회귀 모델 클래스.

    이 모델은 (grid_size, grid_size, 1)의 격자형 입력(lattice)을 Convolution Layer를 통해 처리한 후,
    추가 입력인 ca_int와 dL을 fully connected layer와 합쳐 최종 회귀 출력을 생성합니다.

    Parameters
    ----------
    grid_size : int
        입력 lattice의 한 변 크기 (예: 32이면 입력은 (32, 32) 크기).
    ca_int_dim : int, optional
        추가 입력 ca_int의 feature 차원 수. 기본값은 1.
    dL_dim : int, optional
        추가 입력 dL의 feature 차원 수. 기본값은 1.
    conv_layers : int, optional
        Convolution layer의 개수. 기본값은 5.
    conv_channels : int, optional
        각 Convolution layer의 출력 채널 수. 기본값은 10.
    conv_kernel : int, optional
        Convolution layer의 커널 크기. 기본값은 4.
    conv_stride : int, optional
        Convolution layer의 stride. 기본값은 1.
    pool_kernel : int, optional
        MaxPooling의 커널 크기. 기본값은 2.
    pool_stride : int, optional
        MaxPooling의 stride. 기본값은 2.
    fc_dims : list of int, optional
        Fully connected layer의 hidden 차원들. 기본값은 [1000, 100].
    dropout_rate : float, optional
        dropout rate (0~1). 기본값은 0.1.
    """
    def __init__(self, grid_size, ca_int_dim=1, dL_dim=1, 
                 conv_layers=5, conv_channels=10, 
                 conv_kernel=4, conv_stride=1,
                 pool_kernel=2, pool_stride=2,
                 fc_dims=[1000, 100], dropout_rate=0.1):
        super().__init__()

        # Convolution layer들을 쌓기 위한 리스트
        self.convs = nn.ModuleList()
        in_channels = 1  # 입력 채널은 1 (흑백 이미지와 동일)

        for _ in range(conv_layers):
            self.convs.append(
                nn.Conv2d(in_channels, conv_channels, kernel_size=conv_kernel, stride=conv_stride)
            )
            in_channels = conv_channels  # 다음 layer의 입력 채널 갱신

        # Convolution과 Pooling을 반복한 후 feature map의 최종 크기 계산
        size = grid_size
        for _ in range(conv_layers):
            size = (size - conv_kernel)//conv_stride + 1  # conv output size
            size = (size - pool_kernel)//pool_stride + 1  # pooling output size

        # 마지막 convolution 결과를 flatten했을 때 차원
        conv_out_dim = conv_channels * size * size

        # Pooling layer 및 dropout 정의
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected 입력 차원 = conv 출력 + ca_int + dL
        fc_input_dim = conv_out_dim + ca_int_dim + dL_dim

        # Fully connected layer 정의
        self.fcs = nn.ModuleList()
        in_dim = fc_input_dim
        for h_dim in fc_dims:
            self.fcs.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim

        # 최종 출력 layer (회귀 문제이므로 출력 차원은 1)
        self.out = nn.Linear(in_dim, 1)

    def forward(self, lattice, ca_int, dL):
        """
        순전파 함수.

        Parameters
        ----------
        lattice : torch.Tensor
            입력 lattice 이미지. 크기: (B, 1, grid_size, grid_size)
        ca_int : torch.Tensor
            입력 특성 ca_int. 크기: (B, ca_int_dim)
        dL : torch.Tensor
            입력 특성 dL. 크기: (B, dL_dim)

        Returns
        -------
        out : torch.Tensor
            모델의 예측 결과. 크기: (B, 1)
        """
        x = lattice

        # Conv + ReLU + Pooling 반복
        for conv in self.convs:
            x = F.relu(conv(x))
            x = self.pool(x)

        # feature map flatten
        x = x.view(x.size(0), -1)

        # 추가 특성(concat)
        x = torch.cat([x, ca_int, dL], dim=1)

        # Fully connected layers + dropout
        for fc in self.fcs:
            x = F.relu(fc(x))
            x = self.dropout(x)

        # 최종 출력 (회귀)
        out = self.out(x)
        return out
