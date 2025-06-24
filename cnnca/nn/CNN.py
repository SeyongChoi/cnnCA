import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

class CNNModel(pl.LightningModule):
    """
    2D CNN 기반 회귀 모델 (PyTorch Lightning 기반).

    이 모델은 격자 형태의 2D 이미지(lattice) 데이터를 입력받아,
    convolution layer를 통해 공간적 특징을 추출한 후,
    추가 입력 특성인 ca_int와 dL과 결합하여 fully connected layer를 통해 
    최종적으로 스칼라 값을 회귀 예측합니다.

    일반적으로 재료과학, 격자 기반 물리 시뮬레이션, 구조 기반 예측 등에 사용됩니다.

    Parameters
    ----------
    input_channel : int
        입력 이미지의 채널 수 (예: 흑백이면 1).
    conv_channels : list[int]
        각 convolution layer의 출력 채널 수.
    conv_kernel : int
        Convolution kernel의 크기 (정방형 필터로 가정).
    conv_stride : int
        Convolution stride.
    conv_dropout_rates : list[float]
        각 convolution block 뒤에 적용할 dropout 확률 (생략시 0.0넣어야함).
    pool_kernel : int
        Max pooling의 커널 크기.
    pool_stride : int
        Max pooling의 stride.
    grid_size : int
        입력 lattice 이미지의 한 변의 길이 (입력은 (grid_size, grid_size) 크기).
    ca_int_dim : int
        추가 입력 특성 ca_int의 차원 수.
    dL_dim : int
        추가 입력 특성 dL의 차원 수.
    hidden_dims : list[int]
        Fully connected hidden layer의 차원 목록.
    fc_dropout_rates : list[float]
        각 hidden layer 뒤에 적용할 dropout 확률.
    weight_init : str, optional
        가중치 초기화 방식. {'xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal', 'orthogonal', 'default'}
    lr : float, optional
        학습률.
    loss_fn : str, optional
        손실 함수. {'MSE', 'MAE', 'Huber'}
    optimizer : str, optional
        옵티마이저 이름. {'Adam', 'SGD'} 등. (현재 코드에서는 내부 구현되지 않음, 향후 configure_optimizers에서 활용 가능)
    weight_decay : float, optional (default=0.0)
        가중치 감쇠 (weight decay) 값. L2 정규화에 사용됩니다.

    Example
    -------
    >>> model = CNNModel(
            input_channel=1,
            conv_channels=[10, 10],
            conv_kernel=4,
            conv_stride=1,
            conv_dropout_rates=[0.2, 0.2],
            pool_kernel=2,
            pool_stride=2,
            grid_size=100,
            ca_int_dim=1,
            dL_dim=1,
            hidden_dims=[1000, 100],
            fc_dropout_rates=[0.2, 0.2]
        )
    >>> lattice = torch.randn(32, 1, 100, 100)
    >>> ca_int = torch.randn(32, 1)
    >>> dL = torch.randn(32, 1)
    >>> output = model(lattice, ca_int, dL)  # output.shape: (32, 1)
    """
    def __init__(self, input_channel, conv_channels, conv_kernel, conv_stride, conv_dropout_rates,
                 pool_kernel, pool_stride, grid_size, ca_int_dim, dL_dim, hidden_dims, fc_dropout_rates,
                 weight_init='xavier_uniform', lr=1e-3, loss_fn='MSE', optimizer='Adam', weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters() # 하이퍼파라미터 저장
        self.weight_init = weight_init  # 가중치 초기화 방법
        in_channels = input_channel  # 입력 채널 수

        # Dropout 리스트 길이 검증
        assert len(conv_dropout_rates) == len(conv_channels), \
            "conv_dropout_rates와 conv_channels 길이는 같아야 합니다."
        assert len(fc_dropout_rates) == len(hidden_dims), \
            "fc_dropout_rates와 hidden_dims 길이는 같아야 합니다."
        
        # ----- Convolution blocks ----- #
        # Convolution Block을 쌓기 위한 리스트
        self.conv_blocks = nn.ModuleList()
        for conv_channel, conv_dropout_rate in zip(conv_channels, conv_dropout_rates):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, conv_channel,
                          kernel_size=conv_kernel,
                          stride=conv_stride,
                          padding='same'),       # ← same padding
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
                *( [nn.Dropout(conv_dropout_rate)] if conv_dropout_rate>0 else [] )
            ))
            in_channels = conv_channel
        
        # ----- Calculate conv output shape ----- #
        # Convolution과 Pooling을 반복한 후 feature map의 최종 크기 계산
        size = grid_size
        for _ in conv_channels:
            # same-conv: out = ceil(in / stride)
            size = math.ceil(size / conv_stride)
            # pooling: out = floor((in + 2*0 − kernel)//stride + 1)
            #      = floor((size − pool_kernel)//pool_stride + 1)
            size = (size - pool_kernel)//pool_stride + 1
        conv_out_dim = conv_channels[-1] * size * size

        # ----- Fully connected layers ----- #
        # Fully connected 입력 차원 = conv 출력 + ca_int + dL
        fc_input_dim = conv_out_dim + ca_int_dim + dL_dim

        # Fully connected layer 정의
        fc_layers = []
        in_dim = fc_input_dim
        for h_dim, dropout_rate in zip(hidden_dims, fc_dropout_rates):
            fc_layers.append(nn.Linear(in_dim, h_dim))
            fc_layers.append(nn.ReLU())
            if dropout_rate > 0:
                fc_layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        self.fcs = nn.Sequential(*fc_layers)

        # 최종 출력 layer (회귀 문제이므로 출력 차원은 1)
        self.out = nn.Linear(in_dim, 1)

        # 가중치 초기화 적용
        self.apply(self.initialize_weights)

        # 손실 함수 설정
        loss_fn = loss_fn.lower()  # 소문자로 변환하여 일관성 유지
        if loss_fn == 'mse' or loss_fn == 'mean_squared_error':
            self.criterion = nn.MSELoss()
        elif loss_fn == 'mae' or loss_fn == 'mean_absolute_error':
            self.criterion = nn.L1Loss()
        elif loss_fn == 'huber':
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}. Supported: 'MSE', 'MAE', 'Huber'.")
        
    def initialize_weights(self, m):
        """
        레이어별 가중치 초기화 함수.
        self.weight_init 값에 따라 초기화 방식을 분기 처리함.
        
        Parameters
        ----------
        m : nn.Module
            초기화 대상 레이어 (주로 nn.Linear).
        """
        if isinstance(m, nn.Linear):
            if self.weight_init == 'xavier_uniform':
                init.xavier_uniform_(m.weight)
            elif self.weight_init == 'xavier_normal':
                init.xavier_normal_(m.weight)
            elif self.weight_init in ['he_uniform', 'kaiming_uniform']:
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif self.weight_init in ['he_normal', 'kaiming_normal']:
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif self.weight_init == 'orthogonal':
                init.orthogonal_(m.weight)
            elif self.weight_init == 'default':
                # PyTorch 기본 초기화 (변경하지 않음)
                pass
            else:
                raise ValueError(f"Unknown weight_init method: {self.weight_init}")

            if m.bias is not None:
                init.zeros_(m.bias)

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
        # 입력은 (B, 1, grid_size, grid_size) 형태
        # 출력은 (B, conv_channels[-1], size, size) 형태
        for conv in self.conv_blocks:
            x = conv(x)

        # Convolution 결과를 flatten
        x = torch.flatten(x, start_dim=1)

        # 추가 특성(concat)
        x = torch.cat([x, ca_int, dL], dim=1)

        # Fully connected layers + dropout
        x = self.fcs(x)

        # 최종 출력 (회귀)
        out = self.out(x)
        return out
    
    def training_step(self, batch, batch_idx):
        """
        학습 단계에서 호출되는 함수.
        
        Parameters
        ----------
        batch : tuple
            입력 데이터와 타겟을 포함하는 튜플 (x, y).
        batch_idx : int
            현재 배치의 인덱스.
        
        Returns
        -------
        torch.Tensor
            손실 값.
        """
        lattice, ca_int, dL, y = batch
        y = y.view(-1, 1)  # 안전하게 (B,1)로 맞춤
        y_hat = self(lattice, ca_int, dL)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        검증 단계에서 호출되는 함수.
        
        Parameters
        ----------
        batch : tuple
            입력 데이터와 타겟을 포함하는 튜플 (x, y).
        batch_idx : int
            현재 배치의 인덱스.
        
        Returns
        -------
        torch.Tensor
            손실 값.
        """
        lattice, ca_int, dL, y = batch
        y = y.view(-1, 1)  # 안전하게 (B,1)로 맞춤
        y_hat = self(lattice, ca_int, dL)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        테스트 단계에서 호출되는 함수.

        Parameters
        ----------
        batch : tuple
            입력 데이터와 타겟을 포함하는 튜플 (x, y).
        batch_idx : int
            현재 배치의 인덱스.

        Returns
        -------
        torch.Tensor
            손실 값.
        """
        lattice, ca_int, dL, y = batch
        y = y.view(-1, 1)  # 안전하게 (B,1)로 맞춤
        y_hat = self(lattice, ca_int, dL)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        예측 단계에서 호출되는 함수.

        Parameters
        ----------
        batch : tuple
            입력 데이터와 타겟을 포함하는 튜플 (x, y) 또는 (x,) 형식.
        batch_idx : int
            현재 배치의 인덱스.
        dataloader_idx : int, optional
            데이터로더 인덱스 (멀티로더일 때 사용).

        Returns
        -------
        torch.Tensor
            예측 결과 텐서 (y_hat).
        """
        # batch가 (lattice, ca_int, dL) 또는 (lattice, ca_int, dL, y) 둘 다 처리
        if len(batch) == 3:
            lattice, ca_int, dL = batch
        else:
            lattice, ca_int, dL, _ = batch
        y_hat = self(lattice, ca_int, dL)
        
        return y_hat

    def configure_optimizers(self):
        """
        최적화 알고리즘을 설정하는 함수.
        Returns
        -------
        torch.optim.Optimizer
            설정된 최적화 알고리즘.
        """
        optimizer_type = self.hparams.optimizer.lower()
        lr = self.hparams.lr
        weight_decay = self.hparams.weight_decay  # <-- 여기서 가져옴

        if optimizer_type == 'adam':
            return optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

if __name__ == "__main__":
    from torch.utils.data import DataLoader, Dataset
    from pytorch_lightning import Trainer
    from torchinfo import summary

    class SimpleDataset(Dataset):
        def __init__(self, lattice, ca_int, dL, y):
            self.lattice = lattice
            self.ca_int = ca_int
            self.dL = dL
            self.y = y
        
        def __len__(self):
            return len(self.y)
        
        def __getitem__(self, idx):
            return self.lattice[idx], self.ca_int[idx], self.dL[idx], self.y[idx]

    # 하이퍼파라미터 설정
    grid = 100
    batch_size = 32

    # 모델 하이퍼파라미터
    model = CNNModel(
        input_channel=1,
        conv_channels=[10, 10],
        conv_kernel=4,
        conv_stride=1,
        conv_dropout_rates=[0.2, 0.2],
        pool_kernel=2,
        pool_stride=2,
        grid_size=grid,
        ca_int_dim=1,
        dL_dim=1,
        hidden_dims=[1000, 100],
        fc_dropout_rates=[0.2, 0.2],
        weight_init='he_normal',
        lr=1e-3,
        loss_fn='MSE',
        optimizer='Adam',
        weight_decay=0.0
    )

    # summary (입력 크기: (채널, H, W))
    summary(model, 
            input_data=(
                torch.randn(1, 1, grid, grid),    # lattice
                torch.randn(1, 1),                # ca_int
                torch.randn(1, 1)                 # dL
            ))
    # 임시 데이터 생성
    N = 500
    lattice = torch.randn(N, 1, grid, grid)
    ca_int = torch.randn(N, 1)
    dL = torch.randn(N, 1)
    y = torch.randn(N, 1)

    dataset = SimpleDataset(lattice, ca_int, dL, y)

    # train/val split
    train_dataset = torch.utils.data.Subset(dataset, range(400))
    val_dataset = torch.utils.data.Subset(dataset, range(400, 500))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    trainer = Trainer(max_epochs=10, accelerator="cpu")
    trainer.fit(model, train_loader, val_loader)