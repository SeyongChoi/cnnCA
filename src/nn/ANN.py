import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import pytorch_lightning as pl

class ANNModel(pl.LightningModule):
    """
    인공신경망(ANN) 모델 클래스.
    입력은 1차원 벡터이고, 출력도 1차원 벡터입니다 (예: 회귀 문제).
    
    Parameters
    ----------
    input_dim : int
        입력 벡터의 차원 (예: 특성의 개수).
    hidden_dims : list of int
        각 은닉층(hidden layer)의 뉴런 개수 리스트.
    output_dim : int
        출력층의 뉴런 수. 회귀 문제라면 일반적으로 1.
    dropout_rates : list of float
        각 은닉층에 적용할 드롭아웃 비율 리스트. 각 hidden_dim에 대응.
    weight_init : str, optional (default='xavier_uniform')
        가중치 초기화 방법. 다음 중 선택 가능:
        - 'xavier_uniform'  : Xavier 균등 분포 초기화
        - 'xavier_normal'   : Xavier 정규 분포 초기화
        - 'he_uniform'      : He (Kaiming) 균등 분포 초기화 (ReLU용)
        - 'he_normal'       : He (Kaiming) 정규 분포 초기화 (ReLU용)
        - 'kaiming_uniform' : he_uniform과 동일
        - 'kaiming_normal'  : he_normal과 동일
        - 'orthogonal'      : 직교 행렬 초기화
        - 'default'         : PyTorch 기본 초기화 (변경하지 않음)
    lr: float, optional (default=1e-3)
        학습률 (learning rate). 최적화 알고리즘에 사용됩니다.
    loss_fn : str, optional (default='MSE')
        손실 함수. 다음 중 선택 가능:
        - 'MSE' or 'mean_squared_error' : 평균 제곱 오차 (Mean Squared Error)
        - 'MAE' or 'mean_absolute_error' : 평균 절대 오차 (Mean Absolute Error)
        - 'Huber' : 후버 손실 (Huber Loss)
    optimizer : str, optional (default='Adam')
        최적화 알고리즘. 다음 중 선택 가능:
        - 'Adam' : Adam 최적화 알고리즘
        - 'SGD'  : 확률적 경사 하강법 (Stochastic Gradient Descent)
        - 'RMSprop' : RMSprop 최적화 알고리즘
    weight_decay : float, optional (default=0.0)
        가중치 감쇠 (weight decay) 값. L2 정규화에 사용됩니다.
    Attributes
    ----------
    model : nn.Sequential
        nn.Sequential 형태로 구성된 신경망 모델.
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rates, weight_init='xavier_uniform',
                  lr=1e-3, loss_fn='MSE', optimizer='Adam', weight_decay=0.0):
        super(ANNModel, self).__init__()
        self.save_hyperparameters()  # 하이퍼파라미터 저장
        self.weight_init = weight_init  # 가중치 초기화 방법

        layers = []               # nn.Sequential에 넣을 계층들을 저장할 리스트
        in_dim = input_dim        # 첫 번째 층의 입력 차원
        # Dropout 리스트 길이 검증
        assert len(dropout_rates) == len(hidden_dims), \
            "fc_dropout_rates와 hidden_dims 길이는 같아야 합니다."

        # 은닉층 구성
        for hidden_dim, dropout_rate in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(in_dim, hidden_dim))  # 전결합층 추가
            layers.append(nn.ReLU())                      # 활성화 함수 ReLU 적용
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))  # 드롭아웃 적용
            in_dim = hidden_dim                           # 다음 층 입력 크기 업데이트

        # 출력층 추가 (활성화 함수 없음 - 회귀 문제에 적합)
        layers.append(nn.Linear(in_dim, output_dim))

        # Sequential 모델로 묶기
        self.model = nn.Sequential(*layers)

        # 가중치 초기화 적용
        self.model.apply(self.initialize_weights)

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

    def forward(self, x):
        """
        순전파(forward pass) 정의.
        
        Parameters
        ----------
        x : torch.Tensor
            입력 텐서, shape (batch_size, input_dim)
        
        Returns
        -------
        torch.Tensor
            출력 텐서, shape (batch_size, output_dim)
        """
        return self.model(x)
    
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
        x, y = batch
        y_hat = self(x)
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
        x, y = batch
        y_hat = self(x)
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
        x, y = batch
        y_hat = self(x)
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
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        y_hat = self(x)
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
    from torch.utils.data import DataLoader, TensorDataset
    from pytorch_lightning import Trainer
    from torchsummary import summary

    # 하이퍼파라미터 설정
    grid = 100
    input_dim = grid**2 + 2
    hidden_dims = [1000, 100]
    dropout_rates = [0.2, 0.2]
    output_dim = 1
    weight_init = 'he_normal'

    # 모델 인스턴스
    model = ANNModel(input_dim, hidden_dims, output_dim, dropout_rates, weight_init)

    # summary 확인
    summary(model, input_size=(input_dim,), device='cpu')

    # 임시 학습 데이터
    X = torch.randn(500, input_dim)
    y = torch.randn(500, 1)
    dataset = TensorDataset(X, y)
    train_dataset = torch.utils.data.TensorDataset(X[:400], y[:400])
    val_dataset = torch.utils.data.TensorDataset(X[400:], y[400:])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Lightning Trainer
    trainer = Trainer(max_epochs=10, accelerator="cpu")  # GPU 사용 시 accelerator="gpu"
    trainer.fit(model, train_loader, val_loader)