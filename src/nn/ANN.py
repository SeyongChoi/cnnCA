import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

class ANNModel(nn.Module):
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
    
    Attributes
    ----------
    model : nn.Sequential
        nn.Sequential 형태로 구성된 신경망 모델.
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rates, weight_init='xavier_uniform'):
        super(ANNModel, self).__init__()

        layers = []               # nn.Sequential에 넣을 계층들을 저장할 리스트
        in_dim = input_dim        # 첫 번째 층의 입력 차원
        self.weight_init = weight_init

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

if __name__ == "__main__":
    from torchsummary import summary
    # 예시로 사용할 격자 크기
    grid = 100
    # 모델 인스턴스 생성
    input_dim = grid**2 + 2  # 입력 차원
    hidden_dims = [1000, 100]  # 은닉층 차원   
    dropout_rates = [0.2, 0.2]  # 각 은닉층에 적용할 드롭아웃 비율
    output_dim = 1  # 출력 차원 (회귀 문제)
    weight_init = 'he_normal'  # 가중치 초기화 방법
    
    model = ANNModel(input_dim, hidden_dims, output_dim, dropout_rates, weight_init)
    # 모델 요약 출력
    summary(model, input_size=(input_dim,), device='cpu')