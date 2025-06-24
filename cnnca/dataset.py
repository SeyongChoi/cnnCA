import torch
from torch.utils.data import Dataset

class ANNDataset(Dataset):
    """
    ANN 모델 학습용 PyTorch Dataset.
    lattice를 flatten하여 ca_int, dL과 concat한 1D 벡터를 반환합니다.
    """

    def __init__(self, inputs, outputs):
        """
        Parameters
        ----------
        inputs : list
            각 항목은 (lattice, ca_int, dL)의 튜플로 구성됨
        outputs : list
            각 항목은 target 값 (ca_exp_deg)
        """
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        # 전체 데이터 개수 반환
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        하나의 샘플을 반환합니다.

        Returns
        -------
        input_tensor : torch.Tensor
            Flatten된 lattice (1D)와 ca_int, dL이 연결된 1차원 텐서
        output_tensor : torch.Tensor
            예측 대상 값 (ca_exp_deg)
        """
        lattice, ca_int, dL = self.inputs[idx]

        # lattice: 2D numpy array → 1D tensor
        lattice_flat = torch.tensor(lattice, dtype=torch.float32).flatten()

        # ca_int와 dL을 tensor로 변환 (1D scalar tensor)
        ca_int_tensor = torch.tensor([ca_int], dtype=torch.float32)
        dL_tensor = torch.tensor([dL], dtype=torch.float32)

        # [lattice_flat, ca_int, dL]을 하나의 벡터로 이어붙임
        input_tensor = torch.cat([lattice_flat, ca_int_tensor, dL_tensor], dim=0)

        # target tensor (ca_exp_deg)
        output_tensor = torch.tensor([self.outputs[idx]], dtype=torch.float32)

        return input_tensor, output_tensor
    
class CNNDataset(Dataset):
    """
    CNN 및 SteerableCNN 모델 학습용 PyTorch Dataset.
    lattice는 2D 텐서 형태(이미지처럼), ca_int와 dL은 별도 scalar 텐서로 반환됩니다.
    """

    def __init__(self, inputs, outputs):
        """
        Parameters
        ----------
        inputs : list
            각 항목은 (lattice, ca_int, dL)의 튜플
        outputs : list
            각 항목은 target 값 (ca_exp_deg)
        """
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        # 전체 샘플 개수
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        하나의 샘플을 반환합니다.

        Returns
        -------
        lattice_tensor : torch.Tensor
            Conv layer에 입력될 2D lattice 텐서, shape: (1, grid, grid)
        ca_int_tensor : torch.Tensor
            내부 FC layer 입력용 scalar 텐서
        dL_tensor : torch.Tensor
            내부 FC layer 입력용 scalar 텐서
        output_tensor : torch.Tensor
            예측 대상 값 (ca_exp_deg)
        """
        lattice, ca_int, dL = self.inputs[idx]

        # lattice: numpy array → 2D 텐서로 변환 후 채널 차원 추가 (1, grid, grid)
        lattice_tensor = torch.tensor(lattice, dtype=torch.float32).unsqueeze(0)

        # ca_int와 dL 각각 scalar 텐서로 변환
        ca_int_tensor = torch.tensor([ca_int], dtype=torch.float32)
        dL_tensor = torch.tensor([dL], dtype=torch.float32)

        # target tensor
        output_tensor = torch.tensor([self.outputs[idx]], dtype=torch.float32)

        return lattice_tensor, ca_int_tensor, dL_tensor, output_tensor
    
    

if __name__=="__main__":
    import os
    from reader import DatasetReader

    data_root_dir = "D:\\SteerableCNNCA\\data\\"
    data_fpath = os.path.join(data_root_dir, "cnnCA_input_data.csv")
    norm_ca_int = True
    norm_height = True
    # DatasetBuilder 인스턴스 생성 및 데이터셋 로드
    reader = DatasetReader(data_fpath, norm_ca_int=norm_ca_int, norm_height=norm_height)

    print(reader.dataset.head())
    print(reader.dataset.shape)

    grid = 100
    pbc_step = 15
    model = "CNN"  # 또는 "SteerableCNN", "ANN"

    # 데이터셋 읽기
    inputs, outputs = reader.read(grid, pbc_step, model=model)
    print(f"Number of inputs: {len(inputs)}")
    print(f"Number of outputs: {len(outputs)}")

    if model == "ANN":
        # ANN 모델용 데이터셋 생성
        dataset = ANNDataset(inputs, outputs)
        print(f"ANN Dataset size: {len(dataset)}")
        sample_input, sample_output = dataset[0]
        print(f"Sample input shape: {sample_input.shape}, Sample output: {sample_output.item()}")
    else:
        # CNN 또는 SteerableCNN 모델용 데이터셋 생성
        dataset = CNNDataset(inputs, outputs)
        print(f"CNN Dataset size: {len(dataset)}")
        sample_input, ca_int, dL, sample_output = dataset[0]
        print(f"Sample input shape: {sample_input.shape}, ca_int: {ca_int.item()}, dL: {dL.item()}, Sample output: {sample_output.item()}")
