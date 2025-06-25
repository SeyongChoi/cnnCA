import pandas as pd

from cnnca.unitcell import UnitCell
from typing import Optional, Literal
from sklearn.preprocessing import MinMaxScaler

class DatasetReader:
    """
    DatasetBuilder는 CSV 파일에서 데이터를 불러오고, 정규화 및 모델 타입에 따라
    입력(features)과 출력(targets)을 생성하는 기능을 제공합니다.

    Attributes
    ----------
    data_fpath : str
        CSV 파일 경로.
    norm_ca_int : bool
        ca_int_deg 값을 정규화할지 여부.
    norm_height : bool
        height 값을 정규화할지 여부.
    dataset : pd.DataFrame
        정규화된 데이터셋.
    """

    def __init__(self,
                 data_fpath: str,
                 norm_ca_int: bool = True,
                 norm_height: bool = True):
        """
        DatasetBuilder 초기화 및 데이터셋 로딩.

        Parameters
        ----------
        data_fpath : str
            CSV 파일 경로.
        norm_ca_int : bool, optional
            ca_int_deg 값을 정규화 여부, 기본값 True.
        norm_height : bool, optional
            height 값을 정규화 여부, 기본값 True.
        """
        self.data_fpath = data_fpath
        self.norm_ca_int = norm_ca_int
        self.norm_height = norm_height
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> pd.DataFrame:
        """
        CSV 파일에서 데이터를 불러오고 height, ca_int_deg 컬럼을 정규화함.

        Returns
        -------
        pd.DataFrame
            정규화된 데이터프레임
        """
        dataset = pd.read_csv(self.data_fpath, header=0)

        # height 정규화
        if self.norm_height:
            min_max = MinMaxScaler()
            dataset['norm_height'] = min_max.fit_transform(dataset[['height']])

        # ca_int_deg 정규화
        if self.norm_ca_int:
            min_max = MinMaxScaler()
            dataset['norm_ca_int'] = min_max.fit_transform(dataset[['ca_int_deg']])

        return dataset

    def read(self,
              grid: Optional[int] = 10,
              pbc_step: Optional[int] = 15,
              model: Literal["ANN", "CNN", "SteerableCNN"] = "SteerableCNN"):
        """
        데이터셋을 주어진 모델 형태에 따라 가공하여 입력과 출력을 생성함.

        Parameters
        ----------
        grid : int, optional
            UnitCell 생성 시 사용할 격자(grid) 해상도.
        pbc_step : int, optional
            PBC 적용 시 사용될 step 수 (CNN 전용).
        model : {"ANN", "CNN", "SteerableCNN"}, optional
            사용할 모델 형태.

        Returns
        -------
        inputs : list
            모델 입력 데이터 (lattice, ca_int, dL).
        outputs : list
            모델 타겟 데이터 (ca_exp_deg).
        """
        inputs = []
        outputs = []

        for _, row in self.dataset.iterrows():
            # UnitCell 인스턴스 생성
            unit_cell = UnitCell(
                width=row['width'],
                spacing=row['pitch'],
                height=row.get('norm_height', row['height']),
                shape=row['shape'],
                ca_int=row.get('norm_ca_int', row['ca_int_deg']),
                grid=grid
            )

            if model in ["SteerableCNN"]:
                # SteerableCNN 모델의 입력 형식
                inputs.append([unit_cell.lattice(), unit_cell.ca_int, unit_cell.dL])
                outputs.append(row['ca_exp_deg'])

            else:  # CNN, ANN 모델 처리
                lattices = [unit_cell.lattice()]
                # H, V, D 방향의 periodic boundary condition 적용
                for direction in ['H', 'V', 'D']:
                    lattices.extend(unit_cell.lattice_pbc(direction=direction, pbc_step=pbc_step))
                
                # 각 lattice에 대해 입력 생성
                for lattice in lattices:
                    inputs.append([lattice, unit_cell.ca_int, unit_cell.dL])
                    outputs.append(row['ca_exp_deg'])

        return inputs, outputs
            

if __name__=="__main__":
    import os

    data_root_dir = "D:\\SteerableCNNCA\\data\\"
    data_fpath = os.path.join(data_root_dir, "cnnCA_input_data.csv")

    # DatasetBuilder 인스턴스 생성 및 데이터셋 로드
    reader = DatasetReader(data_fpath, norm_ca_int=True, norm_height=True)

    print(reader.dataset.head())
    print(reader.dataset.shape)

    grid = 100
    pbc_step = 15
    inputs, outputs = reader.read(grid, pbc_step, model="CNN")
    print(f"Number of inputs: {len(inputs)}")
    print(f"Number of outputs: {len(outputs)}")

    