import pandas as pd

from unitcell import UnitCell
from typing import Optional, Literal, List
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_dataset(data_fpath: str, 
                  norm_ca_int: bool = True,
                  norm_height: bool = True):
    """
    Load dataset from a CSV file and normalize the features.
    """
    dataset = pd.read_csv(data_fpath, header=0)

    if norm_height:
        min_max = MinMaxScaler()
        dataset['norm_height'] = min_max.fit_transform(dataset[['height']])

    if norm_ca_int:
        min_max = MinMaxScaler()
        dataset['norm_ca_int'] = min_max.fit_transform(dataset[['ca_int_deg']])
    
    return dataset

def build_dataset(dataset: pd.DataFrame,
                  grid: Optional[int] = 10,
                  pbc_step: Optional[int] = 15,
                  model: Literal["ANN", "CNN", "SteerableCNN"] = "SteerableCNN"):
    
    inputs = []
    outputs = []
    if model == "SteerableCNN" or model == "ANN":
        for _, row in dataset.iterrows():
            unit_cell = UnitCell(
                width=row['width'],
                spacing=row['pitch'],
                height=row['norm_height'] if 'norm_height' in row else row['height'],
                shape=row['shape'],
                ca_int=row['norm_ca_int'] if 'norm_ca_int' in row else row['ca_int_deg'],
                grid=grid  # Default grid size
            )
            inputs.append([unit_cell.lattice(), unit_cell.ca_int, unit_cell.dL])
            outputs.append(row['ca_exp_deg'])
    
    else:
        for _, row in dataset.iterrows():
            unit_cell = UnitCell(
                width=row['width'],
                spacing=row['pitch'],
                height=row['norm_height'] if 'norm_height' in row else row['height'],
                shape=row['shape'],
                ca_int=row['norm_ca_int'] if 'norm_ca_int' in row else row['ca_int_deg'],
                grid=grid  # Default grid size
            )
            lattices = []
            lattices.append(unit_cell.lattice())
            for direction in ['H', 'V', 'D']:
                lattices.extend(unit_cell.lattice_pbc(direction=direction, pbc_step=pbc_step))

            for lattice in lattices:
                inputs.append([lattice, unit_cell.ca_int, unit_cell.dL])
                outputs.append(row['ca_exp_deg'])
    
    return inputs, outputs
            

if __name__=="__main__":
    import os

    data_root_dir = "D:\\SteerableCNNCA\\data\\"
    data_fpath = os.path.join(data_root_dir, "cnnCA_input_data.csv")

    dataset = load_dataset(data_fpath,
                           norm_ca_int=True,
                           norm_height=True)
    
    print(dataset.head())
    print(dataset.shape)

    grid = 100
    pbc_step = 15
    inputs, outputs = build_dataset(dataset, grid, pbc_step, model="CNN")
    print(f"Number of inputs: {len(inputs)}")
    print(f"Number of outputs: {len(outputs)}")

    