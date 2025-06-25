import io
import os
import yaml
import logging
import numpy as np

from cnnca.reader import DatasetReader
from cnnca.dataset import ANNDataset, CNNDataset
from cnnca.nn.ANN import ANNModel
from cnnca.nn.CNN import CNNModel
# from nn.SteerableCNN import SteerableCNNModel

import torch
from torch.utils.data import DataLoader

import wandb
import contextlib
from torchinfo import summary
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def set_seed(seed):
    """
    Set the random seed for reproducibility.
    
    Parameters
    ----------
    seed : int
        The random seed to set.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return

def get_input_dim(train_loader, model_type="ANN"):
    
    if model_type == "ANN":
        inputs, _ = next(iter(train_loader))
        # Flatten the input for ANN
        return inputs.shape[1:]
    elif model_type in ["CNN", "SteerableCNN"]:
        inputs_lattice, inputs_ca_int, inputs_dL, _ = next(iter(train_loader))
        # For CNN, return the shape of the input tensor excluding the batch size
        # Assuming inputs are in shape (batch_size, channels, height, width)
        return inputs_lattice.shape[1:], inputs_ca_int.shape[1:], inputs_dL.shape[1:]  # Return (channels, height, width)

def log_model_summary(model, input_dim, device='cpu', model_type='ANN', logger=None):
    """
    Log the model summary to the logger.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model instance to summarize.
    input_size : tuple
        The input size for the model.
    device : str
        The device type ('cpu' or 'cuda').
    logger : logging.Logger
        The logger instance to log the summary.
    """
    if logger is not None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            if model_type == 'ANN':
                summary(model, input_size=(input_dim,), device=device)
            elif model_type in ['CNN', 'SteerableCNN']:
                inputs_lattice, inputs_ca_int, inputs_dL = input_dim
                summary(
                    model,
                    input_data=(
                        inputs_lattice.to(device),
                        inputs_ca_int.to(device),
                        inputs_dL.to(device)
                    ),
                    device=device
                )
        summary_str = buffer.getvalue()
        logger.info("\n" + summary_str)  # 보기 좋게 줄 바꿈 추가

def initialize_logger(logname):
    logger = logging.getLogger("CAModelTraining")
    logger.setLevel(logging.INFO)

    # ✅ CAModelTraining 로거의 기존 핸들러 제거
    if logger.hasHandlers():
        logger.handlers.clear()

    # ✅ 루트 로거 핸들러도 제거 (필요시)
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 포맷 지정
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 콘솔 핸들러 추가
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 파일 핸들러 추가
    fh = logging.FileHandler(logname, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def print_conditions(logger, config, indent_level=0):
    """
    Print the training conditions to the logger, including nested dictionaries.
    
    Parameters
    ----------
    logger : logging.Logger
        The logger instance to log the conditions.
    config : dict
        Configuration dictionary containing training conditions.
    indent_level : int
        Current indentation level for nested dictionaries.
    """
    INDENT_UNIT = 2
    MAX_KEY_WIDTH = 21
    VALUE_WIDTH = 76
    LINE_WIDTH = MAX_KEY_WIDTH + VALUE_WIDTH + 5  # '| key | value |'

    def format_key(key, level):
        # 들여쓰기 없이 레벨 기반 prefix만 붙여서 정렬 깨지지 않게
        prefix = " → " * level
        display_key = f"{prefix}{key}"
        return f"{display_key:<{MAX_KEY_WIDTH}}"

    if indent_level == 0:
        logger.info('=' * LINE_WIDTH)
        logger.info('|{:^100}|'.format('Conditions of Training the Model for predicting CA'))
        logger.info('=' * LINE_WIDTH)

    for section, settings in config.items():
        if indent_level == 0:
            logger.info('|{:^100}|'.format(f'{section} settings'))
            logger.info('-' * LINE_WIDTH)

        if isinstance(settings, dict):
            for key, value in settings.items():
                if isinstance(value, dict):
                    logger.info(f"| {format_key(key, indent_level)}| {'':<{VALUE_WIDTH}} |")
                    print_conditions(logger, value, indent_level + 1)
                else:
                    logger.info(f"| {format_key(key, indent_level)}| {str(value):<{VALUE_WIDTH}} |")
        else:
            logger.info(f"| {format_key(section, indent_level)}| {str(settings):<{VALUE_WIDTH}} |")

        if indent_level == 0:
            logger.info('-' * LINE_WIDTH)

    if indent_level == 0:
        logger.info('\n')
    return

def load_n_split_dataset(data_fpath, norm_ca_int=True, norm_height=True,
                         grid_size=32, pbc_step=None, model="CNN",
                         data_split=[0.7, 0.1, 0.2], logger=None):
    """
    Load and split the dataset into training, validation, and test sets.
    
    Parameters
    ----------
    data_fpath : str
        Path to the dataset file.
    norm_ca_int : bool
        Whether to normalize ca_int.
    norm_height : bool
        Whether to normalize height.
    grid_size : int
        Grid size for the dataset.
    pbc_step : int or None
        Step size for periodic boundary conditions (if applicable).
    model : str
        Model type (e.g., "CNN", "ANN").
    data_split : list of float
        Ratios for train, validation, and test splits.
    
    Returns
    -------
    tuple of lists
        Inputs and outputs for the model.
    """
    if logger is not None:
        logger.info('=' * 102)
        logger.info(f'Load the dataset and transform it for the model type: {model}')
        logger.info('=' * 102)
    
    # DatasetReader 인스턴스 생성 및 데이터셋 로드
    Loader = DatasetReader(data_fpath, norm_ca_int=norm_ca_int, norm_height=norm_height)
    if logger is not None:
        logger.info(Loader.dataset.head())
        logger.info(Loader.dataset.shape)
    # model_type에 따라 적절한 데이터셋 클래스를 선택
    inputs, outputs = Loader.read(grid_size, pbc_step, model=model)
    if logger is not None:
        logger.info(f"Number of inputs:  {len(inputs)}")
        logger.info(f"Number of outputs: {len(outputs)}")
    
    indices = np.random.permutation(len(inputs))  # Shuffle dataset
    inputs_shuffle = [inputs[i] for i in indices]
    outputs_shuffle = [outputs[i] for i in indices]
    
    # 데이터셋을 train, val, test로 분할
    train_size = int(len(inputs) * data_split[0])
    val_size = int(len(inputs) * data_split[1])
    
    train_inputs = inputs_shuffle[:train_size]
    val_inputs = inputs_shuffle[train_size:train_size + val_size]
    test_inputs = inputs_shuffle[train_size + val_size:]
    
    train_outputs = outputs_shuffle[:train_size]
    val_outputs = outputs_shuffle[train_size:train_size + val_size]
    test_outputs = outputs_shuffle[train_size + val_size:]

    if logger is not None:
        logger.info(f"[input]  Train size: {len(train_inputs)}, Val size: {len(val_inputs)}, Test size: {len(test_inputs)}")
        logger.info(f"[output] Train size: {len(train_outputs)}, Val size: {len(val_outputs)}, Test size: {len(test_outputs)}")
    
    return train_inputs, train_outputs, val_inputs, val_outputs, test_inputs, test_outputs

def create_dataset(model_type, train_inputs, train_outputs,
                   val_inputs, val_outputs, test_inputs, test_outputs,
                   grid_size, logger=None):
    """
    Create a dataset based on the model type.
    
    Parameters
    ----------
    model_type : str
        The type of model (e.g., "ANN", "CNN").
    train_inputs : list
        List of training inputs.
    train_outputs : list
        List of training outputs.
    grid_size : int
        Grid size for the dataset.
    
    Returns
    -------
    Dataset instance
        The created dataset instance.
    """
    if logger is not None:
        logger.info('-' * 102)
        logger.info(f"Creating dataset for {model_type} model...")
        logger.info('-' * 102)


    if model_type == "ANN":
        # ANN 모델용 데이터셋 생성
        train_dataset = ANNDataset(train_inputs, train_outputs)    
        val_dataset = ANNDataset(val_inputs, val_outputs)
        test_dataset = ANNDataset(test_inputs, test_outputs)
    else:
        # CNN 또는 SteerableCNN 모델용 데이터셋 생성
        train_dataset = CNNDataset(train_inputs, train_outputs)
        val_dataset = CNNDataset(val_inputs, val_outputs)
        test_dataset = CNNDataset(test_inputs, test_outputs)

    if logger is not None:
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32, seed=1234, logger=None):
    """
    Create DataLoaders for training, validation, and test datasets.
    
    Parameters
    ----------
    train_dataset : Dataset
        The training dataset.
    val_dataset : Dataset
        The validation dataset.
    test_dataset : Dataset
        The test dataset.
    batch_size : int
        Batch size for the DataLoader.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    tuple of DataLoader instances
        The created DataLoaders for train, validation, and test datasets.
    """
    g = torch.Generator().manual_seed(seed)  # Set random seed for reproducibility
    if logger is not None:
        logger.info('-' * 102)
        logger.info(f"Creating DataLoader...")
        logger.info('-' * 102)

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if logger is not None:
        logger.info(f"Train DataLoader size: {len(train_loader)}")
        logger.info(f"Validation DataLoader size: {len(val_loader)}")
        logger.info(f"Test DataLoader size: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

def train(input_yaml='./input.yaml', seed=1234):
    """
    Main function to run the training process.
    
    Parameters
    ----------
    input_yaml : str
        Path to the input YAML configuration file.
    """
    set_seed(seed)  # Set random seed for reproducibility

    # YAML 파일 읽기
    with open(input_yaml, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # ----- YAML에서 읽어온 값 할당 ----- #
    ### 데이터셋 load 관련 설정
    # 데이터셋의 루트 디렉토리
    data_root_dir = config['dataset']['data_root_dir'] 
    # 데이터셋 파일 경로
    data_fpath = os.path.join(data_root_dir, "cnnCA_input_data.csv")  
    # ca_int 정규화 여부 
    norm_ca_int = config['dataset']['normalize']['ca_int'] if 'ca_int' in config['dataset']['normalize'] else True
    # height 정규화 여부 
    norm_height = config['dataset']['normalize']['height'] if 'height' in config['dataset']['normalize'] else True

    ### 데이터셋 transformation 관련 설정
    # UnitCell 생성 시 사용할 격자(grid) 해상도
    grid_size = config['dataset']['grid_size'] 
    # PBC 적용 시 사용될 step 수 (ANN, CNN 전용)
    pbc_step = config['dataset']['pbc_step'] if 'pbc_step' in config['dataset'] else None 
    # train, val, test split 비율
    # config에 'data_split'이 없으면 기본값 [0.7, 0.1, 0.2] 사용
    # [train_ratio, val_ratio, test_ratio] 형태로 설정
    data_split = config['dataset']['split'] if 'split' in config['dataset'] else [0.7, 0.1, 0.2] 

    ### 모델 관련 설정 (model type & hyperparameters)
    # 사용할 모델의 종류 (ANN, CNN, SteerableCNN 중 하나)
    model_type = config['model']['type']
    # CNN 전용
    # Convolutional Layer의 channel 수
    conv_channels = config['model']['conv_channels'] if 'conv_channels' in config['model'] else [32, 64, 128]
    # Convolutional Layer의 dropout rate
    conv_dropout_rates = config['model']['conv_dropout_rates'] if 'conv_dropout_rates' in config['model'] else [0.2, 0.2, 0.2]
    # Convolutional Layer의 kernel size
    conv_kernel = config['model']['conv_kernel'] if 'conv_kernel' in config['model'] else 2
    # Convolutional Layer의 stride
    conv_stride = config['model']['conv_stride'] if 'conv_stride' in config['model'] else 1
    # Pooling Layer의 kernel size
    pool_kernel = config['model']['pool_kernel'] if 'pool_kernel' in config['model'] else 2
    # Pooling Layer의 stride
    pool_stride = config['model']['pool_stride'] if 'pool_stride' in config['model'] else 2

    # Fully Connected Layer의 hidden layer
    hidden_dims = config['model']['hidden_dims'] if 'hidden_dims' in config['model'] else [1000, 100]
    # Fully Connected Layer의 dropout 비율
    dropout_rates = config['model']['dropout_rates'] if 'dropout_rates' in config['model'] else [0.2, 0.2]
    # weight 초기화 방법
    # 'he_normal', 'xavier_uniform', 'xavier_normal' 등
    weight_init = config['model']['weight_init'] if 'weight_init' in config['model'] else 'he_normal'

    ### 학습 관련 설정 
    # device 설정 (GPU 사용 여부)
    device = config['training']['device'] if 'device' in config['training'] else 'cuda' if torch.cuda.is_available() else 'cpu'
    # Loss 함수
    loss_fn = config['training']['loss_fn'] if 'loss_fn' in config['training'] else 'mse'  # 기본값은 MSE
    # 학습률
    lr = config['training']['lr'] if 'lr' in config['training'] else 0.001
    # 에폭 수
    max_epochs = config['training']['max_epochs'] if 'max_epochs' in config['training'] else 10
    # 배치 크기
    batch_size = config['training']['batch_size'] if 'batch_size' in config['training'] else 32
    # L2 정규화 계수
    weight_decay = config['training']['l2_reg'] if 'l2_reg' in config['training'] else 0.0
    # 최적화 알고리즘
    optimizer = config['training']['optimizer'] if 'optimizer' in config['training'] else 'adam'  # 기본값은 Adam
    
    ### 학습 logging 설정
    logging_config = config['logging'] if 'logging' in config else {}
    # WandB 설정
    use_wandb = logging_config.get("wandb", {}).get("enable", False)
    wandb_project = logging_config.get("wandb", {}).get("project_name", "DefaultProject")
    wandb_run_name = logging_config.get("wandb", {}).get("run_name", None)
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=config,  # 전체 config를 WandB에 기록
            reinit=True  # 이미 초기화된 경우에도 재초기화
        )
    # TensorBoard 설정
    use_tensorboard = logging_config.get("TensorBoard", {}).get("enable", False)
    tensorboard_log_dir = logging_config.get("TensorBoard", {}).get("log_dir", "logs")
    # ---------------------------------------------------------------------------------------------------------------- #
    # ----- logging for whole process & conditions ----- #
    logname = f"{model_type}_training.log"
    logger = initialize_logger(logname)
    
    # 출력하여 확인
    print_conditions(logger, config)
    
    # ---------------------------------------------------------------------------------------------------------------- #
    # Load the dataset and transform it for the model type
    # ---------------------------------------------------------------------------------------------------------------- #    
    train_inputs, train_outputs, val_inputs, val_outputs, test_inputs, test_outputs = \
        load_n_split_dataset(data_fpath, norm_ca_int=norm_ca_int, norm_height=norm_height,
                             grid_size=grid_size, pbc_step=pbc_step, model=model_type, 
                             data_split=data_split, logger=logger)
    # ---------------------------------------------------------------------------------------------------------------- #
    # 데이터셋 생성
    # ---------------------------------------------------------------------------------------------------------------- #
    # DataLoader 생성에 사용할 데이터셋 클래스 선택
    train_dataset, val_dataset, test_dataset = \
        create_dataset(model_type, train_inputs, train_outputs,
                      val_inputs, val_outputs, test_inputs, test_outputs,  
                      grid_size, logger=logger)

    # DataLoader 생성
    train_loader, val_loader, test_loader = \
        create_dataloaders(train_dataset, val_dataset, test_dataset,
                           batch_size=batch_size, seed=seed, logger=logger)
    
    # ---------------------------------------------------------------------------------------------------------------- #
    # 모델 초기화 및 학습 설정
    # ---------------------------------------------------------------------------------------------------------------- #
    logger.info('=' * 102)
    logger.info('|{:^100}|'.format(f'Training the {model_type} model for predicting CA'))
    logger.info('=' * 102)
    logger.info('-' * 102)
    logger.info(f"Initializing the {model_type} model...")
    logger.info('-' * 102)
    # 모델 초기화
    if model_type == "ANN":
        # ANN 모델의 입력 차원 계산
        input_dim = get_input_dim(train_loader, model_type)[0]
        logger.info(f"Input dimension for {model_type} model: {input_dim}")
        # ANN 모델 인스턴스 생성
        model = ANNModel(input_dim, hidden_dims, output_dim=1, dropout_rates=dropout_rates, 
                         weight_init=weight_init, lr=lr, loss_fn=loss_fn,
                         optimizer=optimizer, weight_decay=weight_decay)
        # 모델 요약 출력
        log_model_summary(model, input_dim, device=device, model_type=model_type, logger=logger)
    elif model_type == "CNN":
        # CNN 모델의 입력 차원 계산
        input_dims = get_input_dim(train_loader, model_type)
        for i, dim in enumerate(input_dims):
            if i == 0:
                logger.info(f"Input channel for {model_type} model: {dim[0]}")
                input_channel = dim[0]
            elif i == 1:
                logger.info(f"ca_int_dim for {model_type} model: {dim[0]}")
                ca_int_dim = dim[0]
            elif i == 2:
                logger.info(f"dL_dim for {model_type} model: {dim[0]}")
                dL_dim = dim[0]
        
        # CNN 모델 인스턴스 생성
        model = CNNModel(input_channel, conv_channels, conv_kernel, conv_stride, conv_dropout_rates,
                 pool_kernel, pool_stride, grid_size, ca_int_dim, dL_dim, hidden_dims, dropout_rates,
                 weight_init=weight_init, lr=lr, loss_fn=loss_fn, optimizer=optimizer, weight_decay=weight_decay)
        # 모델 요약 출력
        inputs_lattice, inputs_ca_int, inputs_dL, _ = next(iter(train_loader))
        input_dim = (inputs_lattice, inputs_ca_int, inputs_dL)
        log_model_summary(model, input_dim, device=device, model_type=model_type, logger=logger)
    # elif model_type == "SteerableCNN":
    #     model = SteerableCNNModel(grid_size)
    
    logger.info('-' * 102)
    logger.info(f"Setting up the Lightning Trainer & Logger...")
    logger.info('-' * 102)
    pl_logging = logging.getLogger("pytorch_lightning")
    pl_logging.setLevel(logging.INFO)
    pl_logging.addHandler(logger.handlers[1])  # 기존 파일 핸들러 복사
    
    # 개별 로거 준비
    wandb_logger = None
    tensorboard_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(
            project=wandb_project,
            name=wandb_run_name,
            sync_tensorboard=False,  # TensorBoard와 동기화
            log_model=True
        )
        logger.info(f"Using WandB logger - project: {wandb_project}, run: {wandb_run_name}")

    if use_tensorboard:
        tensorboard_logger = TensorBoardLogger(
            save_dir=tensorboard_log_dir,
            name=wandb_run_name or "default_run"
        )
        logger.info(f"Using TensorBoard logger - log dir: {tensorboard_log_dir}")

    # 로거 리스트 생성
    loggers = []
    if wandb_logger:
        loggers.append(wandb_logger)
    if tensorboard_logger:
        loggers.append(tensorboard_logger)

    pl_logger =  loggers if len(loggers) > 1 else (loggers[0] if loggers else None)

    # Model Checkpoint 설정
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename=f"{model_type}" + "-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
        save_weights_only=True,
    )

    # Lightning Trainer
    trainer = Trainer(max_epochs=max_epochs,
                      accelerator=device,
                      logger=pl_logger,
                      log_every_n_steps=1,
                      callbacks=[checkpoint_callback])  # GPU 사용 시 accelerator="gpu"
    

    trainer.fit(model, train_loader, val_loader)
    # Best model 로드 후 test 평가
    best_model_path = checkpoint_callback.best_model_path
    logger.info(f"Best model path: {best_model_path}")
    if model_type == "ANN":
        best_model = ANNModel.load_from_checkpoint(
            checkpoint_path=best_model_path,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout_rates=dropout_rates,
            weight_init=weight_init,
            lr=lr,
            loss_fn=loss_fn,
        )
    elif model_type == "CNN":
        best_model = CNNModel.load_from_checkpoint(
            checkpoint_path=best_model_path,
            input_channel=input_channel,
            conv_channels=conv_channels,
            conv_kernel=conv_kernel,
            conv_stride=conv_stride,
            conv_dropout_rates=conv_dropout_rates,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            grid_size=grid_size,
            ca_int_dim=ca_int_dim,
            dL_dim=dL_dim,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
            weight_init=weight_init,
            lr=lr,
            loss_fn=loss_fn,
            optimizer=optimizer,
            weight_decay=weight_decay
        )

    test_result = trainer.test(best_model, dataloaders=test_loader)
    logger.info(f"Test results: {test_result}")
    return

if __name__ == "__main__":
    
    # 기본 입력 YAML 파일 경로
    input_yaml = './input.yaml'
    # 랜덤 시드 설정
    seed = 1234

    train(input_yaml=input_yaml, seed=seed)