import os
import argparse

from cnnca.train import train

def main():
    parser = argparse.ArgumentParser(description="SteerableCNNCA Training Script")
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    
    # 기본 입력 YAML 파일 경로
    input_yaml = config_path
    # 랜덤 시드 설정
    seed = 1234

    train(input_yaml=input_yaml, seed=seed)

if __name__ == '__main__':
    main()