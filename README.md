# SteerableCNNCA

**SteerableCNNCA**는 2D 격자 기반 물리 시스템에서 **접촉각 (contact angle)**을 예측하는 딥러닝 모델들을 구현한 프로젝트입니다. 이 프로젝트는 PyTorch 및 PyTorch Lightning 프레임워크를 기반으로 하며, Steerable CNN, 일반 CNN, ANN 모델을 포함하여 다양한 실험 구성이 가능하도록 설계되어 있습니다.

## 📁 프로젝트 구조

```
SteerableCNNCA/
├── config/               # YAML 설정 파일 (모델, 데이터셋, 학습 등)
├── data/                 # 데이터셋 저장 폴더
├── dataset/              # 데이터셋 빌더 및 전처리 모듈
├── nn/                   # ANN, CNN, Steerable CNN 모델 정의
├── trainer/              # 학습 및 평가 루프
├── utils/                # 시각화 및 기타 유틸 함수
├── main.py               # 실행 엔트리포인트
├── requirements.txt      # 필요한 패키지 목록
└── README.md             # 본 문서
```

## 🧠 주요 기능

- 2D 격자(lattice) + 추가 스칼라 특성(ca_int, dL) 기반 회귀 예측
- 회전 변환에 강인한 Steerable CNN(ESCNN 기반) 지원
- PyTorch Lightning 기반 깔끔한 학습/평가 루프
- W&B(Weights & Biases) 로깅 통합
- YAML 기반 유연한 실험 설정
- 결과 시각화 자동화

## ⚙️ 설치 방법

```bash
# Conda 환경 예시
conda create -n steerablecnnca python=3.8
conda activate steerablecnnca

# 저장소 클론
git clone https://github.com/SeyongChoi/SteerableCNNCA.git
cd SteerableCNNCA

# 필수 패키지 설치
pip install -r requirements.txt
```

## 🚀 실행 방법

### 1. 설정 파일 준비

`config/` 디렉토리의 YAML 파일을 수정하여 데이터 경로, 모델 종류, 학습 설정 등을 구성합니다.

```yaml
model:
  type: "SteerableCNN"  # or "CNN", "ANN"

dataset:
  data_root_dir: "./data/"
  grid_size: 100
  ...
```

### 2. 학습 실행

```bash
python main.py --config config/steerablecnn.yaml
```

학습 로그는 W&B와 콘솔에 출력되며, 모델 체크포인트 및 예측 결과는 지정된 출력 폴더에 저장됩니다.

## 🧩 모델 종류

- `ANNModel`: MLP 기반 단순 회귀
- `CNNModel`: 2D ConvNet 기반 회귀
- `SteerableCNNModel`: 회전 변환에 불변한 steerable filter 기반 CNN (ESCNN 사용)

## 📊 예측 결과 예시

- 학습 loss curve  
- 예측값 vs 실제값 scatter plot  
- 격자 데이터 시각화

<p align="center">
  <img src="docs/example_plot.png" width="500">
</p>

## 📦 주요 의존성

- Python 3.8+
- PyTorch
- PyTorch Lightning
- [ESCNN](https://github.com/QUVA-Lab/escnn)
- wandb
- numpy, matplotlib, scikit-learn 등

## ✍️ 작성자

- **Seyong Choi** – [GitHub 프로필](https://github.com/SeyongChoi)

## 📄 라이선스

본 프로젝트는 MIT 라이선스를 따릅니다. (필요 시 명시)

---

이 문서는 학습, 실험, 평가를 효율적으로 관리하고자 하는 사용자와 연구자를 위한 안내서입니다.  
피드백이나 제안이 있다면 언제든지 Issue나 PR을 통해 공유해주세요!
