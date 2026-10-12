# 카톡으로 받은 사진도 판별하는 AI 생성 이미지 판별기

> 컴퓨터비전 미니 프로젝트 | 강남대학교

AI 생성 이미지가 중고거래, 단톡방, SNS에 섞이면서 이미지의 진위를 확인할 필요가 커졌습니다. 기존 판별 모델은 원본 이미지에서는 잘 동작하지만, 카카오톡·인스타그램 등으로 전송되며 **압축·리사이즈된 이미지에서는 성능이 크게 떨어집니다.** 이 프로젝트는 실제 사람들이 이미지를 접하는 환경(메신저·SNS)에서도 동작하는 AI 생성 이미지 판별 서비스를 만드는 것을 목표로 합니다.

## 주요 내용

| 구분 | 기존 구현 | 이번 프로젝트의 추가 구현 |
| --- | --- | --- |
| 판별 대상 | 원본 이미지 | 메신저·SNS로 압축된 이미지 |
| 데이터 | 실제=JPEG, 생성=PNG 형태의 포맷 편향 존재 | 포맷 편향 제거 후 학습 |
| 학습 | 일반 학습 | 압축·리사이즈 증강으로 강건성 확보 |
| 결과 | 진짜 / 가짜 이분법 | 신뢰도 3단계 + Grad-CAM 근거 + 압축 정도 안내 |
| 형태 | 코드 | Gradio 웹 데모 |

## 서비스 흐름

1. 사용자가 이미지를 업로드
2. 메타데이터(EXIF, 생성 이력 정보) 확인 → 남아 있으면 참고 정보로 표시
3. 판별 모델이 이미지 자체를 분석
4. 결과를 3단계로 안내: **AI 생성 가능성 높음 / 판단 어려움 / 실제 사진 가능성 높음**
5. Grad-CAM 히트맵으로 판단 근거 표시, 압축이 심한 경우 신뢰도 안내

## 데이터셋

| 데이터셋 | 내용 | 용도 |
| --- | --- | --- |
| [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | 실제 이미지 60,000장 + Stable Diffusion 생성 이미지 60,000장 (32×32) | 베이스라인, 파이프라인 검증 |
| [GenImage](https://github.com/GenImage-Dataset/GenImage) | 여러 생성 모델(Midjourney, Stable Diffusion 등)의 고해상도 이미지 | 압축 실험, 일반화 테스트 (하위 세트 일부만 사용) |
| 압축 데이터 (자체 생성) | 위 이미지를 JPEG 품질 저하·리사이즈 처리 + 실제 카카오톡 전송본 | 압축 강건성 평가 |
| 최신 모델 테스트셋 (자체 생성) | 최신 생성 AI로 만든 이미지 50~100장 | 일반화 한계 확인 |

> CIFAKE는 32×32 해상도라 JPEG 압축(8×8 블록 단위) 실험에는 적합하지 않아, 핵심 압축 실험은 고해상도 데이터로 진행합니다.

## 참조 코드 및 모델

| 이름 | 설명 | 링크 |
| --- | --- | --- |
| CNNDetection | ResNet-50 기반 생성 이미지 판별기 (Wang et al., CVPR 2020) | [GitHub](https://github.com/PeterWang512/CNNDetection) |
| UnivFD | CLIP ViT 특징 기반, 여러 생성 모델에 일반화하는 판별기 (Ojha et al., CVPR 2023) | [GitHub](https://github.com/WisconsinAIVision/UniversalFakeDetect) |
| CIFAKE 분류 노트북 | CNN·ResNet 전이학습 예제 | [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/code) |
| pytorch-grad-cam | Grad-CAM 시각화 라이브러리 | [GitHub](https://github.com/jacobgil/pytorch-grad-cam) |
| UnbiasedGenImage | GenImage 데이터셋의 포맷·크기 편향 분석 | [GitHub](https://github.com/gendetection/UnbiasedGenImage) |

### 비교 대상 서비스

- [Hive AI-Generated Content Detection](https://hivemoderation.com)
- [AI or Not](https://www.aiornot.com)
- [Illuminarty](https://illuminarty.ai)

## 평가 방법

- **정확도**와 함께 **실제 사진 오탐률**(실제 사진을 AI 생성으로 잘못 판정하는 비율)을 별도로 측정합니다. 서비스 관점에서는 억울한 오탐이 더 치명적이기 때문입니다.
- 조건별 비교: 원본 / 코드로 압축한 이미지 / 실제 카카오톡 전송본 / 최신 생성 모델 이미지

## 일정

| 주차 | 내용 |
| --- | --- |
| 9주차 | 환경 세팅, CIFAKE 베이스라인 재현 |
| 10주차 | 압축 데이터 생성, 포맷 편향 확인, 성능 하락 측정, 기존 서비스 비교 |
| 11~12주차 | 편향 제거 + 증강 재학습, 전후 비교, 최신 모델 테스트 |
| 13주차 | Grad-CAM, 압축 정도 안내, Gradio 웹 데모 |
| 14주차 | 최종 발표 |

## 폴더 구조

```
ai-image-detector/
├── data/          # 데이터셋 (용량 문제로 Git에 올리지 않음)
├── notebooks/     # Colab 실험 노트북
├── src/           # 전처리, 학습, 평가 코드
├── demo/          # Gradio 웹 데모
├── results/       # 실험 결과, 그래프, Grad-CAM 이미지
└── docs/          # 발표 자료
```

## 개발 환경

- 학습: Google Colab (GPU)
- 시연: 로컬 (Spyder)
- 주요 라이브러리: PyTorch, torchvision, OpenCV, Pillow, pytorch-grad-cam, Gradio

## 한계 및 유의사항

- 학습에 포함되지 않은 최신 생성 모델의 이미지는 판별 성능이 낮을 수 있습니다.
- 판별 결과는 참고용이며, 이미지의 진위를 단정하는 근거로 사용해서는 안 됩니다.

## 팀원

| 이름 | 역할 |
| --- | --- |
| 승주 | |
| 이상진 | |
