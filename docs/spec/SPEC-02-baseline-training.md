# SPEC-02: 베이스라인 학습

- 상태: 구현 완료 (GenImage 대기)
- 담당: 안승주
- 관련 결정·실험: DECISIONS.md "모델 입력 규칙", "CIFAKE 입력 크기", "CIFAKE 알려진 한계", "로컬 개발 환경", "분할과 저장", SPEC-01a, SPEC-01b, EXP-001

## 목적
`original` 조건 train으로 판별 모델을 학습해 "기존 판별 모델" 역할을 할 베이스라인 가중치를 만든다.
이 가중치로 SPEC-03에서 압축 조건별 성능 하락(핵심 문장의 X%)을 재고, SPEC-04 증강 재학습 모델과 비교한다.
같은 코드를 CIFAKE에 먼저 돌려 GenImage가 준비되기 전에 학습 파이프라인을 확인한다 (CIFAKE 결과는 발표 수치로 쓰지 않는다).

## 입력
1. manifest와 data_root: SPEC-01a의 ManifestDataset으로 읽는다.
   - CIFAKE: cifake_manifest.csv (sha256 79e176eeb61bb024e6ffbc5163b5c84fa02f23daf74b9e39c2427086bcb3740c)
   - GenImage: manifest.csv (이상진 작성, 준비 후)
2. 설정 파일 (JSON, `src/model/configs/`): 실험 하나에 파일 하나
   - `cifake_baseline.json`, `genimage_baseline.json`
   - 필수 키: run_name, manifest_path, data_root, source, crop_size(정수 또는 null), model, pretrained,
     epochs, batch_size, lr, weight_decay, seed, num_workers, output_dir
3. 사용하는 행: `condition == original`, split은 train과 val만.
   **test 이미지 파일은 열지 않는다** (manifest 검사는 split 겹침 확인을 위해 test를 포함한 전체 행에 실행한다).

## 출력
1. 코드
   - `src/model/transforms.py`: 크롭 좌표 계산과 PIL 크롭, 크롭 감싸기(CropWrapper) (torch를 import하지 않음)
   - `src/model/metrics.py`: 정확도, 실제 사진 오탐률, 생성 이미지 미탐률, AP, ROC-AUC (numpy만 사용, torch를 import하지 않음, SPEC-03에서 재사용)
   - `src/model/config.py`: 설정 파일 읽기와 필수 키 검사 (torch를 import하지 않음. 로컬 테스트를 위해 train.py와 분리)
   - `src/model/resume.py`: 재개 판단 순수 함수 (metrics 행 정리, best.pt 정합성, best 저장 여부, 재개 설정 비교. torch를 import하지 않음)
   - `src/model/run_log.py`: metrics.csv 열 정의와 읽기·쓰기 (torch를 import하지 않음)
   - `src/model/train.py`: 모델 생성, 학습 루프, 체크포인트, 로그 (torch 필요, Colab 실행)
   - `src/model/configs/*.json`
   - `notebooks/run_model.ipynb`에 학습 실행 칸
2. 실행 결과 (`output_dir/{run_name}/`, `output_dir`은 안승주 소유 Drive의 `MyDrive/ai-image-detector-runs/`. GitHub에 올리지 않음)
   - `config.json`: 실행에 쓴 설정 + manifest_sha256 + git 커밋 해시 + torch·torchvision 버전 + GPU 이름
   - `metrics.csv`: epoch별 epoch, train_loss, val_loss, val 정확도, val 실제 사진 오탐률, val 생성 이미지 미탐률, val_ap, val_auc, epoch 소요 시간(초)
   - `last.pt`: 매 epoch 끝 상태 (모델, 옵티마이저, epoch, 난수 상태)
   - `best.pt`: val_loss가 가장 작은 epoch의 모델과 그 epoch, val_loss
   - `train_log.txt`: 화면 출력 전체
3. 실험 기록: `docs/experiments/EXP-00N-*.md` (CIFAKE, GenImage 각각)

## 처리 방식

### 1. 입력 변환 (transforms.py)
크롭 크기 `crop_size`가 정수이면 아래 규칙, null이면 크롭하지 않고 이미지 전체를 쓴다 (CIFAKE 32×32).
어느 경우에도 리사이즈는 하지 않는다.

- **학습 크롭 (RandomCrop)**: 이미지 크기 W×H, 크롭 크기 C일 때
  - 난수 생성기는 `numpy.random.default_rng([seed, epoch, row_index])`로 이미지마다 새로 만든다.
    row_index는 ManifestDataset이 돌려주는 manifest 행 인덱스다. 그래서 DataLoader 워커 수나 배치 순서와 상관없이
    같은 (seed, epoch, row_index)면 같은 좌표가 나오고, epoch가 바뀌면 같은 이미지도 다른 위치를 자른다.
  - 시작 x는 이 생성기로 {0, 8, 16, …, ⌊(W−C)/8⌋×8} 중 균등하게 고른다. y도 같은 방식.
  - W 또는 H가 C보다 작으면 에러를 내고 멈춘다 (해당 path 출력).
    GenImage 해상도 분포는 미해결 2. 확인 전까지 에러로 멈춘다.
- **평가 크롭 (중앙 크롭)**: 시작 좌표 = (⌊(W−C)/2 / 8⌋×8, ⌊(H−C)/2 / 8⌋×8), 즉 중앙 위치를 8의 배수로 내림
  - 예: 512×512 → (128, 128). 300×260 → (16, 0). 256×256 → (0, 0) 즉 이미지 전체 (resize50 포함).
  - 한 변이라도 C보다 작은 이미지의 처리는 미해결 1. 결정 전까지는 에러를 내고 멈춘다.
- 크롭 후: RGB → 텐서(0~1) → 정규화 (train.py에서 torchvision `ToTensor`, `Normalize`로 처리)
  - mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225) (ImageNet 값)
  - `ResNet50_Weights.IMAGENET1K_V1.transforms()`는 쓰지 않는다. 리사이즈와 중앙 크롭이 들어 있어 리사이즈 금지 규칙과 위 크롭 규칙을 깬다.
- 좌우 반전 등 다른 증강은 쓰지 않는다. 베이스라인("일반 학습")과 SPEC-04 증강 학습의 차이를 압축 증강 하나로 한정하기 위해서다.

### 2. 모델
- torchvision `resnet50`, ImageNet 사전학습 가중치(`IMAGENET1K_V1`)로 시작해 전체를 미세조정한다.
- 마지막 fc를 출력 1개로 바꾼다. 출력은 로짓, `sigmoid(로짓)` = fake일 확률(점수).
- 라벨: real=0, fake=1 (dataset.py의 LABEL_TO_INDEX. CIFAKE 논문은 반대이므로 주의).
- 전역 평균 풀링이라 32×32와 256×256 입력을 같은 코드로 처리한다. 32×32 입력은 마지막 특징맵이 1×1이 된다.
- 교체 조건: CIFAKE 1 epoch 시간을 재서 GenImage 학습이 무료 Colab 세션 안에 끝나기 어려우면 ResNet-18로 교체를 검토한다 (미해결 3).
  교체하면 이 스펙부터 고친다.

### 3. 학습
- 손실: BCEWithLogitsLoss. 판정 임계값 0.5 (로짓 0).
- 옵티마이저: Adam, lr 1e-4, weight_decay 0.
- batch_size: CIFAKE 64, GenImage 32. epochs: CIFAKE 5, GenImage 미정 (미해결 4).
- 매 epoch 끝에 val 전체로 val_loss와 지표 5개(정확도, 오탐률, 미탐률, AP, ROC-AUC)를 계산해 metrics.csv에 한 행 추가한다.
- best.pt 선택 기준: val_loss 최소. 임계값에 의존하지 않기 때문이다.
- 하이퍼파라미터 조정은 val로만 한다. test 이미지 파일은 열지 않는다.

### 4. 재현성
- seed 하나로 Python `random`, NumPy, torch(CPU·CUDA), DataLoader 셔플 generator, worker seed를 모두 고정한다.
- 학습 크롭 좌표는 위 1절대로 (seed, epoch, row_index)로 정한다. 워커별 난수 상태에 의존하지 않는다.
- GPU 연산의 비트 단위 재현은 요구하지 않는다 (cudnn 비결정성). 대신 설정, seed, 버전, 커밋 해시를 config.json에 남긴다.

### 5. 체크포인트와 재개
- epoch는 1부터 센다.
- 매 epoch 끝 저장 순서:
  1. metrics.csv에 그 epoch 행을 추가한다.
  2. last.pt를 임시 파일로 저장한 뒤 `os.replace`로 이름을 바꾼다. 저장 도중 세션이 끊겨도 이전 last.pt가 깨지지 않는다.
  3. 그 epoch의 val_loss가 이전 epoch들의 최소보다 작으면(동점이면 앞 epoch 유지) best.pt를 임시 파일로 저장한 뒤 `os.replace`로 이름을 바꾼다.
     best.pt에는 모델과 함께 epoch, val_loss를 저장한다.
  - last.pt가 "이 epoch까지 끝났다"는 기준이다. 1만 되고 2 전에 끊기면 그 epoch는 끝나지 않은 것으로 본다.
- 같은 run_name으로 다시 실행하면 `last.pt`가 있을 때 그 다음 epoch부터 이어서 학습한다.
  - 설정(config 전체)이나 manifest_sha256이 config.json과 다르면 에러를 내고 멈춘다 (다른 데이터·설정으로 이어 학습하는 것을 막기 위함).
  - git 커밋 해시가 다르면 경고만 출력하고 계속한다 (문서 수정 같은 무관한 커밋으로 멈추지 않도록).
  - metrics.csv에서 last.pt의 epoch보다 뒤쪽 행을 지운다 (위 2 전에 끊긴 epoch의 행).
    남은 행이 1~last epoch를 정확히 한 번씩 갖지 않으면 에러.
  - best.pt 정합성 확인:
    1. 남은 metrics.csv 행에서 val_loss 최소 epoch를 구한다 (동점이면 앞 epoch).
    2. best.pt의 epoch(파일이 없으면 "없음")가 최소 epoch와 같으면 그대로 진행한다.
    3. 다르고, 최소 epoch == last.pt epoch이면 위 2와 3 사이에 끊긴 경우이므로 last.pt의 모델로 best.pt를 다시 만든다 (임시 파일 후 `os.replace`).
    4. 그 외 불일치는 에러를 내고 멈춘다.
- 위 재개 판단(metrics 행 정리, best.pt 정합성)은 `src/model/resume.py`의 torch 없는 순수 함수로 둔다.
  입력은 last.pt epoch, metrics 행의 (epoch, val_loss) 목록, best.pt epoch(없으면 None)이고,
  출력은 남길 metrics 행과 best.pt 처리("그대로" / "last.pt로 다시 만들기")다. 불일치는 에러를 낸다.
  파일 읽기·쓰기와 torch 저장은 train.py가 한다.

### 6. 로그
- 학습 시작 시 출력: config 전체, manifest_sha256 (ManifestDataset.manifest_sha256에서만 꺼냄), 사용 split과 label별 행 수, 장치(GPU 이름)
- epoch마다 출력: metrics.csv와 같은 값

### 7. 지표 (metrics.py, numpy만 사용)
라벨 기준 지표. 입력: 정답 라벨 배열, 예측 라벨 배열 (0/1)
- 정확도 = 맞힌 수 / 전체
- 실제 사진 오탐률 = (정답 real이고 예측 fake인 수) / (정답 real 수)
- 생성 이미지 미탐률 = (정답 fake이고 예측 real인 수) / (정답 fake 수)
- 분모가 0이면 0으로 나누지 않고 None을 돌려준다.

점수 기준 지표. 입력: 정답 라벨 배열, 점수 배열 (fake일 확률). 양성은 fake(1)
- AP: 점수 내림차순으로 서로 다른 점수값마다 임계값을 두고, AP = Σ (Rₙ − Rₙ₋₁) × Pₙ (Pₙ, Rₙ은 n번째 임계값의 정밀도·재현율).
  같은 점수는 하나의 임계값으로 묶는다.
- ROC-AUC: 무작위로 고른 fake 하나의 점수가 real 하나보다 클 확률. 모든 (fake, real) 쌍에서 큼=1, 같음=0.5, 작음=0의 평균.
- 정답에 real이나 fake 중 한 라벨만 있으면 AP와 ROC-AUC는 None을 돌려준다.
- sklearn은 쓰지 않는다. 테스트에서 비교 기준은 손 계산 값이다.

## 완료 기준

### A. 로컬 pytest (CPU, torch 없이)
- [x] 학습 크롭: 512×512, C=256에서 (seed, epoch, row_index)를 바꿔 여러 번 뽑아도 시작 좌표가 모두 8의 배수이고 0~256 범위이며, 결과 크기가 256×256이다
- [x] 학습 크롭: 같은 (seed, epoch, row_index)로 두 번 호출하면 좌표가 같고, 호출 순서를 섞어도 각 (seed, epoch, row_index)의 좌표는 같다
- [x] 학습 크롭: 한 변이 C보다 작은 이미지면 에러가 나고 path가 출력된다
- [x] 평가 크롭: 512×512 → (128, 128), 300×260 → (16, 0), 256×256 → (0, 0)이다
- [x] 평가 크롭: 한 변이 C보다 작은 이미지면 에러가 난다 (미해결 1 결정 전 동작)
- [x] crop_size가 null이면 32×32 이미지가 바이트·크기 변화 없이 그대로 나온다
- [x] 지표: 손으로 만든 예시에서 정확도·오탐률·미탐률이 손 계산 값과 같고, real이 0장이면 오탐률이 None이다
- [x] 지표: 손으로 만든 점수 예시(같은 점수 포함)에서 AP·ROC-AUC가 손 계산 값과 같다
- [x] 지표: 정답이 한 라벨만 있으면 AP·ROC-AUC가 None이다
- [x] 설정 파일에 필수 키가 빠지면 에러가 나고 빠진 키 이름이 출력된다
- [x] resume.py: 정상 / last 후 best 전 끊김(복구) / 불일치(에러) 세 경우를 pytest로 확인한다

### B. Colab CIFAKE 실행
- [x] 학습 시작 로그의 manifest_sha256이 79e176eeb61bb024e6ffbc5163b5c84fa02f23daf74b9e39c2427086bcb3740c이다
- [x] 로그의 사용 split이 train·val뿐이고, label별 행 수가 train 40,000/40,000, val 10,000/10,000이다
- [x] 1 epoch 소요 시간이 metrics.csv에 기록된다 (ResNet-18 교체 판단 근거)
- [x] 매 epoch 끝에 Drive의 last.pt와 metrics.csv가 갱신되고, metrics.csv에 val_ap·val_auc 열이 있다
- [x] 1 epoch 뒤 런타임을 끊고 다시 실행하면 다음 epoch부터 이어지고, metrics.csv에 같은 epoch가 두 번 기록되지 않는다
- [x] 설정이나 manifest가 다른 상태로 같은 run_name을 재개하면 에러가 난다
- [x] 학습이 끝난 뒤 val 정확도가 0.5(찍기 수준)보다 높다 (파이프라인 동작 확인용. 값은 실행 로그에서 옮겨 적는다)
- [x] docs/experiments/에 CIFAKE 실험 기록을 남긴다
- [x] (선택) Colab에서 metrics.py의 AP·AUC를 sklearn 결과와 비교해 같다

### C. Colab GenImage 실행 (manifest.csv 준비 후)
- [ ] 학습 시작 전 train·val `original`의 label별 width·height·format 분포를 출력하고, real과 fake의 분포가 같음을 확인한다
- [ ] 학습 입력이 256×256이고, 크롭 시작 좌표가 8의 배수임을 배치 하나에서 로그로 확인한다
- [ ] B의 체크포인트·로그 항목이 GenImage에서도 같게 동작한다
- [ ] docs/experiments/에 GenImage 베이스라인 실험 기록을 남긴다

A·B가 끝나고 C만 남으면 상태를 "구현 완료 (GenImage 대기)"로 둔다.

## 테스트 방법
- 로컬 CPU: pytest에서 PIL로 만든 작은 가짜 이미지(512×512, 300×260, 256×256, 32×32, 200×512)로 transforms.py를, 손으로 만든 라벨·점수 배열로 metrics.py를, 손으로 만든 (epoch, val_loss) 목록으로 resume.py를, 임시 JSON 파일로 config.py를 확인한다 (A).
  train.py는 torch가 필요해 로컬에서 import하지 않는다.
  B-코드에서 추가한 CropWrapper, run_log.py, resume.py 추가 함수도 로컬에서 확인한다 (확정 후 보완 2).
- Colab GPU: run_model.ipynb에서 Drive 연결 → zip 복사·풀기 → manifest 복사 → git clone → train.py 실행 (B, 순서와 위치는 확정 후 보완 3).
  재개 확인은 실제로 런타임을 끊었다가 다시 실행해서 본다.

## 알려진 한계
- CIFAKE: CIFAR-10 근접 중복(test의 3.25%, DECISIONS.md "CIFAKE 알려진 한계")과
  FAKE만 거친 512→32 바이리니어 축소(SPEC-01b 입력 절, Bird & Lotfi 논문 기준) 때문에 CIFAKE 결과는 생성 흔적 탐지 성능의 근거가 아니다.
- ResNet-50에 32×32를 넣으면 마지막 특징맵이 1×1이라 ImageNet 사전학습의 이점이 작을 수 있다. CIFAKE는 파이프라인 확인용이므로 받아들인다.
- ImageNet 사전학습 백본: GenImage의 실제 사진은 ImageNet에서 왔으므로, 백본이 사전학습 때 같은 실제 사진을 이미 봤을 수 있다.
  이 때문에 실제 사진 쪽 특징이 유리하게 잡힐 가능성이 있다 (영향 크기는 미확인). 베이스라인과 SPEC-04가 같은 백본을 쓰므로 전후 비교에는 같은 조건이다.
- GenImage 실제 사진 중복: 같은 ImageNet 사진이 여러 생성기 폴더에 들어 있을 수 있다 (미확인, 데이터 계약 미해결 "실제 사진의 origin_id 규칙").
  origin_id가 이를 반영하지 않으면 split 겹침 검사로 잡히지 않는 누수가 생긴다.

## 범위 밖
- test 평가, 조건별(압축·resize50·kakao) 평가 → SPEC-03
- 압축 증강 학습 → SPEC-04. SPEC-04도 같은 val(`original`)과 같은 best.pt 기준(val_loss 최소)을 쓴다
- Grad-CAM → SPEC-05
- CNNDetection 공개 가중치로 평가만 하기 (학습 없음) → 별도 스펙 후보. 그 가중치가 학습에 쓴 증강 조건(블러·JPEG 비율 등) 확인이 먼저 필요하다
- 하이퍼파라미터 탐색 (이 스펙은 위 값으로 한 번 학습)
- 좌우 반전 등 압축 외 증강

## 미해결
1. 평가 시 한 변이라도 256보다 작은 이미지 처리: 제외 / 에러 / 별도 표시 중 결정 필요 (GenImage 실제 해상도 분포를 본 뒤)
2. GenImage 실제 해상도 분포 (생성기 최종 선택 후 확인. 데이터 계약 미해결 "생성기 최종 선택"과 연결)
3. ResNet-50 → ResNet-18 교체 여부 (CIFAKE 1 epoch 시간 측정 후 판단)
4. 하이퍼파라미터: 확정됨 (본문 3절). GenImage epochs만 미정 (GenImage 데이터 규모 확인 후)
5. best.pt 선택 기준: 확정됨 (val_loss 최소, 본문 3절)
6. 좌우 반전 사용 여부: 확정됨 (쓰지 않음, 본문 1절)
7. 실행 결과 저장 위치: 확정됨 (`MyDrive/ai-image-detector-runs/`, 출력 절)
8. CIFAKE zip의 Drive 업로드 방식 (SPEC-01b 미해결 1, 이상진님 답변 대기). B 묶음 실행 전에 필요

## 확정 후 보완 (2026-11-18, 동작·완료 기준 변경 없음)
- 설정 파일 읽기와 필수 키 검사는 `src/model/config.py`에 둔다. train.py는 torch를 import해 로컬에서 테스트할 수 없으므로 분리했다 (출력 절에 반영).
- resume.py는 val_loss에, metrics.py는 점수에 NaN(또는 무한대)이 있으면 에러를 낸다. NaN이 섞이면 최소값 비교와 정렬이 조용히 틀려지기 때문이다.
- ROC-AUC는 같은 점수에 평균 순위를 주는 순위 합(Mann–Whitney U) 방식으로 계산한다. 7절의 쌍 비교 정의와 같은 값이며(손 계산 테스트로 확인), val 2만 장에서 1억 쌍을 직접 세지 않기 위함이다.
- train.py가 best.pt를 저장할지 정할 때도 resume.py의 `best_epoch_of()`를 쓴다. 저장 규칙과 재개 시 정합성 검사가 같은 동점 규칙(앞 epoch)을 따르게 하기 위함이다 (B에서 적용).

## 확정 후 보완 2 (2026-11-20, B-코드 착수 전, 완료 기준 변경 없음)
train.py에서 torch가 필요 없는 로직을 분리해 로컬 pytest로 먼저 확인한다.
Colab 재개 확인(런타임 끊고 다시 실행)은 한 번 돌리는 데 시간이 오래 걸려, 거기서 처음 버그를 찾지 않기 위함이다.

1. 크롭 감싸기 (transforms.py, CropWrapper)
   - ManifestDataset의 transform은 PIL 이미지만 받아 epoch·row_index를 쓸 수 없다.
     그래서 train.py는 ManifestDataset을 transform=None으로 만들고 CropWrapper로 감싼다.
   - CropWrapper(dataset, crop_size, mode("train"/"eval"), seed, to_tensor)
     - __getitem__: dataset에서 (PIL, label, row_index)를 받아 mode에 맞게 train_crop / eval_crop을 적용하고,
       to_tensor(크롭된 PIL), label, row_index를 돌려준다. 크롭 에러 메시지용 path는 dataset.df에서 꺼낸다.
     - to_tensor는 train.py가 넣는다 (ToTensor + Normalize). CropWrapper는 torch를 import하지 않는다.
     - mode="train"은 set_epoch(epoch)로 현재 epoch를 받는다. set_epoch 전에 꺼내면 에러 (epoch 기본값으로 조용히 도는 것을 막기 위함).
   - train.py는 매 epoch 시작 전 set_epoch(epoch)를 부르고, DataLoader는 persistent_workers=False로 고정한다.
     True면 워커가 첫 epoch 때 복사한 epoch 값을 계속 써서 epoch가 바뀌어도 같은 위치를 자른다.
     CIFAKE는 crop_size=null이라 이 문제가 B에서 드러나지 않으므로 로컬 테스트로 확인한다.

2. metrics.csv 읽기·쓰기 (run_log.py)
   - 열 순서: epoch, train_loss, val_loss, val_accuracy, val_false_positive_rate, val_false_negative_rate, val_ap, val_auc, epoch_seconds
     (지표 열 이름은 metrics.py label_metrics의 키에 val_을 붙인 것)
   - 값이 None인 지표는 빈칸으로 저장하고, 읽을 때 빈칸은 None으로 돌린다. epoch와 val_loss는 빈칸이면 에러.
   - 행 추가: 파일이 없으면 헤더와 함께 만든다. 열이 빠지거나 남으면 에러.
   - 읽기: 헤더가 위 열 순서와 다르면 에러. (epoch, val_loss) 목록을 plan_resume에 넘길 수 있게 돌려준다.
   - 재개 시 다시 쓰기: plan_resume이 남긴 epoch의 행만 원래 순서대로, 임시 파일 후 os.replace로 쓴다.

3. resume.py 추가
   - is_new_best(rows, epoch): 이번 epoch를 포함한 (epoch, val_loss) 목록에서 best_epoch_of(rows) == epoch인지. epoch가 rows에 없으면 에러.
     train.py는 이 값이 참일 때만 best.pt를 저장한다.
   - check_resume_config(saved, current): config.json(저장된 것)과 이번 실행 값을 비교한다.
     - config 전체 또는 manifest_sha256이 다르면 에러, 다른 키 이름을 출력
     - git 커밋 해시, torch·torchvision 버전, GPU 이름이 다르면 경고 목록만 돌려준다
   - 재개할 때 config.json은 덮어쓰지 않는다 (처음 실행 기록 유지). 이번 실행의 커밋·버전·GPU는 train_log.txt에 출력한다.

B-코드 로컬 확인 (pytest, 완료 기준 B는 그대로)
- CropWrapper: 같은 (seed, epoch, row_index)는 꺼내는 순서와 상관없이 같은 좌표, set_epoch로 epoch를 바꾸면 좌표가 바뀜, set_epoch 전 train 모드 호출은 에러, crop_size=null이면 그대로
- run_log: None 지표가 쓰고 읽은 뒤에도 None, 헤더 불일치 에러, 재개 다시 쓰기 후 남은 행이 1~last epoch
- is_new_best: 첫 epoch 참, 동점이면 거짓
- check_resume_config: 설정 다름·manifest 다름은 에러, 커밋만 다름은 경고

## 확정 후 보완 3 (2026-11-20, B-실행 후, 동작·완료 기준 변경 없음)
B-실행에서 본문과 다르게 진행한 실행 절차를 기록한다. 학습 코드의 동작은 바뀌지 않았다.

1. 레포 받기: `git pull` 대신 런타임마다 `git clone`한다. Colab 런타임은 매번 새로 떠서 pull할 기존 사본이 없다.
   (CLAUDE.md "작업 방식"은 clone과 pull을 모두 허용한다)
2. Colab 준비 순서 (런타임마다): GPU 확인 → Drive 연결 → cifake.zip을 /content로 복사하고 sha256 확인
   → /content/data에 압축 풀기(120,000장 확인) → cifake_manifest.csv를 /content/data/에 복사하고 sha256 확인
   → 레포 clone → train.py 실행
   - manifest는 압축을 푼 뒤에 복사한다. 압축 풀기 단계에서 /content/data를 지우고 새로 만들기 때문이다.
   - 복사 위치는 설정 파일의 manifest_path(/content/data/cifake_manifest.csv), data_root(/content/data)와 같다.
3. 입력 파일 위치: cifake.zip과 cifake_manifest.csv는 안승주 Drive의 임시 폴더 `MyDrive/ai-image-detector-tmp/`에 두고 썼다.
   결과 저장 위치(`MyDrive/ai-image-detector-runs/`)와는 분리했다.
   공유 위치는 아직 정해지지 않았으므로 미해결 8은 열어 둔다. 정해지면 준비 셀의 경로만 바꾼다.
4. B5·B6은 본 실행(cifake_baseline)이 아닌 별도 run으로 확인했다. 본 실행의 결과 파일을 재개·에러 시험으로 건드리지 않기 위함이다.
   - B5: cifake_resume_test2 (epochs 3)
   - B6: cifake_resume_test (epochs 3, 완료된 상태에서 시험). lr 변경, manifest 로컬 사본 끝에 빈 줄 추가(해시만 변경)
   - 이 run들의 수치는 실험 결과로 쓰지 않는다 (EXP-001).
