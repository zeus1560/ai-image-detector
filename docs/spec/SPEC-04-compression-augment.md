# SPEC-04: 압축 증강 재학습

- 상태: 확정 (2026-11-28)
- 담당: 안승주
- 관련 결정·실험: DECISIONS.md "분할과 저장", "변형 조건", "기준본 만들기", "모델 입력 규칙", 데이터 계약 규칙 6·8·9, SPEC-02, SPEC-03, EXP-001, EXP-002

## 목적
베이스라인(SPEC-02)과 모든 조건을 같게 두고 **JPEG 압축 증강 하나만 더해** 다시 학습한다.
이 모델을 SPEC-03 코드로 평가해 핵심 문장 "압축하면 기존 판별 모델 성능이 X 떨어지고, 우리 방법으로 Y까지 회복한다"의 Y를 잰다.
X와 Y의 정의는 SPEC-03(목적, 보완 2-4)을 그대로 따른다.

베이스라인과의 차이를 압축 증강 하나로 한정하는 이유: 다른 것(반전, 리사이즈, 에폭, seed)이 함께 바뀌면 회복이 압축 증강 덕분인지 가를 수 없기 때문이다.

## 입력
1. manifest와 data_root: SPEC-02와 같다 (ManifestDataset, train_generators로 행 고르기).
   - 학습: `condition == original`, split train. val: `condition == original`, split val. **test 이미지 파일은 열지 않는다.**
   - train에는 original만 저장되어 있다 (DECISIONS "분할과 저장"). 압축본은 학습 중에 만든다.
2. 설정 파일: `src/model/configs/genimage_aug.json`
   - `genimage_baseline.json`을 복사하고 `run_name`과 아래 `augment` 키만 바꾼다/더한다. 나머지 키는 모두 같아야 한다 (처리 방식 5).
   - 선택 키 `augment` (config.py에서 검사):
     ```json
     "augment": {"jpeg_prob": 0.5, "jpeg_quality_min": 30, "jpeg_quality_max": 95}
     ```
     - `jpeg_prob`: 0 이상 1 이하 숫자. 이미지 한 장에 압축을 적용할 확률
     - `jpeg_quality_min`, `jpeg_quality_max`: 1~100 정수, min ≤ max. 이 범위(양 끝 포함)의 정수에서 균등하게 고른다
     - 위 세 키 외의 키가 있거나, 하나라도 빠지거나, 범위를 벗어나면 ConfigError (이유 출력)
   - `augment` 키가 없으면 지금과 같게 동작한다 (증강 없음). 베이스라인 설정과 기존 config.json의 재개 검사 결과가 바뀌지 않는다 (보완 4의 train_generators와 같은 방식).

## 출력
1. 코드
   - `src/model/config.py`: `augment` 키 검사 추가, 베이스라인과 설정 비교 함수 추가 (처리 방식 5, torch를 import하지 않음)
   - `src/model/transforms.py`: 압축 계획 함수, JPEG 재압축 함수, CropWrapper에 압축 단계 추가 (torch를 import하지 않음)
   - `src/model/train.py`: CropWrapper에 augment 설정 전달, 시작 로그·epoch 로그 추가
   - `src/model/configs/genimage_aug.json` (GenImage 준비 후 값 확정), `src/model/configs/cifake_aug_smoke.json` (파이프라인 확인용)
   - `notebooks/run_model.ipynb`: 증강 학습 칸, 설정 비교 칸
2. 실행 결과 (`MyDrive/ai-image-detector-runs/{run_name}/`): SPEC-02와 같은 파일 (config.json, metrics.csv, last.pt, best.pt, train_log.txt)
   - config.json에 추가: Pillow 버전, libjpeg 버전 (`PIL.features.version("jpg")`)
3. 평가 결과: SPEC-03 코드로 만든 `run_dir/eval/{eval_name}/` (변경 없음)
4. 실험 기록: `docs/experiments/EXP-00N-genimage-aug.md` — 베이스라인과 증강 모델의 전후 비교표 포함 (처리 방식 6)

## 처리 방식

### 1. 압축 계획 (transforms.py, 순수 함수)
`jpeg_plan(seed, epoch, row_index, augment) → None 또는 품질 정수`
- 난수 생성기: `numpy.random.default_rng([seed, epoch, row_index, 1])`
  - 크롭 난수(`[seed, epoch, row_index]`, SPEC-02 1절)와 **다른 난수원**이다. 그래서 같은 (seed, epoch, row_index)면 증강 여부와 상관없이 크롭 좌표가 베이스라인과 같다.
  - 끝의 1은 "압축 증강용"을 나타내는 고정 상수다.
- 순서: `u = rng.random()`, `u < jpeg_prob`이면 `q = rng.integers(min, max + 1)`을 돌려주고, 아니면 None.
- **label을 입력으로 받지 않는다.** 실제와 가짜가 같은 확률·같은 품질 분포로 압축되어야 증강이 새로운 포맷 편향을 만들지 않는다 (CLAUDE.md 규칙 3).
- epoch가 바뀌면 같은 이미지도 압축 여부와 품질이 바뀐다.

### 2. JPEG 재압축 (transforms.py)
`jpeg_recompress(image, quality) → PIL 이미지`
- PIL로 메모리(BytesIO)에 JPEG로 저장한 뒤 다시 열어 RGB로 돌려준다. 파일을 디스크에 쓰지 않는다.
- 크기(W×H)는 바뀌지 않는다. 리사이즈는 하지 않는다.
- 저장 옵션(품질 외)은 transforms.py의 상수 한 곳에 둔다. 값은 val/test 변형본과 같아야 한다. 처음 값은 Pillow 기본값(품질만 지정)이고, GenImage 변형본 파일이 오면 파일에서 서브샘플링·양자화 표를 직접 읽어 비교한 뒤 다르면 상수만 고친다 (미해결 1).
- 라이브러리는 PIL만 쓴다 (데이터 계약 규칙 8).

### 3. 적용 순서 (CropWrapper, mode="train"만)
1. dataset에서 (PIL 이미지 512×512, label, row_index)를 받는다.
2. `q = jpeg_plan(seed, epoch, row_index, augment)`
3. q가 None이 아니면 **이미지 전체(512×512)를** `jpeg_recompress(image, q)`로 바꾼다.
4. 그 다음 SPEC-02 1절의 학습 크롭(RandomCrop 256, 8의 배수)을 적용한다.
5. to_tensor (ToTensor + Normalize) — SPEC-02와 같다.

- 압축을 크롭보다 먼저 하는 이유: val/test 변형본은 "512 기준본 재압축 → 모델이 중앙 크롭"으로 만들어진다 (데이터 계약 규칙 6). 학습도 같은 경로여야 결과의 원인을 가를 수 있다.
- 압축 대상은 Q96 기준본이므로, 학습 중 압축은 val/test 변형본과 같은 "이미 JPEG인 사진의 재압축"이다 (DECISIONS "변형 조건").
- mode="eval"은 압축하지 않는다 (val은 SPEC-02와 같은 original 중앙 크롭).
- augment가 없으면 2~3단계를 건너뛴다 (지금 동작과 같음).
- 품질 상한 95: 기준본이 Q96이라 96 이상 재압축은 원본과 거의 같다. 압축하지 않는 경우는 jpeg_prob로 따로 들어간다.

### 4. 모델·학습
- SPEC-02 2~5절 그대로 (ResNet-50, BCEWithLogitsLoss, Adam lr 1e-4, 체크포인트·재개 규칙).
- best.pt 선택: **val `original`의 val_loss 최소** (SPEC-02 범위 밖에 적힌 기준과 같음). 베이스라인과 선택 기준이 같아야 비교가 공정하다.
- 압축 조건 val 성능은 학습 중에 보지 않고, 학습이 끝난 뒤 SPEC-03 평가(split=val)로 확인만 한다. 그 결과로 best.pt를 바꾸지 않는다.
- 로그 (6절 추가)
  - 시작: augment 설정 (키가 없으면 "augment 없음"), Pillow·libjpeg 버전
  - 매 epoch 시작: 그 epoch의 train 행 전체에 대해 jpeg_plan을 계산해 label별 압축 적용 비율, 품질 평균·최소·최대를 출력한다. 이미지를 열지 않고 계산한다 (계획 함수가 순수 함수이므로).
    real과 fake 값이 비슷한지 눈으로 확인하는 용도다.

### 5. 공정 비교 조건 (config.py, torch를 import하지 않음)
`check_same_setup(baseline_record, aug_record)` — 두 run의 config.json 내용을 비교한다.
- 달라도 되는 키: `run_name`, `augment`
- 기록용 값(git 커밋, torch·torchvision·Pillow·libjpeg 버전, GPU 이름)은 다르면 경고 목록만 돌려준다
- 그 밖의 키(manifest_sha256, train_generators, seed, epochs, batch_size, lr, weight_decay, model, pretrained, crop_size, source 등)가 하나라도 다르면 에러, 다른 키 이름 출력
- 노트북의 설정 비교 칸에서 증강 학습을 시작하기 전과 test 평가 전에 실행한다.

### 6. 평가와 전후 비교
- 증강 모델도 SPEC-03 코드로 평가한다. eval 설정은 베이스라인 eval 설정을 복사하고 `eval_name`, `run_dir`만 바꾼다.
- 순서: 두 모델 val 평가 → 두 체크포인트가 확정된 뒤 test 평가 (SPEC-03 처리 방식 6). test 결과를 보고 augment 값이나 epochs를 바꾸지 않는다. 바꾸면 그 사실과 이전 test 결과를 실험 기록에 남긴다.
- 전후 비교표는 실험 기록에 두 run의 summary.csv·drop.csv 값을 옮겨 적어 만든다.
  - 행: original, jpeg90, jpeg75, jpeg50, jpeg30, resize50, (kakao), (latest), (sns_sim) — manifest에 있는 조건만
  - 열: 베이스라인 / 증강 모델 / 차이(%p). 정확도, 실제 사진 오탐률, 생성 이미지 미탐률, ROC-AUC, 각 그룹 장수
  - 핵심 문장의 X·Y는 `group == generator, source == genimage, train_generator == yes` 행에서 가져온다 (SPEC-03 보완 2-4). 교차 생성기 행(train_generator == no)은 일반화 결과로 따로 적는다.
  - **original 조건의 변화도 반드시 적는다.** 압축 조건이 좋아져도 원본 정확도·오탐률이 나빠지면 그 대가도 결과다.
- 해석 규칙 (테스트 전에 정해 둠)
  - jpeg90~30 회복은 학습에 쓴 품질 범위(30~95) 안의 결과다. "학습에 없던 조건"의 근거는 kakao(와 정의되면 sns_sim)로만 말한다.
  - resize50은 학습 증강에 없는 조건이다.
  - 단일 seed 결과다. 차이는 Wilson 95% 구간과 함께 보고한다.

## 완료 기준

### A. 로컬 pytest (CPU, torch 없이)
- [x] config: augment 키가 없는 설정은 지금과 같게 읽히고, cifake_baseline.json으로 만든 설정이 기존 config.json과 check_resume_config에서 에러 없이 비교된다
- [x] config: jpeg_prob 범위 밖(-0.1, 1.1), 품질이 정수가 아님, min > max, 1~100 밖, 키 누락, 모르는 키가 각각 ConfigError이고 이유가 출력된다
- [x] jpeg_plan: 같은 (seed, epoch, row_index)로 두 번 부르면 같은 결과이고, 호출 순서를 섞어도 같다
- [x] jpeg_plan: row 10,000개에서 적용 비율이 jpeg_prob=0.5 기준 0.47~0.53 안이고, 품질이 모두 30~95 안이며 30과 95가 모두 한 번 이상 나온다
- [x] jpeg_plan: jpeg_prob=0이면 모두 None, 1이면 모두 품질 값이다
- [x] jpeg_plan: epoch만 바꾸면 결과가 달라지는 row가 있다
- [x] jpeg_plan_stats (보완 1-2): 같은 row_indices에서 labels만 바꿔도 각 행의 계획이 같다
- [x] jpeg_plan_stats (보완 1-2): 손으로 만든 예시에서 label별 통계가 손 계산 값과 같다
- [x] jpeg_recompress: 512×512 RGB 입력의 출력이 512×512 RGB이고, 품질 30 결과가 품질 95 결과보다 원본과의 평균 절대 차이가 크다
- [x] CropWrapper(train, augment 있음)와 CropWrapper(train, augment 없음)의 크롭 좌표가 같은 (seed, epoch, row_index)에서 같다
- [x] CropWrapper(train, jpeg_prob=1): 출력이 "512 전체를 jpeg_recompress → 같은 좌표로 크롭"한 결과와 픽셀 단위로 같다 (압축 후 크롭 순서 확인)
- [x] CropWrapper(eval, augment 있음)는 압축하지 않는다 (augment 없는 eval 출력과 픽셀 단위로 같음)
- [x] check_same_setup: run_name·augment만 다름(통과) / num_workers만 다름(통과, 보완 1-3) / 커밋·버전만 다름(경고) / seed 다름(에러) / manifest_sha256 다름(에러) / train_generators 다름(에러)
- [x] 기존 테스트가 모두 통과한다

A 확인 (2026-11-28, 로컬 Windows CPU, Pillow 13.0.0, libjpeg 8.0): `python -m pytest src/model/tests -q` → 235 passed (기존 189 + test_augment.py 46). train.py는 import하지 않고 `py_compile`로 문법만 확인.

### B. Colab CIFAKE 파이프라인 확인 (수치는 쓰지 않음)
CIFAKE는 압축 실험에 쓰지 않는다 (DECISIONS "CIFAKE의 역할"). 코드가 GPU 환경에서 도는지만 본다.
- [ ] cifake_aug_smoke (cifake_baseline + augment, epochs 1)이 끝까지 돌고, 시작 로그에 augment 설정과 Pillow·libjpeg 버전이 나온다
- [ ] epoch 로그에 label별 압축 적용 비율·품질 통계가 나오고, real과 fake 값이 비슷하다 (값은 로그에서 옮겨 적는다)
- [ ] config.json에 Pillow·libjpeg 버전이 저장된다
- [ ] 노트북 설정 비교 칸에서 cifake_baseline과 비교하면 epochs가 다르다고 에러가 난다 (run_name·augment는 달라도 되는 키). 기록용 키는 경고만 (smoke는 epochs가 다르므로 에러가 정상)

### C. Colab GenImage 실행 (manifest.csv와 genimage_baseline 학습 후)
- [ ] 학습 시작 전 check_same_setup(genimage_baseline, genimage_aug)이 에러 없이 통과한다
- [ ] 1 epoch 소요 시간을 metrics.csv에서 베이스라인과 비교해 기록한다 (압축 비용 측정. 값은 로그에서)
- [ ] 크롭 시작 좌표가 8의 배수이고 입력이 256×256임을 배치 하나에서 확인한다 (SPEC-02 C와 같음)
- [ ] 두 모델의 val 평가(SPEC-03)를 마친 뒤 test 평가를 실행한다
- [ ] test 평가 전 check_same_setup을 다시 실행해 통과한다
- [ ] docs/experiments/에 전후 비교표(처리 방식 6)와 해석 규칙에 따른 한계를 기록한다

A·B가 끝나고 C만 남으면 상태를 "구현 완료 (GenImage 대기)"로 둔다.

## 테스트 방법
- 로컬 CPU: PIL로 만든 512×512·32×32 가짜 이미지와 작은 테스트 manifest로 transforms.py(jpeg_plan, jpeg_recompress, CropWrapper)를, 임시 JSON으로 config.py를, 손으로 만든 config.json 쌍으로 check_same_setup을 확인한다 (A). train.py는 로컬에서 import하지 않는다.
- Colab GPU: SPEC-02 보완 3의 준비 순서 후 cifake_aug_smoke 실행 (B). GenImage는 manifest 준비 후 genimage_baseline → genimage_aug → 평가 순서 (C).

## 범위 밖
- 리사이즈 증강, sns_sim 증강 (리사이즈 금지 규칙과 resize50 보간 미해결. 비교 조건을 압축 하나로 유지)
- 좌우 반전 등 다른 증강 (SPEC-02 미해결 10이 반전 사용으로 정해지면 베이스라인과 함께 바꾼다)
- augment 값 탐색 (한 번 학습. 바꾸면 val로만 정하고 기록)
- 여러 seed 반복 (여유가 있으면 별도 실험으로 추가)
- best.pt를 압축 val 기준으로 고르기
- 전후 비교표 자동 생성 스크립트 (미해결 4)
- README의 "압축·리사이즈 증강" 문구 수정 (스펙 확정 후 맞춤)

## 미해결
A·B 구현과 실행을 막는 항목은 없다. 아래는 모두 C(GenImage 실행) 전까지 정하면 된다.

### 안승주가 직접 확인 (이상진님께 묻지 않음)
1. JPEG 저장 옵션 일치: GenImage 변형본 파일이 오면 Pillow로 서브샘플링(`JpegImagePlugin.get_sampling`)과 양자화 표(`im.quantization`)를 읽어, 같은 기준본을 transforms.py 상수로 재압축한 결과와 비교한다. 다르면 상수만 고친다
2. 크롭 격자: 1에서 기준본이 4:2:0이면 색 성분 블록은 16×16이라 8의 배수 크롭은 색 격자를 절반의 경우 맞추지 못한다. 학습 크롭은 모델 측 규칙이므로 확인 결과를 보고 안승주가 정하고, 바꾸면 SPEC-02 1절과 DECISIONS "모델 입력 규칙"을 고친 뒤 이상진님께 알린다 (기준본 제작 방식은 바뀌지 않음)
3. 압축 비용: C의 1 epoch 시간 측정 후 판단. 느리면 num_workers를 먼저 늘린다. "크롭 후 압축"으로 바꾸는 것은 test 경로와 달라지므로 이 스펙을 먼저 고친다
4. 전후 비교표: 실험 기록에서 손으로 만든다. 반복이 많아지면 그때 스크립트를 검토한다
5. 학습 시작 전 설정 비교: 증강 run의 config.json은 학습이 시작돼야 생기므로, 노트북 설정 비교 칸으로는 C의 "학습 시작 전 check_same_setup 통과"를 지킬 수 없다. B에는 영향 없음. C 전에 정한다. 제안: train.py에 베이스라인 run_dir을 받는 선택 인자를 두고, 기록을 만든 직후·학습 전에 check_same_setup을 실행해 에러면 학습하지 않고 멈춘다. 정하면 보완으로 스펙을 먼저 고친다.

### 기존 안건에 이미 포함 (새로 묻는 것 아님)
6. GenImage epochs (SPEC-02 미해결 4). 증강 모델도 같은 값
7. X·Y 단위 (SPEC-03 미해결 2, %p 제안)
8. DECISIONS "분할과 저장"(확정)과 데이터 계약 미해결 마지막 줄의 표기 불일치: 계약 합의 때 함께 정리한다. 이 스펙은 확정을 전제로 한다

### 이 스펙과 무관
- sns_sim 정의: 이 스펙은 sns_sim을 학습에 쓰지 않는다. manifest에 들어오면 SPEC-03 코드가 그대로 그룹으로 잡는다

## 확정 후 보완 1 (2026-11-28, 구현 착수 전)
구현 계획 검토에서 나온 세 가지를 반영한다. 학습·증강 동작은 바뀌지 않는다.

1. 기록용 키에 Pillow·libjpeg 버전 추가 (resume.py ENV_KEYS)
   - config.json에 pillow_version, libjpeg_version을 저장하므로 resume.py의 ENV_KEYS에 두 키를 추가한다. 추가하지 않으면 재개 때 설정이 다르다는 에러가 난다.
   - check_same_setup의 기록용 키도 ENV_KEYS를 그대로 쓴다.
   - 입력 절의 "기존 config.json의 재개 검사 결과가 바뀌지 않는다"는 "에러 없이 비교된다"(완료 기준 A 첫 항목)로 읽는다. 두 키가 없는 기존 config.json(cifake_baseline 등)을 재개 검사하면 경고가 2줄 늘어나며, 이는 허용한다.

2. 압축 통계 집계 함수 (transforms.py)
   - jpeg_plan_stats(seed, epoch, row_indices, labels, augment): 이미지를 열지 않고 jpeg_plan을 행마다 계산해 label별 적용 비율, 품질 평균·최소·최대를 돌려준다. 처리 방식 4의 epoch 로그에 쓴다.
   - label은 집계에만 쓰고 jpeg_plan에 넘기지 않는다.
   - 완료 기준 A 추가: 같은 row_indices에서 labels만 바꿔도 각 행의 계획이 같다. 손으로 만든 예시에서 label별 통계가 손 계산 값과 같다.

3. num_workers는 check_same_setup에서 달라도 되는 키로 둔다
   - 학습 크롭과 압축 난수는 (seed, epoch, row_index)로 정해지고 셔플 generator는 메인 프로세스에서 고정되므로(SPEC-02 4절), 워커 수는 학습 결과에 영향을 주지 않는다.
   - 미해결 3처럼 속도 때문에 증강 run만 num_workers를 늘려도 공정 비교 조건이 깨지지 않게 하기 위함이다.
   - 처리 방식 5의 "달라도 되는 키"는 run_name, augment, num_workers가 된다. 완료 기준 A의 check_same_setup 항목에 "num_workers만 다름(통과)"을 추가한다.
   - 같은 run을 재개할 때의 check_resume_config는 바꾸지 않는다 (SPEC-02 범위).
