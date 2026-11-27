# SPEC-03: 조건별 평가

- 상태: 구현 완료 (GenImage 대기)
- 담당: 안승주
- 관련 결정·실험: 데이터 계약 "manifest 열"·"규칙", DECISIONS.md "변형 조건", "분할과 저장", "모델 입력 규칙", SPEC-01a, SPEC-02, EXP-001

## 목적
학습된 체크포인트(SPEC-02 베이스라인, SPEC-04 증강 모델)를 조건별로 평가해 핵심 문장의 X를 잰다.
"압축하면 기존 판별 모델 성능이 X 떨어진다"의 X는 이 스펙의 출력에서만 가져온다.
SPEC-04는 같은 코드로 증강 모델을 평가해 Y를 잰다. X는 원본 대비 정확도 하락(%p), Y는 증강 모델의 같은 조건 정확도(회복 후 수준)다. 오탐률과 ROC-AUC를 항상 함께 보고한다.

조건은 원본 / 코드 압축(품질별) / resize50 / 실제 카톡 전송본 / 최신 생성 모델로 나눠 따로 보고한다 (CLAUDE.md "평가 지표").

## 입력
1. 평가 설정 파일 (JSON, `src/model/configs/`): 평가 하나에 파일 하나
   - 예: `cifake_baseline_eval_val.json`, `cifake_baseline_eval_test.json`
   - 필수 키: eval_name, run_dir, checkpoint("best" 또는 "last"), manifest_path, data_root, split, source(문자열 또는 null), batch_size, num_workers
   - 선택 키: train_manifest_path (평가 manifest 해시가 학습 때와 다를 때만 필요, 처리 방식 5)
   - split은 "val" 또는 "test"만 허용한다. "train"이면 에러.
   - split이 test면 checkpoint는 best만 허용한다. last면 에러. test에서 두 체크포인트를 보고 고르면 test로 모델을 선택하는 셈이기 때문이다.
   - 모델 이름과 crop_size는 평가 설정에 쓰지 않고 `run_dir/config.json`(학습 기록)에서 읽는다.
     학습과 평가의 크롭 크기가 어긋나는 것을 막기 위함이다.
2. 체크포인트: `run_dir/best.pt` 또는 `last.pt` (SPEC-02 출력)
3. manifest와 data_root: SPEC-01a의 ManifestDataset으로 읽는다.
   - 지정한 split의 행을 condition·generator 구분 없이 모두 읽는다 (source가 null이 아니면 그 source만).
   - 다른 split의 이미지 파일은 열지 않는다 (manifest 검사는 split 겹침 확인을 위해 전체 행에 실행한다).

## 출력
1. 코드
   - `src/model/eval_config.py`: 평가 설정 읽기와 필수 키·split 값 검사 (torch를 import하지 않음)
   - `src/model/eval_report.py`: 예측 표에서 그룹별 지표 계산, 원본 대비 하락 계산, predictions.csv 읽기·쓰기 (torch를 import하지 않음, metrics.py 재사용)
   - `src/model/evaluate.py`: 체크포인트 불러오기, 추론, 결과 저장 (torch 필요, Colab 실행. 모델 생성은 train.py의 build_model을 재사용)
   - `src/model/configs/*_eval_*.json`
   - `notebooks/run_model.ipynb`에 평가 실행 칸
2. 실행 결과 (`run_dir/eval/{eval_name}/`, Drive. GitHub에 올리지 않음)
   - `eval_config.json`: 평가 설정 + 체크포인트 파일의 sha256과 epoch + manifest_sha256 + 학습 때 manifest_sha256(run_dir/config.json에서) + git 커밋 해시 + torch·torchvision 버전 + GPU 이름
   - `predictions.csv`: 이미지 한 장에 한 행. 열: row_index, path, label, source, generator, condition, origin_id, logit, score, pred
     - score = sigmoid(logit) = fake일 확률, pred = 1(fake) if logit ≥ 0 else 0 (SPEC-02 train.py evaluate와 같은 규칙)
   - `summary.csv`: 그룹 한 개에 한 행 (아래 처리 방식 3)
   - `drop.csv`: 원본 대비 하락 (아래 처리 방식 4)
   - `eval_log.txt`: 화면 출력 전체
3. 실험 기록: `docs/experiments/EXP-00N-*.md`

## 처리 방식

### 1. 실행 순서 (evaluate.py)
1. 평가 설정과 `run_dir/config.json`을 읽는다.
2. `run_dir/eval/{eval_name}/`이 이미 있으면 에러를 내고 멈춘다. 이전 평가 결과를 덮어쓰지 않기 위함이다 (특히 test).
3. ManifestDataset을 **한 번만** 만든다 (split 지정, condition은 거르지 않음). manifest 검사도 한 번만 돈다.
4. manifest 버전 확인 (처리 방식 5).
5. 시작 로그: 설정, 체크포인트 epoch·sha256, manifest_sha256, (source, condition, label)별 행 수, 장치, (source, condition, label)별 파일 확장자 분포. 같은 조건에서 real과 fake의 포맷이 다르면 포맷 편향 의심으로 경고를 출력한다 (실행은 계속).
6. CropWrapper(mode="eval")로 감싸 중앙 크롭 → ToTensor → Normalize(ImageNet 값). SPEC-02 1절과 같다.
   - crop_size가 null이면 크롭하지 않는다 (CIFAKE).
   - 한 변이라도 crop_size보다 작은 이미지는 에러를 내고 멈춘다 (SPEC-02 미해결 1이 정해질 때까지).
7. 전체 행을 한 번 추론해 predictions.csv를 쓴다. 셔플하지 않는다.
8. eval_report로 summary.csv, drop.csv를 만들고 화면에 표로 출력한다.

평가에는 난수가 쓰이지 않는다 (중앙 크롭, 셔플 없음, model.eval()).

### 2. 판정 기준
- 임계값은 0.5(로짓 0)로 고정한다. test 결과를 보고 임계값을 바꾸지 않는다.
- 임계값에 의존하지 않는 AP·ROC-AUC를 함께 보고한다. 압축으로 점수가 한쪽으로 쏠려 정확도가 떨어진 것인지, 구분 능력 자체가 떨어진 것인지 나눠 보기 위함이다.

### 3. 그룹별 지표 (summary.csv, eval_report.py)
- 그룹 단위 두 가지:
  1. (source, condition): 핵심 표. 정확도, 실제 사진 오탐률, 생성 이미지 미탐률, AP, ROC-AUC
  2. (source, condition, generator), generator ≠ real: 생성기별 미탐률만. 교차 생성기 일반화를 보기 위함이다.
     생성기별 오탐률은 계산하지 않는다 (실제 사진은 generator가 모두 real이라 생성기별로 나눌 수 없다).
- 모든 행에 n_real, n_fake, 오탐 수, 미탐 수를 함께 쓴다. kakao·latest처럼 장수가 적은 그룹은 비율만으로 읽으면 안 되기 때문이다.
- 지표 계산은 metrics.py(label_metrics, average_precision, roc_auc)를 쓴다. 분모가 0이거나 한 라벨만 있으면 None(빈칸)이다.
- 정확도·오탐률·미탐률에 Wilson 95% 신뢰구간(하한, 상한)을 함께 쓴다. kakao·latest처럼 장수가 적은 그룹의 불확실성을 보이기 위함이다. AP·ROC-AUC 구간은 계산하지 않는다.

### 4. 원본 대비 하락 (drop.csv, eval_report.py)
- 같은 source 안에서 condition ≠ original인 각 그룹에 대해 (original 값 − 그 조건 값)을 정확도, AP, ROC-AUC마다 계산한다.
  오탐률·미탐률은 (그 조건 값 − original 값)으로 계산한다 (증가가 나빠짐).
- 단위는 %p(퍼센트포인트)다. 예: 정확도 0.95 → 0.80이면 15.0%p. (상대 비율 % 와 섞지 않는다. 미해결 2)
- kakao source는 같은 source에 original이 없으므로 genimage의 같은 split original을 기준으로 한다.
  origin_id로 연결되면 같은 origin_id의 original 행만 골라 비교하고, 연결되지 않으면 genimage original 전체와 비교하며 서로 다른 이미지 간 비교라는 한계를 기록한다. 어느 쪽이 될지는 manifest를 만드는 방식에 달려 있다 (미해결 3).
  - 연결 여부는 데이터로 자동 판정한다. kakao 행의 origin_id가 모두 genimage의 같은 split original에 있으면 연결 방식,
    하나도 없으면 전체 비교 방식, 일부만 있으면 에러를 내고 멈춘다 (manifest가 잘못 만들어진 것이므로).
    어느 방식으로 계산했는지 화면과 drop.csv에 표시한다.
- 그 밖에 기준 original이 없는 source는 drop을 빈칸으로 둔다.

### 5. manifest 버전 확인 (데이터 누수 방지)
학습 뒤 manifest.csv가 바뀌면(kakao·latest 추가 등) 학습 때 train에 있던 origin_id가 평가 split에 들어갈 수 있다.
split 겹침 검사는 한 manifest 안에서만 보므로 이것을 잡지 못한다.
- 평가 manifest_sha256 == 학습 때 manifest_sha256이면 그대로 진행한다.
- 다르면 학습 때 manifest(같은 해시의 스냅샷 파일, 설정 키 train_manifest_path)를 읽어 평가 대상 행의 origin_id를 split별로 확인한다.
  - split=val 평가: 평가 대상 origin_id가 학습 manifest의 train origin_id와 겹치면 에러
  - split=test 평가: 학습 manifest의 train 또는 val origin_id와 겹치면 에러 (val은 best epoch 선택에 쓰였기 때문)
  - 학습 manifest에 없던 새 origin_id는 통과
  train_manifest_path가 없거나 그 파일의 해시가 학습 때 해시와 다르면 에러를 내고 멈춘다.
- 이 비교는 eval_report.py(또는 eval_config.py)의 torch 없는 함수로 둔다.

### 6. test 사용 규칙
- 하이퍼파라미터·증강 강도·임계값 조정은 split=val 평가로만 한다 (CLAUDE.md 규칙 5).
- split=test 평가는 비교할 체크포인트가 확정된 뒤 실행하고, 실행한 것은 결과가 나쁘더라도 모두 실험 기록에 남긴다.
- test 결과를 본 뒤 그 결과를 근거로 재학습하거나 설정을 바꾸지 않는다. 바꾸게 되면 그 사실과 이전 test 결과를 실험 기록에 남긴다.

## 완료 기준

### A. 로컬 pytest (CPU, torch 없이)
- [x] 평가 설정에 필수 키가 빠지면 에러가 나고 빠진 키 이름이 출력된다. split이 "train"이면 에러가 난다
- [x] 손으로 만든 예측 표(source 2개, condition 3개, generator 2개)에서 summary의 (source, condition) 행마다 n_real·n_fake·오탐 수·미탐 수·정확도·오탐률·미탐률이 손 계산 값과 같다
- [x] 생성기별 행은 generator ≠ real만 있고, 미탐률이 손 계산 값과 같다
- [x] 한 라벨만 있는 그룹(예: fake만 있는 latest)에서 오탐률·AP·ROC-AUC가 빈칸이다
- [x] drop: 손 계산 예시에서 정확도·오탐률·미탐률 하락이 %p 값과 같고, original이 없는 source는 빈칸이다
- [x] predictions.csv를 쓰고 다시 읽으면 값이 같다 (logit·score 소수점 손실 없이)
- [x] manifest 버전 확인: 해시 같음(통과) / 해시 다름 + val 평가 + 학습 val과만 겹침(통과) / 해시 다름 + val 평가 + 학습 train과 겹침(에러) / 해시 다름 + test 평가 + 학습 val과 겹침(에러) / 학습 manifest 해시 불일치(에러)
- [x] 출력 폴더가 이미 있으면 에러가 난다
- [x] split=test + checkpoint=last면 에러가 난다
- [x] Wilson 구간이 손 계산 값(예: 100장 중 90 맞음)과 소수 4자리까지 같다
- [x] kakao drop: 손으로 만든 표 세 가지에서 확인한다. 모두 연결(연결 방식 값이 손 계산과 같음) / 모두 미연결(전체 비교 값이 손 계산과 같음) / 일부만 연결(에러)
- [x] (보완 2) gray 열 없는 예전 predictions.csv를 읽으면 gray가 모두 False다. 새 형식은 쓰고 읽은 뒤 값이 같다. gray 빈 값은 에러다
- [x] (보완 2) 손으로 만든 표(실제 컬러·흑백, 생성기 2개, 흑백 가짜 포함)에서 condition 행이 gray==False 행만으로 계산된 손 계산 값과 같다
- [x] (보완 2) gray_real 행의 n_real·오탐 수·오탐률·Wilson 구간이 손 계산 값과 같고, 흑백 실제가 0장인 조건은 n_real=0, 오탐률 빈칸이다
- [x] (보완 2) generator 행의 n_real이 공유 실제(gray==False) 장수와 같고, 전체 지표가 손 계산 값과 같다. 생성기 행끼리 오탐률이 같다
- [x] (보완 2) train_generators=["g1"]이면 g1 행이 yes, g2 행이 no이고, 키가 없으면 빈칸이다
- [x] (보완 2) drop.csv의 generator 행 하락이 손 계산 값과 같다 (kakao 연결·전체 두 방식 모두)
- [x] (보완 2) (source, condition) 안에서 origin_id가 중복되면 에러가 난다
- [x] (보완 2) 기존 SPEC-03 A 테스트는 열 추가에 맞춘 기대값 수정 외에는 그대로 통과한다

### B. Colab CIFAKE 실행 (cifake_baseline best.pt)
- [x] split=val 평가의 정확도·오탐률·미탐률이 metrics.csv의 best epoch(2) 값과 같다 (같은 체크포인트·같은 데이터. 다르면 몇 장이 다른지와 그 이미지의 로짓을 기록하고 원인을 확인한다. AP·ROC-AUC는 1e-4 이내)
- [x] split=test 평가 시작 로그의 행 수가 real 10,000 / fake 10,000이고, predictions.csv가 20,000행이다
- [x] summary.csv에 (cifake, original) 행이 있고, AP·ROC-AUC가 같은 predictions.csv로 sklearn 계산값과 같다
- [x] 같은 eval_name으로 다시 실행하면 에러가 나고 기존 결과 파일이 바뀌지 않는다
- [x] docs/experiments/에 기록을 남긴다 (CIFAKE 결과는 발표 수치로 쓰지 않는다)
- [ ] (보완 2) EXP-002의 test predictions.csv(예전 형식)로 summary를 다시 만들면 condition 행 값이 EXP-002 기록과 같다

### C. Colab GenImage 실행 (manifest.csv 준비 후)
- [ ] val과 test 각각에서 original, jpeg90/75/50/30, resize50 그룹이 summary.csv에 있고, 각 그룹의 n_real·n_fake가 manifest 요약표와 같다
- [ ] drop.csv에서 조건별 하락이 계산된다 (값은 실행 결과에서 옮겨 적는다)
- [ ] kakao·latest가 manifest에 있으면 그 그룹도 summary.csv에 있다
- [ ] docs/experiments/에 GenImage 베이스라인 조건별 평가 기록을 남긴다
- [ ] (보완 2) summary.csv의 gray_real 행 n_real이 manifest의 실제 test gray==True 장수(n)와 같다
- [ ] (보완 2) generator 행마다 n_real이 같고, n_fake가 생성기별 test gray==False 장수와 같다

A·B가 끝나고 C만 남으면 상태를 "구현 완료 (GenImage 대기)"로 둔다.

## 테스트 방법
- 로컬 CPU: pytest에서 손으로 만든 predictions 표(DataFrame)로 eval_report.py를, 임시 JSON으로 eval_config.py를, 작은 가짜 manifest 두 개로 manifest 버전 확인을 본다 (A).
  evaluate.py는 torch가 필요해 로컬에서 import하지 않는다.
- Colab GPU: run_model.ipynb에서 셀 2·3 준비 후 평가 칸 실행 (B, C). val 평가를 먼저 돌려 metrics.csv와 맞는지 본 뒤 test를 돌린다.

## 범위 밖
- 압축 증강 학습, 증강 전후 비교표 → SPEC-04 (이 스펙의 summary.csv·drop.csv를 두 체크포인트에 대해 만들어 비교한다)
- 임계값 조정, 신뢰도 3단계 → SPEC-06 (데모)
- Grad-CAM → SPEC-05
- 기존 서비스 비교 (이상진)
- sns_sim 조건 (데이터 계약에서 정의된 뒤 manifest에 들어오면 코드 변경 없이 그룹으로 잡힌다)
- 같은 origin_id의 조건 간 쌍 비교 (predictions.csv의 origin_id로 나중에 계산할 수 있다)

## 미해결
1. 256보다 작은 평가 이미지 처리 (SPEC-02 미해결 1과 같음). kakao 전송본은 카톡이 리사이즈하므로 크기를 먼저 확인해야 한다.
   resize50도 해당된다 (기준본이 256이면 128이 됨). 또 카톡은 큰 사진을 줄여 보내므로 같은 256 중앙 크롭이어도 original과 kakao는 사진에서 차지하는 범위(배율)가 다르다. 기준본 512 여부와 함께 정한다 (이상진님 확인)
2. 핵심 문장의 X·Y 단위: %p(제안) / 상대 비율(%) / 둘 다. 어느 지표 기준인지(정확도 / 오탐률 / AUC)도 정해야 한다 (제안 반영: 목적 절. 이상진님 동의 후 DECISIONS.md 기록)
3. [C 단계 전에 정한다, 이상진님 확인 1순위] kakao manifest를 만드는 방식: 카톡으로 보낸 원본을 genimage test에서 고르고 같은 origin_id로 연결할지(권장, 같은 사진의 전·후 비교가 가능) 여부. 코드는 두 방식을 모두 지원하고 자동 판정한다(처리 방식 4).
4. [C 단계 전에 정한다, 이상진님 확인, 데이터 계약 수정 필요] latest 출처의 condition 값 (original로 둘지) — 데이터 계약에 없음
5. 장수가 적은 그룹(kakao·latest 50~100장)의 신뢰구간(예: Wilson 95%)을 summary에 넣을지. 넣지 않으면 개수만 함께 보고한다 (결정: 비율 지표에 Wilson 95% 구간, 처리 방식 3)
6. 같은 origin_id의 조건 간 쌍 비교 (예: original에서 맞혔는데 jpeg50에서 틀린 비율)를 넣을지. predictions.csv에 origin_id가 있어 나중에 계산할 수 있다 (결정: 범위 밖)
7. 평가할 체크포인트: best.pt만 / best와 last 둘 다 (SPEC-04 비교 기준과 같이 정해야 함) (결정: test는 best만, val은 best·last 허용. 입력 1)
8. 생성기 구성과 교차 test 장수 (계약 합의 후 확인, SPEC-02 미해결 9와 같음)
9. 테스트 입력을 512 전체로 할지 (계약 "모델 측 입력 규칙", 계약 합의 후 확인). 지금은 SPEC-02와 같은 중앙 크롭 256

## 확정 후 보완 1 (2026-11-21, A 구현 후, 동작·완료 기준 변경 없음)
구현하면서 정한 출력 형식을 기록한다. SPEC-04와 실험 기록은 이 형식을 따른다.
1. predictions.csv의 label은 pred와 같은 0(real)/1(fake) 정수다 (manifest의 label은 real/fake 문자열).
2. summary.csv는 한 파일에 두 그룹을 담고 group 열(condition / generator)로 구분한다. generator 행은 미탐률과 그 Wilson 구간만 채운다. (보완 2에서 group 값이 condition / gray_real / generator 셋이 되고, generator 행은 전체 지표로 바뀜)
3. drop.csv의 열 이름은 하락(…_drop_pp)과 증가(…_increase_pp)를 구분한다. 값은 반올림하지 않는다.
   baseline 열(기준 방식)과 baseline_n_real·baseline_n_fake 열(기준 그룹 장수)을 함께 쓴다.
4. 포맷 편향 경고는 (source, condition)별 real·fake의 확장자 비율이 다르면 낸다. 한 라벨만 있는 그룹은 비교하지 않는다.
5. 출력 폴더 생성과 manifest 버전 확인은 eval_config.py에 둔다.

## 확정 후 보완 2 (2026-11-27, 흑백 분리와 생성기별 지표, 동작 변경 있음)
GenImage test는 공유 실제 750장(계약 잠정, 미확인)에 생성기마다 가짜가 붙어 (source, condition) 그룹이 real:fake 1:3이 된다.
그러면 정확도·AP가 가짜 쪽으로 치우치고, X가 test에 넣은 생성기 수에 따라 바뀐다.
또 계약상 test에는 흑백 실제 사진이 gray=True로 남는다. 이를 대표 지표와 분리한다.

1. predictions.csv에 gray 열 (맨 끝)
   - 값은 문자열 True / False. ManifestDataset.df의 gray(SPEC-01a v3, bool)에서 가져온다.
   - 읽기: 헤더가 새 열 목록이면 gray 값이 정확히 True/False인지 검사한다 (빈 값이 먼저 에러).
     헤더가 예전 열 목록(gray 없음)이면 전부 False로 읽고 "gray 열 없는 예전 파일" 로그를 낸다.
     두 목록 외의 헤더는 지금처럼 에러.

2. summary.csv 그룹 (group 열 값 3가지)
   a. condition (대표 지표): (source, condition)에서 gray == False인 행만. 지표는 지금과 같다.
      생성기를 모두 섞은 값이라 real:fake 비율이 생성기 수에 따라 달라진다. X는 여기서 가져오지 않는다 (4번).
      kakao·latest·CIFAKE처럼 생성기 구분 의미가 약한 출처의 기본 보고 단위로 남긴다.
   b. gray_real (신규): (source, condition)에서 label == real & gray == True인 행.
      n_real, 오탐 수, 오탐률과 Wilson 구간만 채운다. 실제 행이 있는 모든 (source, condition)에 한 행씩 쓰고,
      흑백 실제가 0장이면 n_real=0, 오탐률은 빈칸이다. n이 작으면 참고용이다 (계약 "흑백 처리").
   c. generator (변경): (source, condition, generator ≠ real)마다
      "그 (source, condition)의 실제 gray == False 전체 + 그 생성기의 가짜 gray == False" 행으로 전체 지표를 계산한다.
      (이전: 그 생성기 가짜만으로 미탐률만)
      공유 실제를 모든 생성기 행에서 같이 쓰므로, 생성기 행끼리 오탐률은 같고 미탐률·정확도·AP·AUC가 다르다.
   흑백 가짜(fake, gray == True)는 a·c 어디에도 들어가지 않는다. 흑백 가짜 미탐률은 따로 보지 않는다.

3. summary.csv·drop.csv에 train_generator 열 (신규)
   - run_dir/config.json에 train_generators가 있으면 generator 행에 yes / no, 나머지 행은 빈칸
   - 키가 없으면(CIFAKE) 모든 행 빈칸, 로그에 "train_generators 없음 → 학습 생성기 표시 안 함"

4. drop.csv도 생성기별
   - group 열(condition / generator)을 추가하고, 2-a와 2-c 그룹 각각 original 대비 하락을 계산한다.
     기준 그룹도 같은 방식으로 만든다 (generator 행의 기준 = 같은 source original의 실제 gray==False + 같은 생성기 가짜 gray==False).
   - kakao 기준 방식(연결/전체/에러 자동 판정)은 지금 규칙 그대로이고, generator 행에서는 기준 행을 같은 생성기 가짜와 실제로 제한한다.
   - gray_real은 drop을 계산하지 않는다 (장수가 적어서).
   - 핵심 문장의 X는 drop.csv에서 group == generator, source == genimage, train_generator == yes 행에서 가져온다.
     이 정의는 test 평가 전에 정하고 test 결과를 보고 바꾸지 않는다 (처리 방식 6).
     교차 생성기 행(train_generator == no)은 일반화 결과로 따로 보고한다.

5. 누수·중복 검사 추가 (평가 시작 시, 어기면 에러)
   - 평가 대상 행에서 (source, condition) 안에 같은 origin_id가 두 번 이상 나오면 에러.
     공유 실제 사진이 생성기마다 따로 등록되면 생성기 행에서 같은 사진이 중복으로 세어지기 때문이다.
