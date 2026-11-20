# EXP-001: CIFAKE 베이스라인 학습 (파이프라인 동작 확인)

- 날짜: 2026-11-20
- 작성자: 안승주
- 커밋: https://github.com/zeus1560/ai-image-detector/commit/eb1efeb
- 관련 스펙: SPEC-02 (완료 기준 B)

## 가설
SPEC-02 학습 파이프라인(manifest 로드 → 학습 → 매 epoch val 평가 → 체크포인트·재개)이
Colab에서 끝까지 동작하고, val 정확도가 찍기 수준(0.5)보다 높다.
생성 흔적 탐지 성능을 검증하는 실험이 아니다 (아래 해석 참고).

## 설정
- 데이터: cifake_manifest.csv (sha256 79e176eeb61bb024e6ffbc5163b5c84fa02f23daf74b9e39c2427086bcb3740c)
  - source=cifake, condition=original, split train·val만 사용 (test 미사용)
  - train real 40,000 / fake 40,000, val real 10,000 / fake 10,000 (학습 시작 로그 기준)
  - 입력 32×32, 크롭·리사이즈 없음 (crop_size=null), ImageNet mean/std 정규화, 증강 없음
- 모델: torchvision resnet50, IMAGENET1K_V1 사전학습, 전체 미세조정, fc 출력 1개 (real=0, fake=1)
- 하이퍼파라미터: epochs 5, batch_size 64, Adam lr 1e-4, weight_decay 0, seed 42, num_workers 2
  - 손실 BCEWithLogitsLoss, 판정 임계값 0.5, best.pt 기준 val_loss 최소
  - 설정 파일: src/model/configs/cifake_baseline.json, run_name cifake_baseline
- 실행 환경: Colab T4, torch 2.14.1+cu130, torchvision 0.29.1+cu130

## 결과
실제 실행 로그(metrics.csv, train_log.txt)에서 옮긴 수치. 모두 val 기준이며 test는 평가하지 않았다.

| 조건 | 정확도 | 실제 사진 오탐률 | 생성 이미지 미탐률 |
| --- | --- | --- | --- |
| original (val, best.pt = epoch 2) | 0.968350 | 0.024500 | 0.038800 |
| original (val, 마지막 epoch 5) | 0.969650 | 0.031800 | 0.028900 |

epoch별 전체 기록 (실행 04:00:22~04:08:59):

| epoch | train_loss | val_loss | val_acc | val_FPR | val_FNR | val_AP | val_AUC | 초 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.160945 | 0.102080 | 0.960300 | 0.038900 | 0.040500 | 0.993690 | 0.993587 | 99.6 |
| 2 | 0.076434 | 0.081797 | 0.968350 | 0.024500 | 0.038800 | 0.995809 | 0.995869 | 99.1 |
| 3 | 0.050371 | 0.093390 | 0.966450 | 0.034200 | 0.032900 | 0.994986 | 0.994996 | 103.1 |
| 4 | 0.037727 | 0.087679 | 0.969800 | 0.028800 | 0.031600 | 0.995571 | 0.995790 | 102.4 |
| 5 | 0.029433 | 0.086388 | 0.969650 | 0.031800 | 0.028900 | 0.995916 | 0.995715 | 102.2 |

파이프라인 확인 (SPEC-02 완료 기준 B):
- 재개: 별도 run cifake_resume_test2(epochs 3)에서 epoch 1 뒤 강제 종료 → 런타임 삭제 → 재실행 시
  epoch 2부터 이어서 3에서 종료, metrics.csv 3행·중복 0
- 설정 불일치 차단: 별도 run cifake_resume_test에서 lr 변경 → 다른 키 ['lr'] 에러,
  manifest 해시 변경 → 다른 키 ['manifest_sha256'] 에러. 두 경우 결과 파일 변화 없음
- 지표 검증: best.pt로 val 재계산 시 metrics.py와 sklearn 차이 AP 0, AUC 1.11e-16
- 재개·불일치 확인용 run의 수치는 이 기록의 결과에 쓰지 않는다

- 그래프·이미지: `results/EXP-001/cifake_baseline_curves.png` (손실 곡선, val 정확도·오탐률·미탐률)

## 해석
- 가설 지지: 파이프라인이 끝까지 동작했고 val 정확도(0.968350, best.pt)가 0.5보다 높다.
- 이 수치는 생성 흔적 탐지 성능의 근거가 아니다. CIFAKE는 FAKE 쪽만 512→32 바이리니어 축소를 거쳤고
  CIFAR-10 근접 중복이 있다 (SPEC-02 알려진 한계). 높은 정확도에 이 차이가 얼마나 기여했는지는 (미확인).
  카톡 사진 판별 성능과도 무관하다.
- val_loss는 epoch 2에서 최소였고 이후 train_loss는 계속 줄었으나 val_loss는 올랐다(과적합 신호).
  다만 val 정확도·AP는 epoch 4~5에서도 비슷하거나 약간 높아, 임계값 기준 지표와 val_loss가 같은 방향은 아니다.
- 같은 seed로 다른 run을 돌리면 소수점 셋째 자리 수준의 차이가 있었다. SPEC-02 4절이 GPU 비트 단위 재현을
  요구하지 않으므로 스펙 범위 안의 차이다.
- 1 epoch 약 99~103초(T4, 32×32, train 80,000장). GenImage는 해상도·장수가 달라 이 값으로 환산하지 않는다.

## 다음 할 일
- SPEC-02 미해결 4 (GenImage epochs): 이번 val_loss 추이를 참고 자료로만 쓰고, GenImage에서 다시 본다
- SPEC-02 미해결 3 (ResNet-18 교체): GenImage 데이터 준비 후 1 epoch 시간을 직접 재서 판단
- SPEC-02 C (GenImage 실행): manifest.csv 준비 대기 (이상진님 데이터 계약 미해결 항목)
