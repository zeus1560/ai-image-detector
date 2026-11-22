# EXP-002: CIFAKE 베이스라인 조건별 평가 (평가 파이프라인 동작 확인)

- 날짜: 2026-11-23 (KST. 로그 시각은 UTC 2026-11-22 16:05~16:17)
- 작성자: 안승주
- 커밋: 평가 https://github.com/zeus1560/ai-image-detector/commit/daa0844
  (학습은 EXP-001의 eb1efeb. 학습 재실행 없이 기존 best.pt를 평가함)
- 관련 스펙: SPEC-03 (완료 기준 B)

## 가설
SPEC-03 평가 파이프라인(체크포인트 로드 → manifest 검사·버전 확인 → 추론 → summary·drop)이
Colab에서 동작하고, 같은 체크포인트·같은 데이터에서 학습 때 val 지표를 그대로 재현한다.
생성 흔적 탐지 성능을 검증하는 실험이 아니다. **CIFAKE 결과는 발표 수치로 쓰지 않는다.**

## 설정
- 데이터: cifake_manifest.csv (sha256 79e176eeb61bb024e6ffbc5163b5c84fa02f23daf74b9e39c2427086bcb3740c, 학습 때와 같음)
  - source=cifake, condition=original
  - val real 10,000 / fake 10,000, test real 10,000 / fake 10,000 (평가 시작 로그 기준, 모두 .jpg)
- 모델: cifake_baseline best.pt (epoch 2, sha256 b272083b6c6f94dbc34eb6a39a27926deee9ea17d9fdb36b497e9c7f8d0bd27e), resnet50, crop_size=null
- 평가 설정: src/model/configs/cifake_baseline_eval_val.json / cifake_baseline_eval_test.json,
  checkpoint=best, batch_size 64, 임계값 0.5 고정
- 실행 환경: Colab T4, torch 2.14.1+cu130, torchvision 0.29.1+cu130

## 결과
실제 실행 로그에서 옮긴 수치. 괄호는 Wilson 95% 구간.

| 조건 | 정확도 | 실제 사진 오탐률 | 생성 이미지 미탐률 | AP | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| original (val) | 0.96835 (0.965832–0.970688) | 0.0245 (0.021648–0.027718) | 0.0388 (0.035189–0.042766) | 0.995809 | 0.995869 |
| original (test) | 0.9695 (0.967025–0.971794) | 0.0247 (0.021836–0.027929) | 0.0363 (0.032809–0.040148) | 0.995698 | 0.995733 |

- 오탐 수·미탐 수: val 245·388, test 247·363
- 생성기별 미탐률: sd14만 있음 (val 0.0388, test 0.0363, 위 표와 같음)
- drop: original 외 조건이 없어 (없음)
- 추론 시간: val 20.0s, test 15.0s (20,000장)
- 결과 폴더: MyDrive/ai-image-detector-runs/cifake_baseline/eval/cifake_baseline_eval_val, …_eval_test

파이프라인 확인 (SPEC-03 완료 기준 B):
- B1: val 평가의 정확도·오탐률·미탐률·AP·AUC가 metrics.csv epoch 2와 차이 0
- B2: test 시작 로그 real 10,000 / fake 10,000, predictions.csv 20,000행
- B3: sklearn 대비 차이 val AP 1.11e-16·AUC 1.11e-16, test AP 1.11e-16·AUC 0
- B4: 같은 eval_name(val) 재실행 시 OutputExistsError. 출력 폴더 검사가 manifest 읽기·추론보다 먼저라
  추론 전에 멈췄고, 재실행 후 결과 확인 칸의 값이 처음과 같았다 (파일 시각·해시는 따로 비교하지 않음)

- 그래프·이미지: 없음

## 해석
- 가설 지지: 평가 코드가 학습 때 val 지표를 차이 0으로 재현했다. 같은 체크포인트·같은 데이터에서
  train.py와 evaluate.py의 전처리·지표 계산이 일치한다는 뜻이다.
- test가 val과 비슷하고 구간이 겹친다. best epoch를 val로 고른 것이 val에 과하게 맞춘 흔적은 이 수치로는 보이지 않는다.
- 이 수치는 생성 흔적 탐지 성능의 근거가 아니다 (EXP-001 해석과 같은 이유: FAKE 쪽만 512→32 축소, CIFAR-10 근접 중복).
- 누수 검사 한계: CIFAKE manifest는 행마다 origin_id가 달라(120,000행 = origin_id 120,000개) split 겹침 0개가
  자동으로 나온다. 같은 origin_id를 공유하는 GenImage 조건 데이터에서 처음으로 실제 검사가 된다.
- manifest 줄바꿈이 CRLF다 (Colab od 확인). Drive zip에서 온 파일이라 .gitattributes와 무관하다.
  LF로 다시 만들면 내용이 같아도 sha256이 바뀌어 manifest 버전 확인이 "해시 다름"으로 간다.

## 다음 할 일
- SPEC-03 C (GenImage 실행): manifest.csv 준비 대기 (이상진님 데이터 계약 미해결 항목)
- manifest LF 전환 여부: 이상진님 답(정리본 5번)과 함께, 전환 시 CIFAKE 실행과 해시가 달라지는 점까지 합의
- 노트북: 평가 실행 칸의 `!python`은 실패해도 다음 칸이 진행된다. "모두 실행" 시 실패 후에도
  결과 확인 칸이 이전 결과를 보여줄 수 있어, 실패 시 멈추게 할지 정한다 (우선순위 낮음)
