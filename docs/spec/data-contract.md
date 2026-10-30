# 데이터 계약 (Data Contract) — 초안

- 상태: 초안. 9주차 첫 회의에서 함께 확정한다.
- 데이터 측: 이상진 (이 형식으로 데이터를 만든다)
- 모델 측: 안승주 (이 형식으로만 데이터를 읽는다)
- 데이터 담당 영역이므로 이상진이 자유롭게 수정 의견을 낸다.

## 목적

데이터는 이 형식으로 만들고, 모델은 이 형식만 읽는다. 형식을 먼저 정해두면 데이터가 다 만들어지기 전에도 모델 측은 CIFAKE를 같은 형식으로 감싸 코드를 먼저 짤 수 있어서, 서로 기다리지 않고 동시에 작업할 수 있다.

## 저장 위치

데이터는 용량이 커서 GitHub이 아니라 **Google Drive 공유 폴더**에 둔다.

```
(Drive 공유 폴더)/data/
├── images/
│   └── {source}/{condition}/{label}/{파일명}
└── manifest.csv              # 모든 이미지의 목록 (유일한 기준)
```

모델 코드는 폴더를 직접 훑지 않고 **반드시 manifest.csv를 통해서만** 이미지를 찾는다. manifest.csv는 이상진의 Colab 노트북에서 데이터를 만들 때 함께 생성한다.

## manifest.csv 열

| 열 | 예시 | 설명 |
| --- | --- | --- |
| `path` | `images/genimage/jpeg75/fake/sd_00123.jpg` | `data/` 기준 상대 경로 |
| `label` | `real` / `fake` | 정답 |
| `source` | `cifake` / `genimage` / `kakao` / `latest` | 출처 데이터셋 |
| `generator` | `real`, `sd14`, `midjourney`, `adm` ... | 생성 모델 (실제 사진은 `real`) |
| `origin_id` | `genimage_sd_00123` | 원본 이미지 ID. 같은 원본의 모든 변형본은 같은 값 |
| `condition` | `original`, `jpeg95`, `jpeg75`, `jpeg50`, `jpeg30`, `resize50`, `kakao` | 변형 조건 |
| `split` | `train` / `val` / `test` | 데이터 분할 |
| `width`, `height` | `512`, `512` | 저장된 이미지 크기 |
| `format` | `jpg` / `png` | 저장 포맷 |

## 규칙

1. 같은 `origin_id`는 모두 같은 `split`에 있어야 한다 (데이터 누수 방지).
2. `test`는 학습·튜닝에 쓰지 않는다. 튜닝은 `val`로만 한다.
3. `original` 조건에서도 실제/생성 이미지의 포맷, 크기, 색상 모드가 같아야 한다 (포맷 편향 방지).
4. `kakao`와 `latest` 출처는 `test` 전용이다.
5. 이 계약을 바꾸면 `DECISIONS.md`에 남기고 상대에게 알린다.

## 완료 기준 (데이터 측)

- [ ] manifest.csv의 모든 `path`가 실제 파일로 존재한다
- [ ] `origin_id` 기준 split 겹침 검사 통과 (겹치는 ID 0개)
- [ ] `original` 조건에서 label별 포맷·크기 분포가 같다
- [ ] label별, condition별, split별 이미지 수 요약표를 함께 제공한다

## 완료 기준 (모델 측)

- [ ] manifest.csv를 읽어 `source`, `condition`, `split`으로 골라내는 데이터 로더
- [ ] 학습 시작 전 split 겹침 검사를 자동으로 실행하고, 겹치면 학습을 멈춘다
- [ ] 평가 결과를 condition별로 나눠 출력한다

## 미해결 (9주차 합의 필요)

- [ ] 통일할 기준 크기: 256×256? 512×512? (GenImage 원본 크기 확인 후 결정)
- [ ] 통일할 기준 포맷: 모두 PNG로 저장한 뒤 압축 조건만 JPEG로 만들지, 모두 같은 품질의 JPEG로 맞출지
- [ ] GenImage에서 쓸 생성 모델 종류와 생성기당 장수
- [ ] train/val/test 비율 (예: 70/15/15)
- [ ] JPEG 품질 단계 (95/75/50/30 안)
- [ ] Google Drive 공유 폴더 위치와 권한
