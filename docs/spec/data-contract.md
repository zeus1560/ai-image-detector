# 데이터 계약 (Data Contract)

- 상태: 부분 확정 (9주차 회의 반영. "미해결" 절의 항목은 아직 정하지 않음)
- 데이터 측: 이상진 (이 형식으로 데이터를 만든다)
- 모델 측: 안승주 (이 형식으로만 데이터를 읽는다)
- 데이터 담당 영역이므로 이상진이 자유롭게 수정 의견을 낸다.

## 목적

데이터는 이 형식으로 만들고, 모델은 이 형식만 읽는다. 형식을 먼저 정해두면 데이터가 다 만들어지기 전에도 모델 측은 CIFAKE를 같은 형식으로 감싸 코드를 먼저 짤 수 있어서, 서로 기다리지 않고 동시에 작업할 수 있다.

## 저장 위치

데이터는 용량이 커서 GitHub이 아니라 Google Drive에 둔다. 위치는 이상진 소유의 `MyDrive/ai-image-detector/data`이고, 안승주는 보기 권한으로 공유받아 "내 드라이브에 바로가기 추가"로 연결한다.

```
data/
├── images/                     # Drive에는 zip으로 둔다
│   └── {source}/{condition}/{label}/{파일명}
├── manifest.csv                # GenImage·kakao·latest 이미지 목록 (이 데이터의 유일한 기준)
├── manifests/
│   └── manifest_v1.csv ...     # manifest.csv가 바뀔 때마다 남기는 읽기 전용 스냅샷
└── cifake_manifest.csv         # CIFAKE 목록 (안승주가 만든다. manifest.csv와 섞지 않는다)
```

- 이미지는 zip으로 Drive에 두고, Colab에서 `/content`로 풀어 쓴다. Drive에서 작은 파일을 직접 읽으면 매우 느리다. zip을 나누는 단위와 이름은 데이터 측이 정한다.
- 모델 코드는 폴더를 직접 훑지 않고 반드시 manifest 파일을 통해서만 이미지를 찾는다.
- manifest.csv는 이상진의 Colab 노트북에서 데이터를 만들 때 함께 생성한다.

## manifest 열 (manifest.csv, cifake_manifest.csv 공통)

| 열 | 예시 | 설명 |
| --- | --- | --- |
| `path` | `images/genimage/jpeg75/fake/sd14_00123.jpg` | `data/` 기준 상대 경로 |
| `label` | `real` / `fake` | 정답 |
| `source` | `cifake` / `genimage` / `kakao` / `latest` | 출처 데이터셋 |
| `generator` | `real`, `sd14`, `sd15`, `wukong` ... | 생성 모델 (실제 사진은 `real`) |
| `origin_id` | `genimage_sd14_00123` | 원본 이미지 ID. 같은 원본의 모든 변형본은 같은 값 |
| `condition` | `original`, `jpeg90`, `jpeg75`, `jpeg50`, `jpeg30`, `resize50`, `sns_sim`, `kakao` | 변형 조건 (`sns_sim`의 정의는 미해결) |
| `split` | `train` / `val` / `test` | 데이터 분할 |
| `width`, `height` | `512`, `512` | 저장된 이미지 크기 |
| `format` | `jpg` / `png` | 저장 포맷 |
| `color_mode` | `RGB` / `L` / `CMYK` | 변환 전 원본의 PIL 색상 모드. 저장된 파일은 모두 RGB |

## 기준본(original)과 변형본 만드는 방법 (GenImage)

1. 실제 사진은 가로·세로 모두 512 이상인 것만 쓴다.
2. 512×512로 중앙 크롭한다. 크롭 시작 좌표는 8의 배수로 맞춘다. 실제 사진에 남아 있는 원래 JPEG의 8×8 블록 격자가 어긋나지 않게 하기 위해서다.
3. CMYK는 RGB로 변환한다. 흑백(L)은 규칙 6을 따른다. 변환 전 모드는 `color_mode`에 기록한다.
4. 실제·생성 모두 PIL로 JPEG 품질 96으로 저장한다. 이것이 `original`이다. ImageNet 실제 사진은 이미 JPEG라서 PNG로 저장해도 기존 JPEG 흔적이 남기 때문에, 양쪽을 같은 JPEG 조건으로 맞춘다.
5. 압축 변형본은 Q96 기준본을 다시 열어 품질 90/75/50/30으로 재압축한다(`jpeg90`~`jpeg30`). 이미 JPEG인 사진을 다시 압축하는 카톡 상황과 같은 순서다.
6. `resize50`은 기준본을 가로·세로 50%로 줄인 평가용 변형이다.
7. train split에는 `original`만 저장한다. 압축 변형본은 val/test에만 만든다. 학습용 압축은 모델 측이 학습 중에 즉석으로 만든다(SPEC-04).

## 규칙

1. 같은 `origin_id`는 모두 같은 `split`에 있어야 한다 (데이터 누수 방지).
2. `test`는 학습·튜닝에 쓰지 않는다. 튜닝은 `val`로만 한다.
3. `original` 조건에서도 실제/생성 이미지의 포맷, 크기, 색상 모드가 같아야 한다 (포맷 편향 방지).
4. `kakao`와 `latest` 출처는 `test` 전용이다.
5. 이 계약을 바꾸면 `DECISIONS.md`에 남기고 상대에게 알린다.
6. 흑백(`color_mode`가 `L`) 실제 사진은 train/val에 넣지 않는다 (A안). test에서의 처리는 미해결.
7. 분할 비율은 70/15/15로 시작하고, `origin_id` 단위로 나눈다. 변형본을 뺀 원본 수가 적으면 60/20/20을 검토한다.
8. manifest.csv를 바꿀 때마다 `manifests/manifest_vN.csv`로 스냅샷을 남긴다. 스냅샷은 엑셀이나 pandas로 다시 저장하지 않고 파일 복사로 만든다. 모델 측이 파일 해시로 버전을 대조하기 때문이다.
9. CIFAKE는 `cifake_manifest.csv`에만 넣고, 압축 실험에는 쓰지 않는다. CIFAKE는 실제·생성 12만 장 모두 이미 JPEG 품질 75로 저장되어 있다 (2026-11 전수 확인).

## 완료 기준 (데이터 측)

- [ ] manifest.csv의 모든 `path`가 실제 파일로 존재한다
- [ ] `origin_id` 기준 split 겹침 검사 통과 (겹치는 ID 0개)
- [ ] `original` 조건에서 label별 포맷·크기 분포가 같다
- [ ] 저장된 이미지 파일이 모두 RGB다 (별도 검사)
- [ ] label별, condition별, split별 이미지 수 요약표를 함께 제공한다
- [ ] manifest.csv를 바꿀 때마다 스냅샷을 남긴다

## 완료 기준 (모델 측)

- [x] manifest를 읽어 `source`, `condition`, `split`으로 골라내는 데이터 로더 (SPEC-01a)
- [x] 학습 시작 전 split 겹침 검사를 자동으로 실행하고, 겹치면 데이터셋을 만들지 않아 학습이 시작되지 않는다 (SPEC-01a)
- [ ] manifest를 읽을 때 파일 해시를 출력해 실험에 쓴 버전을 남긴다 (SPEC-01a v2 예정)
- [ ] 평가 결과를 condition별로 나눠 출력한다

## 확인된 수치

- 가로·세로 512 이상 실제 사진: 66,657장 / 전체 1,256,098장 (5.3%). 이상진 확인. 생성기 폴더 간 중복을 뺀 수인지는 (미확인)

## 미해결

- [ ] `sns_sim` 조건의 정의 (리사이즈 크기, 압축 품질, 적용 순서)
- [ ] 생성기 최종 선택 (후보: SD v1.4 / SD v1.5 / Wukong)과 생성기당 장수
- [ ] 흑백 비율 (3,000장 샘플로 확인 예정)과 test에서의 흑백 처리
- [ ] 실제 사진의 `origin_id` 규칙: 같은 ImageNet 사진이 여러 생성기 폴더에 들어 있으면 같은 `origin_id`가 되어야 한다 (예: ImageNet 원래 파일 이름 사용)
- [ ] split을 나눌 때 층화(label·generator별) 여부와 seed 고정 방식
- [ ] `resize50`의 보간 방법과 저장 포맷, JPEG 저장 시 크로마 서브샘플링 설정
- [ ] CIFAKE 이미지의 Drive 위치와 `data/cifake/` 편집 권한
