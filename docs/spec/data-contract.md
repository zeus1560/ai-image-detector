# 데이터 계약 (Data Contract) — 수정안

- 상태: **수정안 (2026-11-26, 안승주와 합의 전)**. 9주차 회의에서 함께 확정한다.
- 데이터 측: 이상진 (이 형식으로 데이터를 만든다)
- 모델 측: 안승주 (이 형식으로만 데이터를 읽는다)
- 표기: **(이상진 결정)** 데이터 영역이라 이상진이 정함, 합의 전 · **(제안)** 합의가 필요한 변경 · **(미확인)** 실행해 보지 못함

## 목적

데이터는 이 형식으로 만들고, 모델은 이 형식만 읽는다. 형식을 먼저 정해두면 데이터가 다 만들어지기 전에도 모델 측은 CIFAKE를 같은 형식으로 감싸 코드를 먼저 짤 수 있어서, 서로 기다리지 않고 동시에 작업할 수 있다.

## 저장 위치

데이터는 용량이 커서 GitHub이 아니라 **Google Drive 공유 폴더**에 둔다.

```
내 드라이브/ai-image-detector/
├── data/                     # 이미지 + manifest만 둔다
│   ├── images/{source}/{condition}/{label}/{파일명}
│   └── manifest.csv          # 모든 이미지의 목록 (유일한 기준)
├── cifake/                   # CIFAKE zip + cifake_manifest.csv (안승주가 전달, data/ 밖)
├── meta/                     # metadata.csv, selection.csv 등 선별 기록
└── logs/                     # 설정·환경·변환 기록
```

- 모델 코드는 폴더를 직접 훑지 않고 **반드시 manifest.csv를 통해서만** 이미지를 찾는다.
- **권한 (이상진 결정):** `data/`(이미지 + manifest.csv)는 이상진만 편집하고 안승주는 뷰어다. `cifake/`만 안승주가 편집자다.
- manifest.csv는 이상진의 Colab 노트북에서 데이터를 만들 때 함께 생성한다.

## manifest.csv 열

| 열 | 예시 | 설명 |
| --- | --- | --- |
| `path` | `images/genimage/jpeg75/fake/sd14_00123.jpg` | `data/` 기준 상대 경로 |
| `label` | `real` / `fake` | 정답 |
| `source` | `cifake` / `genimage` / `kakao` / `latest` | 출처 데이터셋 |
| `generator` | `real`, `sd14`, `sd15`, `wukong` ... | 생성 모델 (실제 사진은 `real`) |
| `origin_id` | `imagenet_n02807133_2314`, `sd14_176_sd14_52` | 원본 이미지 ID. 같은 원본의 모든 변형본은 같은 값. 실제는 `imagenet_{파일명}`, 가짜는 `{생성기}_{파일명}` |
| `condition` | `original`, `jpeg90`, `jpeg75`, `jpeg50`, `jpeg30`, `resize50`, `sns_sim`(제안), `kakao` | 변형 조건 |
| `split` | `train` / `val` / `test` | 데이터 분할 |
| `width`, `height` | `512`, `512` | 저장된 이미지 크기 |
| `format` | `jpg` / `png` | 저장 포맷 |
| `color_mode` | `RGB` / `L` / `CMYK` / `other` | **(제안, 신규)** 변환 전 원본 색상 모드. 저장 파일은 모두 RGB 3채널이다 |
| `gray` | `True` / `False` | **(제안, 신규)** 흑백 여부. 아래 "흑백 처리" 참고 |

### `gray` 열 규칙 (제안)

- 값은 문자열 `True` / `False`로 고정한다. 로더는 `df['gray'].astype(str) == 'True'`처럼 문자열로 비교한다.
- 판정: 원본이 `L` 모드이거나, RGB로 바꿔 64×64로 줄인 뒤 채널 간 평균 차이가 `GRAY_CHROMA_TH`(잠정 2.0) 미만이면 `True`. 실제·가짜에 같은 기준을 쓴다.
- **`gray` 열이 아예 없는 manifest(`cifake_manifest.csv`)는 전부 `False`로 본다.** CIFAKE는 32×32 컬러 이미지라 판정 대상이 아니고, 다시 만들면 sha256이 바뀌기 때문이다.
- 열은 있는데 값이 비어 있는 행은 오류로 처리한다.

## 규칙

1. 같은 `origin_id`는 모두 같은 `split`에 있어야 한다 (데이터 누수 방지).
2. `test`는 학습·튜닝에 쓰지 않는다. 튜닝은 `val`로만 한다.
3. `original` 조건에서도 실제/생성 이미지의 포맷, 크기, 색상 모드가 같아야 한다 (포맷 편향 방지). **(이상진 결정)** GenImage의 `original`은 실제·생성 모두 512×512, JPEG Q96, RGB 3채널이다. PNG로 통일하지 않는다.
4. `kakao`와 `latest` 출처는 `test` 전용이다. `sns_sim`(제안)은 `val`/`test` 전용이다.
5. 이 계약을 바꾸면 `DECISIONS.md`에 남기고 상대에게 알린다.
6. **(제안, 신규)** 변형본(`jpeg90` 등)은 저장된 `original` 파일을 다시 열어 만든다. 원본 이미지에서 직접 만들지 않는다. (이미 JPEG인 사진을 재압축하는 실제 전송 과정과 비슷하게 하려는 목적)
7. **(이상진 결정, 신규)** 크기 규칙: 생성 이미지는 원래 512×512인 모델(SD v1.4, SD v1.5, Wukong)만 써서 손대지 않는다. 실제 사진은 짧은 변이 512 이상인 것만 중앙 크롭한다. 리사이즈는 `resize50`(과 제안 중인 `sns_sim`) 조건에서만 쓴다. 짧은 변 256 미만 이미지는 확대·패딩 없이 제외한다.
8. **(이상진 결정, 신규)** JPEG 저장은 PIL로 통일하고 `logs/data_config.json`의 `jpeg_library`에 기록한다. 라이브러리를 섞지 않는다.
9. **(이상진 결정, 신규)** JPEG 품질 단계는 90 / 75 / 50 / 30이다. (`jpeg95`는 `original`(Q96)과 거의 겹쳐서 뺐다)
10. **(이상진 결정, 신규)** 분할은 `origin_id` 단위, label·generator 층화, 70/15/15, seed 42다.
11. **(제안, 신규)** 같은 파일이 두 번 들어 있으면 한 장만 남긴다 (평가 장수가 부풀려지지 않게). 현재는 파일명(path) 기준으로만 확인했고, 이미지 내용(md5 등) 비교는 이미지 확보 후 한다 **(미확인)**.
12. **(제안, 신규)** manifest는 읽어서 다시 저장하지 않고 바이트 그대로 복사한다. 만든 시점의 sha256을 `logs/manifest_sha256.txt`에 기록하고, `cifake_manifest.csv`도 만든 시점의 sha256을 함께 전달한다.

## 모델 측 입력 규칙 (이상진 제안, 합의 전)

- 학습: `RandomCrop(256)` + 좌우 반전. 시작 좌표는 8의 배수로 한다 (JPEG 8×8 블록 격자를 유지).
- 검증: 고정 중앙 크롭 256. 테스트: 512 전체 또는 중앙 크롭 256.
- `Resize`, `RandomResizedCrop`은 쓰지 않는다. (리사이즈가 생성 흔적(고주파)을 지울 수 있고, 압축·리사이즈 증강의 효과를 깨끗하게 보려는 목적)

## 흑백 처리 (제안)

흑백 실제 사진이 많으면 모델이 "흑백이면 진짜"라는 지름길을 배울 수 있어서 다음처럼 다룬다.

| 구분 | 실제 사진 | 가짜 |
| --- | --- | --- |
| train / val | 흑백이면 같은 클래스의 컬러 reserve로 **교체**하고 split을 물려받는다. reserve가 없으면 모든 생성기의 같은 클래스·split 가짜를 1장씩 같이 제외한다 (클래스×split의 real=fake 장수 유지) | 표시만 하고 남긴다 (`gray=True`) |
| test | 교체하지 않고 **`gray=True` 행으로 남긴다** | 표시만 하고 남긴다 |

- test 실제 사진은 전체 750장(core 기준) 안에 흑백이 n장 들어 있는 구조다. 컬러는 750 − n장이다. n은 이미지 확보 후 측정한다 **(미확인)**.
- 평가: 대표 지표는 `gray==False`, 흑백 실제 오탐률은 `label=='real' & gray==True`로 따로 보고한다. n이 작으면 오차가 커서 참고용으로만 쓴다.
- 이 방식(A안)은 학습에서 흑백 실제 사진을 못 보므로 서비스에서 흑백 입력의 오탐이 늘 수 있다. README 한계에 적는다.

## 데이터 선별 방법 (제안, 폴더 제한은 `data_02_selection` 재실행 후 확정)

- 실제 사진 후보: GenImage `nature`이고 짧은 변 ≥ 512인 이미지 (metadata 기준 66,657장). 클래스 번호가 0장인 1개 클래스는 양쪽에서 제외한다.
- **폴더 제한:** 실제 사진을 가짜와 같은 SD v1.4 / v1.5 / Wukong 폴더에서만 뽑는 안으로 `data_02_selection`을 재실행하는 중이다. 재실행 결과를 보고 이 절을 확정한다 **(미확인)**.
- 클래스에 균등 배분하고, 클래스 안은 `numpy.random.default_rng(42)`로 무작위 선택한다. 흑백 교체 대비 reserve를 클래스당 최대 2장 둔다.
- 가짜는 생성기별로 클래스·split 장수를 실제와 똑같이 맞춘다 (클래스가 특정 label과 엮이는 지름길 방지).
- 생성기 구성과 장수(잠정): SD v1.4 / v1.5 / Wukong 3개, 생성기당 5,000장. 실제 사진은 공유 세트 하나(core 5,000장)를 쓴다. SD v1.4로 학습하고 나머지를 교차 생성기 test로 쓴다.
- 재현: `notebooks/data_02_selection.ipynb`, 기록: `meta/selection.csv`.

## 완료 기준 (데이터 측)

- [ ] manifest.csv의 모든 `path`가 실제 파일로 존재한다
- [ ] `origin_id` 기준 split 겹침 검사 통과 (겹치는 ID 0개)
- [ ] `original` 조건에서 label별 포맷·크기·색상 모드 분포가 같다
- [ ] `gray` 값이 모두 `True`/`False` 문자열이고 빈 값이 없다
- [ ] 클래스×split의 real/fake 장수가 맞는다
- [ ] label별, condition별, split별, `gray`별 이미지 수 요약표를 함께 제공한다
- [ ] manifest.csv의 sha256을 `logs/manifest_sha256.txt`에 기록한다

## 완료 기준 (모델 측)

- [ ] manifest.csv를 읽어 `source`, `condition`, `split`으로 골라내는 데이터 로더
- [ ] `gray` 열이 없으면 전부 `False`로 읽고, 값이 비어 있으면 오류를 낸다
- [ ] 학습 시작 전 split 겹침 검사를 자동으로 실행하고, 겹치면 학습을 멈춘다
- [ ] 평가 결과를 condition별로 나눠 출력한다 (대표 지표 `gray==False`, 흑백 실제 오탐률 별도)

## 미해결 (9주차 합의 필요)

원래 초안의 미해결 항목 중 아래는 이상진이 정한 값이 있어 **합의만 남았다**: 기준 크기(512), 기준 포맷(JPEG Q96), 분할 비율(70/15/15), JPEG 품질 단계(90/75/50/30), Drive 폴더와 권한.

- [ ] 위 "합의만 남은" 항목과 신규 규칙 6~12, 모델 측 입력 규칙에 대한 안승주 확인
- [ ] 생성기 최종 선택(SD v1.4 / v1.5 / Wukong)과 생성기당 장수
- [ ] 흑백 처리 A안(제외) vs B안(생성 이미지도 같은 비율로 흑백 변환), `GRAY_CHROMA_TH` 값
- [ ] `color_mode`, `gray` 열 추가
- [ ] `sns_sim` 조건 도입 여부, `resize50`의 보간·저장 방식
- [ ] 실제 사진 폴더 제한 여부 (`data_02_selection` 재실행 결과 확인 후)
- [ ] 이중 압축 흔적 대응 (실제는 원래 JPEG에 Q96이 한 번 더 들어가고 가짜는 PNG에서 한 번만 압축됨): 지금 방식 유지 / 가짜도 사전 JPEG 압축 / 소규모 비교 실험
- [ ] 실제 사진의 ImageNet train/val 출처 (풀 A의 97.2%가 train 출신이라, 사전학습 백본이면 오탐률이 낙관적일 수 있음)
- [ ] 계약 밖 안건: train은 `original`만 저장하고 압축은 학습 중 증강으로 할지, CIFAKE manifest 분리(`cifake_manifest.csv`), manifest 버전 스냅샷(`data/manifests/`)
