# SPEC-01b: CIFAKE manifest 만들기

- 상태: 구현 완료
- 담당: 안승주
- 관련 결정·실험: docs/spec/data-contract.md, DECISIONS.md "CIFAKE의 역할", "manifest 운영", "SPEC-01 분리", SPEC-01a

## 목적
CIFAKE를 데이터 계약 형식(cifake_manifest.csv와 계약 폴더 구조)으로 바꿔서,
GenImage 데이터가 준비되기 전에 SPEC-02 학습 코드를 실제 데이터로 돌려볼 수 있게 한다.
CIFAKE는 파이프라인 확인용이며 압축 실험에는 쓰지 않는다.

## 입력
- CIFAKE 원본 폴더 (로컬 C:\data\cifake): train/REAL, train/FAKE, test/REAL, test/FAKE
- 확인된 사실 (2026-11 로컬 확인)
  - 장수: train/REAL 50,000, train/FAKE 50,000, test/REAL 10,000, test/FAKE 10,000
  - 포맷: 모두 JPEG(.jpg). 12만 장 전수 확인 결과 양자화 테이블 1종(IJG 품질 75), 4:2:0 서브샘플링, 읽기 실패 0개
  - 크기·모드: 32×32 RGB (폴더당 정렬 순서 앞 200장 확인. 전수는 이 스펙의 실행에서 확인)
  - 파일 이름: "번호.jpg" 또는 "번호 (2~10).jpg". 번호 표기는 폴더마다 다르다
    (train/REAL·test/REAL은 0000처럼 4자리, test/FAKE는 0처럼 0을 붙이지 않음, train/FAKE는 1000~5999)
  - 같은 split 안에서 REAL과 FAKE의 파일 이름이 train에서 40,000개 겹친다 (test는 0개)
  - 생성 방법 (Bird & Lotfi 논문, arXiv 2303.14126 / IEEE Access 2024):
    FAKE는 Stable Diffusion 1.4로 512px에서 생성(50단계, Euler Ancestral)한 뒤 바이리니어 보간으로 32px로 축소했다.
    REAL은 CIFAR-10 원본(32×32)이다. 논문은 REAL=1, FAKE=0으로 쓰며, 이 프로젝트는 real=0, fake=1이다.

## 출력
1. 출력 폴더(인자로 받음, 계약의 data/ 역할) 아래:
   - images/cifake/original/real/, images/cifake/original/fake/ : 이름을 바꿔 복사한 이미지
   - cifake_manifest.csv
2. 코드: src/model/build_cifake_manifest.py (명령행 실행: 원본 폴더, 출력 폴더를 인자로 받음)
3. 실행 로그: 행 수, label × split 장수표, validate 결과, manifest 해시, color_mode 분포

## 처리 방식
1. 파일 이름 해석: 이름이 "번호.jpg" 또는 "번호 (n).jpg" 규칙에 맞지 않으면 에러를 내고 멈춘다(해당 파일 이름 출력).
   번호는 정수로 바꾸고(0000 → 0), 괄호가 없는 이름은 순번 1로 본다.
2. origin_id: cifake_{label}_{원래 split}_{번호}_{순번}  (예: cifake_real_train_0_10)
   - label을 넣어 REAL·FAKE의 같은 파일 이름을 구분한다.
   - "원래 split"은 CIFAKE의 원래 폴더(train/test)이며, 이 스펙이 정하는 split과 다를 수 있다
     (train에서 떼어낸 val 이미지도 ID에는 train이 남는다).
   - origin_id가 중복되면 에러를 내고 멈춘다.
3. 이미지 복사: images/cifake/original/{label}/{origin_id}.jpg 로 복사한다.
   PIL로 다시 저장하지 않고 파일을 바이트 그대로 복사한다 (다시 저장하면 JPEG가 재압축된다).
4. 열 값:
   - label: REAL → real, FAKE → fake
   - source: cifake / condition: original
   - generator: real 또는 sd14 (논문 기준 Stable Diffusion 1.4)
   - width, height, format, color_mode: 가정하지 않고 실제 파일을 PIL로 열어 읽은 값 (format은 jpg)
5. split: 원래 test → test. 원래 train에서 label별로 10,000장씩 무작위로 val로 떼고 나머지는 train.
   seed는 상수로 고정하고 로그에 출력한다.
6. 마지막에 SPEC-01a의 load_manifest와 validate(check_files="all")를 실행하고, summary를 출력한다.

## 완료 기준
### A. 미니 폴더 pytest (로컬 CPU)
- [x] "0000 (10).jpg" → 번호 0·순번 10, "0000.jpg" → 순번 1, "0 (2).jpg" → 번호 0·순번 2로 해석한다
- [x] REAL과 FAKE에 같은 파일 이름이 있어도 origin_id가 서로 다르다
- [x] 규칙에 맞지 않는 파일 이름이 있으면 에러가 나고 그 이름이 출력된다
- [x] origin_id가 중복되는 입력이면 에러가 난다
- [x] 복사된 파일이 원본과 바이트 단위로 같다
- [x] 같은 seed로 두 번 실행하면 split 열이 완전히 같다
- [x] 결과 manifest가 SPEC-01a validate를 통과한다

### B. 실제 CIFAKE 로컬 실행
- [x] manifest 행 수가 120,000이고 원본 파일 수와 같다
- [x] label × split 장수가 train 40,000/40,000, val 10,000/10,000, test 10,000/10,000이다
- [x] validate 결과에 "split 겹침 origin_id 0개"와 manifest 해시가 출력된다
- [x] summary에서 포맷·크기 분포 경고가 나오지 않는다
- [x] color_mode, width, height 분포가 전수 기준으로 출력된다

### C. Colab 실행 (Drive 업로드 후)
- [x] zip을 /content에 풀고 ManifestDataset과 PyTorch DataLoader로 배치 하나를 꺼낸다
      (ToTensor 기준 이미지 크기 (배치, 3, 32, 32), label은 0과 1만 있음)

A·B가 끝나고 C만 남으면 상태를 "구현 완료 (Colab 확인 대기)"로 둔다.

## 테스트 방법
- 로컬 CPU: pytest에서 임시 폴더에 CIFAKE와 같은 이름 규칙의 미니 폴더(폴더당 몇 장, 8×8 JPEG)를 만들어 A를 확인한다.
- 로컬 실제 실행: C:\data\cifake로 B를 확인한다. 출력 로그 전체를 남긴다.
- Colab: 업로드 후 C를 확인한다.

## 알려진 한계
- CIFAKE의 실제 사진은 CIFAR-10에서 왔고, CIFAR-10 test의 3.25%(325장, train과 중복 286 + test 내부 중복 39)가 근접 중복이다
  (Barz & Denzler, J. Imaging 2020, ciFAIR). 파일이 서로 달라 origin_id 검사로는 잡히지 않는다.
  CIFAKE는 파이프라인 확인용이므로 고치지 않고 한계로 기록한다 (DECISIONS.md "CIFAKE 알려진 한계").
- FAKE만 512px → 32px 바이리니어 축소를 거쳤다. 모델이 생성 흔적이 아니라 축소 흔적으로 구분할 수 있으므로,
  CIFAKE 결과는 압축 강건성이나 생성 흔적 탐지의 근거로 쓰지 않는다 (DECISIONS.md "CIFAKE의 역할"과 같은 이유).

## 범위 밖
- 학습·평가 (SPEC-02 이후)
- GenImage manifest (이상진)
- 압축·리사이즈 변형 (CIFAKE는 압축 실험에 쓰지 않음)

## 미해결
1. Drive 업로드 방법: data/images/cifake/와 data/cifake_manifest.csv에 쓸 편집 권한을 받을지,
   zip을 이상진님께 전달해 올려달라고 할지 (C 묶음만 영향)

## 구현 중 추가 사항 (2026-11-18, 스펙 요구사항 변경 없음)
- build()에 테스트용 인자 val_per_label 추가. 명령행 실행은 기본값 10,000 고정.
- 출력 폴더가 원본 폴더와 같거나 서로 포함 관계이면 에러.
- 로그에 원본 파일 수와 manifest 행 수 일치 여부 출력.

## 실행 결과 (B)
- manifest: C:\data\cifake_out\cifake_manifest.csv
- sha256: 79e176eeb61bb024e6ffbc5163b5c84fa02f23daf74b9e39c2427086bcb3740c
- 행 120,000 / split 겹침 0 / 없는 파일 0
- 실행 로그: C:\data\cifake_out\spec01b_build_log.txt (레포 밖)

## Colab 확인 결과 (C, 2026-11-20)
- 코드: 7d39f41 / 실행: Colab(브라우저), 확인 셀은 레포 밖
- 입력: cifake.zip (133,195,351 B, sha256 cde8714fbe847e9b670be15b5b42a37b6b87f3f5481efc3032b0445351c54567),
  cifake_manifest.csv (ds.manifest_sha256 79e176eeb61bb024e6ffbc5163b5c84fa02f23daf74b9e39c2427086bcb3740c)
- /content/data에 unzip: 이미지 120,000개
- ManifestDataset validate 통과: split 겹침 0, label 불일치 0, 없는 파일 0
- split×label: train 40,000/40,000, val 10,000/10,000, test 10,000/10,000 (real/fake)
- DataLoader 배치: (64, 3, 32, 32), label {0, 1}, torch.int64
- 소요 시간: Drive→/content 복사 6.3s(첫 실행), unzip 12.6s·18.0s(두 번 실행)
