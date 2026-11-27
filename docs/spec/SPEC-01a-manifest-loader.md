# SPEC-01a: manifest 로더와 검증

- 상태: 구현 완료 (v3)
- 담당: 안승주
- 관련 결정·실험: docs/spec/data-contract.md, DECISIONS.md "SPEC-01 분리", "manifest 운영", "색상 모드"

## 목적
모델 코드가 데이터를 manifest.csv로만 읽게 하는 공통 입구를 만든다.
split 간 origin_id 겹침을 학습 전에 자동으로 차단해서, 핵심 결과 문장의 수치가
누수로 부풀려지지 않게 한다. 데이터 출처(CIFAKE, GenImage 등)와 무관하게
데이터 계약 형식만 따르면 동작한다.

## 입력
- 데이터 계약 형식의 manifest CSV 경로
- data_root: manifest의 path가 기준으로 삼는 data/ 폴더의 실제 위치
  (로컬, Drive, /content 어디든 가능)

## 출력
코드 (src/model/)
- manifest.py: load_manifest, validate, summary, filter
- dataset.py: ManifestDataset
- tests/: pytest 테스트

## 처리 방식
### manifest.py
- load_manifest(path, data_root)
  필수 열과 허용값(label, split, source, color_mode)을 확인한다.
  - 필수 열: 데이터 계약의 열 11개 (기존 10개 + color_mode)
  - color_mode 허용값: RGB / L / CMYK (변환 전 원본의 PIL 모드. 저장 파일은 모두 RGB)
  실패하면 문제가 된 열과 행 번호를 담은 에러를 낸다.
  읽을 때 manifest 경로, 행 수, 파일의 sha256 해시를 출력한다.
  - gray는 선택 열이다 (REQUIRED_COLUMNS는 11개 그대로).
  - gray 열이 없으면: 모든 행을 False로 채우고 로그에 "gray 열 없음 → 전부 False (행 N개)"를 출력한다.
    파일은 건드리지 않으므로 sha256은 그대로다. (cifake_manifest.csv 대상)
  - gray 열이 있으면 아래 순서로 검사한다.
    1. 빈 값('' 또는 공백만) 행 → 빈 값 위반으로 따로 모은다
    2. 빈 값이 아닌 행 중 값이 정확히 'True' / 'False'가 아닌 행 → 허용값 위반
       ('true', 'TRUE', '1', ' True', 'nan' 모두 위반. 공백 제거·대소문자 변환을 하지 않는다)
    빈 값 검사를 문자열 비교보다 먼저 한다. 빈 값이 비교에서 조용히 False로 처리되지 않게 하기 위함이다.
    위반은 기존 허용값 검사 위반과 함께 모아 ManifestValidationError 하나로 보고한다 (열 이름, 줄 번호, 빈 값/허용값 구분 포함).
  - 통과하면 gray 열을 bool(True/False)로 바꿔 df에 담는다.
    df.attrs["gray_column"]에 "present" 또는 "absent"를 기록한다.
  - 열 위치(마지막 열인지)는 검사하지 않는다. 열 이름으로 읽는다.
- manifest_sha256(path): manifest 파일 바이트 전체의 sha256 값을 반환한다.
  (manifest.csv는 바뀔 때마다 data/manifests/manifest_vN.csv로 스냅샷이 남으므로,
   출력된 해시를 스냅샷과 대조해 실험에 쓴 버전을 확인한다)
- validate(df, data_root, check_files="all")
  아래 1~4를 모두 검사한 뒤, 위반을 모아 하나의 에러로 보고한다.
  (처음 위반에서 멈추지 않는다. 데이터 측이 한 번에 고칠 수 있게 하기 위함)
  - 에러 메시지에는 위반 종류별 개수와 예시가 모두 들어간다.
  - 위반 중 1번(split 겹침)이 있으면 LeakageError, 없으면 ManifestValidationError를 낸다.
    LeakageError는 ManifestValidationError의 하위 클래스로 한다.
  1. 한 origin_id가 두 개 이상의 split에 존재 → LeakageError
     (겹치는 ID 개수와 예시 최대 10개 출력)
  2. 한 origin_id 안에서 label이 다름
  3. source가 kakao 또는 latest인데 split이 test가 아님
  4. path가 가리키는 파일이 없음
     - check_files="all": 전체 검사 (기본값)
     - check_files=정수 n: seed로 고정한 n개만 검사. 샘플 검사라는 사실을 로그에 출력한다
  5. label == real, gray == True인데 split이 train 또는 val → ManifestValidationError (LeakageError 아님)
     (계약: train/val의 흑백 실제 사진은 컬러 reserve로 교체됨. 남아 있으면 "흑백이면 진짜" 지름길이 생김)
     흑백 가짜(gray=True, fake)는 어느 split에 있어도 위반이 아니다.
  통과 로그에 "train/val 흑백 실제 0개"와 "흑백 행: 실제 test n장, 가짜 split별 장수"를 추가한다.
  통과하면 "split 겹침 origin_id 0개"를 포함한 검사 결과를 출력한다.
- summary(df): label × condition × split 장수표.
  original 조건에서 label별 format·크기 분포를 출력한다.
  (분포가 다르면 경고만 하고 멈추지 않는다)
  기존 표 다음에 label × split × gray 장수표를 출력한다. (흑백 실제 test 장수 n을 여기서 확인)
- filter(df, source=None, condition=None, split=None): 조건에 맞는 행만 반환

### dataset.py: ManifestDataset
- torch를 import하지 않는다. __len__과 __getitem__만 갖춘 클래스로 만들고,
  PyTorch DataLoader에 그대로 넘길 수 있게 한다.
  (로컬 Windows에서 torch DLL이 앱 제어 정책으로 차단되어, 로컬 테스트를 가능하게 하기 위함.
   DataLoader 동작 확인은 SPEC-01b의 Colab 테스트에서 한다)
- 생성할 때 필터 전 전체 manifest에 validate를 실행한다.
  (필터한 일부만 검사하면 split 간 겹침을 발견할 수 없기 때문)
  통과하지 못하면 Dataset이 생성되지 않는다.
- __getitem__: PIL로 열고 RGB로 변환 → transform 적용 → (이미지, label, 행 인덱스)
  - label: real = 0, fake = 1
- transform은 외부에서 넘겨받는다. 정규화와 입력 크기는 정하지 않는다.
- 생성 후 manifest_sha256 속성에 읽은 manifest의 해시를 담는다. (SPEC-02에서 실험 기록에 남길 때 사용)

## 완료 기준
- [x] 정상 manifest로 validate를 통과하고, 로그에 "split 겹침 origin_id 0개"가 출력된다
- [x] 같은 origin_id를 train과 test에 넣은 manifest를 주면 LeakageError가 나고 Dataset이 생성되지 않는다
- [x] train만 필터해서 Dataset을 만들어도, 전체 manifest에 겹침이 있으면 LeakageError가 난다
- [x] 필수 열 하나를 뺀 manifest를 주면 빠진 열 이름이 담긴 에러가 난다
- [x] origin_id 내 label 불일치, kakao 출처의 train 배정이 각각 ManifestValidationError로 잡히고, LeakageError는 아니다
- [x] split 겹침과 label 불일치를 함께 넣은 manifest를 주면, LeakageError 하나가 나고 메시지에 두 위반이 모두 나온다
- [x] 존재하지 않는 path가 있으면 check_files="all"에서 ManifestValidationError(LeakageError 아님)가 나고, 해당 path가 출력된다
- [x] check_files=n이면 정확히 n개만 검사하고, 로그에 샘플 검사임이 표시된다
- [x] filter 결과 장수가 summary 표의 해당 칸과 같다
- [x] Dataset[i]가 (이미지, label, 행 인덱스)를 반환하고, label=real 행은 0, fake 행은 1이다
- [x] 위 항목이 모두 src/model/tests/의 pytest로 로컬 CPU에서 통과한다
- [x] color_mode 열을 뺀 manifest를 주면 "color_mode"가 담긴 에러가 난다
- [x] color_mode에 허용값 밖의 값(예: RGBA)이 있으면 열 이름과 줄 번호가 담긴 에러가 난다
- [x] load_manifest가 출력하는 sha256이 hashlib로 따로 계산한 파일 해시와 같고, 파일을 1바이트만 바꿔도 값이 달라진다
- [x] ManifestDataset.manifest_sha256이 load_manifest가 출력한 값과 같다
- [x] v1의 기존 테스트가 모두 그대로 통과한다 (테스트용 manifest에 color_mode 열 추가 외에는 변경 없이)
- [x] (v3) gray 열이 없는 manifest는 오류 없이 읽히고, 모든 행의 gray가 False이며, 로그에 "gray 열 없음"이 나온다
- [x] (v3) gray 열이 없는 manifest를 읽기 전후로 파일 sha256이 같다
- [x] (v3) gray가 'True'/'False'인 manifest를 읽으면 df["gray"]가 bool이고 값이 파일과 같다
- [x] (v3) gray 칸이 빈 행이 있으면 ManifestValidationError가 나고, 메시지에 "gray", 줄 번호, 빈 값이라는 표시가 있다 (False로 읽히지 않는다)
- [x] (v3) gray가 'true', 'TRUE', '1', ' True' 중 하나인 행이 있으면 각각 열 이름과 줄 번호가 담긴 허용값 에러가 난다
- [x] (v3) 빈 값과 허용값 밖 값이 함께 있으면 에러 하나에 두 위반이 모두 나온다
- [x] (v3) real·gray=True가 train(또는 val)에 있으면 ManifestValidationError가 나고 LeakageError는 아니다
- [x] (v3) real·gray=True가 test에만 있고 fake·gray=True가 train에 있는 manifest는 통과한다
- [x] (v3) summary의 label × split × gray 표에서 (real, test, True) 칸이 df로 직접 센 장수와 같다
- [x] (v3) v2의 기존 테스트가 모두 변경 없이 통과한다 (gray 열 없는 픽스처 그대로)

## 테스트 방법
- 로컬 CPU: 테스트 안에서 임시 폴더에 8×8 크기의 가짜 이미지 몇 장과 작은 manifest CSV를 만들어 위 완료 기준을 확인한다.
  CIFAKE는 필요하지 않다.
- Colab: 이 스펙에서는 하지 않는다. 실제 데이터 검증은 SPEC-01b에서 한다.

## 범위 밖
- CIFAKE를 manifest로 변환하는 일 (SPEC-01b)
- 학습·평가·증강 (SPEC-02~04)
- 정규화와 입력 크기 결정 (SPEC-02)
- 흑백 판정 자체 (L 모드·채도 기준 GRAY_CHROMA_TH). 데이터 측이 한다
- gray 기준 평가 지표 (SPEC-03 보완 2)

## 미해결
1. (해결됨 → v3) 흑백 규칙은 "color_mode=L 금지"가 아니라 gray 열 기준으로 검사한다
2. color_mode 허용값에 other 추가 (계약 합의 후 확인)
3. color_mode=L인데 gray=False인 행을 위반으로 볼지 (판정 기준과 GRAY_CHROMA_TH가 계약 미해결이라 보류)

## 변경 이력
- v1 (a1fcc20 확정, 95365bb 구현 완료): 최초 버전
- v2 (2e67b1b 확정, 구현 완료): color_mode 열 추가, manifest 해시 출력 추가 (9주차 회의 후속)
- v3 (d6d7c14 확정, 구현 완료): gray 선택 열 검증, 흑백 실제의 train/val 배정 검사, gray 장수표 (흑백 규칙 11/26 합의, 계약 fc748cc)
