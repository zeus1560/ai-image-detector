# SPEC-01a: manifest 로더와 검증

- 상태: 확정
- 담당: 안승주
- 관련 결정·실험: docs/spec/data-contract.md, DECISIONS.md "SPEC-01 분리"(추가 예정)

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
  필수 열과 허용값(label, split, source)을 확인한다.
  실패하면 문제가 된 열과 행 번호를 담은 에러를 낸다.
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
  통과하면 "split 겹침 origin_id 0개"를 포함한 검사 결과를 출력한다.
- summary(df): label × condition × split 장수표.
  original 조건에서 label별 format·크기 분포를 출력한다.
  (분포가 다르면 경고만 하고 멈추지 않는다)
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

## 완료 기준
- [ ] 정상 manifest로 validate를 통과하고, 로그에 "split 겹침 origin_id 0개"가 출력된다
- [ ] 같은 origin_id를 train과 test에 넣은 manifest를 주면 LeakageError가 나고 Dataset이 생성되지 않는다
- [ ] train만 필터해서 Dataset을 만들어도, 전체 manifest에 겹침이 있으면 LeakageError가 난다
- [ ] 필수 열 하나를 뺀 manifest를 주면 빠진 열 이름이 담긴 에러가 난다
- [ ] origin_id 내 label 불일치, kakao 출처의 train 배정이 각각 ManifestValidationError로 잡히고, LeakageError는 아니다
- [ ] split 겹침과 label 불일치를 함께 넣은 manifest를 주면, LeakageError 하나가 나고 메시지에 두 위반이 모두 나온다
- [ ] 존재하지 않는 path가 있으면 check_files="all"에서 ManifestValidationError(LeakageError 아님)가 나고, 해당 path가 출력된다
- [ ] check_files=n이면 정확히 n개만 검사하고, 로그에 샘플 검사임이 표시된다
- [ ] filter 결과 장수가 summary 표의 해당 칸과 같다
- [ ] Dataset[i]가 (이미지, label, 행 인덱스)를 반환하고, label=real 행은 0, fake 행은 1이다
- [ ] 위 항목이 모두 src/model/tests/의 pytest로 로컬 CPU에서 통과한다

## 테스트 방법
- 로컬 CPU: 테스트 안에서 임시 폴더에 8×8 크기의 가짜 이미지 몇 장과 작은 manifest CSV를 만들어 위 완료 기준을 확인한다.
  CIFAKE는 필요하지 않다.
- Colab: 이 스펙에서는 하지 않는다. 실제 데이터 검증은 SPEC-01b에서 한다.

## 범위 밖
- CIFAKE를 manifest로 변환하는 일 (SPEC-01b)
- 학습·평가·증강 (SPEC-02~04)
- 정규화와 입력 크기 결정 (SPEC-02)

## 미해결
없음. 열 구성과 kakao/latest test 전용 규칙은 9주차 회의에서 변경 없음 확인
(data-contract.md 반영은 이상진님 회의 정리 후)
