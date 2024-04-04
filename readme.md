ref : Notion page

Temporal Population Encoding 기법을 적용하는 파일

형식 : 입력 데이터를 Encoding 레이어를 통해 변환함(SNN MLP로 학습하는 것은 옵션으로 두며, 추후 필요할 시 구현예정)

파라미터 입력값(.json 파일로 지정)

- json 파일 경로
    - 데이터에 해당하는 1차원 list csv파일 경로 : 진행중인 ECG 프로젝트에 맞춰 해당 형식부터 구현 예정
    - 차원 수 : 해당 데이터의 차원을 얼마나 늘릴 것인가?(1 이하는 무의미하니 안됨)
    - 학습 옵션 : 일단 배제하지만 옵션값 자체는 binary(false)로 지정
    - 출력결과 저장할 폴더 경로

출력값

- 인코딩된 데이터(2차원 list csv파일, 파일명은 원본 데이터파일명_encoded.csv)
- 인코딩에 사용된 1차원 뉴런들의 $\tau$, $g$ 값 들어간 데이터(2차원 list csv파일, 파일명은 원본 데이터파일명_parameters.csv)