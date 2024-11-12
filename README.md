# 초등 수학 학습 결손 추적 AI 모델</br>
빅데이터 8기 해조팀</br>
팀원: 김솔미, 김영규, 박지석, 이현희, 허현강</br>

#### R & R</br>
1) 개발환경 구성 : 김솔미, 김영규
2) 데이터 전처리 : 이현희, 박지석
3) 관련 기술 분석 : 허현강
4) DataBase : 김영규
5) 모델 구성 및 학습 : 허현강, 박지석, 이현희
6) API, 웹 UI 구현 : 김솔미</br>

![image](https://github.com/user-attachments/assets/bfc3a3a2-70a8-4ca8-b0af-17990cd6316e)</br>



패키지 버전 requirements 참조


사용 방법
1. 데이터를 이용해 raw_data_to_data 실행
2. DKT 모델 학습(model 폴더)
3. ![ERD (1)](https://github.com/user-attachments/assets/531c7e3f-1c29-4e8c-8224-32cefc9f5769)
![ERD1 (1)](https://github.com/user-attachments/assets/315e4132-81fd-4b6b-8b35-8d5bbf1fd8c1)
사진과 같이 RDS 및 S3에 적재
4. code_ALL_1031_rds 실행
5. gkt_pipeline 실행
6. 첨삭을 희망하는 학생 ID 입력
7. 결과 확인
