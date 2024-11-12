import pandas as pd
from sqlalchemy import create_engine, inspect

# RDS 접속 정보 설정
db_user = ''
db_password = ''
db_host = ''  
db_port = 3306
db_name = ''

# SQLAlchemy 엔진 생성 (PostgreSQL 예시)
engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?charset=utf8')

# 인스펙터(inspector)를 사용하여 테이블 목록 가져오기
inspector = inspect(engine)
tables = inspector.get_table_names()

# 테이블 목록 출력
if tables:
    print("RDS에 저장된 테이블 목록:")
    for table in tables:
        print(f"- {table}")
else:
    print("RDS에 테이블이 존재하지 않습니다.")

print("RDS 테이블 확인 완료.")

# 테이블 목록 출력 및 각 테이블의 정보 확인
if tables:
    print("RDS에 저장된 테이블 목록:")
    for table in tables:
        print(f"- {table}")

        try:
            # 테이블을 데이터프레임으로 불러오기
            df = pd.read_sql_table(table, con=engine)
            
            # 행 수와 열 수 출력
            print(f"테이블: {table}")
            print(f"행 x 열: {df.shape}")
            
            # 열 이름 출력
            print(f"열 이름: {df.columns.tolist()}")
            print("="*50)
        except Exception as e:
            print(f"테이블 '{table}'을(를) 불러오는 중 오류가 발생했습니다: {e}")
else:
    print("RDS에 테이블이 존재하지 않습니다.")

print("RDS 테이블 정보 확인 완료.")