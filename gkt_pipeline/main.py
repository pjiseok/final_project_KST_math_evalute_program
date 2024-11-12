# main.py

import boto3  # S3에서 데이터 로드를 위한 boto3 라이브러리 추가
from db_connection import load_env_from_s3, load_model_from_s3, get_db_connection
from data_retrieval import load_main_tables, load_user_specific_data
from model_loader import get_model_tensors

# 새로 만든 처리 함수 임포트
from existing_learner import process_existing_learner
from new_learner import process_new_learner
from after_learning import process_after_learning

import logging
from logging.handlers import RotatingFileHandler

# -------------------- 로깅 설정 --------------------
# 로거 생성 및 중복 핸들러 제거
logger = logging.getLogger('FormativeEvaluationLogger')
if logger.hasHandlers():
    logger.handlers.clear()

# 로그 포맷 설정
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 파일 핸들러 설정 (5MB 파일 크기 제한, 백업 파일 2개)
file_handler = RotatingFileHandler('application.log', maxBytes=5*1024*1024, backupCount=2)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# 콘솔 핸들러 설정
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# 로거에 핸들러 추가
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def main():
    # S3 버킷과 파일 경로 설정
    env_bucket_name = "dev-team-haejo-backup"
    env_file_key = "env/env_file.txt"

    # S3에서 .env 파일을 로드하여 환경 변수 설정
    load_env_from_s3(env_bucket_name, env_file_key)

    # 데이터베이스 연결
    engine = get_db_connection()

    # 주요 테이블 로드
    main_tables = load_main_tables(engine)
    chunjae_math = main_tables["chunjae_math"]
    label_math_ele_12 = main_tables["label_math_ele_12"]
    

    # UserID 입력
    user_id = input("검색할 UserID를 입력하세요: ")

    # UserID 필터링 및 데이터 로드 (진행률 바가 표시됨)
    user_specific_data = load_user_specific_data(engine, user_id, chunjae_math)

    # S3 버킷과 모델 경로 정보 설정
    s3_bucket_name = "dev-team-haejo-backup"
    s3_model_key_prefix = "model/"

    # 모델 파일 로드
    session, graph = load_model_from_s3(s3_bucket_name, s3_model_key_prefix)
    input_tensor, output_tensor = get_model_tensors(graph)

    # 기존 학습자 여부 확인 및 처리
    if not user_specific_data["result_df_all"].empty:
        process_existing_learner(user_id, user_specific_data, chunjae_math, label_math_ele_12)
    else:
        # 신규 학습자일 경우 'engine' 객체를 추가하여 함수 호출
        process_new_learner(user_id, user_specific_data, chunjae_math, label_math_ele_12, engine)

    # 형성평가 후 처리
    process_after_learning(
        user_id, user_specific_data, session, input_tensor, output_tensor,
        main_tables, chunjae_math, label_math_ele_12
    )


if __name__ == "__main__":
    main()