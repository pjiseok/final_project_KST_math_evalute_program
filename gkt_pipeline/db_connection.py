# db_connection.py
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import boto3
import tempfile
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_env_from_s3(bucket_name, env_file_key):
    """
    S3에서 .env 파일을 다운로드하고 환경 변수로 로드합니다.
    
    Parameters:
        bucket_name (str): S3 버킷 이름
        env_file_key (str): S3의 환경 변수 파일 키 (파일 경로)
    """
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, env_file_key, '/tmp/env_file.txt')
    load_dotenv('/tmp/env_file.txt')
    logging.info(f"S3에서 .env 파일을 다운로드: 버킷={bucket_name}, 키={env_file_key}")

def get_db_connection():
    """
    환경 변수에서 데이터베이스 연결 정보를 불러와 연결 객체를 반환합니다.

    Returns:
        engine: SQLAlchemy 엔진 객체
    """
    # S3 버킷과 파일 경로 설정
    bucket_name = 'dev-team-haejo-backup'  # S3 버킷 이름
    env_file_key = 'env/env_file.txt'  # S3 내 파일 경로
    
    # S3에서 환경 변수 파일 로드
    load_env_from_s3(bucket_name, env_file_key)
    
    # 환경 변수에서 데이터베이스 정보 불러오기
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT')
    db_name = os.getenv('DB_NAME')
    
    # 환경 변수 값이 비어 있을 경우 오류 발생
    if not all([user, password, host, port, db_name]):
        raise EnvironmentError("S3에서 다운로드된 환경 변수 파일에 데이터베이스 연결 정보가 필요합니다.")
    
    engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}?charset=utf8')
    logging.info("데이터베이스 연결이 성공적으로 설정되었습니다.")
    return engine

def load_model_from_s3(bucket_name, model_key_prefix):
    """
    S3에서 모델 파일을 다운로드하고 TensorFlow 모델을 로드합니다.

    Args:
        bucket_name (str): S3 버킷 이름
        model_key_prefix (str): 모델 파일이 저장된 S3 경로(prefix)

    Returns:
        session: TensorFlow 세션 객체
        graph: TensorFlow 그래프 객체
    """
    # 환경 변수에서 AWS 자격 증명을 가져옴
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION")

    # S3 클라이언트 생성
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )

    # 모델 파일 목록 가져오기
    logging.info(f"S3에서 모델 파일 목록 가져오기: 버킷={bucket_name}, 경로={model_key_prefix}")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=model_key_prefix)
    files = response.get('Contents', [])

    # 필요한 파일 분류
    meta_file = None
    index_file = None
    data_files = []

    for obj in files:
        key = obj['Key']
        if key.endswith('.meta'):
            meta_file = key
        elif key.endswith('.index'):
            index_file = key
        elif '.data-' in key:
            data_files.append(key)

    if not meta_file or not index_file or not data_files:
        logging.error("S3에 필요한 모델 파일이 모두 존재하지 않습니다.")
        raise ValueError("S3에 모델 파일이 완전하지 않습니다.")

    # 임시 디렉토리 생성
    with tempfile.TemporaryDirectory() as tmpdirname:
        logging.info(f"모델 파일을 임시 디렉토리에 다운로드: {tmpdirname}")
        # 각 파일 다운로드
        meta_local_path = os.path.join(tmpdirname, os.path.basename(meta_file))
        s3.download_file(bucket_name, meta_file, meta_local_path)

        index_local_path = os.path.join(tmpdirname, os.path.basename(index_file))
        s3.download_file(bucket_name, index_file, index_local_path)

        for data_key in data_files:
            data_local_path = os.path.join(tmpdirname, os.path.basename(data_key))
            s3.download_file(bucket_name, data_key, data_local_path)

        # 체크포인트 경로 설정
        checkpoint_path = os.path.join(tmpdirname, os.path.basename(index_file).replace('.index', ''))

        # TensorFlow 세션 및 그래프 로드
        logging.info("TensorFlow 모델을 로드 중입니다.")
        session = tf.compat.v1.Session()
        saver = tf.compat.v1.train.import_meta_graph(f"{checkpoint_path}.meta")
        saver.restore(session, checkpoint_path)
        graph = tf.compat.v1.get_default_graph()
        logging.info("TensorFlow 모델이 성공적으로 로드되었습니다.")

    return session, graph
