from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, APIRouter
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine
import boto3
import botocore
import logging
import os
import urllib.parse
from datetime import datetime
import pytz
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import plotly
import io
from matplotlib import font_manager
import shutil
import openai
import json
import re
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import tempfile
import warnings



# FastAPI 인스턴스 생성
app = FastAPI()


# 정적 파일 서빙을 위한 경로 설정
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define CORS settings
origins = ["*"]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 루트 경로로 접속 시 static 디렉토리의 api.html 파일 반환
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/api.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

# S3 설정
s3 = boto3.client('s3')
# download_bucket = 'fast-solmi2'
backup_bucket = 'fastapi-backup'
local_data_dir = "./data"
os.makedirs(local_data_dir, exist_ok=True)


# KST 시간대 설정
KST = pytz.timezone('Asia/Seoul')

# 로그 설정 (KST 적용)
def get_kst_time(*args):
    return datetime.now(KST).timetuple()

logging.basicConfig(
    filename="api_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.Formatter.converter = get_kst_time



def load_env_from_s3(bucket_name, env_file_key):

    s3 = boto3.client('s3')
    s3.download_file(bucket_name, env_file_key, '/tmp/env_file.txt')
    load_dotenv('/tmp/env_file.txt')
    logging.info(f"S3에서 .env 파일을 다운로드: 버킷={bucket_name}, 키={env_file_key}")

def get_db_connection():

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
    
    engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}?charset=utf8mb4')
    logging.info("데이터베이스 연결이 성공적으로 설정되었습니다.")
    return engine

engine = get_db_connection()

def load_model_from_s3(bucket_name, model_key_prefix):

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

def load_checkpoint_model(checkpoint_dir):
    sess = tf.compat.v1.Session()
    checkpoint_path = checkpoint_dir
    saver = tf.compat.v1.train.import_meta_graph(checkpoint_path + ".meta")
    saver.restore(sess, checkpoint_path)
    graph = tf.compat.v1.get_default_graph()
    return sess, graph

def get_model_tensors(graph):
    input_tensor = graph.get_tensor_by_name('X:0')
    output_tensor = graph.get_tensor_by_name('output_layer/preds:0')
    return input_tensor, output_tensor




# UserID 입력 데이터 모델
class UserIDInput(BaseModel):
    user_id: str

user_id = None
data = None

@app.post("/check_user_id/", response_class=HTMLResponse)
async def check_user_id(user_id_input: UserIDInput):
    global user_id, result_df_all, chunjae_math, label_math_ele_12, data

    # 이전에 로드된 user_id와 같다면 데이터를 다시 로드하지 않고 캐싱된 데이터 사용
    if user_id_input.user_id == user_id and data is not None:
        logging.info(f"기존 user_id {user_id}에 대한 데이터가 이미 캐싱되어 있습니다.")
    else:
        # 새로운 user_id가 입력되었으므로 데이터 초기화 및 로드
        user_id = user_id_input.user_id
        data = None
        result_df_all = None
        chunjae_math = None
        label_math_ele_12 = None

        logging.info(f"UserID {user_id}가 입력되었습니다.")
        # 데이터 로드 및 전처리 실행
        try:
            data = load_data_from_rds(user_id)
            result_df_all = data.get("result_df_all", pd.DataFrame())
            chunjae_math = data.get("chunjae_math")
            label_math_ele_12 = data.get("label_math_ele_12")
        except Exception as e:
            logging.error(f"데이터 로드 중 오류 발생: {str(e)}")
            raise HTTPException(status_code=500, detail="Database Load Error")


    # 정적 파일 저장 경로 설정
    base_path = f"./static/{user_id}/"
    os.makedirs(base_path, exist_ok=True)

    # 기존 학습자 처리
    if not result_df_all.empty and 'UserID' in result_df_all.columns and user_id in result_df_all["UserID"].values:
        logging.info(f"UserID {user_id}는 기존 학습자입니다.")
        
        # 기존 학습자 처리 함수 호출하여 HTML 문자열 및 area_avg_scores 반환받기
        full_graph_html, detailed_graph_html, area_report_existing, area_avg_scores = process_existing_learner(user_id, data, chunjae_math, label_math_ele_12, base_path=base_path)

        # Radar Chart 경로 설정 및 생성
        plot_radar_chart(area_avg_scores, user_id, title=f"{user_id} Radar Chart")

        # HTML 파일 저장
        with open(f"{base_path}{user_id}_gkt_model_existing.html", "w", encoding="utf-8") as file:
            file.write(full_graph_html)
            logging.info(f"Full graph HTML saved at {base_path}{user_id}_gkt_model_existing.html")

        with open(f"{base_path}{user_id}_gkt_model_existing_detailed.html", "w", encoding="utf-8") as file:
            file.write(detailed_graph_html)
            logging.info(f"Detailed graph HTML saved at {base_path}{user_id}_gkt_model_existing_detailed.html")


        # HTML iframe 및 이미지로 파일들을 표시
        combined_content = f"""
        <h2 style="color: purple">지금까지 나의 학습 상태를 확인해볼까요?</h2>
        <h3 style="text-align: center;">개념별 내 이해도는 어떨까요?</h3>
        <iframe src="/static/{user_id}/{user_id}_gkt_model_existing.html" width="100%" height="400"></iframe>
        <iframe src="/static/{user_id}/{user_id}_gkt_model_existing_detailed.html" width="100%" height="400"></iframe>
        <h3 style="text-align: center;">영역별 나의 학습 상태는?</h3>
        <img src="/static/{user_id}/{user_id}_radar_chart.png" alt="Radar Chart" width="100%">
        <h3 style="text-align: center;">영역별 나의 학습 상태에 대한 평가</h3>
        <p>{area_report_existing}</p>
        """
        return HTMLResponse(content=combined_content)

    else:
        # 신규 학습자 처리
        logging.info(f"UserID {user_id}는 신규 학습자입니다.")
        s3_client = boto3.client("s3")
        bucket_name = "dev-team-haejo-backup"
        base_path = "mean_data/"

        # GKT 모델 및 상세 모델 파일 다운로드
        s3_client.download_file(bucket_name, f"{base_path}exsiting_gkt_model_mean.html", f"/tmp/{user_id}_gkt_model_mean.html")
        s3_client.download_file(bucket_name, f"{base_path}exsiting_gkt_model_mean_new_detailed.html", f"/tmp/{user_id}_gkt_model_mean_new_detailed.html")
        s3_client.download_file(bucket_name, f"{base_path}기존_학습자의_전체_영역_평균_보고서_Radar_Chart_20241105_082705.png", f"/tmp/{user_id}_기존_학습자의_전체_영역_평균_보고서.png")

        # 파일을 현재 경로에 user_id 명칭으로 복사
        dest_path = f"./static/{user_id}/"
        os.makedirs(dest_path, exist_ok=True)
        shutil.copy(f"/tmp/{user_id}_gkt_model_mean.html", f"{dest_path}{user_id}_gkt_model_mean.html")
        shutil.copy(f"/tmp/{user_id}_gkt_model_mean_new_detailed.html", f"{dest_path}{user_id}_gkt_model_mean_new_detailed.html")
        shutil.copy(f"/tmp/{user_id}_기존_학습자의_전체_영역_평균_보고서.png", f"{dest_path}{user_id}_radar_chart.png")

        # LLM 보고서 생성
        student_data = {"이름": user_id}
        system_mean_prompt = get_mean_area_prompt(prohibited_words)
        area_report_new = generate_area_evaluation_report(
            student=student_data,
            summary_df=load_summary_mean_df(engine),  # summary_mean_df 로드
            system_prompt=system_mean_prompt
        )

        # 신규 학습자 HTML 컨텐츠 생성
        combined_content = f"""
        <h2>다른 친구들은 어떻게 학습하고 있을까요?</h2>
        <h3 style="text-align: center;">다른 친구들의 개념별 이해도는 어떨까요?</h3>
        <iframe src="/static/{user_id}/{user_id}_gkt_model_mean.html" width="100%" height="400"></iframe>
        <iframe src="/static/{user_id}/{user_id}_gkt_model_mean_new_detailed.html" width="100%" height="400"></iframe>
        <h3 style="text-align: center;">다른 친구들의 영역별 학습 상태는?</h3>
        <img src="/static/{user_id}/{user_id}_radar_chart.png" alt="Radar Chart" width="100%">
        <h3 style="text-align: center;">다른 친구들의 영역별 학습에 대한 평가</h3>
        <p>{area_report_new}</p>
        """
        return HTMLResponse(content=combined_content)


        

# 데이터 로드 및 전처리 함수
def load_data_from_rds(user_id=None):

    queries = {
        "label_math_ele_12": "SELECT * FROM math_label",
        "chunjae_math": "SELECT * FROM chunjae_math",
        "final_questions": "SELECT * FROM final_questions"
    }
    data = {}
    
    try:
        # 주요 테이블 로드
        for key, query in queries.items():
            logging.info(f"Loading data for: {key}")
            data[key] = pd.read_sql(query, engine)
            logging.info(f"Data loaded for {key}: {len(data[key])} rows")  # 로드된 행 개수 기록
        
        # final_questions 테이블 전처리
        final_questions = data['final_questions']
        if 'Unnamed: 0' in final_questions.columns:
            final_questions = final_questions.drop(['Unnamed: 0'], axis=1)
        final_questions = final_questions.rename(columns={
            '학년': 'grade',
            '학기': 'semester',
            '단원 순서': 'o_chapter',
            '대단원': 'f_lchapter_nm',
            '중단원': 'f_mchapter_nm'
        })
        data['final_questions'] = final_questions
        logging.info("final_questions data preprocessing complete")
        
        # 특정 UserID에 따른 데이터 필터링 및 병합
        data.update(load_user_specific_data(engine, user_id, data["chunjae_math"]))
        logging.info("Data loading and preprocessing complete.")
    except Exception as e:
        logging.error(f"DB 로드 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="Database Load Error")

    logging.info("All data successfully loaded from RDS")
    
    return data


def load_user_specific_data(engine, user_id, chunjae_math):

    result_df_all = pd.DataFrame()  # 기본값으로 빈 데이터프레임을 설정

    if user_id:
        # 먼저 UserID가 존재하는지 확인
        query_user_exists = f"SELECT UserID FROM student_state WHERE UserID = '{user_id}' LIMIT 1"
        user_exists = pd.read_sql(query_user_exists, engine)

        if not user_exists.empty:
            # 기존 학습자 데이터만 불러오기
            query_result = f"SELECT * FROM student_state WHERE UserID = '{user_id}'"
            result_df_all = pd.read_sql(query_result, engine)
            result_df_all = result_df_all.rename(columns={'학습 상태': 'Learning_state'})

            # chunjae_math에서 필요한 칼럼과 병합
            columns_to_merge = ['knowledgeTag', 'grade', 'semester', 'f_lchapter_nm', 'f_mchapter_nm', 'area2022']
            result_df_all = result_df_all.merge(chunjae_math[columns_to_merge], on='knowledgeTag', how='left')

            # 칼럼 재정렬
            column_order = ['UserID', 'grade', 'semester', 'area2022', 'f_lchapter_nm', 'f_mchapter_nm',
                            'knowledgeTag', 'Prediction_Probability', 'Predicted', 'Correct', 'Learning_state']
            result_df_all = result_df_all[column_order]
        else:
            # 기존 학습자가 아닌 경우 기본 구조의 빈 데이터프레임 반환
            print(f"UserID '{user_id}'가 존재하지 않습니다.")
            result_df_all = pd.DataFrame(columns=['UserID', 'grade', 'semester', 'area2022', 'f_lchapter_nm',
                                                  'f_mchapter_nm', 'knowledgeTag', 'Prediction_Probability',
                                                  'Predicted', 'Correct', 'Learning_state'])

    # 결과 반환
    return {"result_df_all": result_df_all}



    
    # chunjae_math와 병합
    columns_to_merge = ['knowledgeTag', 'grade', 'semester', 'f_lchapter_nm', 'f_mchapter_nm', 'area2022']
    result_df_all = result_df_all.merge(chunjae_math[columns_to_merge], on='knowledgeTag', how='left')
    logging.info("result_df_all successfully merged with chunjae_math")
    logging.info(f"result_df_all 칼럼 목록: {result_df_all.columns}")

    # 비교 결과에 따른 Learning_state 설정
    comparison_results = []
    for actual, predicted in zip(result_df_all['Correct'], result_df_all['Predicted']):
        if predicted == 1 and actual == 1:
            comparison_results.append("알고 있다")
        elif predicted == 1 and actual == 0:
            comparison_results.append("실수")
        elif predicted == 0 and actual == 1:
            comparison_results.append("찍음")
        else:
            comparison_results.append("모른다")

    result_df_all['Learning_state'] = comparison_results
    logging.info(f"'Learning_state' 칼럼이 생성되었습니다. 총 {len(result_df_all)}개의 행이 있습니다.")

    # 칼럼 재정렬 전에 'Learning_state' 컬럼이 존재하는지 확인
    expected_columns = ['UserID', 'grade', 'semester', 'area2022', 'f_lchapter_nm', 'f_mchapter_nm',
                        'knowledgeTag', 'Prediction_Probability', 'Predicted', 'Correct', 'Learning_state']

    # 'Learning_state'가 없으면 에러 메시지 로깅
    if 'Learning_state' not in result_df_all.columns:
        logging.error("result_df_all 데이터프레임에 'Learning_state' 컬럼이 없습니다.")
        raise HTTPException(status_code=500, detail="Required column 'Learning_state' is missing in result_df_all.")

    # 칼럼 재정렬
    result_df_all = result_df_all[[col for col in expected_columns if col in result_df_all.columns]]

    # 칼럼 재정렬
    column_order = ['UserID', 'grade', 'semester', 'area2022', 'f_lchapter_nm', 'f_mchapter_nm',
                    'knowledgeTag', 'Prediction_Probability', 'Predicted', 'Correct', 'Learning_state']
    result_df_all = result_df_all[column_order]
    
    logging.info(f"Filtered and merged data for UserID: {user_id}")
    
    return {
        "result_df_all": result_df_all
    }

def load_summary_mean_df(engine):

    summary_mean_df = pd.read_sql("SELECT * FROM mean_summary", engine)
    return summary_mean_df




# GKT 모델 및 그래프 함수들


# 상태에 따른 가중치 설정
weights = {'알고 있다': 1.0, '실수': 0.3, '찍음': 0.5, '모른다': 0.0}

class EnhancedGKTModel:
    def __init__(self, relationships, all_middlesections):
        logging.info("EnhancedGKTModel을 초기화합니다.")
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(relationships)
        self.graph.add_nodes_from(all_middlesections)
        self.knowledge_state = {node: None for node in self.graph.nodes}
        self.weighted_scores = {}

    def update_knowledge(self, knowledge_element, Learning_state_counts):
        if knowledge_element not in self.graph.nodes:
            logging.warning(f"'{knowledge_element}' 개념이 그래프에 없습니다. 건너뜁니다.")
            return

        Learning_state_counts = {state: Learning_state_counts.get(state, 0) for state in weights.keys()}
        total_counts = sum(Learning_state_counts.values())
        weighted_score = (
            sum(Learning_state_counts[state] * weights[state] for state in weights.keys()) / total_counts
            if total_counts > 0 else 0
        )

        self.weighted_scores[knowledge_element] = weighted_score
        self.knowledge_state[knowledge_element] = 'green' if weighted_score >= 0.8 else 'red'

# 나머지 2번 코드 로직들 계속 포함...
def initialize_existing_model(chunjae_math, label_math_ele_12):
    logging.info("천재교육_계열화 데이터와 개념 관계 데이터를 매핑하여 GKT 모델을 초기화합니다.")
    kt_to_middlesection = chunjae_math.set_index('knowledgeTag')['f_mchapter_nm'].to_dict()
    label_math_ele_12['from_middlesection'] = label_math_ele_12['from_id'].map(kt_to_middlesection)
    label_math_ele_12['to_middlesection'] = label_math_ele_12['to_id'].map(kt_to_middlesection)
    label_math_ele_12 = label_math_ele_12.dropna(subset=['from_middlesection', 'to_middlesection'])

    relationships = list(set(zip(label_math_ele_12['to_middlesection'], label_math_ele_12['from_middlesection'])))
    all_middlesections = chunjae_math['f_mchapter_nm'].unique()
    model = EnhancedGKTModel(relationships, all_middlesections)
    
    return model

def update_model_with_learner_data(model, learner_data, chunjae_math):
    kt_to_middlesection = chunjae_math.set_index('knowledgeTag')['f_mchapter_nm'].to_dict()
    knowledge_tag_status = learner_data.groupby('knowledgeTag')['Learning_state'].value_counts().unstack().fillna(0)

    for knowledge_tag, counts in knowledge_tag_status.iterrows():
        middlesection = kt_to_middlesection.get(knowledge_tag)
        if middlesection is None:
            logging.warning(f"KnowledgeTag '{knowledge_tag}'를 매핑할 수 없습니다. 건너뜁니다.")
            continue
        model.update_knowledge(middlesection, counts.to_dict())
        
def visualize_existing_gkt_model(model, chunjae_math, learner_id, title='GKT Model 3D Visualization'):
    logging.info(f"학습자 '{learner_id}'의 GKT 모델을 시각화합니다.")
    f_mchapter_nm_to_knowledgeTags = chunjae_math.groupby('f_mchapter_nm')['knowledgeTag'].apply(list).to_dict()
    pos = nx.spring_layout(model.graph, dim=3, k=0.7, seed=42)

       
    # 노드 상태별로 분류
    node_groups = {}
    for node in model.graph.nodes():
        state = model.knowledge_state.get(node)
        if state not in node_groups:
            node_groups[state] = []
        node_groups[state].append(node)
    
    # 엣지 상태별로 분류 (노드 그룹과 연동)
    edge_traces = []
    state_color_mapping = {'green': 'green', 'red': 'red', None: 'gray'}
    state_name_mapping = {'green': '후속 학습 필요', 'red': '선수 학습 필요', None: '학습하지 않음'}

    for state, nodes in node_groups.items():
        # 해당 상태의 노드와 연결된 엣지를 수집
        edges = []
        for edge in model.graph.edges():
            if edge[0] in nodes or edge[1] in nodes:
                x0, y0, z0 = pos[edge[0]]
                x1, y1, z1 = pos[edge[1]]
                edges.append(dict(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    z=[z0, z1, None]
                ))
        # 엣지 트레이스 생성
        edge_trace = go.Scatter3d(
            x=sum([e['x'] for e in edges], []),
            y=sum([e['y'] for e in edges], []),
            z=sum([e['z'] for e in edges], []),
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines',
            showlegend=False,  # 범례에 엣지 중복 방지
            legendgroup=state_name_mapping[state]  # 노드 그룹과 동일한 legendgroup으로 설정
        )
        edge_traces.append(edge_trace)

    # 노드 트레이스 생성
    node_traces = []
    for state, nodes in node_groups.items():
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        for node in nodes:
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            weighted_score = model.weighted_scores.get(node, 0)
            # 연결된 선수 및 후속 개념 가져오기
            predecessors = list(model.graph.predecessors(node))
            successors = list(model.graph.successors(node))
            knowledge_tags = f_mchapter_nm_to_knowledgeTags.get(node, [])
            knowledge_tags_str = ', '.join(map(str, knowledge_tags))
            node_text.append(
                f"개념: {node} (KnowledgeTags: {knowledge_tags_str})<br>"
                f"지식 상태: {state_name_mapping[state]}<br>"
                f"Weighted Score: {weighted_score:.2f}<br>"
                f"연결된 선수 학습 개념: {', '.join(predecessors) if predecessors else '없음'}<br>"
                f"연결된 후속 학습 개념: {', '.join(successors) if successors else '없음'}"
            )
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=4,
                color=state_color_mapping[state],
                opacity=0.8
            ),
            text=nodes,
            textposition='top center',
            hoverinfo='text',
            hovertext=node_text,
            name=state_name_mapping[state],
            legendgroup=state_name_mapping[state],  # 엣지와 동일한 legendgroup으로 설정
            showlegend=True  # 범례에 표시
        )
        node_traces.append(node_trace)

    # Figure 생성
    fig = go.Figure(data=edge_traces + node_traces)

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        showlegend=True
    )

    return fig


# 기존 학습자 보고서 생성
def generate_existing_report(model):
    logging.info("기존 전체 보고서를 생성합니다.")
    
    # 지식 상태 분류
    known_areas = [node for node, state in model.knowledge_state.items() if state == 'green']
    deficient_areas = [node for node, state in model.knowledge_state.items() if state == 'red']
    unlearned_areas = [node for node, state in model.knowledge_state.items() if state is None]

    # 상태 데이터프레임 생성
    knowledge_state_df = pd.DataFrame.from_dict(model.knowledge_state, orient='index', columns=['Node Color'])
    weighted_scores_df = pd.DataFrame.from_dict(model.weighted_scores, orient='index', columns=['Weighted Score'])
    weighted_scores_df['Node Color'] = weighted_scores_df.index.map(model.knowledge_state)
    weighted_scores_df.reset_index(inplace=True)
    weighted_scores_df.rename(columns={'index': 'Node'}, inplace=True)
    
    return weighted_scores_df

            
def generate_existing_detailed_model_and_visualization(model, learner_id, chunjae_math, title='GKT Model 3D Visualization (세분화된 그래프)'):
    # red 노드만 추출하여 서브그래프 생성
    logging.info("세분화된 그래프를 위한 모델을 생성합니다.")
    red_nodes = [node for node, state in model.knowledge_state.items() if state == 'red']
    subgraph = model.graph.subgraph(red_nodes).copy()

    # 서브그래프를 기반으로 세부 모델 인스턴스 생성
    detailed_model = EnhancedGKTModelDetailed(subgraph.edges(), subgraph.nodes())
    
    # 지식 상태 업데이트
    for node in subgraph.nodes():
        weighted_score = model.weighted_scores.get(node, 0)
        detailed_model.weighted_scores[node] = weighted_score
        # weighted_score에 따라 색상 분류
        if weighted_score >= 0.7:
            detailed_model.knowledge_state[node] = 'yellow'  # 보통 이해도
        elif weighted_score >= 0.5:
            detailed_model.knowledge_state[node] = 'orange'  # 낮은 이해도
        else:
            detailed_model.knowledge_state[node] = 'red'     # 매우 낮은 이해도

    # 세분화된 그래프 시각화
    fig_detailed = visualize_gkt_model_3d_detailed(detailed_model, chunjae_math, learner_id, title)
    return fig_detailed


def visualize_gkt_model_3d_detailed(model, chunjae_math, learner_id, title='GKT Model 3D Visualization (선수학습이 필요한 노드)'):
    logging.info(f"학습자 '{learner_id}'의 상세 GKT 모델을 시각화합니다.")
    f_mchapter_nm_to_knowledgeTags = chunjae_math.groupby('f_mchapter_nm')['knowledgeTag'].apply(list).to_dict()
    pos = nx.spring_layout(model.graph, dim=3, k=0.7, seed=42)
    
    # 노드 상태별로 분류
    node_groups = {}
    for node in model.graph.nodes():
        state = model.knowledge_state.get(node)
        if state not in node_groups:
            node_groups[state] = []
        node_groups[state].append(node)

    # 엣지 상태별로 분류 (노드 그룹과 연동)
    edge_traces = []
    for state, nodes in node_groups.items():
        # 해당 상태의 노드와 연결된 엣지를 수집
        edges = []
        for edge in model.graph.edges():
            if edge[0] in nodes or edge[1] in nodes:
                x0, y0, z0 = pos[edge[0]]
                x1, y1, z1 = pos[edge[1]]
                edges.append(dict(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    z=[z0, z1, None]
                ))
        # 엣지 트레이스 생성
        edge_trace = go.Scatter3d(
            x=sum([e['x'] for e in edges], []),
            y=sum([e['y'] for e in edges], []),
            z=sum([e['z'] for e in edges], []),
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines',
            showlegend=False,
            legendgroup=state
        )
        edge_traces.append(edge_trace)

    # 노드 트레이스 생성
    node_traces = []
    state_color_mapping = {'yellow': 'yellow', 'orange': 'orange', 'red': 'red'}
    state_name_mapping = {'yellow': '보통 이해도', 'orange': '낮은 이해도', 'red': '매우 낮은 이해도'}
    for state, nodes in node_groups.items():
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        for node in nodes:
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            weighted_score = model.weighted_scores.get(node, 0)
             # 연결된 선수 및 후속 개념 가져오기
            predecessors = list(model.graph.predecessors(node))
            successors = list(model.graph.successors(node))
            knowledge_tags = f_mchapter_nm_to_knowledgeTags.get(node, [])
            knowledge_tags_str = ', '.join(map(str, knowledge_tags))
            node_text.append(
                f"개념: {node} (KnowledgeTags: {knowledge_tags_str})<br>"
                f"지식 상태: {state_name_mapping[state]}<br>"
                f"Weighted Score: {weighted_score:.2f}<br>"
                f"연결된 선수 학습 개념: {', '.join(predecessors) if predecessors else '없음'}<br>"
                f"연결된 후속 학습 개념: {', '.join(successors) if successors else '없음'}"
            )
        node_trace = go.Scatter3d(


            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=4,
                color=state_color_mapping[state],
                opacity=0.8
            ),
            text=nodes,
            textposition='top center',
            hoverinfo='text',
            hovertext=node_text,
            name=state_name_mapping[state],
            legendgroup=state,
            showlegend=True
        )
        node_traces.append(node_trace)

    # Figure 생성
    fig = go.Figure(data=edge_traces + node_traces)

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        showlegend=True
    )
    
    return fig






# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. f_mchapter_nm 별 knowledgeTags 매핑 딕셔너리 생성 함수
def create_knowledge_tag_mapping(df):
    return df.groupby('f_mchapter_nm')['knowledgeTag'].apply(list).to_dict()

# 2. Node에 해당하는 knowledgeTags를 가져오는 함수
def get_knowledge_tags(node, tag_mapping):
    knowledge_tags = tag_mapping.get(node, [])
    return ', '.join(map(str, knowledge_tags))

# 3. knowledgeTag의 첫 숫자를 기반으로 area 카테고리 지정 함수
def categorize_area(knowledge_tags):
    if not knowledge_tags:
        return '기타'
    first_digit = str(knowledge_tags[0])[0]
    area_mapping = {
        '1': '수와 연산',
        '2': '변화와 관계',
        '3': '도형과 측정(측정)',
        '4': '도형과 측정(도형)',
        '5': '자료와 가능성'
    }
    return area_mapping.get(first_digit, '기타')

# 4. area별 평균 점수 계산 함수
def calculate_area_avg_scores(df):
    return df.groupby('area')['Weighted Score'].mean().reindex(
        ['수와 연산', '변화와 관계', '도형과 측정(측정)', '도형과 측정(도형)', '자료와 가능성'], fill_value=0)

# 5. area별 최고, 최저 점수 Node 찾기 함수
def find_extreme_nodes(df):
    # NaN 값 제거 또는 0으로 대체
#     df = df.dropna(subset=['Weighted Score'])  # NaN이 있는 행 제거
    df['Weighted Score'] = df['Weighted Score'].fillna(0)  # 또는 NaN을 0으로 채우기

    # 각 영역(area)에서 'Weighted Score'가 가장 높은 Node 선택
    highest_nodes = df.loc[df.groupby('area')['Weighted Score'].idxmax()][['area', 'Node', 'Weighted Score']]
    # 각 영역(area)에서 'Weighted Score'가 가장 낮은 Node 선택
    lowest_nodes = df.loc[df.groupby('area')['Weighted Score'].idxmin()][['area', 'Node', 'Weighted Score']]
    
    return highest_nodes, lowest_nodes

# 6. 영역에 대한 점수가 없는 경우 메시지 출력 함수
def check_missing_scores(area_avg_scores):
    for area in area_avg_scores.index:
        if area_avg_scores[area] == 0:
            print(f"{area} 영역에 대한 문제가 없습니다. 해당 영역에 대한 문제를 풀어야 합니다.")

# 7. 오각형 그래프 생성 함수
def plot_radar_chart(area_avg_scores, user_id, title="Radar Chart"):
    # 한글 폰트를 적용하여 그래프 설정
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    labels = area_avg_scores.index
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # 데이터와 각도를 원형으로 닫기
    scores = area_avg_scores.values
    scores = np.concatenate((scores, [scores[0]]))
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, color='blue', alpha=0.25)
    ax.plot(angles, scores, color='blue', linewidth=2)
    
    # 각 축에 레이블 추가
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    # y축 레이블 표시
    num_ticks = 10
    ticks = np.linspace(0, 1, num_ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{tick:.1f}" for tick in ticks], color="gray", fontsize=8)
    
    # 점수 텍스트 추가
    for angle, score in zip(angles, scores):
        ax.text(angle, score, f'{score:.2f}', ha='center', va='center', fontsize=10, color='black')

    plt.title(title)

    # `user_id` 기반으로 파일명 생성 및 저장
    radar_chart_filename = f"./static/{user_id}/{user_id}_radar_chart.png"
    plt.savefig(radar_chart_filename, dpi=300)
    logging.info(f"Radar chart saved as {radar_chart_filename}")
    plt.close(fig)

# 실행 함수: 전체 분석 수행
def perform_analysis(df, chunjae_math, base_path, user_id, report_type="Report"):
    # Knowledge Tag 매핑 생성 및 추가
    tag_mapping = create_knowledge_tag_mapping(chunjae_math)
    df['knowledgeTag'] = df['Node'].apply(get_knowledge_tags, args=(tag_mapping,))
    
    # Area 열 추가
    df['area'] = df['knowledgeTag'].apply(lambda tags: categorize_area([int(tag) for tag in tags.split(', ') if tag]))

    # Area별 평균 점수 계산
    area_avg_scores = calculate_area_avg_scores(df)

    # 최고/최저 점수 Node 찾기
    highest_nodes, lowest_nodes = find_extreme_nodes(df)

    # 점수 없는 영역 체크 및 알림 메시지 출력
    check_missing_scores(area_avg_scores)


    # Radar Chart 저장
    radar_chart_path = f"{base_path}{user_id}_radar_chart.png"
    plot_radar_chart(area_avg_scores, user_id, title=f"{user_id} Radar Chart")
    logging.info(f"Radar chart saved at {radar_chart_path}")  # 파일 저장 확인 로그

    
    # area_avg_scores 반환
    return area_avg_scores

# 영역별 평균 점수, 최고 점수, 최저 점수를 데이터프레임으로 정리하는 함수
def create_summary_df(all_area_score):
    # 각 영역별 평균 점수 계산
    area_avg_scores = all_area_score.groupby('area')['Weighted Score'].mean().reindex(
        ['수와 연산', '변화와 관계', '도형과 측정(측정)', '도형과 측정(도형)', '자료와 가능성'], fill_value=np.nan
    )

    # 각 영역별 최고 및 최저 점수 Node 찾기 (데이터가 없으면 NaN)
    highest_nodes = all_area_score.loc[all_area_score.groupby('area')['Weighted Score'].idxmax(), ['area', 'Node', 'Weighted Score']].set_index('area').reindex(area_avg_scores.index)
    lowest_nodes = all_area_score.loc[all_area_score.groupby('area')['Weighted Score'].idxmin(), ['area', 'Node', 'Weighted Score']].set_index('area').reindex(area_avg_scores.index)

    # 영역별 요약 정보 데이터프레임 생성
    summary_df = pd.DataFrame({
        'Average_Score': area_avg_scores,
        'Highest_Node': highest_nodes['Node'],
        'Highest_Score': highest_nodes['Weighted Score'],
        'Lowest_Node': lowest_nodes['Node'],
        'Lowest_Score': lowest_nodes['Weighted Score']
    })
    
    # 평균 점수 또는 최고/최저 점수가 없는 영역에 대해 메시지 출력
    for area, row in summary_df.iterrows():
        if pd.isna(row['Average_Score']):
            print(f"{area} 영역에 대한 문제풀이 데이터가 없습니다.")
    
    return summary_df

    # existing_learner.py


def process_existing_learner(user_id, user_specific_data, chunjae_math, label_math_ele_12, base_path):
    print(f"기존 학습자 '{user_id}'의 데이터를 처리합니다.")

    # 기존 학습자 모델 생성
    gkt_model = initialize_existing_model(chunjae_math, label_math_ele_12)
    learner_data = user_specific_data["result_df_all"]

    for knowledge_tag, counts in learner_data.groupby('knowledgeTag')['Learning_state'].value_counts().unstack().fillna(0).iterrows():
        gkt_model.update_knowledge(
            chunjae_math.set_index('knowledgeTag')['f_mchapter_nm'].to_dict().get(knowledge_tag),
            counts.to_dict()
        )

    # 모델 시각화 - 전체 그래프
    fig = visualize_existing_gkt_model(gkt_model, chunjae_math, user_id, title=f"{user_id} (기존 학습자 전체 그래프)")
    full_graph_html = fig.to_html(full_html=False)
    print(f"{user_id}_gkt_model_existing 그래프 HTML로 생성되었습니다.")

    # 세분화된 모델 생성 및 시각화
    fig_detailed = generate_existing_detailed_model_and_visualization(
        gkt_model, user_id, chunjae_math, title=f"{user_id} (기존 학습자 세분화 그래프)"
    )
    detailed_graph_html = fig_detailed.to_html(full_html=False)
    print(f"{user_id}_gkt_model_existing_detailed 세분화 그래프 HTML로 생성되었습니다.")

    # 학습자 보고서 생성 및 출력
    report_df = generate_existing_report(gkt_model)
    area_avg_scores = perform_analysis(report_df, chunjae_math, base_path, user_id, report_type=f"기존 학습자 영역 보고서_{user_id}")  # area_avg_scores 받아오기
    summary_df = create_summary_df(report_df)
    print(summary_df)
    # 학습자 보고서 생성 및 출력 (LLM 사용)
    student_data = {"이름": user_id}
    system_existing_prompt = get_area_prompt(prohibited_words)
    area_report_existing = generate_area_evaluation_report(
        student=student_data,
        summary_df=summary_df,
        system_prompt=system_existing_prompt
    )
    print(area_report_existing)

    # 생성된 HTML 문자열들을 반환
    return full_graph_html, detailed_graph_html, area_report_existing, area_avg_scores



# llm.py

__all__ = ["filter_bad_words", "generate_area_evaluation_report", "get_area_prompt", "get_test_prompt", "generate_test_evaluation_report", "prohibited_words"]

# -------------------- 로깅 설정 --------------------
# 중복된 핸들러 제거
logger = logging.getLogger('FormativeEvaluationLogger')
if logger.hasHandlers():
    logger.handlers.clear()

# S3 버킷 및 파일 키 지정
bucket_name = "dev-team-haejo-backup"  # S3 버킷 이름
env_file_key = "env/env_file.txt"    # 환경 변수 파일 경로
badwords_file_key = "LLM/badwords.json"  # 금지어 파일 경로 업데이트
# -------------------- S3에서 환경 변수 파일 및 금지어 파일 불러오기 --------------------
prohibited_words = []  # 전역 변수 초기화

def load_env_from_s3_llm(bucket_name, env_file_key, badwords_file_key):
    s3 = boto3.client('s3')
    logger.info(f"S3에서 .env 파일 다운로드 중: 버킷 - {bucket_name}, 키 - {env_file_key}")
    
    # .env 파일 다운로드 및 환경 변수 로드
    s3.download_file(bucket_name, env_file_key, '/tmp/env_file.txt')
    load_dotenv('/tmp/env_file.txt')
    
    # badwords.json 파일 다운로드
    try:
        logger.info(f"S3에서 금지어 파일 다운로드 중: 버킷 - {bucket_name}, 키 - {badwords_file_key}")
        s3.download_file(bucket_name, badwords_file_key, '/tmp/badwords.json')
        with open('/tmp/badwords.json', 'r', encoding='utf-8') as f:
            badwords_data = json.load(f)
        global badwords
        badwords = badwords_data.get('badwords', [])
        logger.info(f"금지 단어 목록이 S3에서 성공적으로 로드되었습니다. 총 {len(badwords)}개의 단어가 로드되었습니다.")
    except Exception as e:
        logger.error(f"S3에서 badwords.json 파일을 로드하는 데 실패했습니다: {str(e)}")
        badwords = []

# 환경 변수 및 금지어 목록 로드
load_env_from_s3_llm(bucket_name, env_file_key, badwords_file_key)

# OpenAI API 키 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    logger.info("OpenAI API 키가 설정되었습니다.")
else:
    logger.error("OpenAI API 키가 설정되지 않았습니다.")

# -------------------- 정규식 패턴 설정 --------------------
try:
    badwords_regex = re.compile(
        r'[시씨씪슈쓔쉬쉽쒸쓉]([0-9]*|[0-9]+ *)[바발벌빠빡빨뻘파팔펄]|'
        r'[섊좆좇졷좄좃좉졽썅춍봊]|'
        r'[ㅈ조][0-9]*까|ㅅㅣㅂㅏㄹ?|ㅂ[0-9]*ㅅ|'
        r'[ㅄᄲᇪᄺᄡᄣᄦᇠ]|[ㅅㅆᄴ][0-9]*[ㄲㅅㅆᄴㅂ]|'
        r'[존좉좇][0-9 ]*나|[자보][0-9]+지|보빨|'
        r'[봊봋봇봈볻봁봍] *[빨이]|'
        r'[후훚훐훛훋훗훘훟훝훑][장앙]|'
        r'[엠앰]창|애[미비]|애자|[가-탏탑-힣]색기|'
        r'([샊샛세쉐쉑쉨쉒객갞갟갯갰갴겍겎겏겤곅곆곇곗곘곜걕걖걗걧걨걬] *[끼키퀴])|'
        r'새 *[키퀴]|'
        r'[병븅][0-9]*[신딱딲]|미친[가-닣닥-힣]|[믿밑]힌|'
        r'[염옘][0-9]*병|'
        r'[샊샛샜샠섹섺셋셌셐셱솃솄솈섁섂섓섔섘]기|'
        r'[섹섺섻쎅쎆쎇쎽쎾쎿섁섂섃썍썎썏][스쓰]|'
        r'[지야][0-9]*랄|니[애에]미|갈[0-9]*보[^가-힣]|'
        r'[뻐뻑뻒뻙뻨][0-9]*[뀨큐킹낑)|꼬[0-9]*추|'
        r'곧[0-9]*휴|[가-힣]슬아치|자[0-9]*박꼼|빨통|'
        r'[사싸](이코|가지|[0-9]*까시)|육[0-9]*시[랄럴]|'
        r'육[0-9]*실[알얼할헐]|즐[^가-힣]|찌[0-9]*(질이|랭이)|'
        r'찐[0-9]*따|찐[0-9]*찌버거|창[녀놈]|[가-힣]{2,}충[^가-힣]|'
        r'[가-힣]{2,}츙|부녀자|화냥년|환[양향]년|호[0-9]*[구모]|'
        r'조[선센][징]|조센|[쪼쪽쪾]([발빨]이|[바빠]리)|盧|무현|'
        r'찌끄[레래]기|(하악){2,}|하[앍앜]|[낭당랑앙항남담람암함][ ]?[가-힣]+[띠찌]|'
        r'느[금급]마|文在|在寅|(?<=[^\n])[家哥]|속냐|[tT]l[qQ]kf|Wls|[ㅂ]신|'
        r'[ㅅ]발|[ㅈ]밥'
    )
    logger.info("금지 단어 정규식이 성공적으로 컴파일되었습니다.")
except re.error as e:
    logger.error(f"정규식 컴파일 오류: {str(e)}")
    badwords_regex = None

# -------------------- 금지어 필터링 함수 --------------------

def filter_bad_words(text):
    """
    주어진 텍스트에서 금지 단어를 감지하고 대체합니다.

    Parameters:
    - text (str): 필터링할 텍스트

    Returns:
    - str: 금지 단어가 대체된 텍스트
    """
    if not text:
        logger.warning("빈 텍스트가 필터링되었습니다.")
        return text

    original_text = text
    try:
        # 정규식으로 금지 단어 감지 및 대체
        if badwords_regex:
            text = badwords_regex.sub('***', text)
            logger.debug("정규식 필터링이 적용되었습니다.")
        
        # 명시적 금지 단어 목록을 확인하여 대체
        for word in badwords:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            text, count = pattern.subn('***', text)
            if count > 0:
                logger.debug(f"명시적 필터링: '{word}'가 {count}번 대체되었습니다.")
        
        if original_text != text:
            logger.info("금지 단어가 필터링되었습니다.")
        return text
    except Exception as e:
        logger.error(f"금지 단어 필터링 중 오류 발생: {str(e)}")
        return original_text  # 필터링 실패 시 원본 텍스트 반환

# --------- 첨삭 보고서 생성 ----------

def get_area_prompt(prohibited_words):
    """한 학생의 영역별 분석 프롬프트를 반환합니다."""
    prohibited_words_list = ", ".join(prohibited_words)  # 금지어 목록 문자열 생성
    
    prompt = f"""
당신은 초등학생과 그 학부모를 위한 교육 평가 전문가입니다. 주어진 학생의 영역별 요약 데이터를 분석하여 다음 작업을 수행합니다:

1. **영역별 분석**:
    - `summary_df` 데이터를 기반으로 각 영역별 평균 점수, 가장 높은 노드 및 점수, 가장 낮은 노드 및 점수를 설명합니다.

**중요:** 다음 금지된 언어 목록에 있는 단어들은 절대 사용하지 마세요: {prohibited_words_list}

**추가 지침:**
- **모든 개념 코드에 대해 첨삭 내용을 제공하세요. 누락이 없도록 하세요.**
- 보고서의 마지막 문장은 항상 학습자를 칭찬하고 격려하는 말로 끝내세요.
- 비속어, 줄임말, 부적절한 언어를 사용하지 마세요.
- 언어를 초등학생과 학부모가 이해하기 쉽게 단순하고 명확하게 작성하세요.

**보고서 형식:**

---
학생 이름: {{이름}}

**1. 영역별 분석:**
- **수와 연산:**
    -   평균 점수:  {{Average_Score}}
    -   가장 잘한 부분: {{Highest_Node}} ({{Highest_Score}})
    -   가장 못한 부분: {{Lowest_Node}} ({{Lowest_Score}})

- **도형과 측정(도형):**
    -   평균 점수: {{Average_Score}}
    -   가장 잘한 부분:  {{Highest_Node}} ({{Highest_Score}})
    -   가장 못한 부분:  {{Lowest_Node}} ({{Lowest_Score}})

- **도형과 측정(측정):**
    -   평균 점수: {{Average_Score}}
    -   가장 잘한 부분:  {{Highest_Node}} ({{Highest_Score}})
    -   가장 못한 부분:  {{Lowest_Node}} ({{Lowest_Score}})

- **변화와 관계:**
    -   평균 점수: {{Average_Score}}
    -   가장 잘한 부분:  {{Highest_Node}} ({{Highest_Score}})
    -   가장 못한 부분:  {{Lowest_Node}} ({{Lowest_Score}})

- **자료와 가능성:**
    -   평균 점수: {{Average_Score}}
    -   가장 잘한 부분:  {{Highest_Node}} ({{Highest_Score}})
    -   가장 못한 부분:  {{Lowest_Node}} ({{Lowest_Score}})

**종합 정리:**
{{격려 메시지}}

---
"""

    return prompt

def get_mean_area_prompt(prohibited_words):
    """학생 전체의 영역별 분석 프롬프트를 반환합니다."""
    prohibited_words_list = ", ".join(prohibited_words)  # 금지어 목록 문자열 생성
    
    prompt = f"""
당신은 초등학생과 그 학부모를 위한 교육 평가 전문가입니다. 주어진 학생의 영역별 요약 데이터를 분석하여 다음 작업을 수행합니다:

1. **영역별 분석**:
    - `summary_df` 데이터를 기반으로 각 영역별 평균 점수, 가장 높은 노드 및 점수, 가장 낮은 노드 및 점수를 설명합니다.

**중요:** 다음 금지된 언어 목록에 있는 단어들은 절대 사용하지 마세요: {prohibited_words_list}

**추가 지침:**
- **모든 개념 코드에 대해 첨삭 내용을 제공하세요. 누락이 없도록 하세요.**
- 보고서의 마지막 문장은 항상 학습자를 칭찬하고 격려하는 말로 끝내세요.
- 비속어, 줄임말, 부적절한 언어를 사용하지 마세요.
- 언어를 초등학생과 학부모가 이해하기 쉽게 단순하고 명확하게 작성하세요.
- **종합 정리:**에는 적힌 말 이외 다른 말을 반환하지 마세요.

**보고서 형식:**

---
학생 이름: {{이름}}

**1. 영역별 분석:**
- **수와 연산:**
    -   평균 점수: {{Average_Score}}
    -   가장 잘한 부분:  {{Highest_Node}} ({{Highest_Score}})
    -   가장 못한 부분:  {{Lowest_Node}} ({{Lowest_Score}})

- **도형과 측정(도형):**
    -   평균 점수: {{Average_Score}}
    -   가장 잘한 부분:  {{Highest_Node}} ({{Highest_Score}})
    -   가장 못한 부분:  {{Lowest_Node}} ({{Lowest_Score}})

- **도형과 측정(측정):**
    -   평균 점수: {{Average_Score}}
    -   가장 잘한 부분:  {{Highest_Node}} ({{Highest_Score}})
    -   가장 못한 부분:  {{Lowest_Node}} ({{Lowest_Score}})

- **변화와 관계:**
    -   평균 점수: {{Average_Score}}
    -   가장 잘한 부분:  {{Highest_Node}} ({{Highest_Score}})
    -   가장 못한 부분:  {{Lowest_Node}} ({{Lowest_Score}})

- **자료와 가능성:**
    -   평균 점수: {{Average_Score}}
    -   가장 잘한 부분:  {{Highest_Node}} ({{Highest_Score}})
    -   가장 못한 부분:  {{Lowest_Node}} ({{Lowest_Score}})

**종합 정리:**
{{이 보고서는 기존 학습자들의 평균 점수입니다. 나만의 보고서를 제공받기 위해서는 문제를 풀어주세요. 그럼 함께 수학 공부하러 가볼까요 ?}}

---
"""

    return prompt


# -------------------- 보고서 생성 함수 --------------------
def generate_area_evaluation_report(student, summary_df, system_prompt):

    try:
        logger.info(f"보고서 생성을 시작합니다. 학생 이름: {student['이름']}")

        # 학생 이름 추출 및 필터링
        name = filter_bad_words(student['이름'])
        logger.debug(f"필터링된 학생 이름: {name}")

        
        # 영역별 데이터 처리
        summary_df = summary_df.reset_index().rename(columns={'index': '영역'})
        areas = summary_df.to_dict(orient='records')
        # 예시로 summary_df의 특정 컬럼에서 개념 코드 리스트를 생성
        concept_codes = summary_df['knowledgeTag'].unique().tolist() if 'knowledgeTag' in summary_df.columns else []

        # 모든 데이터를 딕셔너리에 저장
        data = {
            '이름': name,
            '개념코드목록': concept_codes,
            '영역목록': areas
        }

        # 데이터를 JSON 문자열로 변환
        data_json = filter_bad_words(json.dumps(data, ensure_ascii=False))

        logger.debug("데이터가 JSON 문자열로 변환되고 필터링되었습니다.")

        # 사용자 프롬프트 구성
        user_prompt = f"{data_json}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        logger.info("OpenAI API 호출을 시작합니다.")

        # API 호출
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            top_p=0.8,
            max_tokens=5000  # 필요에 따라 조정
        )

        # 응답 추출
        report = completion.choices[0].message.content.strip()
        logger.info("보고서가 성공적으로 생성되었습니다.")

        # 토큰 사용량 추출 및 로그 기록
        usage = completion.usage
        if usage:
            try:
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens
                logger.info(f"토큰 사용량 - 프롬프트: {prompt_tokens}, 컴플리션: {completion_tokens}, 총: {total_tokens}")
            except AttributeError as ae:
                logger.error(f"토큰 사용량 정보 접근 오류: {str(ae)}")
        else:
            logger.warning("API 응답에 토큰 사용량 정보가 포함되어 있지 않습니다.")

        return report

    except openai.OpenAIError as oe:
        logger.error(f"OpenAI API 오류 발생: {str(oe)}")
        return f"오류 발생: {str(oe)}"
    except Exception as e:
        logger.error(f"보고서 생성 중 일반 오류 발생: {str(e)}")
        return f"오류 발생: {str(e)}"

# ------ 형성평가 첨삭 ------
def get_test_prompt(prohibited_words, f_lchapter_nm, nodes_str, concept_codes_json, understanding_levels_str, learning_suggestions_str, related_concepts_str):
    """형성평가 분석 프롬프트를 반환합니다."""
    prohibited_words_list = ", ".join(prohibited_words)
    
    prompt = f"""
    당신은 초등학생과 그 학부모를 위한 교육 평가 전문가입니다. 주어진 학생의 문제풀이 요약 데이터와 모든 개념의 세부 점수 데이터를 분석하여 다음 작업을 수행합니다:
    
    1. **형성평가 첨삭 방식**:
        - 학생 이름: {{이름}} 아래에 형성평가 첨삭 방식과 관련된 설명을 적습니다.
        - 본 형성평가는 {f_lchapter_nm} 단원과 관련된 형성평가입니다.
        - 본 형성평가에 포함된 개념 목록은 다음과 같습니다 (Node와 knowledgeTag를 포함):
            - **개념 목록:** {nodes_str}

    2. **개념별 첨삭**:
        - 각 Node와 개념 코드(knowledgeTag)를 적고 그와 관련된 문제들을 제시합니다.
        - **이해도 수준**: 각 개념의 이해도 수준은 다음과 같습니다: {understanding_levels_str}.
        - **개선 사항**: 이해도 수준에 따라 학생의 학습을 향상시킬 수 있는 구체적인 제안을 제공합니다.
        - **학습 제안**: 각 개념별 학습 제안은 다음과 같습니다: {learning_suggestions_str}.
        - **관련 개념**: 각 개념별 관련 개념은 다음과 같습니다: {related_concepts_str}.
        
    3. **전체 첨삭**:
        - 형성평가 전체에 대한 종합적인 첨삭을 제공합니다.
        - 500~600자 정도로 작성합니다.

    **중요:** 다음 금지된 언어 목록에 있는 단어들은 절대 사용하지 마세요: {prohibited_words_list}
    
    **추가 지침:**
    - **모든 개념 코드에 대해 첨삭 내용을 제공하세요. 누락이 없도록 하세요.**
    - 보고서의 마지막 문장은 항상 학습자를 칭찬하고 격려하는 말로 끝내세요.
    - 비속어, 줄임말, 부적절한 언어를 사용하지 마세요.
    - 초등학생과 학부모가 이해하기 쉽게 단순하고 명확하게 작성하세요.

    **보고서 형식:**

    ---
    학생 이름: {{이름}}

    본 형성평가는 {f_lchapter_nm} 단원과 관련된 형성평가입니다. 본 형성평가에 포함된 개념은 {nodes_str}입니다. 다음은 학습자가 형성평가를 푼 결과를 바탕으로 개념별 학습자의 이해도를 첨삭 및 평가한 결과입니다.

    **1. 개념별 첨삭:**
    - **개념: {{Node}} ({{knowledgeTag}}):**

        - 연관된 문제:
            - 문제1: {{Question}}
                - 학생 답변: {{UserAnswer}}
                - 정답: {{Answer}}
                - 정오표시: {{Correct_OX}}

            - 문제8: {{Question}}
                - 학생 답변: {{UserAnswer}}
                - 정답: {{Answer}}
                - 정오표시: {{Correct_OX}}<br>

        - *이해도 수준: {{이해도 수준}}
        - *학습 제안: {{학습 제안}}
        - *관련 개념: {{관련 개념}}
        - *개선 사항: {{개선 내용}}

    ...

    **2. 전체 첨삭:**
    {{전체 첨삭 내용}}

    ---
    """

    return prompt


# -------------------- 보고서 생성 함수 --------------------
def generate_test_evaluation_report(student, knowledge_tag_summary, knowledge_tag_weighted_score, system_prompt):
    try:
        logger.info(f"보고서 생성을 시작합니다. 학생 이름: {student['이름']}")

        # 학생 이름 추출 및 필터링
        name = filter_bad_words(student['이름'])
        logger.debug(f"필터링된 학생 이름: {name}")

        # `knowledge_tag_summary`와 `knowledge_tag_weighted_score` 데이터에서 f_lchapter_nm와 nodes_str을 추출
        f_lchapter_nm = knowledge_tag_summary['f_lchapter_nm'].iloc[0]
        all_nodes_and_tags = [
            f"{row['Node']} ({row['knowledgeTag']})"
            for _, row in knowledge_tag_weighted_score.iterrows()
        ]
        nodes_str = ', '.join(all_nodes_and_tags)  # get_test_prompt에 전달할 개념 목록
        
        # 데이터 유형을 문자열로 통일
        knowledge_tag_summary['knowledgeTag'] = knowledge_tag_summary['knowledgeTag'].astype(str)
        knowledge_tag_weighted_score['knowledgeTag'] = knowledge_tag_weighted_score['knowledgeTag'].astype(str)

        # 개념 코드별로 문제와 이해도 분류 정보 추출
        concept_codes = []
        understanding_levels = []
        learning_suggestions = []
        related_concepts = []

        for knowledge_tag, group in knowledge_tag_summary.groupby('knowledgeTag'):
            problems = group.to_dict(orient='records')

            # 개념 코드에 대한 정보를 가져오기
            score_row = knowledge_tag_weighted_score[knowledge_tag_weighted_score['knowledgeTag'] == knowledge_tag]

            if not score_row.empty:
                node = score_row.iloc[0]['Node']  # Node 값 가져오기
                weighted_score = score_row.iloc[0]['Weighted Score']
                node_color = score_row.iloc[0]['Node Color']
                predecessors = score_row.iloc[0]['Predecessors']
                successors = score_row.iloc[0]['Successors']
            else:
                node = 'unknown'
                weighted_score = 0
                node_color = 'unknown'
                predecessors = '없음'
                successors = '없음'

            # 이해도 수준 판별
            if weighted_score >= 0.8:
                understanding_level = "높은 이해도"
            elif weighted_score >= 0.7:
                understanding_level = "보통 이해도"
            elif weighted_score >= 0.5:
                understanding_level = "낮은 이해도"
            else:
                understanding_level = "매우 낮은 이해도"

            # 학습 제안 결정
            if node_color == 'green':
                learning_suggestion = "후속 학습을 추천합니다."
                related_concepts_value = successors
            else:
                learning_suggestion = "선수 학습을 추천합니다."
                related_concepts_value = predecessors

            # 개념 코드 정보 생성
            concept_code = {
                '개념코드': str(knowledge_tag),
                'Node': node,
                '문제목록': problems,
                '이해도 수준': understanding_level,
                'Weighted Score': weighted_score,
                'Node Color': node_color,
                '학습 제안': learning_suggestion,
                '관련 개념': related_concepts_value
            }
            concept_codes.append(concept_code)
            understanding_levels.append(f"{node} (이해도 수준: {understanding_level})")
            learning_suggestions.append(f"{node}: {learning_suggestion}")
            related_concepts.append(f"{node} 관련 개념: {related_concepts_value}")

        # 문자열로 결합
        understanding_levels_str = ', '.join(understanding_levels)
        learning_suggestions_str = ', '.join(learning_suggestions)
        related_concepts_str = ', '.join(related_concepts)

        # JSON 문자열로 변환 및 필터링
        data = {
            '학생 이름': name,
            '개념 목록': concept_codes
        }
        concept_codes_json = filter_bad_words(json.dumps(data, ensure_ascii=False))

        # 사용자 프롬프트 구성
        system_prompt_with_chapter = get_test_prompt(
            prohibited_words,
            f_lchapter_nm,
            nodes_str,
            concept_codes_json,
            understanding_levels_str,
            learning_suggestions_str,
            related_concepts_str
        )

        messages = [
            {"role": "system", "content": system_prompt_with_chapter},
            {"role": "user", "content": concept_codes_json}
        ]

        logger.info("OpenAI API 호출을 시작합니다.")


        # API 호출
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            top_p=0.8,
            max_tokens=5000  # 필요에 따라 조정
        )

        # 응답 추출
        report = completion.choices[0].message.content.strip()
        logger.info("보고서가 성공적으로 생성되었습니다.")

        return report

    except openai.OpenAIError as oe:
        logger.error(f"OpenAI API 오류 발생: {str(oe)}")
        return f"오류 발생: {str(oe)}"
    except Exception as e:
        logger.error(f"보고서 생성 중 일반 오류 발생: {str(e)}")
        return f"오류 발생: {str(e)}"


# 특정 테이블만 로드하는 함수 정의
def load_final_questions():

    query = "SELECT * FROM final_questions"
    try:
        final_questions = pd.read_sql(query, engine)
        logging.info(f"final_questions 데이터가 성공적으로 로드되었습니다: {len(final_questions)}개의 행이 로드됨.")
        # 전처리 코드가 필요하면 여기서 추가할 수 있음
        if 'Unnamed: 0' in final_questions.columns:
            final_questions = final_questions.drop(['Unnamed: 0'], axis=1)
        final_questions = final_questions.rename(columns={
            '학년': 'grade',
            '학기': 'semester',
            '단원 순서': 'o_chapter',
            '대단원': 'f_lchapter_nm',
            '중단원': 'f_mchapter_nm'
        })
        logging.info("final_questions 전처리가 완료되었습니다.")
        return final_questions
    except Exception as e:
        logging.error(f"final_questions 로드 오류: {str(e)}")
        raise HTTPException(status_code=500, detail="Database Load Error")


class QuestionRequest(BaseModel):
    grade: int
    semester: int
    f_lchapter_nm: str



@app.post("/get_questions/")
def get_questions(question_request: QuestionRequest):
    global grade, semester, f_lchapter_nm
    try:
        grade = question_request.grade
        semester = question_request.semester
        f_lchapter_nm = question_request.f_lchapter_nm

        # 데이터가 로드되었는지 확인
        final_questions = load_final_questions()
        
        if final_questions is None:
            raise HTTPException(status_code=500, detail="데이터 로드 오류")

        # 입력된 학년, 학기, 대단원명으로 문제 필터링
        filtered_questions = final_questions[
            (final_questions['grade'] == grade) &
            (final_questions['semester'] == semester) &
            (final_questions['f_lchapter_nm'] == f_lchapter_nm)
        ]
        
        if len(filtered_questions) < 10:
            raise HTTPException(status_code=404, detail="해당 조건에 맞는 문제가 충분하지 않습니다.")

        questions = filtered_questions.head(10).to_dict(orient='records')
        return JSONResponse(content={"questions": questions}, media_type="application/json; charset=utf-8")

    except Exception as e:
        logging.error(f"Error during get_questions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


# TensorFlow eager execution 비활성화
tf.compat.v1.disable_eager_execution()

# 예측을 위한 입력 데이터 준비 함수
def prepare_input_data(knowledge_tags, num_features=10004):
    input_data = np.zeros((len(knowledge_tags), 1, num_features))
    for idx, quiz_code in enumerate(knowledge_tags):
        if quiz_code < num_features:
            input_data[idx][0][quiz_code] = 1
        else:
            logging.warning(f"QuizCode {quiz_code}는 num_features({num_features})를 초과합니다.")
    return input_data

# 예측 함수
def predict_with_model(session, input_tensor, output_tensor, input_data):
    try:
        dummy_labels = np.zeros((input_data.shape[0], input_data.shape[1], 5002))
        predictions = session.run(output_tensor, feed_dict={input_tensor: input_data, 'y_corr:0': dummy_labels})
        predictions = predictions[:, -1, :]
        binary_predictions = (predictions >= 0.5).astype(int)
        return predictions, binary_predictions.flatten()
    except Exception as e:
        logging.error(f"예측 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# 학습자 결과 분석 함수
def analyze_student_results(session, result_df_all, final_questions, quiz_session, input_tensor, output_tensor, user_id):
    if result_df_all.empty:
        logging.info(f"학습자 {user_id}의 데이터가 없습니다. 신규 학습자입니다.")
        learner_questions = full_quiz_session(session, final_questions, result_df_all, input_tensor, output_tensor, user_id, user_answers)
        if learner_questions is None:
            logging.error("퀴즈 세션에서 학습자의 답안이 없습니다. 프로그램을 종료합니다.")
            return None
    else:
        learner_data = result_df_all[result_df_all['UserID'] == user_id]
        if learner_data.empty:
            logging.info(f"학습자 {user_id}의 데이터가 없습니다. 신규 학습자로 처리합니다.")
            learner_questions = full_quiz_session(session, final_questions, result_df_all, input_tensor, output_tensor, user_id, user_answers)
            if learner_questions is None:
                logging.error("퀴즈 세션에서 학습자의 답안이 없습니다. 프로그램을 종료합니다.")
                return None
        else:
            quiz_code_counts = learner_data['QuizCode'].value_counts()
            sufficient_quiz_codes = quiz_code_counts[quiz_code_counts >= 3].index
            learner_questions = final_questions[final_questions['QuizCode'].isin(sufficient_quiz_codes)]
            knowledge_tags = learner_questions['knowledgeTag'].tolist()
            input_data = prepare_input_data(knowledge_tags)
            
            predictions, binary_predictions = predict_with_model(session, input_tensor, output_tensor, input_data)
            
            if predictions is not None:
                predictions = predictions[:len(learner_questions)]
                learner_questions['Predicted'] = predictions
            else:
                learner_questions['Predicted'] = None
    
    learner_questions['Correct'] = learner_questions['Correct'].map({'O': 1, 'X': 0})
    
    comparison_results = []
    for actual, predicted in zip(learner_questions['Correct'], learner_questions['Predicted']):
        if predicted == 1 and actual == 1:
            comparison_results.append("알고 있다")
        elif predicted == 1 and actual == 0:
            comparison_results.append("실수")
        elif predicted == 0 and actual == 1:
            comparison_results.append("찍음")
        else:
            comparison_results.append("모른다")
    
    learner_questions['Learning_state'] = comparison_results
    return learner_questions



# quiz_session 함수 수정
def quiz_session(final_questions, result_df_all, user_id, user_answers):
    # user_answers의 키(QuizCode)를 int로 변환하여 매핑 가능하도록 함
    user_answers = {int(k): v for k, v in user_answers.items()}
    # result_df_all이 비어 있거나 'UserID' 열이 없는 경우 신규 학습자로 처리
    if result_df_all.empty or 'UserID' not in result_df_all.columns:
        print(f"학습자 {user_id}의 데이터가 없습니다. 신규 학습자로 퀴즈 세션을 시작합니다.")
        learner_data = pd.DataFrame()  # 빈 DataFrame을 생성하여 신규 학습자 처리
    else:
        # 기존 학습자의 경우 learner_data 필터링
        learner_data = result_df_all[result_df_all['UserID'] == user_id]
        if learner_data.empty:
            print(f"학습자 {user_id}의 데이터가 없습니다. 신규 학습자로 퀴즈 세션을 시작합니다.")
        else:
            # 학습자가 풀었던 문제 목록 출력
            quiz_code_counts = learner_data['knowledgeTag'].value_counts()
            sufficient_quiz_codes = quiz_code_counts[quiz_code_counts <= 3].index
            if sufficient_quiz_codes.empty:
                print(f"학습자 {user_id}은 충분한 문제를 푼 knowledgeTag가 없습니다. 하지만 퀴즈 세션을 계속 진행합니다.")

            # 학습자가 풀었던 대단원 목록 출력
            chapter_counts = learner_data['f_lchapter_nm'].value_counts()
            print(f"학습자가 풀었던 대단원 목록 및 문제 개수:")
            for chapter, count in chapter_counts.items():
                print(f"- {chapter}: {count} 문제")


    filtered_questions = final_questions[
        (final_questions['grade'] == grade) &
        (final_questions['semester'] == semester) &
        (final_questions['f_lchapter_nm'] == f_lchapter_nm)
    ].copy()
    logging.info(f"사용자 입력에 따라 문제필터링.")

    

    if filtered_questions.empty:
        print("해당 학년, 학기, 대단원명에 해당하는 문제가 없습니다.")
        return None

    # UserID, UserAnswer, Correct 열을 추가
    filtered_questions['UserID'] = user_id
    filtered_questions['UserAnswer'] = filtered_questions['QuizCode'].map(user_answers)  # user_answers에서 QuizCode에 매핑된 답안 가져옴
    filtered_questions['Correct'] = filtered_questions.apply(
        lambda row: 'O' if row['UserAnswer'] == row['Answer'] else 'X', axis=1
    )
    logging.info(f"UserID, UserAnswer, Correct 열을 추가.")
    print(filtered_questions)
    return filtered_questions  # full_quiz_session에서 사용
    logging.info(f"full_quiz_session()")

# 전체 퀴즈 세션
def full_quiz_session(session, final_questions, result_df_all, input_tensor, output_tensor, user_id, user_answers):
    knowledge_tags = final_questions['knowledgeTag'].tolist()
    input_data = prepare_input_data(knowledge_tags)
    predictions, binary_predictions = predict_with_model(session, input_tensor, output_tensor, input_data)
    
    if predictions is not None and binary_predictions is not None:
        final_questions['Prediction_Probability'] = predictions.max(axis=1)
        final_questions['Predicted'] = binary_predictions[:len(final_questions)]
    else:
        final_questions['Prediction_Probability'] = None
        final_questions['Predicted'] = None
    
    final_questions = quiz_session(final_questions, result_df_all, user_id, user_answers)
    final_questions['Correct'] = final_questions['Correct'].map({'O': 1, 'X': 0})
    
    return final_questions
    logging.info(f"full_quiz_session()")

# 점수 계산 및 JSON 응답 생성
def calculate_and_display_scores(result_df):
    comparison_results = []
    for actual, predicted in zip(result_df['Correct'], result_df['Predicted']):
        if predicted == 1 and actual == 1:
            comparison_results.append("알고 있다")
        elif predicted == 1 and actual == 0:
            comparison_results.append("실수")
        elif predicted == 0 and actual == 1:
            comparison_results.append("찍음")
        else:
            comparison_results.append("모른다")

    result_df['Learning_state'] = comparison_results
    points_per_question = 10
    predicted_score = result_df['Predicted'].sum() * points_per_question
    actual_score = result_df['Correct'].sum() * points_per_question

    response_data = {
        "learning_state": result_df['Learning_state'].tolist(),
        "predicted_score": predicted_score,
        "actual_score": actual_score,
        "detailed_results": result_df[['Question', 'Answer', 'UserAnswer', 'Predicted', 'Correct', 'Learning_state']].to_dict(orient='records')
    }

    return response_data
    logging.info(f"full_quiz_session()")




        
def initialize_gkt_model_after_existing(result_df_all, learner_id, result_df, chunjae_math, label_math_ele_12):
    kt_to_middlesection = chunjae_math.set_index('knowledgeTag')['f_mchapter_nm'].to_dict()
    label_math_ele_12['from_middlesection'] = label_math_ele_12['from_id'].map(kt_to_middlesection)
    label_math_ele_12['to_middlesection'] = label_math_ele_12['to_id'].map(kt_to_middlesection)
    label_math_ele_12 = label_math_ele_12.dropna(subset=['from_middlesection', 'to_middlesection'])

    relationships = list(set(zip(label_math_ele_12['to_middlesection'], label_math_ele_12['from_middlesection'])))
    all_middlesections = chunjae_math['f_mchapter_nm'].unique()
    
    learner_data = result_df_all[result_df_all['UserID'] == learner_id]
    learner_result_df = result_df[result_df['UserID'] == learner_id] if 'UserID' in result_df.columns else result_df.copy()
    combined_learner_data = pd.concat([learner_data, learner_result_df], ignore_index=True)
    
    model = EnhancedGKTModel(relationships, all_middlesections)
    knowledge_tag_status = combined_learner_data.groupby('knowledgeTag')['Learning_state'].value_counts().unstack().fillna(0)

    for knowledge_tag, counts in knowledge_tag_status.iterrows():
        middlesection = kt_to_middlesection.get(knowledge_tag)
        if middlesection is None:
            logging.warning(f"KnowledgeTag '{knowledge_tag}'를 매핑할 수 없습니다. 건너뜁니다.")
            continue
        Learning_state_counts = counts.to_dict()
        model.update_knowledge(middlesection, Learning_state_counts)

    return model


def visualize_after_gkt_model(model, chunjae_math, learner_id, title='GKT Model 3D Visualization'):
    logging.info(f"학습자 '{learner_id}'의 GKT 모델을 시각화합니다.")
    f_mchapter_nm_to_knowledgeTags = chunjae_math.groupby('f_mchapter_nm')['knowledgeTag'].apply(list).to_dict()
    pos = nx.spring_layout(model.graph, dim=3, k=0.7, seed=42)

       
    # 노드 상태별로 분류
    node_groups = {}
    for node in model.graph.nodes():
        state = model.knowledge_state.get(node)
        if state not in node_groups:
            node_groups[state] = []
        node_groups[state].append(node)
    
    # 엣지 상태별로 분류 (노드 그룹과 연동)
    edge_traces = []
    state_color_mapping = {'green': 'green', 'red': 'red', None: 'gray'}
    state_name_mapping = {'green': '후속 학습 필요', 'red': '선수 학습 필요', None: '학습하지 않음'}

    for state, nodes in node_groups.items():
        # 해당 상태의 노드와 연결된 엣지를 수집
        edges = []
        for edge in model.graph.edges():
            if edge[0] in nodes or edge[1] in nodes:
                x0, y0, z0 = pos[edge[0]]
                x1, y1, z1 = pos[edge[1]]
                edges.append(dict(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    z=[z0, z1, None]
                ))
        # 엣지 트레이스 생성
        edge_trace = go.Scatter3d(
            x=sum([e['x'] for e in edges], []),
            y=sum([e['y'] for e in edges], []),
            z=sum([e['z'] for e in edges], []),
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines',
            showlegend=False,  # 범례에 엣지 중복 방지
            legendgroup=state_name_mapping[state]  # 노드 그룹과 동일한 legendgroup으로 설정
        )
        edge_traces.append(edge_trace)

    # 노드 트레이스 생성
    node_traces = []
    for state, nodes in node_groups.items():
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        for node in nodes:
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            weighted_score = model.weighted_scores.get(node, 0)
            # 연결된 선수 및 후속 개념 가져오기
            predecessors = list(model.graph.predecessors(node))
            successors = list(model.graph.successors(node))
            knowledge_tags = f_mchapter_nm_to_knowledgeTags.get(node, [])
            knowledge_tags_str = ', '.join(map(str, knowledge_tags))
            node_text.append(
                f"개념: {node} (KnowledgeTags: {knowledge_tags_str})<br>"
                f"지식 상태: {state_name_mapping[state]}<br>"
                f"Weighted Score: {weighted_score:.2f}<br>"
                f"연결된 선수 학습 개념: {', '.join(predecessors) if predecessors else '없음'}<br>"
                f"연결된 후속 학습 개념: {', '.join(successors) if successors else '없음'}"
            )
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=4,
                color=state_color_mapping[state],
                opacity=0.8
            ),
            text=nodes,
            textposition='top center',
            hoverinfo='text',
            hovertext=node_text,
            name=state_name_mapping[state],
            legendgroup=state_name_mapping[state],  # 엣지와 동일한 legendgroup으로 설정
            showlegend=True  # 범례에 표시
        )
        node_traces.append(node_trace)

    # Figure 생성
    fig = go.Figure(data=edge_traces + node_traces)

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        showlegend=True
    )

    return fig


def generate_existing_after_report(model):
    logging.info("형성평가 후 전체 보고서를 생성합니다.")
    
    known_areas = [node for node, state in model.knowledge_state.items() if state == 'green']
    deficient_areas = [node for node, state in model.knowledge_state.items() if state == 'red']
    unlearned_areas = [node for node, state in model.knowledge_state.items() if state is None]

    # 상태 데이터프레임 생성
    knowledge_state_df = pd.DataFrame.from_dict(model.knowledge_state, orient='index', columns=['Node Color'])
    report_df = pd.DataFrame.from_dict(model.weighted_scores, orient='index', columns=['Weighted Score'])
    report_df['Node Color'] = report_df.index.map(model.knowledge_state)
    report_df.reset_index(inplace=True)
    report_df.rename(columns={'index': 'Node'}, inplace=True)

    return report_df



            
def generate_after_detailed_model_and_visualization(model, learner_id, chunjae_math, title='GKT Model 3D Visualization (세분화된 그래프)'):
    # red 노드만 추출하여 서브그래프 생성
    logging.info("세분화된 그래프를 위한 모델을 생성합니다.")
    red_nodes = [node for node, state in model.knowledge_state.items() if state == 'red']
    subgraph = model.graph.subgraph(red_nodes).copy()

    # 서브그래프를 기반으로 세부 모델 인스턴스 생성
    detailed_model = EnhancedGKTModelDetailed(subgraph.edges(), subgraph.nodes())
    
    # 지식 상태 업데이트
    for node in subgraph.nodes():
        weighted_score = model.weighted_scores.get(node, 0)
        detailed_model.weighted_scores[node] = weighted_score
        # weighted_score에 따라 색상 분류
        if weighted_score >= 0.7:
            detailed_model.knowledge_state[node] = 'yellow'  # 보통 이해도
        elif weighted_score >= 0.5:
            detailed_model.knowledge_state[node] = 'orange'  # 낮은 이해도
        else:
            detailed_model.knowledge_state[node] = 'red'     # 매우 낮은 이해도

    # 세분화된 그래프 시각화
    fig_detailed = visualize_gkt_model_3d_detailed(detailed_model, chunjae_math, learner_id, title)
    return fig_detailed


def visualize_gkt_model_3d_detailed(model, chunjae_math, learner_id, title='GKT Model 3D Visualization (선수학습이 필요한 노드)'):
    logging.info(f"학습자 '{learner_id}'의 상세 GKT 모델을 시각화합니다.")
    f_mchapter_nm_to_knowledgeTags = chunjae_math.groupby('f_mchapter_nm')['knowledgeTag'].apply(list).to_dict()
    pos = nx.spring_layout(model.graph, dim=3, k=0.7, seed=42)
    
    # 노드 상태별로 분류
    node_groups = {}
    for node in model.graph.nodes():
        state = model.knowledge_state.get(node)
        if state not in node_groups:
            node_groups[state] = []
        node_groups[state].append(node)

    # 엣지 상태별로 분류 (노드 그룹과 연동)
    edge_traces = []
    for state, nodes in node_groups.items():
        # 해당 상태의 노드와 연결된 엣지를 수집
        edges = []
        for edge in model.graph.edges():
            if edge[0] in nodes or edge[1] in nodes:
                x0, y0, z0 = pos[edge[0]]
                x1, y1, z1 = pos[edge[1]]
                edges.append(dict(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    z=[z0, z1, None]
                ))
        # 엣지 트레이스 생성
        edge_trace = go.Scatter3d(
            x=sum([e['x'] for e in edges], []),
            y=sum([e['y'] for e in edges], []),
            z=sum([e['z'] for e in edges], []),
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines',
            showlegend=False,
            legendgroup=state
        )
        edge_traces.append(edge_trace)

    # 노드 트레이스 생성
    node_traces = []
    state_color_mapping = {'yellow': 'yellow', 'orange': 'orange', 'red': 'red'}
    state_name_mapping = {'yellow': '보통 이해도', 'orange': '낮은 이해도', 'red': '매우 낮은 이해도'}
    for state, nodes in node_groups.items():
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        for node in nodes:
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            weighted_score = model.weighted_scores.get(node, 0)
             # 연결된 선수 및 후속 개념 가져오기
            predecessors = list(model.graph.predecessors(node))
            successors = list(model.graph.successors(node))
            knowledge_tags = f_mchapter_nm_to_knowledgeTags.get(node, [])
            knowledge_tags_str = ', '.join(map(str, knowledge_tags))
            node_text.append(
                f"개념: {node} (KnowledgeTags: {knowledge_tags_str})<br>"
                f"지식 상태: {state_name_mapping[state]}<br>"
                f"Weighted Score: {weighted_score:.2f}<br>"
                f"연결된 선수 학습 개념: {', '.join(predecessors) if predecessors else '없음'}<br>"
                f"연결된 후속 학습 개념: {', '.join(successors) if successors else '없음'}"
            )
        node_trace = go.Scatter3d(


            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=4,
                color=state_color_mapping[state],
                opacity=0.8
            ),
            text=nodes,
            textposition='top center',
            hoverinfo='text',
            hovertext=node_text,
            name=state_name_mapping[state],
            legendgroup=state,
            showlegend=True
        )
        node_traces.append(node_trace)

    # Figure 생성
    fig = go.Figure(data=edge_traces + node_traces)

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        showlegend=True
    )
    
    return fig

# gkt_model_after_new.py
def initialize_gkt_model_after_new(result_df, chunjae_math, label_math_ele_12, user_id):
    logging.info("Loading and mapping data for new learner post-assessment model.")
    
    # 중단원 매핑
    kt_to_middlesection = chunjae_math.set_index('knowledgeTag')['f_mchapter_nm'].to_dict()
    logging.info("Mapping 'from_id' and 'to_id' to '중단원'.")
    
    label_math_ele_12['from_middlesection'] = label_math_ele_12['from_id'].map(kt_to_middlesection)
    label_math_ele_12['to_middlesection'] = label_math_ele_12['to_id'].map(kt_to_middlesection)
    label_math_ele_12 = label_math_ele_12.dropna(subset=['from_middlesection', 'to_middlesection'])

    relationships = list(set(zip(label_math_ele_12['to_middlesection'], label_math_ele_12['from_middlesection'])))
    all_middlesections = chunjae_math['f_mchapter_nm'].unique()
    
    # EnhancedGKTModel 초기화
    model = EnhancedGKTModel(relationships, all_middlesections)
    knowledge_tag_status = result_df.groupby('knowledgeTag')['Learning_state'].value_counts().unstack().fillna(0)

    for knowledge_tag, counts in knowledge_tag_status.iterrows():
        middlesection = kt_to_middlesection.get(knowledge_tag)
        if middlesection is None:
            logging.warning(f"KnowledgeTag '{knowledge_tag}'를 매핑할 수 없습니다. 건너뜁니다.")
            continue
        Learning_state_counts = counts.to_dict()
        model.update_knowledge(middlesection, Learning_state_counts)

    return model, user_id


def visualize_new_after_gkt_model(model, chunjae_math, user_id , title='GKT Model 3D Visualization'):
    logging.info(f"신규 학습자 '{user_id}'의 형성평가 GKT 모델을 시각화합니다.")
    f_mchapter_nm_to_knowledgeTags = chunjae_math.groupby('f_mchapter_nm')['knowledgeTag'].apply(list).to_dict()
    pos = nx.spring_layout(model.graph, dim=3, k=0.7, seed=42)
    
    # 노드 상태별로 분류
    node_groups = {}
    for node in model.graph.nodes():
        state = model.knowledge_state.get(node)
        if state not in node_groups:
            node_groups[state] = []
        node_groups[state].append(node)

    # 엣지 상태별로 분류 (노드 그룹과 연동)
    edge_traces = []
    state_color_mapping = {'green': 'green', 'red': 'red', None: 'gray'}
    state_name_mapping = {'green': '후속 학습 필요', 'red': '선수 학습 필요', None: '해당 단원과 관련 없음'}

    for state, nodes in node_groups.items():
        # 해당 상태의 노드와 연결된 엣지를 수집
        edges = []
        for edge in model.graph.edges():
            if edge[0] in nodes or edge[1] in nodes:
                x0, y0, z0 = pos[edge[0]]
                x1, y1, z1 = pos[edge[1]]
                edges.append(dict(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    z=[z0, z1, None]
                ))
        if not edges:
            continue
        # 엣지 트레이스 생성
        edge_trace = go.Scatter3d(
            x=sum([e['x'] for e in edges], []),
            y=sum([e['y'] for e in edges], []),
            z=sum([e['z'] for e in edges], []),
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines',
            showlegend=False,  # 범례에 엣지 중복 방지
            legendgroup=state_name_mapping[state]  # 노드 그룹과 동일한 legendgroup으로 설정
        )
        edge_traces.append(edge_trace)

    # 노드 트레이스 생성
    node_traces = []
    for state, nodes in node_groups.items():
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        for node in nodes:
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            weighted_score = model.weighted_scores.get(node, 0)
            # 연결된 선수 및 후속 개념 가져오기
            predecessors = list(model.graph.predecessors(node))
            successors = list(model.graph.successors(node))
            knowledge_tags = f_mchapter_nm_to_knowledgeTags.get(node, [])
            knowledge_tags_str = ', '.join(map(str, knowledge_tags))
            node_text.append(
                f"개념: {node} (KnowledgeTags: {knowledge_tags_str})<br>"
                f"지식 상태: {state_name_mapping[state]}<br>"
                f"Weighted Score: {weighted_score:.2f}<br>"
                f"연결된 선수 학습 개념: {', '.join(predecessors) if predecessors else '없음'}<br>"
                f"연결된 후속 학습 개념: {', '.join(successors) if successors else '없음'}"
            )
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=4,
                color=state_color_mapping[state],
                opacity=0.8
            ),
            text=nodes,
            textposition='top center',
            hoverinfo='text',
            hovertext=node_text,
            name=state_name_mapping[state],
            legendgroup=state_name_mapping[state],  # 엣지와 동일한 legendgroup으로 설정
            showlegend=True  # 범례에 표시
        )
        node_traces.append(node_trace)

    # Figure 생성
    fig = go.Figure(data=edge_traces + node_traces)

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        showlegend=True
    )
    
    return fig

def generate_new_after_report(model, detailed_model, result_df, chunjae_math):
    # 학습자 문제 풀이 데이터를 knowledge_tag_summary 데이터프레임에 저장
    knowledge_tag_summary = pd.DataFrame({
        'f_lchapter_nm': result_df['f_lchapter_nm'].tolist(),
        'Question_number': list(result_df.index + 1),
        'knowledgeTag': result_df['knowledgeTag'].tolist(),
        'Question': result_df['Question'].tolist(),
        'UserAnswer': result_df['UserAnswer'].tolist(),
        'Answer': result_df['Answer'].tolist(),
        'Correct_OX': ['O' if ua == ans else 'X' for ua, ans in zip(result_df['UserAnswer'], result_df['Answer'])],
        'Learning_state': result_df['Learning_state'].tolist()
    })

    # f_mchapter_nm 별 knowledgeTags 매핑 딕셔너리 생성
    f_mchapter_nm_to_knowledgeTags = chunjae_math.groupby('f_mchapter_nm')['knowledgeTag'].apply(list).to_dict()

    # Node에 해당하는 knowledgeTags를 가져오는 함수
    def get_knowledge_tags(node):
        knowledge_tags = f_mchapter_nm_to_knowledgeTags.get(node, [])
        return ', '.join(map(str, knowledge_tags))

    # 노드의 선수학습(Predecessors)와 후수학습(Successors)를 가져오는 함수
    def get_predecessors(node):
        predecessors = list(model.graph.predecessors(node))
        return ', '.join(predecessors) if predecessors else '없음'

    def get_successors(node):
        successors = list(model.graph.successors(node))
        return ', '.join(successors) if successors else '없음'

    # 모델에서 기본 색상 정보를 가져오고, 필요 시 detailed_model의 색상 정보를 추가
    knowledge_state_df = pd.DataFrame.from_dict(model.knowledge_state, orient='index', columns=['Node Color'])
    weighted_scores_df = pd.DataFrame.from_dict(model.weighted_scores, orient='index', columns=['Weighted Score'])
    weighted_scores_df['Node Color'] = weighted_scores_df.index.map(model.knowledge_state)
    weighted_scores_df.reset_index(inplace=True)
    weighted_scores_df.rename(columns={'index': 'Node'}, inplace=True)

    # 세분화된 모델의 정보 병합
    for node in detailed_model.knowledge_state:
        if node in weighted_scores_df['Node'].values:
            weighted_scores_df.loc[weighted_scores_df['Node'] == node, 'Node Color'] = detailed_model.knowledge_state[node]
            weighted_scores_df.loc[weighted_scores_df['Node'] == node, 'Weighted Score'] = detailed_model.weighted_scores[node]

    # Node와 매핑된 knowledgeTag, Predecessors, Successors 추가
    weighted_scores_df['knowledgeTag'] = weighted_scores_df['Node'].apply(get_knowledge_tags)
    weighted_scores_df['Predecessors'] = weighted_scores_df['Node'].apply(get_predecessors)
    weighted_scores_df['Successors'] = weighted_scores_df['Node'].apply(get_successors)

    # 최종 데이터프레임 생성
    knowledge_tag_weighted_score = weighted_scores_df[['Node', 'knowledgeTag', 'Weighted Score', 'Node Color', 'Predecessors', 'Successors']]
    
    return knowledge_tag_summary, knowledge_tag_weighted_score


class EnhancedGKTModelDetailed(EnhancedGKTModel):
    def update_knowledge(self, knowledge_element, user_Learning_state_counts):
        if knowledge_element not in self.graph.nodes:
            return

        # 모든 가능한 상태에 대한 초기화
        user_Learning_state_counts = {state: user_Learning_state_counts.get(state, 0) for state in weights.keys()}

        # 가중치 점수 계산
        total_counts = sum(user_Learning_state_counts.values())
        if total_counts == 0:
            weighted_score = 0
        else:
            weighted_score = sum(user_Learning_state_counts[state] * weights[state] for state in weights.keys()) / total_counts

        # 가중치 점수 저장
        self.weighted_scores[knowledge_element] = weighted_score

        # 가중치 점수에 따라 지식 상태 결정
        if weighted_score >= 0.7:
            self.knowledge_state[knowledge_element] = 'yellow'  # 보통 이해도
        elif weighted_score >= 0.5:
            self.knowledge_state[knowledge_element] = 'orange'  # 낮은 이해도
        else:
            self.knowledge_state[knowledge_element] = 'red'     # 매우 낮은 이해도

# Red 노드를 기반으로 세분화된 모델 생성 함수
def create_after_new_detailed_gkt_model(model, learner_id, chunjae_math, title='GKT Model 3D Visualization (세분화된 그래프)'):
    logging.info("세분화된 모델을 위한 red 노드만을 포함한 서브그래프를 생성합니다.")
    red_nodes = [node for node, state in model.knowledge_state.items() if state == 'red']
    subgraph = model.graph.subgraph(red_nodes).copy()

    detailed_model = EnhancedGKTModelDetailed(subgraph.edges(), subgraph.nodes())
    
    # 지식 상태 업데이트 (원래 모델의 weighted_score와 knowledge_state 사용)
    for node in subgraph.nodes():
        weighted_score = model.weighted_scores.get(node, 0)
        detailed_model.weighted_scores[node] = weighted_score
        # weighted_score에 따라 색상 분류
        if weighted_score >= 0.7:
            detailed_model.knowledge_state[node] = 'yellow'  # 보통 이해도
        elif weighted_score >= 0.5:
            detailed_model.knowledge_state[node] = 'orange'  # 낮은 이해도
        else:
            detailed_model.knowledge_state[node] = 'red'     # 매우 낮은 이해도
            
    return detailed_model  # 모델 객체를 반환

# 세분화된 GKT 모델 시각화 함수
def visualize_after_new_gkt_model_detailed(model, chunjae_math, learner_id, title='GKT Model 3D Visualization (선수학습이 필요한 노드의 Learning_state)'):
    logging.info(f"학습자 '{learner_id}'의 형성평가 상세 GKT 모델을 시각화합니다.")
    f_mchapter_nm_to_knowledgeTags = chunjae_math.groupby('f_mchapter_nm')['knowledgeTag'].apply(list).to_dict()
    pos = nx.spring_layout(model.graph, dim=3, k=0.7, seed=42)

    # 노드 상태별로 분류
    node_groups = {}
    for node in model.graph.nodes():
        state = model.knowledge_state.get(node)
        if state not in node_groups:
            node_groups[state] = []
        node_groups[state].append(node)

    # 엣지 상태별로 분류
    edge_traces = []
    state_color_mapping = {'yellow': 'yellow', 'orange': 'orange', 'red': 'red'}
    state_name_mapping = {'yellow': '보통 이해도', 'orange': '낮은 이해도', 'red': '매우 낮은 이해도'}

    for state, nodes in node_groups.items():
        edges = [edge for edge in model.graph.edges() if edge[0] in nodes or edge[1] in nodes]
        if edges:
            edge_trace = go.Scatter3d(
                x=sum([[pos[e[0]][0], pos[e[1]][0], None] for e in edges], []),
                y=sum([[pos[e[0]][1], pos[e[1]][1], None] for e in edges], []),
                z=sum([[pos[e[0]][2], pos[e[1]][2], None] for e in edges], []),
                line=dict(width=2, color='gray'),
                hoverinfo='none',
                mode='lines',
                showlegend=False,
                legendgroup=state
            )
            edge_traces.append(edge_trace)

    # 노드 트레이스 생성
    node_traces = []
    for state, nodes in node_groups.items():
        node_x, node_y, node_z, node_text = [], [], [], []
        for node in nodes:
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            weighted_score = model.weighted_scores.get(node, 0)
            predecessors = list(model.graph.predecessors(node))
            successors = list(model.graph.successors(node))
            knowledge_tags = f_mchapter_nm_to_knowledgeTags.get(node, [])
            knowledge_tags_str = ', '.join(map(str, knowledge_tags))
            node_text.append(
                f"개념: {node}<br>지식 상태: {state_name_mapping[state]}<br>"
                f"Weighted Score: {weighted_score:.2f}<br>"
                f"연결된 선수 학습 개념: {', '.join(predecessors) if predecessors else '없음'}<br>"
                f"연결된 후속 학습 개념: {', '.join(successors) if successors else '없음'}"
            )
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(size=4, color=state_color_mapping[state], opacity=0.8),
            text=nodes,
            textposition='top center',
            hoverinfo='text',
            hovertext=node_text,
            name=state_name_mapping[state],
            legendgroup=state,
            showlegend=True
        )
        node_traces.append(node_trace)

    # Figure 생성
    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(xaxis=dict(showbackground=False), yaxis=dict(showbackground=False), zaxis=dict(showbackground=False)),
        showlegend=True
    )

    return fig

# after_learning.py


def process_after_learning(user_id, user_specific_data, session, input_tensor, output_tensor, main_tables, chunjae_math, label_math_ele_12, base_path):
    global learner_id
    result_df = full_quiz_session(session, main_tables, user_specific_data["result_df_all"], input_tensor, output_tensor, user_id, user_answers)
    calculate_and_display_scores(result_df)

    # 학습 후 데이터 및 시각화 생성
    if not user_specific_data["result_df_all"].empty and 'UserID' in user_specific_data["result_df_all"].columns:
        if user_id in user_specific_data["result_df_all"]["UserID"].values:
            print(f"기존 학습자 '{user_id}'의 학습 후 GKT 모델을 생성합니다.")
            after_gkt_model = initialize_gkt_model_after_existing(user_specific_data["result_df_all"], user_id, result_df, chunjae_math, label_math_ele_12)

            # 전체 그래프 생성 및 HTML로 변환
            fig_after = visualize_after_gkt_model(after_gkt_model, chunjae_math, user_id, title=f"{user_id} (학습 후 전체 그래프)")
            full_after_graph_html = fig_after.to_html(full_html=False)
            print(f"{user_id}_gkt_model_after.html 파일로 저장되었습니다.")
            with open(f"{base_path}{user_id}_gkt_model_after.html", "w", encoding="utf-8") as file:
                file.write(full_after_graph_html)

            # 세분화된 그래프 생성 및 HTML로 변환
            fig_after_detailed = generate_after_detailed_model_and_visualization(after_gkt_model, user_id, chunjae_math, title=f"{user_id} (학습 후 세분화 그래프)")
            detailed_after_graph_html = fig_after_detailed.to_html(full_html=False)
            print(f"{user_id}_gkt_model_after_detailed.html 파일로 저장되었습니다.")
            with open(f"{base_path}{user_id}_gkt_model_after_detailed.html", "w", encoding="utf-8") as file:
                file.write(detailed_after_graph_html)

            # 학습 후 요약 보고서 생성 및 데이터프레임 반환
            summary_after_report_df = generate_existing_after_report(after_gkt_model)
            area_avg_scores = perform_analysis(summary_after_report_df, chunjae_math, base_path, user_id, report_type=f"학습 후 영역 보고서_{user_id}")
            summary_after_report_df = create_summary_df(summary_after_report_df)
            print("기존 학습자를 위한 학습 후 요약 보고서:")
            print(summary_after_report_df)

            # 학습자 보고서 생성 및 출력 (LLM 사용)
            student_data = {"이름": user_id}
            system_after_prompt = get_area_prompt(prohibited_words)
            area_report_after = generate_area_evaluation_report(
                student=student_data,
                summary_df=summary_after_report_df,
                system_prompt=system_after_prompt
            )
            print(area_report_after)

            # 형성평가 데이터 기반 GKT 모델 생성 및 시각화
            test_gkt_model, learner_id = initialize_test_gkt_model(result_df, user_specific_data["result_df_all"], chunjae_math, label_math_ele_12)

            if test_gkt_model:
                # GKT 모델 시각화 생성 및 저장
                fig_test = visualize_test_gkt_model(test_gkt_model, chunjae_math, learner_id, title=f"{learner_id} (형성평가 기반 GKT 모델)")
                full_test_graph_html = fig_test.to_html(full_html=False)
                print(f"{learner_id}_gkt_model_test_existing.html 파일로 저장되었습니다.")
                with open(f"{base_path}{learner_id}_gkt_model_test_existing.html", "w", encoding="utf-8") as file:
                    file.write(full_test_graph_html)

                # 세분화된 모델 시각화 생성 및 저장
                fig_test_detailed = visualize_test_gkt_model_detailed(
                    create_test_detailed_gkt_model(test_gkt_model, learner_id, chunjae_math), chunjae_math, learner_id, title=f"{learner_id} (형성평가 세분화 그래프)"
                )
                detailed_test_graph_html = fig_test_detailed.to_html(full_html=False)
                print(f"{learner_id}_gkt_model_test_existing_detailed.html 파일로 저장되었습니다.")
                with open(f"{base_path}{learner_id}_gkt_model_test_existing_detailed.html", "w", encoding="utf-8") as file:
                    file.write(detailed_test_graph_html)

                # 세분화된 모델 객체 생성
                test_gkt_model_detailed = create_test_detailed_gkt_model(test_gkt_model, learner_id, chunjae_math)

                # 학습자 문제 풀이 요약 및 가중치 점수 보고서 생성
                knowledge_tag_summary, knowledge_tag_weighted_score = generate_test_existing_report(test_gkt_model, test_gkt_model_detailed, result_df, chunjae_math)

                # 생성된 데이터프레임 출력
                print("\nKnowledge Tag Summary:")
                print(knowledge_tag_summary)

                print("\nKnowledge Tag Weighted Score:")
                print(knowledge_tag_weighted_score)

                # f_lchapter_nm 값을 knowledge_tag_summary에서 추출 (모든 행이 동일하다고 가정)
                f_lchapter_nm = knowledge_tag_summary['f_lchapter_nm'].iloc[0]

                # Node와 knowledgeTag를 결합하여 nodes_str 생성
                nodes_str = ', '.join(
                    f"{row['Node']} ({row['knowledgeTag']})" for _, row in knowledge_tag_weighted_score.iterrows()
                )

                # 개념 코드별로 추가 정보 추출
                concept_codes = []
                understanding_levels = []
                learning_suggestions = []
                related_concepts = []

                knowledge_tag_summary['knowledgeTag'] = knowledge_tag_summary['knowledgeTag'].astype(str)
                knowledge_tag_weighted_score['knowledgeTag'] = knowledge_tag_weighted_score['knowledgeTag'].astype(str)

                for knowledge_tag, group in knowledge_tag_summary.groupby('knowledgeTag'):
                    problems = group.to_dict(orient='records')
                    score_row = knowledge_tag_weighted_score[knowledge_tag_weighted_score['knowledgeTag'] == knowledge_tag]

                    if not score_row.empty:
                        node = score_row.iloc[0]['Node']
                        weighted_score = score_row.iloc[0]['Weighted Score']
                        node_color = score_row.iloc[0]['Node Color']
                        predecessors = score_row.iloc[0]['Predecessors']
                        successors = score_row.iloc[0]['Successors']
                    else:
                        node = 'unknown'
                        weighted_score = 0
                        node_color = 'unknown'
                        predecessors = '없음'
                        successors = '없음'

                    if weighted_score >= 0.8:
                        understanding_level = "높은 이해도"
                    elif weighted_score >= 0.7:
                        understanding_level = "보통 이해도"
                    elif weighted_score >= 0.5:
                        understanding_level = "낮은 이해도"
                    else:
                        understanding_level = "매우 낮은 이해도"

                    if node_color == 'green':
                        learning_suggestion = "후속 학습을 추천합니다."
                        related_concepts_value = successors
                    else:
                        learning_suggestion = "선수 학습을 추천합니다."
                        related_concepts_value = predecessors

                    concept_code = {
                        '개념코드': str(knowledge_tag),
                        'Node': node,
                        '문제목록': problems,
                        '이해도 수준': understanding_level,
                        'Weighted Score': weighted_score,
                        'Node Color': node_color,
                        '학습 제안': learning_suggestion,
                        '관련 개념': related_concepts_value
                    }
                    concept_codes.append(concept_code)
                    understanding_levels.append(f"{node} (이해도 수준: {understanding_level})")
                    learning_suggestions.append(f"{node}: {learning_suggestion}")
                    related_concepts.append(f"{node} 관련 개념: {related_concepts_value}")

                understanding_levels_str = ', '.join(understanding_levels)
                learning_suggestions_str = ', '.join(learning_suggestions)
                related_concepts_str = ', '.join(related_concepts)

                concept_codes_json = json.dumps(concept_codes, ensure_ascii=False)

                # 보고서 생성 (LLM 사용)
                student_data = {"이름": user_id}
                system_after_test_prompt = get_test_prompt(
                    prohibited_words,
                    f_lchapter_nm,
                    nodes_str,
                    concept_codes_json,
                    understanding_levels_str,
                    learning_suggestions_str,
                    related_concepts_str
                )
                test_report = generate_test_evaluation_report(
                    student=student_data,
                    knowledge_tag_summary=knowledge_tag_summary,
                    knowledge_tag_weighted_score=knowledge_tag_weighted_score,
                    system_prompt=system_after_test_prompt
                )
                print(test_report)

                # 최종 결과 리턴 추가
                return full_after_graph_html, detailed_after_graph_html, area_report_after, area_avg_scores, full_test_graph_html, detailed_test_graph_html, test_report


    else:
        # 신규 학습자에 대한 처리
        print(f"신규 학습자 '{user_id}'의 학습 후 GKT 모델을 생성합니다.")
        after_new_gkt_model, user_id = initialize_gkt_model_after_new(result_df, chunjae_math, label_math_ele_12, user_id)

        # 신규 학습자 모델 시각화
        fig_after_new = visualize_new_after_gkt_model(after_new_gkt_model, chunjae_math, user_id, title=f"{user_id} (학습 후 전체 그래프)")
        full_after_graph_html_new = fig_after_new.to_html(full_html=False)
        print(f"{user_id}_gkt_model_after_new.html 파일로 저장되었습니다.")
        with open(f"{base_path}{user_id}_gkt_model_after_new.html", "w", encoding="utf-8") as file:
            file.write(full_after_graph_html_new)

        # 세분화된 GKT 모델 생성 및 시각화
        detailed_after_new_model = create_after_new_detailed_gkt_model(after_new_gkt_model,user_id, chunjae_math)
        
        fig_after_new_detailed = visualize_after_new_gkt_model_detailed(
            create_after_new_detailed_gkt_model(after_new_gkt_model, user_id, chunjae_math), chunjae_math, user_id, title=f"{user_id} (학습 후 세분화 그래프)"
        )
        detailed_after_graph_html_new = fig_after_new_detailed.to_html(full_html=False)
        print(f"{user_id}_gkt_model_after_new_detailed.html 파일로 저장되었습니다.")
        with open(f"{base_path}{user_id}_gkt_model_after_new_detailed.html", "w", encoding="utf-8") as file:
            file.write(detailed_after_graph_html_new)

        # 신규 학습자 문제 풀이 요약 및 가중치 점수 보고서 생성
        knowledge_tag_new_summary, knowledge_tag_new_weighted_score = generate_new_after_report(after_new_gkt_model,detailed_after_new_model, result_df, chunjae_math)

        # 생성된 데이터프레임 출력
        print("\n신규 학습자의 학습 후 Knowledge Tag Summary:")
        print(knowledge_tag_new_summary)

        print("\n신규 학습자의 학습 후 Knowledge Tag Weighted Score:")
        print(knowledge_tag_new_weighted_score)

        
        # f_lchapter_nm 값을 knowledge_tag_summary에서 추출 (모든 행이 동일하다고 가정)
        f_lchapter_nm = knowledge_tag_new_summary['f_lchapter_nm'].iloc[0]

        # Node와 knowledgeTag를 결합하여 nodes_str 생성
        nodes_str = ', '.join(
            f"{row['Node']} ({row['knowledgeTag']})" for _, row in knowledge_tag_new_weighted_score.iterrows()
        )

        # 개념 코드별로 추가 정보 추출
        concept_codes = []
        understanding_levels = []
        learning_suggestions = []
        related_concepts = []

        knowledge_tag_new_summary['knowledgeTag'] = knowledge_tag_new_summary['knowledgeTag'].astype(str)
        knowledge_tag_new_weighted_score['knowledgeTag'] = knowledge_tag_new_weighted_score['knowledgeTag'].astype(str)

        for knowledge_tag, group in knowledge_tag_new_summary.groupby('knowledgeTag'):
            problems = group.to_dict(orient='records')
            score_row = knowledge_tag_new_weighted_score[knowledge_tag_new_weighted_score['knowledgeTag'] == knowledge_tag]

            if not score_row.empty:
                node = score_row.iloc[0]['Node']
                weighted_score = score_row.iloc[0]['Weighted Score']
                node_color = score_row.iloc[0]['Node Color']
                predecessors = score_row.iloc[0]['Predecessors']
                successors = score_row.iloc[0]['Successors']
            else:
                node = 'unknown'
                weighted_score = 0
                node_color = 'unknown'
                predecessors = '없음'
                successors = '없음'

            if weighted_score >= 0.8:
                understanding_level = "높은 이해도"
            elif weighted_score >= 0.7:
                understanding_level = "보통 이해도"
            elif weighted_score >= 0.5:
                understanding_level = "낮은 이해도"
            else:
                understanding_level = "매우 낮은 이해도"

            if node_color == 'green':
                learning_suggestion = "후속 학습을 추천합니다."
                related_concepts_value = successors
            else:
                learning_suggestion = "선수 학습을 추천합니다."
                related_concepts_value = predecessors

            concept_code = {
                '개념코드': str(knowledge_tag),
                'Node': node,
                '문제목록': problems,
                '이해도 수준': understanding_level,
                'Weighted Score': weighted_score,
                'Node Color': node_color,
                '학습 제안': learning_suggestion,
                '관련 개념': related_concepts_value
            }
            concept_codes.append(concept_code)
            understanding_levels.append(f"{node} (이해도 수준: {understanding_level})")
            learning_suggestions.append(f"{node}: {learning_suggestion}")
            related_concepts.append(f"{node} 관련 개념: {related_concepts_value}")

        understanding_levels_str = ', '.join(understanding_levels)
        learning_suggestions_str = ', '.join(learning_suggestions)
        related_concepts_str = ', '.join(related_concepts)

        concept_codes_json = json.dumps(concept_codes, ensure_ascii=False)

        # 보고서 생성 (LLM 사용)
        student_data_new = {"이름": user_id}
        system_prompt_new = get_test_prompt(
            prohibited_words,
            f_lchapter_nm,
            nodes_str,
            concept_codes_json,
            understanding_levels_str,
            learning_suggestions_str,
            related_concepts_str
        )
        new_test_report = generate_test_evaluation_report(
            student=student_data_new,
            knowledge_tag_summary=knowledge_tag_new_summary,
            knowledge_tag_weighted_score=knowledge_tag_new_weighted_score,
            system_prompt=system_prompt_new
    )
        print(new_test_report)

        return full_after_graph_html_new, detailed_after_graph_html_new, new_test_report, knowledge_tag_new_summary
        


def initialize_test_gkt_model(result_df, result_df_all, chunjae_math, label_math_ele_12):
    logging.info("Loading and mapping data from 천재교육_계열화.")
    kt_to_middlesection = chunjae_math.set_index('knowledgeTag')['f_mchapter_nm'].to_dict()
    label_math_ele_12['from_middlesection'] = label_math_ele_12['from_id'].map(kt_to_middlesection)
    label_math_ele_12['to_middlesection'] = label_math_ele_12['to_id'].map(kt_to_middlesection)
    label_math_ele_12 = label_math_ele_12.dropna(subset=['from_middlesection', 'to_middlesection'])

    relationships = list(set(zip(label_math_ele_12['to_middlesection'], label_math_ele_12['from_middlesection'])))
    all_middlesections = chunjae_math['f_mchapter_nm'].unique()

    # learner_id를 result_df에서 추출
    learner_id = result_df['UserID'].iloc[0]  # 첫 번째 행의 UserID를 사용
    
    learner_data = result_df_all[result_df_all['UserID'] == learner_id]
    learner_result_df = result_df[result_df['UserID'] == learner_id] if 'UserID' in result_df.columns else result_df.copy()
    combined_learner_data = pd.concat([learner_data, learner_result_df], ignore_index=True)
    
    unique_knowledge_tags = result_df['knowledgeTag'].unique()
    filtered_learner_data = combined_learner_data[combined_learner_data['knowledgeTag'].isin(unique_knowledge_tags)]
    knowledge_tag_status = filtered_learner_data.groupby('knowledgeTag')['Learning_state'].value_counts().unstack().fillna(0)

    model = EnhancedGKTModel(relationships, all_middlesections)

    for knowledge_tag, counts in knowledge_tag_status.iterrows():
        middlesection = kt_to_middlesection.get(knowledge_tag)
        if middlesection is None:
            logging.warning(f"KnowledgeTag '{knowledge_tag}'를 매핑할 수 없습니다. 건너뜁니다.")
            continue
        Learning_state_counts = counts.to_dict()
        model.update_knowledge(middlesection, Learning_state_counts)

    return model, learner_id  # learner_id를 반환합니다.



def visualize_test_gkt_model(model, chunjae_math, learner_id, title='GKT Model 3D Visualization'):
    logging.info(f"학습자 '{learner_id}'의 형성평가 GKT 모델을 시각화합니다.")
    f_mchapter_nm_to_knowledgeTags = chunjae_math.groupby('f_mchapter_nm')['knowledgeTag'].apply(list).to_dict()
    pos = nx.spring_layout(model.graph, dim=3, k=0.7, seed=42)

       
    # 노드 상태별로 분류
    node_groups = {}
    for node in model.graph.nodes():
        state = model.knowledge_state.get(node)
        if state not in node_groups:
            node_groups[state] = []
        node_groups[state].append(node)
    
    # 엣지 상태별로 분류 (노드 그룹과 연동)
    edge_traces = []
    state_color_mapping = {'green': 'green', 'red': 'red', None: 'gray'}
    state_name_mapping = {'green': '후속 학습 필요', 'red': '선수 학습 필요', None: '해당 단원과 관련 없음'}

    for state, nodes in node_groups.items():
        # 해당 상태의 노드와 연결된 엣지를 수집
        edges = []
        for edge in model.graph.edges():
            if edge[0] in nodes or edge[1] in nodes:
                x0, y0, z0 = pos[edge[0]]
                x1, y1, z1 = pos[edge[1]]
                edges.append(dict(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    z=[z0, z1, None]
                ))
        # 엣지 트레이스 생성
        edge_trace = go.Scatter3d(
            x=sum([e['x'] for e in edges], []),
            y=sum([e['y'] for e in edges], []),
            z=sum([e['z'] for e in edges], []),
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines',
            showlegend=False,  # 범례에 엣지 중복 방지
            legendgroup=state_name_mapping[state]  # 노드 그룹과 동일한 legendgroup으로 설정
        )
        edge_traces.append(edge_trace)

    # 노드 트레이스 생성
    node_traces = []
    for state, nodes in node_groups.items():
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        for node in nodes:
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            weighted_score = model.weighted_scores.get(node, 0)
            # 연결된 선수 및 후속 개념 가져오기
            predecessors = list(model.graph.predecessors(node))
            successors = list(model.graph.successors(node))
            knowledge_tags = f_mchapter_nm_to_knowledgeTags.get(node, [])
            knowledge_tags_str = ', '.join(map(str, knowledge_tags))
            node_text.append(
                f"개념: {node} (KnowledgeTags: {knowledge_tags_str})<br>"
                f"지식 상태: {state_name_mapping[state]}<br>"
                f"Weighted Score: {weighted_score:.2f}<br>"
                f"연결된 선수 학습 개념: {', '.join(predecessors) if predecessors else '없음'}<br>"
                f"연결된 후속 학습 개념: {', '.join(successors) if successors else '없음'}"
            )
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=4,
                color=state_color_mapping[state],
                opacity=0.8
            ),
            text=nodes,
            textposition='top center',
            hoverinfo='text',
            hovertext=node_text,
            name=state_name_mapping[state],
            legendgroup=state_name_mapping[state],  # 엣지와 동일한 legendgroup으로 설정
            showlegend=True  # 범례에 표시
        )
        node_traces.append(node_trace)

    # Figure 생성
    fig = go.Figure(data=edge_traces + node_traces)

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        showlegend=True
    )

    return fig

def generate_test_existing_report(model, detailed_model, result_df, chunjae_math):
    # 학습자 문제 풀이 데이터를 knowledge_tag_summary 데이터프레임에 저장
    knowledge_tag_summary = pd.DataFrame({
        'f_lchapter_nm': result_df['f_lchapter_nm'].tolist(),
        'Question_number': list(result_df.index + 1),
        'knowledgeTag': result_df['knowledgeTag'].tolist(),
        'Question': result_df['Question'].tolist(),
        'UserAnswer': result_df['UserAnswer'].tolist(),
        'Answer': result_df['Answer'].tolist(),
        'Correct_OX': ['O' if ua == ans else 'X' for ua, ans in zip(result_df['UserAnswer'], result_df['Answer'])],
        'Learning_state': result_df['Learning_state'].tolist()
    })

    # f_mchapter_nm 별 knowledgeTags 매핑 딕셔너리 생성
    f_mchapter_nm_to_knowledgeTags = chunjae_math.groupby('f_mchapter_nm')['knowledgeTag'].apply(list).to_dict()

    # Node에 해당하는 knowledgeTags를 가져오는 함수
    def get_knowledge_tags(node):
        knowledge_tags = f_mchapter_nm_to_knowledgeTags.get(node, [])
        return ', '.join(map(str, knowledge_tags))

    # 노드의 선수학습(Predecessors)와 후수학습(Successors)를 가져오는 함수
    def get_predecessors(node):
        predecessors = list(model.graph.predecessors(node))
        return ', '.join(predecessors) if predecessors else '없음'

    def get_successors(node):
        successors = list(model.graph.successors(node))
        return ', '.join(successors) if successors else '없음'

    # 모델에서 기본 색상 정보를 가져오고, 필요 시 detailed_model의 색상 정보를 추가
    knowledge_state_df = pd.DataFrame.from_dict(model.knowledge_state, orient='index', columns=['Node Color'])
    weighted_scores_df = pd.DataFrame.from_dict(model.weighted_scores, orient='index', columns=['Weighted Score'])
    weighted_scores_df['Node Color'] = weighted_scores_df.index.map(model.knowledge_state)
    weighted_scores_df.reset_index(inplace=True)
    weighted_scores_df.rename(columns={'index': 'Node'}, inplace=True)

    # 세분화된 모델의 정보 병합
    for node in detailed_model.knowledge_state:
        if node in weighted_scores_df['Node'].values:
            weighted_scores_df.loc[weighted_scores_df['Node'] == node, 'Node Color'] = detailed_model.knowledge_state[node]
            weighted_scores_df.loc[weighted_scores_df['Node'] == node, 'Weighted Score'] = detailed_model.weighted_scores[node]

    # Node와 매핑된 knowledgeTag, Predecessors, Successors 추가
    weighted_scores_df['knowledgeTag'] = weighted_scores_df['Node'].apply(get_knowledge_tags)
    weighted_scores_df['Predecessors'] = weighted_scores_df['Node'].apply(get_predecessors)
    weighted_scores_df['Successors'] = weighted_scores_df['Node'].apply(get_successors)

    # 최종 데이터프레임 생성
    knowledge_tag_weighted_score = weighted_scores_df[['Node', 'knowledgeTag', 'Weighted Score', 'Node Color', 'Predecessors', 'Successors']]
    
    return knowledge_tag_summary, knowledge_tag_weighted_score

def create_test_detailed_gkt_model(model, learner_id, chunjae_math, title='GKT Model 3D Visualization (세분화된 그래프)'):
    # `red` 노드들만 추출
    logging.info("세분화된 모델을 위한 red 노드만을 포함한 서브그래프를 생성합니다.")
    red_nodes = [node for node, state in model.knowledge_state.items() if state == 'red']
    subgraph = model.graph.subgraph(red_nodes).copy()

    # 서브그래프를 기반으로 세분화된 모델 생성
    detailed_model = EnhancedGKTModelDetailed(subgraph.edges(), subgraph.nodes())

    # 지식 상태 업데이트 (원래 모델의 weighted_score와 knowledge_state 사용)
    for node in subgraph.nodes():
        weighted_score = model.weighted_scores.get(node, 0)
        detailed_model.weighted_scores[node] = weighted_score
        # weighted_score에 따라 색상 분류
        if weighted_score >= 0.7:
            detailed_model.knowledge_state[node] = 'yellow'  # 보통 이해도
        elif weighted_score >= 0.5:
            detailed_model.knowledge_state[node] = 'orange'  # 낮은 이해도
        else:
            detailed_model.knowledge_state[node] = 'red'     # 매우 낮은 이해도
            
    return detailed_model  # 모델 객체를 반환


def visualize_test_gkt_model_detailed(model, chunjae_math, learner_id, title='GKT Model 3D Visualization (선수학습이 필요한 노드)'):
    logging.info(f"학습자 '{learner_id}'의 형성평가 상세 GKT 모델을 시각화합니다.")
    f_mchapter_nm_to_knowledgeTags = chunjae_math.groupby('f_mchapter_nm')['knowledgeTag'].apply(list).to_dict()
    pos = nx.spring_layout(model.graph, dim=3, k=0.7, seed=42)
    
    # 노드 상태별로 분류
    node_groups = {}
    for node in model.graph.nodes():
        state = model.knowledge_state.get(node)
        if state not in node_groups:
            node_groups[state] = []
        node_groups[state].append(node)

    # 엣지 상태별로 분류 (노드 그룹과 연동)
    edge_traces = []
    for state, nodes in node_groups.items():
        # 해당 상태의 노드와 연결된 엣지를 수집
        edges = []
        for edge in model.graph.edges():
            if edge[0] in nodes or edge[1] in nodes:
                x0, y0, z0 = pos[edge[0]]
                x1, y1, z1 = pos[edge[1]]
                edges.append(dict(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    z=[z0, z1, None]
                ))
        # 엣지 트레이스 생성
        edge_trace = go.Scatter3d(
            x=sum([e['x'] for e in edges], []),
            y=sum([e['y'] for e in edges], []),
            z=sum([e['z'] for e in edges], []),
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines',
            showlegend=False,
            legendgroup=state
        )
        edge_traces.append(edge_trace)

    # 노드 트레이스 생성
    node_traces = []
    state_color_mapping = {'yellow': 'yellow', 'orange': 'orange', 'red': 'red'}
    state_name_mapping = {'yellow': '보통 이해도', 'orange': '낮은 이해도', 'red': '매우 낮은 이해도'}
    for state, nodes in node_groups.items():
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        for node in nodes:
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            weighted_score = model.weighted_scores.get(node, 0)
             # 연결된 선수 및 후속 개념 가져오기
            predecessors = list(model.graph.predecessors(node))
            successors = list(model.graph.successors(node))
            knowledge_tags = f_mchapter_nm_to_knowledgeTags.get(node, [])
            knowledge_tags_str = ', '.join(map(str, knowledge_tags))
            node_text.append(
                f"개념: {node} (KnowledgeTags: {knowledge_tags_str})<br>"
                f"지식 상태: {state_name_mapping[state]}<br>"
                f"Weighted Score: {weighted_score:.2f}<br>"
                f"연결된 선수 학습 개념: {', '.join(predecessors) if predecessors else '없음'}<br>"
                f"연결된 후속 학습 개념: {', '.join(successors) if successors else '없음'}"
            )
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=4,
                color=state_color_mapping[state],
                opacity=0.8
            ),
            text=nodes,
            textposition='top center',
            hoverinfo='text',
            hovertext=node_text,
            name=state_name_mapping[state],
            legendgroup=state,
            showlegend=True
        )
        node_traces.append(node_trace)

    # Figure 생성
    fig = go.Figure(data=edge_traces + node_traces)

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        showlegend=True
    )
    
    return fig




# AnswerInput 데이터 모델
class AnswerInput(BaseModel):
    answers: dict  # QuizCode와 답변이 매핑된 딕셔너리로 변경

@app.post("/submit_answers/", response_class=HTMLResponse)
async def submit_answers(answer_input: AnswerInput):
    global user_id, input_tensor, output_tensor, user_answers, session, result_df_all, chunjae_math, label_math_ele_12, data
    user_answers = answer_input.answers

    logging.info(f"Submitting answers for user_id: {user_id}")

    if not user_id:
        raise HTTPException(status_code=400, detail="User ID가 입력되지 않았습니다. 먼저 /check_user_id 엔드포인트를 호출하세요.")

    try:
        # 모델 로드 확인
        if 'input_tensor' not in globals() or 'output_tensor' not in globals():
            s3_bucket_name = "dev-team-haejo-backup"
            s3_model_key_prefix = "model/"
            session, graph = load_model_from_s3(s3_bucket_name, s3_model_key_prefix)
            if session is None or graph is None:
                raise ValueError("모델 세션이나 그래프를 로드할 수 없습니다.")
            input_tensor, output_tensor = get_model_tensors(graph)

        # 캐싱된 데이터 사용
        if result_df_all is None or chunjae_math is None or label_math_ele_12 is None:
            raise HTTPException(status_code=500, detail="데이터가 로드되지 않았습니다. /check_user_id 엔드포인트를 먼저 호출하세요.")
        final_questions = load_final_questions()

        # 정적 파일 저장 경로 설정
        base_path = f"./static/{user_id}/"
        os.makedirs(base_path, exist_ok=True)

        # 기존 학습자 처리
        if not result_df_all.empty and 'UserID' in result_df_all.columns and user_id in result_df_all["UserID"].values:
            logging.info(f"UserID {user_id}는 기존 학습자입니다.")
            
            # process_after_learning 함수 호출
            full_after_graph_html, detailed_after_graph_html, area_report_after, area_avg_scores, full_test_graph_html, detailed_test_graph_html, test_report = process_after_learning(
                user_id, data, session, input_tensor, output_tensor, final_questions, chunjae_math, label_math_ele_12, base_path
            )

            # Radar Chart 생성 및 저장
            plot_radar_chart(area_avg_scores, user_id, title=f"{user_id} Radar Chart")

            # HTML 파일 저장
            with open(f"{base_path}{user_id}_gkt_model_after.html", "w", encoding="utf-8") as file:
                file.write(full_after_graph_html)
            with open(f"{base_path}{user_id}_gkt_model_after_detailed.html", "w", encoding="utf-8") as file:
                file.write(detailed_after_graph_html)
            with open(f"{base_path}{user_id}_gkt_model_test_existing.html", "w", encoding="utf-8") as file:
                file.write(full_test_graph_html)
            with open(f"{base_path}{user_id}_gkt_model_test_existing_detailed.html", "w", encoding="utf-8") as file:
                file.write(detailed_test_graph_html)

            # HTML 콘텐츠 병합
            combined_content = f"""
            <h3 style="text-align: center;">개념별 내 이해도는 어떨까요?</h3>
            <iframe src="/static/{user_id}/{user_id}_gkt_model_after.html" width="100%" height="400"></iframe>
            <iframe src="/static/{user_id}/{user_id}_gkt_model_after_detailed.html" width="100%" height="400"></iframe>
            <h3 style="text-align: center;">영역별 나의 학습 상태는?</h3>
            <img src="/static/{user_id}/{user_id}_radar_chart.png" alt="Radar Chart" width="100%">
            <h3 style="text-align: center;">영역별 나의 학습 상태에 대한 평가</h3>
            <p>{area_report_after}</p>
            <br><br>
            <h3 style="text-align: center;">개념별 내 이해도는 어떨까요?</h3>
            <iframe src="/static/{user_id}/{user_id}_gkt_model_test_existing.html" width="100%" height="400"></iframe>
            <iframe src="/static/{user_id}/{user_id}_gkt_model_test_existing_detailed.html" width="100%" height="400"></iframe>
            <h3 style="text-align: center;">개념별 나의 학습 상태에 대한 평가</h3>
            <p>{test_report}</p>
            """

            return HTMLResponse(content=combined_content)

        else:
            # 신규 학습자 처리
            logging.info(f"UserID {user_id}는 신규 학습자입니다.")
            full_after_graph_html_new, detailed_after_graph_html_new, new_test_report, _ = process_after_learning(
                user_id, data, session, input_tensor, output_tensor, final_questions, chunjae_math, label_math_ele_12, base_path
            )

            # 신규 학습자 파일 저장
            with open(f"{base_path}{user_id}_gkt_model_after_new.html", "w", encoding="utf-8") as file:
                file.write(full_after_graph_html_new)
            with open(f"{base_path}{user_id}_gkt_model_after_new_detailed.html", "w", encoding="utf-8") as file:
                file.write(detailed_after_graph_html_new)

            # 신규 학습자 HTML 콘텐츠 생성
            combined_content = f"""
            <h3 style="text-align: center;">개념별 내 이해도는 어떨까요?</h3>
            <iframe src="/static/{user_id}/{user_id}_gkt_model_after_new.html" width="100%" height="400"></iframe>
            <iframe src="/static/{user_id}/{user_id}_gkt_model_after_new_detailed.html" width="100%" height="400"></iframe>
            <h3 style="text-align: center;">개념별 나의 학습 상태에 대한 평가</h3>
            <p>{new_test_report}</p>
            """

            return HTMLResponse(content=combined_content)

    except Exception as e:
        logging.error(f"Error in submit_answers endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

