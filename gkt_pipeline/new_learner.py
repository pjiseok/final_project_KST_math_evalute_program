from gkt_model_existing_area import perform_analysis, create_summary_df, plot_radar_chart
from data_retrieval import load_summary_mean_df
from llm import get_mean_area_prompt, generate_area_evaluation_report, prohibited_words
import boto3
import shutil
from datetime import datetime
import matplotlib.pyplot as plt

def process_new_learner(user_id, user_specific_data, chunjae_math, label_math_ele_12, engine):
    print(f"신규 학습자 '{user_id}'는 기존 학습자의 전체 평균 데이터를 기반으로 GKT 모델을 생성합니다.")

    # 조건을 명확히 하여 신규 학습자 분기 실행
    if not user_id or user_specific_data["result_df_all"].empty:
        s3_client = boto3.client("s3")
        bucket_name = "dev-team-haejo-backup"
        base_path = "mean_data/"

        # 기존 학습자의 평균 그래프와 보고서 불러오기
        s3_client.download_file(bucket_name, f"{base_path}exsiting_gkt_model_mean.html", "/tmp/exsiting_gkt_model_mean.html")
        s3_client.download_file(bucket_name, f"{base_path}exsiting_gkt_model_mean_new_detailed.html", "/tmp/exsiting_gkt_model_mean_new_detailed.html")
        s3_client.download_file(bucket_name, f"{base_path}기존_학습자의_전체_영역_평균_보고서_Radar_Chart_20241105_082705.png", "/tmp/기존_학습자의_전체_영역_평균_보고서_Radar_Chart_20241105_082705.png")
        
        # 현재 디렉토리에 {user_id}로 파일명 저장
        shutil.copy("/tmp/exsiting_gkt_model_mean.html", f"./{user_id}_exsiting_gkt_model_mean.html")
        print("기존 학습자의 전체 평균 데이터를 기반으로 한 GKT 모델 입니다.")
        shutil.copy("/tmp/exsiting_gkt_model_mean_new_detailed.html", f"./{user_id}_exsiting_gkt_model_mean_new_detailed.html")
        print("기존 학습자의 전체 평균 데이터를 기반으로 한 세분화 GKT 모델 입니다.")
        shutil.copy("/tmp/기존_학습자의_전체_영역_평균_보고서_Radar_Chart_20241105_082705.png", f"./{user_id}_기존_학습자의_전체_영역_평균_보고서.png")
        print("기존 학습자의 전체 평균 데이터를 기반으로 한 전체 영역 평균 보고서 입니다.")
        
        # summary_mean_df를 RDS에서 불러오기
        summary_mean_df = load_summary_mean_df(engine)
        print("summary_mean_df 로드 완료:", summary_mean_df.head())

        # # 평균 보고서 레이더 차트 생성
        # plot_radar_chart(summary_mean_df["Average_Score"], title=f"{user_id}_기존 학습자의 전체 영역 평균 보고서")
        
    else:
        print("기존 학습자로 확인되어 다른 분기로 이동합니다.")
        summary_mean_df = None

    # 이어지는 코드
    if summary_mean_df is not None:
        student_data = {"이름": user_id if user_id else "신규 학습자"}
        system_mean_prompt = get_mean_area_prompt(prohibited_words)
        area_report_new = generate_area_evaluation_report(
            student=student_data,
            summary_df=summary_mean_df,
            system_prompt=system_mean_prompt
        )
        print("LLM을 사용한 학습자 보고서 생성 완료")
        print(area_report_new)
