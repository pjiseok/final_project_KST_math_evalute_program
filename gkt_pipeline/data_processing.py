# data_processing.py
import pandas as pd

def create_result_df_all_mean(result_df_all):
    """
    각 knowledgeTag별 Correct와 Predicted의 평균을 계산하고 학습 상태를 결정합니다.

    Parameters:
        result_df_all (pd.DataFrame): 학습자 데이터
    
    Returns:
        pd.DataFrame: 각 knowledgeTag에 대한 평균값을 포함한 result_df_all_mean
    """
    # 각 knowledgeTag별 Correct와 Predicted의 평균 계산
    result_df_all_mean = result_df_all.groupby('knowledgeTag').agg({
        'Correct': 'mean',
        'Predicted': 'mean'
    }).reset_index()

    # 평균값을 기준으로 mean_correct 및 mean_predict 컬럼 생성
    result_df_all_mean['mean_correct'] = result_df_all_mean['Correct'].apply(lambda x: 1 if x > 0.779731053627856 else 0)
    result_df_all_mean['mean_predict'] = result_df_all_mean['Predicted'].apply(lambda x: 1 if x > 0.5030735414547344 else 0)

    # 학습 상태를 결정하는 함수 정의
    def get_mean_learning_state(row):
        if row['mean_correct'] == 1 and row['mean_predict'] == 1:
            return '알고 있다'
        elif row['mean_correct'] == 1 and row['mean_predict'] == 0:
            return '찍음'
        elif row['mean_correct'] == 0 and row['mean_predict'] == 1:
            return '실수'
        elif row['mean_correct'] == 0 and row['mean_predict'] == 0:
            return '모른다'

    # mean_학습상태 컬럼 추가
    result_df_all_mean['mean_Learning_state'] = result_df_all_mean.apply(get_mean_learning_state, axis=1)
    
    return result_df_all_mean
