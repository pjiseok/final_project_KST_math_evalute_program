# data_retrieval.py
import pandas as pd

def load_main_tables(engine):
    """
    주요 테이블 데이터를 로드하여 반환합니다.
    
    Parameters:
        engine: SQLAlchemy 엔진 객체로 데이터베이스 연결
    
    Returns:
        dict: 로드된 주요 테이블 데이터
    """
    # 주요 테이블 불러오기
    label_math_ele_12 = pd.read_sql("SELECT * FROM math_label", engine)
    chunjae_math = pd.read_sql("SELECT * FROM chunjae_math", engine)
    final_questions = pd.read_sql("SELECT * FROM final_questions", engine)
    # 4. 테이블 'final_questions' 불러오기 및 열 이름 변경
    final_questions = pd.read_sql("SELECT * FROM final_questions", engine)
    if 'Unnamed: 0' in final_questions.columns:
        final_questions = final_questions.drop(['Unnamed: 0'], axis=1)
    final_questions = final_questions.rename(columns={
        '학년': 'grade',
        '학기': 'semester',
        '단원 순서': 'o_chapter',
        '대단원': 'f_lchapter_nm',
        '중단원': 'f_mchapter_nm'
    })
    
    return {
        "label_math_ele_12": label_math_ele_12,
        "chunjae_math": chunjae_math,
        "final_questions": final_questions
    }


def load_user_specific_data(engine, user_id, chunjae_math):
    """
    특정 UserID에 해당하는 student_df_12와 result_df_all 데이터를 불러오고 전처리합니다.
    신규 학습자의 경우 데이터 불러오기를 생략합니다.
    
    Parameters:
        engine: SQLAlchemy 엔진 객체로 데이터베이스 연결
        user_id (str): 검색할 UserID
        chunjae_math (pd.DataFrame): chunjae_math 데이터프레임 (병합에 사용)

    Returns:
        dict: 특정 UserID에 해당하는 데이터 또는 빈 데이터
    """
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
            print(f"UserID '{user_id}'가 존재하지 않습니다.")
            result_df_all = pd.DataFrame()  # 빈 데이터 반환
    else:
        result_df_all = pd.DataFrame()  # 빈 데이터 반환
    
    return {
        "result_df_all": result_df_all
    }

def load_summary_mean_df(engine):
    """
    RDS에서 summary_mean 테이블을 불러와 데이터프레임으로 반환합니다.
    
    Parameters:
        engine: SQLAlchemy 엔진 객체로 데이터베이스 연결

    Returns:
        pd.DataFrame: summary_mean 테이블 데이터
    """
    summary_mean_df = pd.read_sql("SELECT * FROM mean_summary", engine)
    return summary_mean_df
