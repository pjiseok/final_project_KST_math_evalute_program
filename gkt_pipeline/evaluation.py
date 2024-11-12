import pandas as pd
import numpy as np
import tensorflow as tf
from model_loader import load_checkpoint_model, get_model_tensors

# TensorFlow eager execution 비활성화
tf.compat.v1.disable_eager_execution()

# 예측을 위한 입력 데이터 준비 함수
def prepare_input_data(knowledge_tags, num_features=10004):
    input_data = np.zeros((len(knowledge_tags), 1, num_features))
    for idx, quiz_code in enumerate(knowledge_tags):
        if quiz_code < num_features:
            input_data[idx][0][quiz_code] = 1
        else:
            print(f"QuizCode {quiz_code}는 num_features({num_features})를 초과합니다.")
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
        print(f"예측 중 오류 발생: {str(e)}")
        return None, None


# 학습자 결과 분석 함수
def analyze_student_results(session, result_df_all, final_questions, quiz_session, input_tensor, output_tensor, user_id):
    # result_df_all이 비어 있는지 확인
    if result_df_all.empty:
        print(f"학습자 {user_id}의 데이터가 없습니다. 신규 학습자입니다. 바로 퀴즈 세션으로 이동합니다.")
        learner_questions = full_quiz_session(session, final_questions, result_df_all, input_tensor, output_tensor, user_id)
        if learner_questions is None:
            print("퀴즈 세션에서 학습자의 답안이 없습니다. 프로그램을 종료합니다.")
            return None
    else:
        # 기존 학습자의 경우 learner_data를 필터링
        learner_data = result_df_all[result_df_all['UserID'] == user_id]
        if learner_data.empty:
            print(f"학습자 {user_id}의 데이터가 없습니다. 신규 학습자입니다. 바로 퀴즈 세션으로 이동합니다.")
            learner_questions = full_quiz_session(session, final_questions, result_df_all, input_tensor, output_tensor, user_id)
            if learner_questions is None:
                print("퀴즈 세션에서 학습자의 답안이 없습니다. 프로그램을 종료합니다.")
                return None
        else:
            quiz_code_counts = learner_data['QuizCode'].value_counts()
            sufficient_quiz_codes = quiz_code_counts[quiz_code_counts >= 3].index
            learner_questions = final_questions[final_questions['QuizCode'].isin(sufficient_quiz_codes)]
            knowledge_tags = learner_questions['knowledgeTag'].tolist()
            input_data = prepare_input_data(knowledge_tags)
            
            # 모델 예측
            predictions, binary_predictions = predict_with_model(session, input_tensor, output_tensor, input_data)
            
            # 예측 결과의 길이가 learner_questions와 맞지 않을 경우 처리
            if predictions is not None:
                predictions = predictions[:len(learner_questions)]  # 예측 결과를 learner_questions 길이에 맞춤
                learner_questions['Predicted'] = predictions
            else:
                learner_questions['Predicted'] = None
    
    # 학습자 정답 데이터를 이진 값으로 매핑
    learner_questions['Correct'] = learner_questions['Correct'].map({'O': 1, 'X': 0})
    
    # 학습 상태 비교
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


# 학년, 학기, 대단원명 유효성 검사 함수
def get_valid_input(prompt, valid_options):
    while True:
        user_input = input(prompt).strip()
        if user_input in valid_options:
            return user_input
        else:
            print("잘못된 입력입니다. 다시 입력해주세요.")

# quiz_session 함수 수정
def quiz_session(final_questions, result_df_all, user_id):
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

    # 학년, 학기, 대단원 목록 생성
    valid_grades = final_questions['grade'].astype(str).unique().tolist()
    valid_semesters = final_questions['semester'].astype(str).unique().tolist()
    valid_chapters = final_questions['f_lchapter_nm'].unique().tolist()

    # 유효성 검사 적용된 입력 받기
    grade = int(get_valid_input("학년을 입력하세요: ", valid_grades))
    semester = int(get_valid_input("학기를 입력하세요: ", valid_semesters))
    l_chapter = get_valid_input("대단원명을 입력하세요: ", valid_chapters)
    
    # 사용자 입력에 따라 문제 필터링
    filtered_questions = final_questions[
        (final_questions['grade'] == grade) &
        (final_questions['semester'] == semester) &
        (final_questions['f_lchapter_nm'] == l_chapter)
    ]

    if filtered_questions.empty:
        print("해당 학년, 학기, 대단원명에 해당하는 문제가 없습니다.")
        return None

    # 문제를 출력하고 사용자 답안을 기록
    user_answers = []
    correct_answers = []

    for idx, row in filtered_questions.iterrows():
        print('---------------------------')
        print(f"문제 {idx + 1}: {row['Question']}")
        user_answer = input("답을 입력하세요: ")
        user_answers.append(user_answer)
        correct_answers.append('O' if user_answer == row['Answer'] else 'X')

    # 결과 데이터프레임에 사용자 답안과 정답 여부 추가
    filtered_questions['UserID'] = user_id
    filtered_questions['UserAnswer'] = user_answers
    filtered_questions['Correct'] = correct_answers
    
    pd.set_option('display.max_rows', None)  # 모든 행을 표시
    pd.set_option('display.max_columns', None)
    print(filtered_questions)
    print(filtered_questions.dtypes)
    return filtered_questions


# 전체 퀴즈 세션
def full_quiz_session(session, final_questions, result_df_all, input_tensor, output_tensor, user_id):
    knowledge_tags = final_questions['knowledgeTag'].tolist()
    input_data = prepare_input_data(knowledge_tags)
    predictions, binary_predictions = predict_with_model(session, input_tensor, output_tensor, input_data)
    
    if predictions is not None and binary_predictions is not None:
        # 이진 예측 결과 저장
        final_questions['Predicted'] = binary_predictions[:len(final_questions)]

        # 확률 값을 저장 - Predicted 값에 따라 조정
        final_questions['Prediction_Probability'] = final_questions.apply(
            lambda row: predictions.min(axis=1)[row.name] if row['Predicted'] == 0 else predictions.max(axis=1)[row.name],axis=1
        )
    else:
        final_questions['Prediction_Probability'] = None
        final_questions['Predicted'] = None

    
    final_questions = quiz_session(final_questions, result_df_all, user_id)
    final_questions['Correct'] = final_questions['Correct'].map({'O': 1, 'X': 0})
    
    return final_questions

# 점수 계산 및 출력
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

    columns_to_drop = ['Unnamed: 0', 'unique_content_nm', 'f_lchapter_id', 'f_mchapter_id', 'o_chapter']
    result_df = result_df.drop(columns=columns_to_drop, errors='ignore')
    columns_order = ['UserID', 'grade', 'semester', 'f_lchapter_nm', 'f_mchapter_nm', 'mCode', 'QuizCode', 'knowledgeTag', 
                     'Question', 'Answer', 'UserAnswer', 'Predicted', 'Correct', 'Prediction_Probability', 'Learning_state']
    result_df = result_df[columns_order]
    
    # pandas 설정 변경: 출력할 최대 행 수와 최대 열 수 설정
    pd.set_option('display.max_rows', None)  # 모든 행을 표시
    pd.set_option('display.max_columns', None)
#     result_df = pd.read_csv('./mixed_data.csv', index_col=None, encoding='utf-8-sig')
    print("\n학습자 답변과 모델 예측 결과:")
    print(result_df)
    # Answer와 UserAnswer 열의 데이터 타입 출력
    print("\nAnswer와 UserAnswer 열의 데이터 타입:")
    print("Answer 타입:", result_df['Answer'].dtype)
    print("UserAnswer 타입:", result_df['UserAnswer'].dtype)
    print(f"\n예측 점수: {predicted_score} / {len(result_df) * points_per_question}")
    print(f"실제 점수: {actual_score} / {len(result_df) * points_per_question}")
