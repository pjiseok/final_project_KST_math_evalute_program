# after_learning.py


import pandas as pd
import json
from evaluation import full_quiz_session, calculate_and_display_scores
from gkt_model_after_existing import (
    initialize_gkt_model_after_existing,
    visualize_after_gkt_model,
    generate_existing_after_report,
    generate_after_detailed_model_and_visualization
)
from gkt_model_after_new import (
    initialize_gkt_model_after_new,
    visualize_new_after_gkt_model,
    create_after_new_detailed_gkt_model,
   visualize_after_new_gkt_model_detailed,
    generate_new_after_report
)
from gkt_model_test_existing import (
    initialize_test_gkt_model,
    visualize_test_gkt_model,
    create_test_detailed_gkt_model,
    visualize_test_gkt_model_detailed,
    generate_test_existing_report
)
from gkt_model_existing_area import perform_analysis, create_summary_df
from llm import get_area_prompt, get_test_prompt, generate_area_evaluation_report, generate_test_evaluation_report, prohibited_words

def process_after_learning(user_id, user_specific_data, session, input_tensor, output_tensor, main_tables, chunjae_math, label_math_ele_12):
    # 형성평가 진행 및 학습 후 GKT 모델 생성
    result_df = full_quiz_session(session, main_tables["final_questions"], user_specific_data["result_df_all"], input_tensor, output_tensor, user_id)
    # 가상의 1000문제 입력--------------------
#     result_df = pd.read_csv('./82mixed_data.csv', index_col=None, encoding='utf-8-sig')
    #-----------------------------------------
    calculate_and_display_scores(result_df)

    if not user_specific_data["result_df_all"].empty and 'UserID' in user_specific_data["result_df_all"].columns:
        if user_id in user_specific_data["result_df_all"]["UserID"].values:
            print(f"기존 학습자 '{user_id}'의 학습 후 GKT 모델을 생성합니다.")
            after_gkt_model = initialize_gkt_model_after_existing(user_specific_data["result_df_all"], user_id, result_df, chunjae_math, label_math_ele_12)

            fig_after = visualize_after_gkt_model(after_gkt_model, chunjae_math, user_id, title=f"{user_id} (학습 후 전체 그래프)")
            fig_after.write_html(f"{user_id}_gkt_model_after.html")
            print(f"{user_id}_gkt_model_after.html 파일로 저장되었습니다.")

            fig_after_detailed = generate_after_detailed_model_and_visualization(after_gkt_model, user_id, chunjae_math, title=f"{user_id} (학습 후 세분화 그래프)")
            fig_after_detailed.write_html(f"{user_id}_gkt_model_after_detailed.html")
            print(f"{user_id}_gkt_model_after_detailed.html 파일로 저장되었습니다.")

            # 학습 후 보고서 생성 및 출력
            summary_after_report_df = generate_existing_after_report(after_gkt_model)
            perform_analysis(summary_after_report_df, chunjae_math, report_type=f"학습 후 영역 보고서_{user_id}")
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
                fig_test.write_html(f"{learner_id}_gkt_model_test_existing.html")
                print(f"{learner_id}_gkt_model_test_existing.html 파일로 저장되었습니다.")

                # 세분화된 모델 객체 생성
                test_gkt_model_detailed = create_test_detailed_gkt_model(test_gkt_model, learner_id, chunjae_math)

                # 세분화된 모델 시각화 생성 및 저장
                fig_test_detailed = visualize_test_gkt_model_detailed(
                    test_gkt_model_detailed, chunjae_math, learner_id, title=f"{learner_id} (형성평가 세분화 그래프)"
                )
                fig_test_detailed.write_html(f"{learner_id}_gkt_model_test_existing_detailed.html")
                print(f"{learner_id}_gkt_model_test_existing_detailed.html 파일로 저장되었습니다.")

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


    else:
        # 신규 학습자에 대한 처리
        print(f"신규 학습자 '{user_id}'의 학습 후 GKT 모델을 생성합니다.")
        after_new_gkt_model, user_id = initialize_gkt_model_after_new(result_df, chunjae_math, label_math_ele_12, user_id)

        # 학습 후 GKT 모델 시각화
        fig_after_new = visualize_new_after_gkt_model(after_new_gkt_model, chunjae_math, user_id, title=f"{user_id} (학습 후 전체 그래프)")
        fig_after_new.write_html(f"{user_id}_gkt_model_after_new.html")
        print(f"{user_id}_gkt_model_after_new.html 파일로 저장되었습니다.")

       
        # 세분화된 GKT 모델 생성 및 시각화
        detailed_after_new_model = create_after_new_detailed_gkt_model(after_new_gkt_model,user_id, chunjae_math)
        
        fig_after_new_detailed = visualize_after_new_gkt_model_detailed(
            detailed_after_new_model, chunjae_math, user_id, title=f"{user_id} (학습 후 세분화 그래프)"
        )
        fig_after_new_detailed.write_html(f"{user_id}_gkt_model_after_new_detailed.html")
        print(f"{user_id}_gkt_model_after_new_detailed.html 파일로 저장되었습니다.")

        # 신규 학습자 문제 풀이 요약 및 가중치 점수 보고서 생성
        knowledge_tag_new_summary, knowledge_tag_new_weighted_score = generate_new_after_report(after_new_gkt_model,detailed_after_new_model, result_df, chunjae_math )

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
        
