# existing_learner.py

from gkt_model_existing import (
    initialize_existing_model,
    visualize_existing_gkt_model,
    generate_existing_detailed_model_and_visualization,
    generate_existing_report
)
from gkt_model_existing_area import perform_analysis, create_summary_df
from llm import get_area_prompt, generate_area_evaluation_report, prohibited_words

def process_existing_learner(user_id, user_specific_data, chunjae_math, label_math_ele_12):
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
    fig = visualize_existing_gkt_model(gkt_model, chunjae_math, user_id,  title=f"{user_id} (기존 학습자 전체 그래프)")
    fig.write_html(f"{user_id}_gkt_model_existing.html")
    print(f"{user_id}_gkt_model_existing.html 파일로 저장되었습니다.")

    # 세분화된 모델 생성 및 시각화
    fig_detailed = generate_existing_detailed_model_and_visualization(
        gkt_model, user_id, chunjae_math, title=f"{user_id} (기존 학습자 세분화 그래프)"
    )
    fig_detailed.write_html(f"{user_id}_gkt_model_existing_detailed.html")
    print(f"{user_id}_gkt_model_existing_detailed.html 파일로 저장되었습니다.")

    # 학습자 보고서 생성 및 출력
    report_df = generate_existing_report(gkt_model)
    perform_analysis(report_df, chunjae_math, report_type=f"기존 학습자 영역 보고서_{user_id}")
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
