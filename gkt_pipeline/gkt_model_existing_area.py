import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
from datetime import datetime
from matplotlib import font_manager
import warnings



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
def plot_radar_chart(area_avg_scores, title="Radar Chart"):
    # 폰트 경로 설정
    font_path = "/home/jovyan/work/NanumFont/NanumGothic.ttf"
    fontprop = font_manager.FontProperties(fname=font_path)
    # 특정 경고 무시 (matplotlib findfont 경고)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    # 한글 폰트를 적용하여 그래프 설정
    plt.rcParams['font.family'] = fontprop.get_name()
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
    
    # 각 축에 레이블 추가 및 평균 점수 표시
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontproperties=fontprop)
#     ax.set_yticklabels([])
    num_ticks = 10  # 내부 원의 개수를 정할 수 있음
    ticks = np.linspace(0, 1, num_ticks)  # 0에서 1 사이의 값을 지정
    ax.set_yticks(ticks)
    ax.set_yticklabels([], color="gray", fontsize=8, fontproperties=fontprop)  # 소수점 2자리까지 표시
    
    # 각 축 근처에 평균 점수 텍스트 추가
    for angle, score, label in zip(angles, scores, labels):
        x = np.cos(angle) * (score + 0.1)
        y = np.sin(angle) * (score + 0.1)
        ax.text(angle, score, f'{score:.2f}', ha='center', va='center', fontsize=10, color='black')
    

    plt.title(title, fontproperties=fontprop)

    # 현재 날짜와 시간을 사용하여 파일 이름 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{title.replace(' ', '_')}_{timestamp}.png"
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    
    logging.info(f"Radar chart saved as {filename}")

# 실행 함수: 전체 분석 수행
def perform_analysis(df, chunjae_math, report_type="Report"):
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

    # 오각형 그래프 생성 및 저장
    plot_radar_chart(area_avg_scores, title=f"{report_type} Radar Chart")

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
