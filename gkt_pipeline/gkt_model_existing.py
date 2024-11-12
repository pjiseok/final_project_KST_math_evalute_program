# gkt_model.py
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
                size=8,
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

class EnhancedGKTModelDetailed(EnhancedGKTModel):
    def update_knowledge(self, knowledge_element, Learning_state_counts):
        if knowledge_element not in self.graph.nodes:
            return

        Learning_state_counts = {state: Learning_state_counts.get(state, 0) for state in weights.keys()}
        total_counts = sum(Learning_state_counts.values())
        weighted_score = (
            sum(Learning_state_counts[state] * weights[state] for state in weights.keys()) / total_counts
            if total_counts > 0 else 0
        )

        self.weighted_scores[knowledge_element] = weighted_score
        if weighted_score >= 0.7:
            self.knowledge_state[knowledge_element] = 'yellow'
        elif weighted_score >= 0.5:
            self.knowledge_state[knowledge_element] = 'orange'
        else:
            self.knowledge_state[knowledge_element] = 'red'
            
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
                size=8,
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
