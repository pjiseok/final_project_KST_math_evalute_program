# llm.py
import openai
import os
import json
import re
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import boto3

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

def load_env_from_s3(bucket_name, env_file_key, badwords_file_key):
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
load_env_from_s3(bucket_name, env_file_key, badwords_file_key)

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
    - **평균 점수:** {{Average_Score}}
    - **가장 잘한 부분:** {{Highest_Node}} ({{Highest_Score}})
    - **가장 못한 부분:** {{Lowest_Node}} ({{Lowest_Score}})
- **도형과 측정(도형):**
    - **평균 점수:** {{Average_Score}}
    - **가장 잘한 부분:** {{Highest_Node}} ({{Highest_Score}})
    - **가장 못한 부분:** {{Lowest_Node}} ({{Lowest_Score}})
- **도형과 측정(측정):**
    - **평균 점수:** {{Average_Score}}
    - **가장 잘한 부분:** {{Highest_Node}} ({{Highest_Score}})
    - **가장 못한 부분:** {{Lowest_Node}} ({{Lowest_Score}})
- **변화와 관계:**
    - **평균 점수:** {{Average_Score}}
    - **가장 잘한 부분:** {{Highest_Node}} ({{Highest_Score}})
    - **가장 못한 부분:** {{Lowest_Node}} ({{Lowest_Score}})
- **자료와 가능성:**
    - **평균 점수:** {{Average_Score}}
    - **가장 잘한 부분:** {{Highest_Node}} ({{Highest_Score}})
    - **가장 못한 부분:** {{Lowest_Node}} ({{Lowest_Score}})

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
    - **평균 점수:** {{Average_Score}}
    - **가장 잘한 부분:** {{Highest_Node}} ({{Highest_Score}})
    - **가장 못한 부분:** {{Lowest_Node}} ({{Lowest_Score}})
- **도형과 측정(도형):**
    - **평균 점수:** {{Average_Score}}
    - **가장 잘한 부분:** {{Highest_Node}} ({{Highest_Score}})
    - **가장 못한 부분:** {{Lowest_Node}} ({{Lowest_Score}})
- **도형과 측정(측정):**
    - **평균 점수:** {{Average_Score}}
    - **가장 잘한 부분:** {{Highest_Node}} ({{Highest_Score}})
    - **가장 못한 부분:** {{Lowest_Node}} ({{Lowest_Score}})
- **변화와 관계:**
    - **평균 점수:** {{Average_Score}}
    - **가장 잘한 부분:** {{Highest_Node}} ({{Highest_Score}})
    - **가장 못한 부분:** {{Lowest_Node}} ({{Lowest_Score}})
- **자료와 가능성:**
    - **평균 점수:** {{Average_Score}}
    - **가장 잘한 부분:** {{Highest_Node}} ({{Highest_Score}})
    - **가장 못한 부분:** {{Lowest_Node}} ({{Lowest_Score}})

**종합 정리:**
{{이 보고서는 기존 학습자들의 평균 점수입니다. 나만의 보고서를 제공받기 위해서는 문제를 풀어주세요. 그럼 함께 수학 공부하러 가볼까요 ?}}

---
"""

    return prompt


# -------------------- 보고서 생성 함수 --------------------
def generate_area_evaluation_report(student, summary_df, system_prompt):
    """
    단일 학생의 전체 영역 보고서를 생성하는 함수.

    Parameters:
    - student (dict): 학생의 데이터 (이름, 문제풀이, 지식맵)
    - summary_df (pd.DataFrame): 영역별 요약 데이터프레임
    - system_prompt (str): 모델에게 제공할 시스템 프롬프트 area_prompt

    Returns:
    - str: 생성된 보고서
    """
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

    2. **개념 코드별 첨삭**:
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

    **1. 개념 코드별 첨삭:**
    - **개념: {{Node}} ({{knowledgeTag}}):**
        - **연관된 문제:**
            - **문제1:** {{Question}}
                - **학생 답변:** {{UserAnswer}}
                - **정답:** {{Answer}}
                - **정오표시:** {{Correct_OX}}
            - **문제8:** {{Question}}
                - **학생 답변:** {{UserAnswer}}
                - **정답:** {{Answer}}
                - **정오표시:** {{Correct_OX}}
        - **이해도 수준:** {{이해도 수준}}
        - **학습 제안:** {{학습 제안}}
        - **관련 개념:** {{관련 개념}}
        - **개선 사항:** {{개선 내용}}

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
