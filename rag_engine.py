"""
rag_engine.py
=============
v5 RAG 데모 로직을 리팩토링한 엔진.

핵심 차이:
1. 하드코딩된 Q1~Q10 대신 자유 질문 라우팅
2. 사용자 추가 인스턴스 인지 (instance_manager 연동)
3. Streamlit 앱에서 호출 가능한 함수 형태

라우팅 흐름:
  사용자 질문
    ↓
  [route_question]
    ├─ 펀드명 매칭 + 진행상태 키워드 → ontology_template (Q1~Q5 패턴)
    ├─ 펀드명 매칭 + RWA/규제 키워드 → instance_with_concept (Q10 패턴 + 인스턴스 컨텍스트)
    ├─ 위험가중치 키워드 → deterministic_lookup (Q9 패턴)
    ├─ 일반 규제 키워드 → rag_concept (Q6~Q8 패턴)
    └─ 매칭 안됨 → guidance (안내 메시지)
"""

import json
import os
import re
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS

INV = Namespace("http://company.com/investment-ontology#")

# ============================================
# 모델 설정
# ============================================
# 환경변수 OLLAMA_MODEL 로 모델 변경 가능
# - 페이퍼 실험 모델: gemma4:e4b (Gemma 4, ~4B effective params, edge device용)
# - 시연 환경에 따라 더 작은 모델로 swap 가능 (예: gemma4:e2b, gemma3:4b 등)
# - 환경변수 미설정 시 기본 gemma4:e4b
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")

# ============================================
# 설정
# ============================================

# Streamlit 앱 폴더에서 상대 경로로 데이터 로드
DATA_DIR = Path(__file__).parent / "data"


# ============================================
# 1. 데이터 로드
# ============================================

def load_ttl(ttl_path):
    g = Graph()
    g.parse(str(ttl_path), format='turtle')
    return g


def load_chunks(jsonl_path):
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def load_alias(alias_path):
    with open(alias_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_lookup(lookup_path):
    with open(lookup_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================
# 2. SPARQL 쿼리 (v5 로직 그대로)
# ============================================

def query_investment_branches(g, investment_iri):
    """특정 Investment의 모든 브랜치와 상태 조회"""
    q = f"""
    PREFIX inv: <http://company.com/investment-ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?branchLabel ?productLabel ?amount ?stageLabel ?stateLabel ?stateOrder WHERE {{
      inv:{investment_iri} inv:hasBranch ?branch .
      ?branch rdfs:label ?branchLabel ;
              inv:hasProductType ?product ;
              inv:hasInvestmentAmount ?amount ;
              inv:hasCurrentStage ?stage ;
              inv:hasBranchState ?state .
      ?product rdfs:label ?productLabel .
      ?stage rdfs:label ?stageLabel .
      ?state rdfs:label ?stateLabel ;
             inv:hasStateOrder ?stateOrder .
      FILTER(LANG(?branchLabel)="ko")
      FILTER(LANG(?productLabel)="ko")
      FILTER(LANG(?stageLabel)="ko")
      FILTER(LANG(?stateLabel)="ko")
    }} ORDER BY ?branchLabel
    """
    branches = []
    for row in g.query(q):
        branches.append({
            "label": str(row.branchLabel),
            "product": str(row.productLabel),
            "amount": int(float(str(row.amount))),
            "stage": str(row.stageLabel),
            "state": str(row.stateLabel),
            "state_order": int(float(str(row.stateOrder))),
        })
    return branches


def query_investment_meta(g, investment_iri):
    """Investment 메타 정보"""
    q = f"""
    PREFIX inv: <http://company.com/investment-ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?label ?gpLabel ?fundLabel WHERE {{
      inv:{investment_iri} rdfs:label ?label .
      OPTIONAL {{ inv:{investment_iri} inv:managedByGP ?gp . ?gp rdfs:label ?gpLabel . FILTER(LANG(?gpLabel)="ko") }}
      OPTIONAL {{ inv:{investment_iri} inv:hasIndirectTarget ?fund . ?fund rdfs:label ?fundLabel . FILTER(LANG(?fundLabel)="ko") }}
      FILTER(LANG(?label)="ko")
    }}
    """
    for row in g.query(q):
        return {
            "label": str(row.label),
            "gp": str(row.gpLabel) if row.gpLabel else "",
            "fund": str(row.fundLabel) if row.fundLabel else "",
        }
    return {}


def query_all_investments_with_label(g):
    """모든 Investment 인스턴스의 IRI와 label 조회 (펀드명 매칭용).
    분기가 있는 인스턴스를 우선 정렬하여 매칭 시 빈 껍데기 인스턴스를 회피."""
    q = """
    PREFIX inv: <http://company.com/investment-ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?inv ?label ?fundLabel (COUNT(?branch) AS ?nBranches) WHERE {
      ?inv a inv:Investment ;
           rdfs:label ?label .
      OPTIONAL { ?inv inv:hasBranch ?branch }
      OPTIONAL { ?inv inv:hasIndirectTarget ?fund . ?fund rdfs:label ?fundLabel . FILTER(LANG(?fundLabel)="ko") }
      FILTER(LANG(?label)="ko")
    } GROUP BY ?inv ?label ?fundLabel
    ORDER BY DESC(?nBranches)
    """
    results = []
    for row in g.query(q):
        results.append({
            "iri": str(row.inv).split("#")[-1],
            "label": str(row.label),
            "fund_label": str(row.fundLabel) if row.fundLabel else "",
            "n_branches": int(float(str(row.nBranches))),
        })
    return results


# ============================================
# 3. 답변 템플릿
# ============================================

def format_amount(amount):
    if amount >= 100000000:
        return f"{amount // 100000000}억 원"
    return f"{amount:,}원"


def template_investment_status(meta, branches):
    """검토건 진행 상태 템플릿 (Q1~Q3 패턴)"""
    if not branches:
        return f"{meta.get('label', '해당 검토건')}의 분기 정보를 찾을 수 없습니다."
    
    fund_name = meta.get('fund', meta.get('label', '해당 검토건'))
    n_branches = len(branches)
    
    lines = [f"**[{fund_name} 검토 현황]**", ""]
    
    if n_branches == 1:
        lines.append(f"이 검토 건은 단일 분기로 진행 중입니다.")
    else:
        lines.append(f"이 검토 건은 총 {n_branches}개의 분기로 진행 중입니다.")
    lines.append("")
    
    for idx, b in enumerate(branches, start=1):
        amount_str = format_amount(b["amount"])
        lines.append(
            f"**분기 {idx}**: {b['product']} ({amount_str})"
        )
        lines.append(f"  - 현재 단계: {b['stage']}")
        lines.append(f"  - 브랜치 상태: {b['state']}")
        lines.append("")
    
    return "\n".join(lines).strip()


def template_stage_threshold(g, threshold_order=5):
    """약정·실행 도달 검토건 추출 (Q4 패턴)"""
    q = f"""
    PREFIX inv: <http://company.com/investment-ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?invLabel ?branchLabel ?stageLabel ?stateLabel ?stateOrder WHERE {{
      ?inv a inv:Investment ;
           rdfs:label ?invLabel ;
           inv:hasBranch ?branch .
      ?branch rdfs:label ?branchLabel ;
              inv:hasCurrentStage ?stage ;
              inv:hasBranchState ?state .
      ?stage rdfs:label ?stageLabel .
      ?state rdfs:label ?stateLabel ;
             inv:hasStateOrder ?stateOrder .
      FILTER(?stateOrder >= {threshold_order})
      FILTER(LANG(?invLabel)="ko")
      FILTER(LANG(?branchLabel)="ko")
      FILTER(LANG(?stageLabel)="ko")
      FILTER(LANG(?stateLabel)="ko")
    }} ORDER BY ?invLabel
    """
    
    results = []
    for row in g.query(q):
        results.append({
            "inv_label": str(row.invLabel),
            "branch_label": str(row.branchLabel),
            "stage": str(row.stageLabel),
            "state": str(row.stateLabel),
        })
    
    if not results:
        return "약정 단계 이후로 진행된 검토 건이 없습니다."
    
    lines = [f"**약정 단계 이후로 진행된 검토 건 ({len(results)}건)**", ""]
    for r in results:
        lines.append(f"- **{r['inv_label']}** / {r['branch_label']}")
        lines.append(f"  단계: {r['stage']}, 상태: {r['state']}")
    return "\n".join(lines)


def template_review_stalled(g):
    """정체 상태 검토건 추출 (Q5 패턴)"""
    q = """
    PREFIX inv: <http://company.com/investment-ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?invLabel ?branchLabel ?stageLabel WHERE {
      ?inv a inv:Investment ;
           rdfs:label ?invLabel ;
           inv:hasBranch ?branch .
      ?branch rdfs:label ?branchLabel ;
              inv:hasCurrentStage ?stage ;
              inv:hasBranchState inv:State_ReviewStalled .
      ?stage rdfs:label ?stageLabel .
      FILTER(LANG(?invLabel)="ko")
      FILTER(LANG(?branchLabel)="ko")
      FILTER(LANG(?stageLabel)="ko")
    } ORDER BY ?invLabel
    """
    
    results = []
    for row in g.query(q):
        results.append({
            "inv_label": str(row.invLabel),
            "branch_label": str(row.branchLabel),
            "stage": str(row.stageLabel),
        })
    
    if not results:
        return "현재 정체 상태인 검토 건이 없습니다."
    
    lines = [f"**예비검토 단계에서 정체된 검토 건 ({len(results)}건)**", ""]
    for r in results:
        lines.append(f"- **{r['inv_label']}** / {r['branch_label']}")
        lines.append(f"  단계: {r['stage']}, 상태: 예비검토 정체")
    return "\n".join(lines)


# ============================================
# 4. 결정론적 위험가중치 lookup (Q9 패턴)
# ============================================

def lookup_risk_weight(lookup, asset_class, credit_rating=None):
    """
    위험가중치 결정론적 조회 (Q9 패턴 — LLM 호출 없이 직접 표 조회).
    
    실제 데이터 구조:
        lookup = {
          "asset_classes": {
            "중앙정부": {
              "asset_id": "CentralGov",
              "clause_id": "BSER_App3_Asset_CentralGov",
              "lookup_method": "신용등급",
              "table": { "AAA~AA-": "0%", "A+~A-": "20%", ... },
              "special_rules": { ... }
            }, ...
          }
        }
    
    Args:
        lookup: 전체 lookup dict
        asset_class: 자산 분류 (예: "중앙정부")
        credit_rating: 신용등급 (예: "AAA~AA-", "AAA", "A+~A-")
    
    Returns:
        dict {
            "weight": "0%",          # 위험가중치 문자열
            "asset_class": "중앙정부",
            "credit_rating": "AAA~AA-",
            "matched_key": "AAA~AA-",
            "clause_id": "BSER_App3_Asset_CentralGov",
        }
        또는 None (매칭 실패)
    """
    asset_classes = lookup.get("asset_classes", {})
    asset_data = asset_classes.get(asset_class)
    if not asset_data:
        return None
    
    table = asset_data.get("table", {})
    
    # 1) 정확 매칭
    if credit_rating and credit_rating in table:
        return {
            "weight": table[credit_rating],
            "asset_class": asset_class,
            "credit_rating": credit_rating,
            "matched_key": credit_rating,
            "clause_id": asset_data.get("clause_id", ""),
        }
    
    # 2) 부분 매칭 (예: "AAA" → "AAA~AA-" 키 찾기)
    if credit_rating:
        for key in table.keys():
            # 사용자가 "AAA"라고 입력했고 키가 "AAA~AA-"라면 매칭
            if credit_rating in key:
                return {
                    "weight": table[key],
                    "asset_class": asset_class,
                    "credit_rating": credit_rating,
                    "matched_key": key,
                    "clause_id": asset_data.get("clause_id", ""),
                }
            # 키가 "AAA"이고 사용자가 "AAA~AA-"라고 입력했다면 매칭 (역방향)
            if key in credit_rating:
                return {
                    "weight": table[key],
                    "asset_class": asset_class,
                    "credit_rating": credit_rating,
                    "matched_key": key,
                    "clause_id": asset_data.get("clause_id", ""),
                }
    
    return None


def template_risk_weight_answer(result):
    """
    위험가중치 lookup 결과를 한국어 답변으로 포맷.
    
    Args:
        result: lookup_risk_weight() 반환값 (dict)
    """
    if result is None:
        return None
    
    return (
        f"**[위험가중치 조회 결과]**\n\n"
        f"- 자산 분류: **{result['asset_class']}**\n"
        f"- 신용등급: **{result['credit_rating']}** (매칭 키: `{result['matched_key']}`)\n"
        f"- 위험가중치: **{result['weight']}**\n\n"
        f"※ 출처: 은행업감독업무시행세칙 별표 3 (표준방법 기준)\n"
        f"※ 본 답변은 LLM 호출 없이 정형 lookup table에서 직접 조회되었습니다."
    )

def render_supporting_chunks(chunks_list, max_n=2, max_text=300):
    """위험가중치 답변에 보조로 붙일 RAG 청크 요약"""
    if not chunks_list:
        return ""
    lines = ["\n\n---\n\n**📚 보조 근거 (관련 규제 청크):**\n"]
    for c in chunks_list[:max_n]:
        cid = c.get("id", "?")
        text = c.get("text", "")[:max_text]
        lines.append(f"- **{cid}**: {text}{'...' if len(c.get('text', '')) > max_text else ''}")
    return "\n\n".join(lines)


# ============================================
# 5. RAG 청크 검색
# ============================================

def search_chunks_by_concept(chunks, concept_id, top_k=5, prefer_summary=False):
    """Concept ID로 RAG 청크 검색"""
    matched = []
    for c in chunks:
        meta = c.get("metadata", {})
        concepts = meta.get("regulatory_concepts", [])
        if concept_id in concepts:
            matched.append(c)
    
    # 요약 청크 우선 (Q10 패턴)
    if prefer_summary:
        summary = [c for c in matched if "Summary" in c.get("id", "") or c.get("metadata", {}).get("is_summary")]
        non_summary = [c for c in matched if c not in summary]
        matched = summary + non_summary
    
    return matched[:top_k]


def search_chunks_by_keyword(chunks, keywords, top_k=5):
    """키워드 매칭으로 청크 검색 (concept ID 매칭 실패 시 fallback)"""
    scored = []
    for c in chunks:
        text = c.get("text", "").lower()
        score = sum(1 for kw in keywords if kw.lower() in text)
        if score > 0:
            scored.append((score, c))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


# ============================================
# 6. Gemma 호출 (Streamlit 환경에서 지연 import)
# ============================================

def call_gemma(question, context, mode="standard", model_name=None):
    """
    Gemma 호출. ollama 패키지 + Ollama 서버 + 모델이 모두 준비되어야 함.
    어느 단계에서 실패해도 예외를 던지지 않고 None 또는 안내 문자열을 반환.
    
    Args:
        question: 사용자 질문
        context: 컨텍스트 (인스턴스 정보 + 청크)
        mode: "standard" 또는 "polish"
        model_name: 모델명. None이면 환경변수 OLLAMA_MODEL 또는 DEFAULT_MODEL 사용
    
    Returns:
        (answer_str, success_bool)
        - 성공: (Gemma 답변, True)
        - 실패: (None, False)  → 호출자가 fallback 처리해야 함
    """
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    try:
        import ollama
    except ImportError:
        return None, False
    
    if mode == "polish":
        # 다듬기 모드: 컨텍스트의 사실을 그대로 유지하며 자연스럽게 표현
        system = (
            "당신은 한국 LP출자 도메인 전문 금융회사 직원의 보조 AI입니다. "
            "아래 컨텍스트의 사실을 그대로 유지하면서 자연스러운 한국어로 답변을 다듬어 주세요. "
            "사실을 추가하거나 추측하지 마세요. 제공된 정보만 사용하세요."
        )
    else:
        # 표준 모드: 컨텍스트 기반 답변
        system = (
            "당신은 한국 LP출자 도메인 전문 금융회사 직원의 보조 AI입니다. "
            "아래 컨텍스트(온톨로지 + 규제 청크)의 정보만 사용하여 답변하세요. "
            "컨텍스트에 없는 내용은 추측하지 말고, 모르면 모른다고 답하세요. "
            "마크다운 강조(**)나 이모지를 과하게 사용하지 마세요. "
            "한국어 격식체로 간결하게 답변하세요."
        )
    
    full_prompt = f"[컨텍스트]\n{context}\n\n[질문]\n{question}\n\n[답변]"
    
    try:
        resp = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": full_prompt},
            ],
            options={"num_predict": 1500, "temperature": 0.3, "repeat_penalty": 1.15},
        )
        return resp['message']['content'], True
    except Exception:
        # 모델 미존재, 서버 미실행, 네트워크 오류 등 모두 여기서 처리
        return None, False


# ============================================
# 7. 라우터 (자유 질문 분류)
# ============================================

# 키워드 사전
PROGRESS_KEYWORDS = ["어디까지", "어떻게 진행", "현재", "진행 상태", "상태", "어디", "어느 단계", "단계가 어떻게"]
RWA_KEYWORDS = ["RWA", "위험가중치", "rwa", "익스포져", "익스포저", "자산분류"]
ALTERNATIVE_KEYWORDS = ["대체투자", "대체 투자"]
SUITABILITY_KEYWORDS = ["적합성", "적합성 원칙"]
EXPLANATION_KEYWORDS = ["설명의무", "설명 의무"]
CONSUMER_KEYWORDS = ["전문금융소비자", "일반금융소비자", "금융소비자"]
SCREENING_KEYWORDS = ["사전심사", "사전 심사", "사전협의", "사전 협의"]
PRODUCT_KEYWORDS = ["펀드 형태", "펀드 종류", "어떤 펀드", "LP출자 대상"]
THRESHOLD_KEYWORDS = ["약정 후", "약정 이후", "약정 단계 이후", "실행 단계", "약정 단계"]
STALLED_KEYWORDS = ["정체", "결재 안", "막힌", "지체", "올렸는데"]


def detect_concept_from_question(question):
    """질문에서 RegulatoryConcept ID 추출"""
    q = question.lower()
    
    if any(kw in question for kw in RWA_KEYWORDS):
        return "Concept_RWA_Calculation"
    if any(kw in question for kw in ALTERNATIVE_KEYWORDS):
        return "Concept_AlternativeInvestmentClassification"
    if any(kw in question for kw in SUITABILITY_KEYWORDS):
        return "Concept_SuitabilityCheck"
    if any(kw in question for kw in EXPLANATION_KEYWORDS):
        return "Concept_ExplanationDuty"
    if any(kw in question for kw in CONSUMER_KEYWORDS):
        return "Concept_ConsumerClassification"
    if any(kw in question for kw in SCREENING_KEYWORDS):
        return None  # 사전심사 단계는 stage_overview로 처리
    if any(kw in question for kw in PRODUCT_KEYWORDS):
        return "Concept_ProductDefinition"
    
    return None


def detect_risk_weight_query(question, lookup):
    """
    위험가중치 질의 자동 감지 + 자산분류·신용등급 추출.
    
    Returns:
        dict {"asset_class": str, "credit_rating": str|None} 또는 None
    """
    # RWA/위험가중치 키워드가 없으면 무관한 질문
    if not any(kw in question for kw in RWA_KEYWORDS):
        return None
    
    # 자산분류 탐지 (lookup의 asset_classes 키 사용)
    asset_classes = lookup.get("asset_classes", {})
    asset_class = None
    for asset_name in asset_classes.keys():
        if asset_name in question:
            asset_class = asset_name
            break
    
    if not asset_class:
        return None
    
    # 신용등급 탐지 - 실제 lookup table 키와 사용자 표현을 모두 시도
    table_keys = list(asset_classes[asset_class].get("table", {}).keys())
    credit_rating = None
    
    # 1) lookup table의 실제 키 직접 매칭
    for key in table_keys:
        if key in question:
            credit_rating = key
            break
    
    # 2) 일반 신용등급 패턴 매칭 (사용자가 "AAA"만 입력한 경우 등)
    if not credit_rating:
        # 긴 패턴부터 매칭 (AAA가 AA보다 먼저)
        rating_patterns = [
            "AAA~AA-", "AAA", "AA-", "AA+", "AA",
            "A+~A-", "A+", "A-", "A",
            "BBB+~BBB-", "BBB+", "BBB-", "BBB",
            "BB+~B-", "BB+", "BB-", "BB",
            "B-미만", "B-이하", "B-",
            "투자등급", "투기등급", "무등급",
        ]
        for pattern in rating_patterns:
            if pattern in question:
                credit_rating = pattern
                break
    
    return {"asset_class": asset_class, "credit_rating": credit_rating}


# ============================================
# 8. 메인 라우팅 함수
# ============================================

def answer_question(question, g, chunks, alias_dict, lookup, user_instances=None,
                    use_gemma=True, model_name="gemma2:2b"):
    """
    자유 질문 처리 메인 함수.
    
    Args:
        question: 사용자 질문 (자연어)
        g: rdflib Graph (TTL + 사용자 추가 인스턴스 포함)
        chunks: RAG 청크 리스트
        alias_dict: 동의어 사전
        lookup: 위험가중치 lookup
        user_instances: 사용자가 추가한 인스턴스 record 리스트 (instance_manager.add_user_investment 결과)
        use_gemma: Gemma 호출 여부 (False면 템플릿/lookup만, 빠른 시연용)
        model_name: Ollama 모델명
    
    Returns:
        dict: {
            "answer": str,           # 최종 답변
            "route": str,            # 라우팅 결과 (디버깅용)
            "context_summary": str,  # 사용된 컨텍스트 요약
        }
    """
    # ============================================
    # 1. 펀드명 매칭 — 사용자 추가 인스턴스 우선
    # ============================================
    matched_investment = None
    matched_source = None  # "user" 또는 "demo"
    
    # 1-1. 사용자가 추가한 인스턴스 매칭
    if user_instances:
        from instance_manager import find_user_investment_by_keyword
        # 질문에서 펀드명 후보 추출 (간단히 단어 단위로)
        question_words = re.findall(r'[가-힣A-Za-z0-9]{2,}', question)
        for word in question_words:
            record = find_user_investment_by_keyword(user_instances, word)
            if record:
                matched_investment = record["investment_iri"]
                matched_source = "user"
                break
    
    # 1-2. 데모 인스턴스 매칭
    # 일반 단어 (질문 의도 키워드)는 매칭 후보에서 제외
    GENERIC_WORDS = {
        "검토", "검토건", "검토 건", "단계", "진행", "어디", "상태", "약정", "출자",
        "어떻게", "지금", "현재", "상황", "처리", "이후", "올렸", "결재", "정체",
        "관리", "신청", "등록", "승인", "산정", "분류", "원칙", "기준",
        "펀드", "투자", "금융", "규제", "조항", "법령", "계약", "체결",
        "예비", "사전", "협의", "실무", "품의", "대체", "적합성", "설명",
        "그리고", "또는", "그래서", "하지만", "지금까지", "그동안",
        "RWA", "rwa", "위험가중치", "익스포져", "익스포저", "자산", "자산분류",
        "lp출자", "LP출자", "lp", "LP", "한도", "공여", "신용", "전문",
        "가상", "기준", "조건", "방법",
    }
    
    if not matched_investment:
        all_investments = query_all_investments_with_label(g)
        for inv in all_investments:
            # 빈 껍데기 인스턴스 (분기 0개) 스킵
            if inv.get("n_branches", 0) == 0:
                continue
            # 이미 매칭된 사용자 인스턴스면 스킵
            if user_instances and any(r["investment_iri"] == inv["iri"] for r in user_instances):
                continue
            
            # label/fund_label에서 단어 추출, 일반 단어 제거
            label_words = re.findall(r"[가-힣A-Za-z0-9]{2,}", inv["label"] + " " + inv["fund_label"])
            distinctive_words = [w for w in label_words if w not in GENERIC_WORDS]
            
            for word in distinctive_words:
                if len(word) >= 2 and word in question:
                    matched_investment = inv["iri"]
                    matched_source = "demo"
                    break
            if matched_investment:
                break
    
    # ============================================
    # 2. 라우팅
    # ============================================
    
    # 2-1. 약정 단계 도달 검토건 (Q4 패턴) — 펀드명 매칭이 없을 때만
    if not matched_investment and any(kw in question for kw in THRESHOLD_KEYWORDS):
        answer = template_stage_threshold(g, threshold_order=1)
        return {
            "answer": answer,
            "route": "stage_threshold",
            "context_summary": "약정·실행 도달 검토건 SPARQL",
        }
    
    # 2-2. 정체 상태 (Q5 패턴) — 펀드명 매칭이 없을 때만
    if not matched_investment and any(kw in question for kw in STALLED_KEYWORDS):
        answer = template_review_stalled(g)
        return {
            "answer": answer,
            "route": "review_stalled",
            "context_summary": "정체 상태 검토건 SPARQL",
        }
    
    # 2-3. 위험가중치 (Q9 패턴) — Lookup 우선, RAG 청크는 보조 근거
    # 페이퍼 §4.5의 핵심 메시지: "결정론적 사실은 코드가 보장, 자연어만 LLM"
    # - asset_class 감지되면 무조건 lookup 시도 (LLM 호출 없음)
    # - 매칭 성공: lookup 결과를 메인 답변으로, 관련 청크는 보조 근거로 첨부
    # - 매칭 실패: 보조 안내 + 일반 RWA 청크로 fallback
    rw_query = detect_risk_weight_query(question, lookup)
    if rw_query:
        result = lookup_risk_weight(lookup, rw_query["asset_class"], rw_query["credit_rating"])
        
        if result is not None:
            # 메인 답변: lookup 결과 (LLM 호출 없음)
            main_answer = template_risk_weight_answer(result)
            
            # 보조 근거: 관련 RAG 청크 1~2개
            supporting = search_chunks_by_concept(chunks, "Concept_RWA_Calculation",
                                                  top_k=2, prefer_summary=True)
            supporting_text = render_supporting_chunks(supporting, max_n=2, max_text=300)
            
            return {
                "answer": main_answer + supporting_text,
                "route": "deterministic_lookup",
                "context_summary": (
                    f"Lookup 직접 조회: {result['asset_class']} / {result['credit_rating']} "
                    f"(매칭: {result['matched_key']}) "
                    f"+ 보조 청크 {len(supporting)}개"
                ),
            }
        else:
            # 자산 분류는 감지됐으나 신용등급이 명확하지 않은 경우
            # → asset_class 표 전체를 안내 + RAG 청크
            asset_classes = lookup.get("asset_classes", {})
            asset_data = asset_classes.get(rw_query["asset_class"], {})
            table = asset_data.get("table", {})
            
            table_lines = [f"**[{rw_query['asset_class']} 자산 분류 위험가중치 표]**", ""]
            for k, v in table.items():
                table_lines.append(f"- {k}: **{v}**")
            
            supporting = search_chunks_by_concept(chunks, "Concept_RWA_Calculation",
                                                  top_k=2, prefer_summary=True)
            supporting_text = render_supporting_chunks(supporting, max_n=2, max_text=300)
            
            note = "\n\n신용등급을 명확히 지정하시면 정확한 가중치를 조회할 수 있습니다 (예: 'AAA~AA-')."
            
            return {
                "answer": "\n".join(table_lines) + note + supporting_text,
                "route": "lookup_table_overview",
                "context_summary": f"Lookup 표 전체: {rw_query['asset_class']} + 보조 청크 {len(supporting)}개",
            }
    
    # 2-4. 펀드명 매칭 + 진행상태 (Q1~Q3 패턴)
    if matched_investment and any(kw in question for kw in PROGRESS_KEYWORDS):
        meta = query_investment_meta(g, matched_investment)
        branches = query_investment_branches(g, matched_investment)
        answer = template_investment_status(meta, branches)
        return {
            "answer": answer,
            "route": f"investment_status ({matched_source})",
            "context_summary": f"인스턴스 SPARQL: {matched_investment}",
        }
    
    # 2-5. 펀드명 매칭 + 규제 키워드 (인스턴스 + 청크 컨텍스트)
    concept_id = detect_concept_from_question(question)
    
    if matched_investment and concept_id:
        # 인스턴스 컨텍스트 + 청크 컨텍스트 결합
        meta = query_investment_meta(g, matched_investment)
        branches = query_investment_branches(g, matched_investment)
        
        instance_context = template_investment_status(meta, branches)
        
        prefer_summary = (concept_id == "Concept_RWA_Calculation")
        relevant_chunks = search_chunks_by_concept(chunks, concept_id, top_k=3, prefer_summary=prefer_summary)
        
        chunk_context = "\n\n".join([
            f"[규제 근거 {i+1}] {c.get('id', '')}\n{c.get('text', '')[:1500]}"
            for i, c in enumerate(relevant_chunks)
        ])
        
        full_context = f"{instance_context}\n\n---\n\n[관련 규제 근거]\n{chunk_context}"
        
        if use_gemma:
            gemma_answer, gemma_ok = call_gemma(question, full_context, mode="standard", model_name=model_name)
        else:
            gemma_answer, gemma_ok = None, False
        
        if gemma_ok:
            answer = gemma_answer
        else:
            # Gemma 미사용/실패 시: 시연 모드 (구조화된 raw 컨텍스트 표시)
            answer = (
                f"{instance_context}\n\n"
                f"---\n\n"
                f"**관련 규제 근거 ({len(relevant_chunks)}개 청크 검색됨):**\n\n"
                + "\n\n".join([f"📄 **{c.get('id', '')}**\n{c.get('text', '')[:500]}..." for c in relevant_chunks])
            )
        
        return {
            "answer": answer,
            "route": f"instance_with_concept ({matched_source}, {concept_id})",
            "context_summary": f"인스턴스 + {len(relevant_chunks)}개 규제 청크",
        }
    
    # 2-6. 일반 규제 질문 (펀드명 없음, Q6~Q8 패턴)
    if concept_id:
        prefer_summary = (concept_id == "Concept_RWA_Calculation")
        relevant_chunks = search_chunks_by_concept(chunks, concept_id, top_k=3, prefer_summary=prefer_summary)
        
        if not relevant_chunks:
            return {
                "answer": "관련 규제 정보를 찾을 수 없습니다. 다른 키워드로 다시 질문해주세요.",
                "route": "rag_concept_no_match",
                "context_summary": f"concept {concept_id} 청크 0개",
            }
        
        chunk_context = "\n\n".join([
            f"[근거 {i+1}] {c.get('id', '')}\n{c.get('text', '')[:1500]}"
            for i, c in enumerate(relevant_chunks)
        ])
        
        if use_gemma:
            gemma_answer, gemma_ok = call_gemma(question, chunk_context, mode="standard", model_name=model_name)
        else:
            gemma_answer, gemma_ok = None, False
        
        if gemma_ok:
            answer = gemma_answer
        else:
            answer = (
                f"**관련 규제 근거 ({len(relevant_chunks)}개 청크 검색됨):**\n\n"
                + "\n\n".join([f"📄 **{c.get('id', '')}**\n{c.get('text', '')[:500]}..." for c in relevant_chunks])
            )
        
        return {
            "answer": answer,
            "route": f"rag_concept ({concept_id})",
            "context_summary": f"{len(relevant_chunks)}개 규제 청크",
        }
    
    # 2-7. 매칭 안 됨 — 안내
    fallback = (
        "이 시스템은 다음과 같은 질문에 답변할 수 있습니다:\n\n"
        "**1. 검토건 진행 상태** (예시 검토건 또는 직접 등록한 검토건 대상)\n"
        "  • \"ABC펀드 검토건은 어디까지 갔어?\"\n"
        "  • \"방금 등록한 펀드는 단계가 어떻게 돼?\"\n\n"
        "**2. 단계별 검토건 추출**\n"
        "  • \"약정 단계 이후로 진행된 검토건은?\"\n"
        "  • \"정체된 검토건 있어?\"\n\n"
        "**3. 위험가중치 조회**\n"
        "  • \"AAA 등급 중앙정부 익스포져 위험가중치는?\"\n\n"
        "**4. 규제 설명**\n"
        "  • \"LP출자한 펀드의 RWA는 어떻게 산정해?\"\n"
        "  • \"적합성 원칙은 어떻게 확인해?\"\n"
        "  • \"대체투자 분류 기준은?\"\n\n"
        "📌 위 범위 외의 질문(일반 금융 상식, 회사 실제 데이터 등)에는 정확한 답변이 어렵습니다."
    )
    
    return {
        "answer": fallback,
        "route": "guidance",
        "context_summary": "매칭 실패, 가이드 응답",
    }
