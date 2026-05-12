"""
demo_app.py
===========
LP출자 온톨로지 RAG 시스템 — 공모전 라이브 시연 콘솔

설계 원칙
- 기존 app.py, rag_engine.py, instance_manager.py, data/는 수정하지 않는다.
- 사이드바를 쓰지 않고 모든 제어를 메인 콘솔에 통합한다.
- 질문 예시 클릭 → 입력창 자동 채우기는 pending_question 패턴으로만 처리한다.
- Streamlit 위젯 key 값을 위젯 생성 이후 직접 수정하지 않는다.
"""

from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

import rag_engine
from instance_manager import (
    PRODUCT_TYPES,
    STATES_SIMPLE,
    STAGES_FULL,
    add_user_investment,
    list_all_user_investments,
)

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="온톨로지 RAG 시연 콘솔",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
STYLE_PATH = APP_DIR / "style.css"
GROUND_TRUTH_PATH = APP_DIR / "ground_truth.json"

BUSINESS_STAGE_LABELS = {
    "CustomerRegistration": "고객 등록",
    "InvestmentConsultation": "투자 상담",
    "PreliminaryScreening": "사전 협의 신청",
    "PreliminaryReview": "예비 검토",
    "WorkingLevelReview": "실무협의회",
    "InvestmentProposal": "개별 품의",
    "LimitCommitment": "한도 약정",
    "DrawdownProposal": "개별 실행",
    "PostManagement": "사후 관리",
}

DEFAULT_QUESTION = "ABC펀드 검토건은 지금 어디까지 갔어?"
MODEL_OPTIONS = ["Gemma 4 e4b 로컬", "Sonnet 4.6 API", "구조화 모드"]

PRESET_GROUPS = [
    {
        "title": "출자 전산 진행 상황",
        "items": [
            ("Q1", "검토 건 상태 조회", "ABC펀드", "ABC펀드 검토건은 지금 어디까지 갔어?"),
            ("Q2", "인수금융 병행건 조회", "1q2w", "1q2w 펀드 검토건은 지금 어떻게 진행 중이야?"),
            ("Q3", "검토 건 상태 조회", "babymonster", "babymonster 검토건은 어디까지 갔어?"),
        ],
    },
    {
        "title": "특정 단계 현황 조회",
        "items": [
            ("Q4", "약정 단계 조회", "", "약정 단계 이후로 진행된 검토건은?"),
            ("Q5", "예비검토 단계 조회", "", "예비검토 올렸는데 결재 안 난 건들은?"),
        ],
    },
    {
        "title": "규제 조회",
        "items": [
            ("Q6", "LP출자 관련 법률", "", "LP출자 검토 시 적합성 원칙은 어떻게 확인해?"),
            ("Q7", "대체투자 분류 기준", "", "대체투자 분류 기준은?"),
            ("Q8", "확인 필요 거래상대방", "", "거래상대방을 식별할 때 누구를 고려해야 해?"),
            ("Q9", "세부 조건", "위험가중치", "AAA 등급 중앙정부 익스포져의 위험가중치는 얼마야?"),
            ("Q10", "자연어 조건 판별", "", "LP출자한 펀드의 RWA는 어떻게 산정해?"),
        ],
    },
]

# -----------------------------------------------------------------------------
# Assets and state
# -----------------------------------------------------------------------------


def inject_assets() -> None:
    """Load external fonts and local CSS."""
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        """,
        unsafe_allow_html=True,
    )
    if STYLE_PATH.exists():
        st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def init_state() -> None:
    defaults = {
        "active_tab": "demo",
        "model_choice": "구조화 모드",
        "show_examples": False,
        "pending_question": None,
        "last_question": "",
        "last_left": None,
        "last_right": None,
        "user_instances": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    if "graph" not in st.session_state:
        st.session_state.graph = rag_engine.load_ttl(DATA_DIR / "investment_ontology_v1_10.ttl")

    # 질문 예시 버튼은 text_area 생성 이후 widget state를 바꾸지 않는다.
    # pending_question을 rerun 직후, text_area 생성 전에만 question_input으로 반영한다.
    pending = st.session_state.get("pending_question")
    if pending:
        st.session_state["question_input"] = pending
        st.session_state["pending_question"] = None


@st.cache_data(show_spinner=False)
def load_chunks() -> List[Dict[str, Any]]:
    return rag_engine.load_chunks(DATA_DIR / "regulations_chunks_v14.jsonl")


@st.cache_data(show_spinner=False)
def load_alias() -> Dict[str, Any]:
    return rag_engine.load_alias(DATA_DIR / "alias_dictionary.json")


@st.cache_data(show_spinner=False)
def load_lookup() -> Dict[str, Any]:
    return rag_engine.load_lookup(DATA_DIR / "risk_weight_lookup.json")


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def render_html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def compact_text(text: str, limit: int = 900) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def chunk_text(chunk: Dict[str, Any]) -> str:
    return chunk.get("text") or chunk.get("content") or ""


def chunk_metadata(chunk: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not chunk:
        return {}
    return chunk.get("metadata") or {}


def format_list(values: Any, mapping: Optional[Dict[str, str]] = None) -> str:
    if not values:
        return "-"
    if isinstance(values, str):
        values = [values]
    mapped = [mapping.get(v, v) if mapping else v for v in values]
    return ", ".join(str(v) for v in mapped if v)


def get_law_names(chunks: List[Dict[str, Any]]) -> List[str]:
    names = sorted({c.get("metadata", {}).get("law_name", "") for c in chunks if c.get("metadata", {}).get("law_name")})
    return names


def search_rag_chunks(question: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """Very small deterministic lexical search for RAG-only comparison."""
    question_l = question.lower()
    tokens = [t.lower() for t in question.replace("?", " ").replace("/", " ").split() if len(t) >= 2]
    concept_hints = []
    if any(k in question for k in ["적합성", "일반금융소비자", "전문금융소비자"]):
        concept_hints.append("Concept_SuitabilityCheck")
    if any(k in question for k in ["대체투자", "대체 투자"]):
        concept_hints.append("Concept_AlternativeInvestmentClassification")
    if any(k in question for k in ["거래상대방", "GP", "피투자", "상대방"]):
        concept_hints.append("Concept_CounterpartyIdentification")
    if any(k in question for k in ["RWA", "rwa", "위험가중", "집합투자증권", "익스포져", "익스포저"]):
        concept_hints.append("Concept_RWA_Calculation")

    scored: List[Tuple[int, Dict[str, Any]]] = []
    for chunk in chunks:
        text = chunk_text(chunk)
        meta = chunk_metadata(chunk)
        haystack = (text + " " + json.dumps(meta, ensure_ascii=False)).lower()
        score = 0
        for token in tokens:
            if token in haystack:
                score += 2
        for concept in concept_hints:
            if concept in meta.get("regulatory_concepts", []):
                score += 10
        if "lp출자" in question_l and "LP" in haystack:
            score += 1
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def get_primary_metadata(question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    matches = search_rag_chunks(question, chunks, top_k=1)
    return chunk_metadata(matches[0]) if matches else {}


def run_rag_only(question: str, chunks: List[Dict[str, Any]], model_choice: str) -> Dict[str, Any]:
    start = time.perf_counter()
    found = search_rag_chunks(question, chunks, top_k=3)
    elapsed = time.perf_counter() - start

    if not found:
        answer = "관련 RAG 청크를 찾지 못했습니다. 본 프로토타입의 지원 범위 내 질문인지 확인해 주십시오."
        preview = "-"
    else:
        lead = found[0]
        answer = (
            "RAG only 비교군은 온톨로지/SPARQL/Lookup을 사용하지 않고, "
            "검색된 청크 텍스트만 기준으로 답변 후보를 구성합니다.\n\n"
            f"{compact_text(chunk_text(lead), 520)}"
        )
        preview = "\n\n---\n\n".join(compact_text(chunk_text(c), 520) for c in found[:2])

    return {
        "answer": answer,
        "route": "- (없음)",
        "context_summary": f"단순 청크 검색 {len(found)}개",
        "elapsed": elapsed,
        "chunks": found,
        "preview": preview,
    }


def run_left_system(question: str, graph: Any, chunks: List[Dict[str, Any]], alias: Dict[str, Any], lookup: Dict[str, Any], model_choice: str) -> Dict[str, Any]:
    use_gemma = model_choice == "Gemma 4 e4b 로컬"
    # 1차 안정화: Sonnet은 아직 rag_engine 내부 래퍼가 없으므로 구조화 모드로 처리한다.
    if model_choice == "Sonnet 4.6 API":
        use_gemma = False

    start = time.perf_counter()
    result = rag_engine.answer_question(
        question,
        graph,
        chunks,
        alias,
        lookup,
        user_instances=st.session_state.get("user_instances", []),
        use_gemma=use_gemma,
        model_name="gemma4:e4b",
    )
    elapsed = time.perf_counter() - start
    result = dict(result)
    result["elapsed"] = elapsed
    result["metadata"] = get_primary_metadata(question, chunks)
    return result


# -----------------------------------------------------------------------------
# HTML component renderers
# -----------------------------------------------------------------------------


def render_header() -> None:
    render_html(
        """
<section class="app-hero">
  <div>
    <div class="eyebrow">IBK Capital · LP Investment Knowledge System</div>
    <h1>온톨로지 RAG 시연 콘솔</h1>
    <p>LP출자 검토 업무를 RDF/TTL 온톨로지, RAG 청크, 결정론적 Lookup으로 구조화한 공모전 시연 앱입니다.</p>
  </div>
  <div class="hero-badge">SIDE-BY-SIDE<br><span>TRACEABLE</span></div>
</section>
        """
    )


def render_model_and_kpi(model_choice: str) -> str:
    render_html("<section class='console-panel top-panel'>")
    c1, c2, c3, c4, c5 = st.columns([1.6, 1, 1, 1, 1])
    with c1:
        st.session_state.model_choice = st.radio(
            "모델 선택",
            MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(model_choice),
            key="model_radio",
            horizontal=False,
            label_visibility="visible",
        )
        selected = st.session_state.model_choice
        if selected == "Gemma 4 e4b 로컬" and os.environ.get("STREAMLIT_CLOUD"):
            st.warning("Gemma 4 e4b 로컬 모델은 로컬 Ollama 환경에서만 사용 가능합니다.")
        elif selected == "Sonnet 4.6 API" and not os.environ.get("ANTHROPIC_API_KEY"):
            st.info("Sonnet 4.6 API는 2차 래퍼 연결 대상입니다. 현재는 구조화 모드로 시연됩니다.")
    for col, label, value, sub, extra_class in [
        (c2, "ONTOLOGY", "v1.10", "3,726 triples", ""),
        (c3, "RAG CHUNKS", "274", "legal chunks", ""),
        (c4, "LOOKUP CLASSES", "9", "deterministic", ""),
        (c5, "SIDE-BY-SIDE", "ON", "comparison", "success"),
    ]:
        with col:
            render_html(
                f"""
<div class="kpi-card {extra_class}">
  <div class="kpi-label">{esc(label)}</div>
  <div class="kpi-value">{esc(value)}</div>
  <div class="kpi-sub">{esc(sub)}</div>
</div>
                """
            )
    render_html("</section>")
    return st.session_state.model_choice


def render_scope_notice() -> None:
    render_html(
        """
<section class="notice-box">
  <div class="notice-title">📋 시연 가능 범위</div>
  <div class="scope-grid">
    <div>✓ 검토건 진행 상태/단계 조회</div>
    <div>✓ LP출자 거래상대방 분류 (적합성)</div>
    <div>✓ 대체투자 분류 (LBA/MBA/FBA)</div>
    <div>✓ 위험가중치 산정 (집합투자증권)</div>
  </div>
  <div class="notice-warn">⚠ 일반 금융상식·실시간 시장 정보는 답변 범위에서 제외됩니다.</div>
</section>
        """
    )


def set_tab(tab_name: str) -> None:
    st.session_state.active_tab = tab_name


def render_tab_bar() -> str:
    current = st.session_state.get("active_tab", "demo")
    cols = st.columns(3)
    tabs = [("demo", "💬 시연"), ("manage", "📝 검토건 관리"), ("info", "ℹ️ 시스템 정보")]
    for col, (key, label) in zip(cols, tabs):
        with col:
            active = "active" if current == key else ""
            if st.button(label, key=f"tab_{key}", use_container_width=True):
                set_tab(key)
                st.rerun()
            render_html(f"<div class='tab-marker {active}'></div>")
    return st.session_state.get("active_tab", "demo")


def render_examples() -> None:
    toggle_label = "📋 질문 예시 보기 ▲" if st.session_state.show_examples else "📋 질문 예시 보기 ▼"
    left, right = st.columns([1, 1])
    with left:
        if st.button(toggle_label, key="toggle_examples", use_container_width=True):
            st.session_state.show_examples = not st.session_state.show_examples
            st.rerun()
    with right:
        if st.button("▶ 질문 실행", key="run_question", use_container_width=True, type="primary"):
            question = st.session_state.get("question_input", "").strip()
            if question:
                execute_question(question)
            else:
                st.session_state.last_left = None
                st.session_state.last_right = None
                st.warning("질문을 입력해 주세요.")

    if not st.session_state.show_examples:
        return

    render_html("<section class='examples-panel'>")
    for group in PRESET_GROUPS:
        render_html(f"<div class='example-group-title'>{esc(group['title'])}</div>")
        cols = st.columns(3)
        for idx, (qid, title, detail, question) in enumerate(group["items"]):
            with cols[idx % 3]:
                label = f"{qid}  {title}" if not detail else f"{qid}  {title}\n{detail}"
                if st.button(label, key=f"preset_{qid}", use_container_width=True):
                    st.session_state["pending_question"] = question
                    st.rerun()
    render_html("</section>")


def execute_question(question: str) -> None:
    graph = st.session_state.graph
    chunks = load_chunks()
    alias = load_alias()
    lookup = load_lookup()
    model_choice = st.session_state.get("model_choice", "구조화 모드")

    left = run_left_system(question, graph, chunks, alias, lookup, model_choice)
    right = run_rag_only(question, chunks, model_choice)

    st.session_state.last_question = question
    st.session_state.last_left = left
    st.session_state.last_right = right


def render_answer_markdown(answer: str) -> None:
    if not answer:
        st.markdown("질문을 실행하면 답변이 표시됩니다.")
        return
    st.markdown(answer)


def render_audit_left(result: Optional[Dict[str, Any]]) -> None:
    meta = (result or {}).get("metadata", {}) if result else {}
    with st.expander("🔍 AI 추론 근거 보기", expanded=False):
        rows = [
            ("📚 참조 법령", meta.get("article_label") or "-"),
            ("🏷️ 규제 개념", format_list(meta.get("regulatory_concepts"))),
            ("📌 관련 업무 단계", format_list(meta.get("business_stages"), BUSINESS_STAGE_LABELS)),
            ("🔑 핵심 개념", format_list(meta.get("legal_concepts"))),
            ("Route", (result or {}).get("route", "-")),
            ("Context summary", (result or {}).get("context_summary", "-")),
        ]
        for label, value in rows:
            render_html(f"<div class='audit-row'><div class='audit-label'>{esc(label)}</div><div class='audit-value'>{esc(value)}</div></div>")


def render_audit_right(result: Optional[Dict[str, Any]]) -> None:
    with st.expander("🔍 AI 추론 근거 보기", expanded=False):
        preview = (result or {}).get("preview", "-") if result else "-"
        render_html("<div class='audit-row'><div class='audit-label'>📄 참조 청크</div></div>")
        st.caption(preview)


def render_comparison_cards() -> None:
    left_result = st.session_state.get("last_left")
    right_result = st.session_state.get("last_right")

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        render_html(
            """
<div class="answer-card ontology-card">
  <div class="card-head">
    <div>
      <div class="card-title">🧩 온톨로지 + RAG</div>
      <div class="card-subtitle">근거 추적 가능 · 검토건/규제/Lookup</div>
    </div>
    <div class="pill traceable">TRACEABLE</div>
  </div>
</div>
            """
        )
        meta = "질문 대기"
        if left_result:
            meta = f"{left_result.get('elapsed', 0):.2f}s · {left_result.get('route', '-')}"
        render_html(f"<div class='meta-line'>{esc(meta)}</div>")
        render_html("<div class='answer-box'>")
        if left_result:
            render_answer_markdown(left_result.get("answer", ""))
        else:
            st.info("질문을 실행하면 좌측에 온톨로지+RAG 답변이 표시됩니다.")
        render_html("</div>")
        render_audit_left(left_result)

    with col_right:
        render_html(
            """
<div class="answer-card baseline-card">
  <div class="card-head">
    <div>
      <div class="card-title baseline-title">📄 RAG only (비교군)</div>
      <div class="card-subtitle">근거 추적 불가 · 청크 단순 검색</div>
    </div>
    <div class="pill baseline">BASELINE</div>
  </div>
</div>
            """
        )
        meta = "질문 대기"
        if right_result:
            meta = f"{right_result.get('elapsed', 0):.2f}s · - (없음)"
        render_html(f"<div class='meta-line'>{esc(meta)}</div>")
        render_html("<div class='answer-box'>")
        if right_result:
            render_answer_markdown(right_result.get("answer", ""))
        else:
            st.info("질문을 실행하면 우측에 RAG only 비교군 답변이 표시됩니다.")
        render_html("</div>")
        render_audit_right(right_result)


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------


def render_demo_tab() -> None:
    render_html("<section class='question-panel'>")
    st.text_area(
        "QUESTION",
        key="question_input",
        height=78,
        placeholder="예: ABC펀드 검토건은 지금 어디까지 갔어?",
        label_visibility="visible",
    )
    render_examples()
    render_html("</section>")
    render_comparison_cards()


def render_manage_tab() -> None:
    render_html("<section class='console-panel'>")
    st.subheader("📝 검토건 관리")
    records = list_all_user_investments(st.session_state.get("user_instances", []))
    if records:
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
    else:
        st.info("현재 세션에 추가된 검토건이 없습니다.")

    st.markdown("### 새 검토건 추가")
    with st.form("new_investment_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            fund_name = st.text_input("펀드명", placeholder="예: 테스트 신기술조합")
            gp_name = st.text_input("GP", placeholder="예: 테스트운용")
            amount = st.number_input("금액(억)", min_value=1, max_value=10000, value=10, step=1)
        with c2:
            stage_label_to_iri = {label: iri for iri, label in STAGES_FULL}
            state_label_to_iri = {label: iri for iri, label, _ in STATES_SIMPLE}
            product_label_to_iri = {label: iri for iri, label in PRODUCT_TYPES}
            product_label = st.selectbox("상품/분기 유형", list(product_label_to_iri.keys()), index=0)
            stage_label = st.selectbox("단계", list(stage_label_to_iri.keys()), index=1 if len(stage_label_to_iri) > 1 else 0)
            state_label = st.selectbox("상태", list(state_label_to_iri.keys()), index=0)
        submitted = st.form_submit_button("등록", use_container_width=True)
        if submitted:
            if not fund_name.strip() or not gp_name.strip():
                st.error("펀드명과 GP를 입력해 주세요.")
            else:
                record = add_user_investment(
                    st.session_state.graph,
                    fund_name_kr=fund_name.strip(),
                    gp_name_kr=gp_name.strip(),
                    fund_type_iri="RecipientFund",
                    branches_data=[
                        {
                            "product_iri": product_label_to_iri[product_label],
                            "amount": int(amount),
                            "stage_iri": stage_label_to_iri[stage_label],
                            "state_iri": state_label_to_iri[state_label],
                        }
                    ],
                )
                st.session_state.user_instances.append(record)
                st.success("검토건을 세션에 등록했습니다.")
                st.rerun()

    if st.button("🔄 세션 초기화", use_container_width=True):
        st.session_state.graph = rag_engine.load_ttl(DATA_DIR / "investment_ontology_v1_10.ttl")
        st.session_state.user_instances = []
        st.session_state.last_left = None
        st.session_state.last_right = None
        st.session_state.last_question = ""
        st.success("세션 검토건과 최근 답변을 초기화했습니다.")
        st.rerun()
    render_html("</section>")


def render_info_tab(chunks: List[Dict[str, Any]], lookup: Dict[str, Any]) -> None:
    render_html("<section class='console-panel'>")
    st.subheader("ℹ️ 시스템 정보")

    c1, c2, c3 = st.columns(3)
    with c1:
        render_html("<div class='info-card'><div class='kpi-label'>Ontology TTL</div><div class='kpi-value'>v1.10</div><div class='kpi-sub'>3,726 triples</div></div>")
    with c2:
        render_html(f"<div class='info-card'><div class='kpi-label'>RAG chunks</div><div class='kpi-value'>{len(chunks)}</div><div class='kpi-sub'>regulations_chunks_v14</div></div>")
    with c3:
        n_lookup = len((lookup.get("asset_classes") or {}).keys())
        render_html(f"<div class='info-card'><div class='kpi-label'>Lookup classes</div><div class='kpi-value'>{n_lookup}</div><div class='kpi-sub'>risk_weight_lookup</div></div>")

    st.markdown("### 🧩 9단계 업무 프로세스")
    stage_df = pd.DataFrame([{"영문": k, "한글": v} for k, v in BUSINESS_STAGE_LABELS.items()])
    st.dataframe(stage_df, use_container_width=True, hide_index=True)

    st.markdown("### 📚 참조 법령 목록")
    law_names = get_law_names(chunks)
    if law_names:
        st.markdown(" · ".join(f"`{name}`" for name in law_names))
    else:
        st.caption("law_name 메타데이터를 찾지 못했습니다.")

    st.markdown("### 🏛️ 5개 분류 원칙")
    principles = [
        "펀드는 전 케이스 전문금융소비자",
        "법인은 일반적 전문, 케이스별 판단",
        "GP는 전 케이스 전문",
        "LP출자는 무조건 집합투자증권 익스포져 (RWA)",
        "LP출자 대체투자 분류는 펀드 최종 투자대상에 따른 조건부 분류",
    ]
    for i, item in enumerate(principles, start=1):
        st.markdown(f"{i}. {item}")
    render_html("</section>")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    inject_assets()
    init_state()

    chunks = load_chunks()
    lookup = load_lookup()

    render_html("<main class='app-shell'>")
    render_header()
    model_choice = render_model_and_kpi(st.session_state.get("model_choice", "구조화 모드"))
    st.session_state.model_choice = model_choice
    render_scope_notice()
    active_tab = render_tab_bar()

    if active_tab == "demo":
        render_demo_tab()
    elif active_tab == "manage":
        render_manage_tab()
    else:
        render_info_tab(chunks, lookup)
    render_html("</main>")


if __name__ == "__main__":
    main()
