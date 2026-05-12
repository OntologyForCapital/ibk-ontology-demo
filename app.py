"""
app.py — LP출자 온톨로지 + RAG + Gemma 4 e4b 시연 앱
====================================================

Streamlit UI로 구성:
- 좌측: 챗봇 (예시 질문 버튼 + 자유 질문 입력 + 채팅 히스토리)
- 우측: 인스턴스 관리 (현재 등록된 검토건 목록 + 새 검토건 추가 폼)

평가자(임원) 대상 시연용. 직접 만져보고 답변 확인 가능.

실행:
    streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import time

# 프로젝트 모듈
import rag_engine
import instance_manager as im

# ============================================
# 페이지 설정
# ============================================

st.set_page_config(
    page_title="LP출자 AI 챗봇 — 온톨로지 + RAG 시연",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 데이터 경로
DATA_DIR = Path(__file__).parent / "data"
TTL_PATH = DATA_DIR / "investment_ontology_v1_10.ttl"
JSONL_PATH = DATA_DIR / "regulations_chunks_v14.jsonl"
ALIAS_PATH = DATA_DIR / "alias_dictionary.json"
LOOKUP_PATH = DATA_DIR / "risk_weight_lookup.json"


# ============================================
# 데이터 로드 (캐시)
# ============================================

@st.cache_resource
def load_static_data():
    """온톨로지·청크·사전·lookup은 변경 없으니 캐시"""
    chunks = rag_engine.load_chunks(JSONL_PATH)
    alias = rag_engine.load_alias(ALIAS_PATH)
    lookup = rag_engine.load_lookup(LOOKUP_PATH)
    return chunks, alias, lookup


def get_session_graph():
    """세션별 독립 그래프. 사용자 인스턴스 추가가 다른 세션에 영향 X."""
    if "graph" not in st.session_state:
        st.session_state.graph = rag_engine.load_ttl(TTL_PATH)
    return st.session_state.graph


# ============================================
# 세션 초기화
# ============================================

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_instances" not in st.session_state:
        st.session_state.user_instances = []
    if "use_gemma" not in st.session_state:
        # 단계별 안전한 체크 — 실패해도 앱 죽지 않음
        try:
            available, status_msg = check_gemma_available()
        except Exception as e:
            available, status_msg = False, f"체크 실패 (시연 모드): {type(e).__name__}"
        st.session_state.use_gemma = available
        st.session_state.gemma_status = status_msg
    if "show_full_options" not in st.session_state:
        st.session_state.show_full_options = False  # 단계·상태 전체/축약 토글
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None  # 예시 버튼 클릭 시 자동 입력


def check_gemma_available():
    """
    Gemma 호출 가능 여부를 단계별로 체크.
    어느 단계에서 실패하든 False 반환 (앱은 죽지 않음).
    
    Returns:
        (available_bool, status_msg)
        - True: Gemma 호출 가능
        - False: 시연 모드 (구조화된 raw 컨텍스트 표시)
    """
    import os
    model_name = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
    
    # 1) ollama 패키지
    try:
        import ollama
    except ImportError:
        return False, "ollama 패키지 미설치 (시연 모드로 동작)"
    
    # 2) 서버 응답 확인 (list 호출)
    try:
        models_resp = ollama.list()
    except Exception as e:
        return False, f"Ollama 서버 응답 없음 (시연 모드): {type(e).__name__}"
    
    # 3) 모델 존재 확인
    try:
        # ollama.list() 응답 형식이 버전에 따라 다를 수 있음
        # 최신: {"models": [{"name": "gemma4:e4b", ...}, ...]}
        # 또는: list of dicts
        models_list = models_resp.get("models", []) if isinstance(models_resp, dict) else models_resp
        model_names = []
        for m in models_list:
            if isinstance(m, dict):
                model_names.append(m.get("name") or m.get("model") or "")
            else:
                model_names.append(str(m))
        
        if model_name in model_names or any(model_name in n for n in model_names):
            return True, f"Gemma 호출 가능: {model_name}"
        else:
            return False, (
                f"모델 '{model_name}' 미설치 (시연 모드). "
                f"`ollama pull {model_name}` 실행 후 재시작하세요."
            )
    except Exception as e:
        # 모델 확인 단계에서 예외 발생해도 시연 모드로 안전 동작
        return False, f"모델 확인 실패 (시연 모드): {type(e).__name__}"


# ============================================
# 사이드바 / 상단 안내
# ============================================

def render_header():
    st.title("💬 LP출자 AI 챗봇")
    st.caption(
        "온톨로지(OWL/Turtle) + RAG(JSONL) + Gemma 4 e4b 결합 프로토타입 — "
        "여신전문금융회사 LP출자 업무 보조 데모"
    )
    
    with st.expander("📌 시연 시작 전 — 시스템 소개 & 사용 방법", expanded=False):
        st.markdown("""
**이 시스템은 어떤 시스템인가요?**

회사가 업무에 LLM(Gemma 4 e4b 등)을 도입할 때, **단순히 LLM에게 모든 판단을 맡기는 것이 아니라**,  
**온톨로지(업무 지식 구조도) + 규제 청크(RAG) + 정형 조회 표(Lookup)**를 통해 답변의 근거를 제공하고,  
**LLM은 그 근거를 자연어로 풀어주는 역할**만 하도록 설계한 프로토타입입니다.

---

**시연 가능한 질문 유형:**

1. **검토건 진행 상태**: "ABC펀드 검토건은 지금 어디까지 갔어?"
2. **단계별 검토건 추출**: "약정 단계 이후로 진행된 검토건은?"
3. **정체 검토건 조회**: "예비검토 올렸는데 결재 안 난 건들은?"
4. **위험가중치 조회**: "AAA 등급 중앙정부 익스포져의 위험가중치는?"
5. **규제 설명**: "LP출자한 펀드의 RWA는 어떻게 산정해?"
6. **사용자 추가 검토건 질의**: 우측에서 직접 등록한 검토건에 대해 자유 질의

---

**시연 범위 외 질문은 답변이 어렵습니다:**

- 일반 금융 상식 (예: "삼성전자 주가는?")
- 실제 회사 데이터 (예: "지난 분기 매출은?") — 본 데모는 가상 인스턴스만 포함
- 개인정보, 회사 기밀 정보

위 범위 외 질문에는 시스템이 "범위 외 질문" 안내를 표시합니다.
        """)


# ============================================
# 좌측: 챗봇
# ============================================

def render_chat_panel(graph, chunks, alias, lookup):
    st.subheader("💬 챗봇")
    
    # 예시 질문 버튼 (3열 × 4행)
    st.markdown("**예시 질문** (클릭하면 자동 입력):")
    
    example_questions = [
        ("ABC펀드 검토건은 어디까지 갔어?", "📋 진행 상태"),
        ("1q2w 펀드 검토건은 어떻게 진행 중이야?", "📋 진행 상태"),
        ("babymonster 검토건 상태는?", "📋 진행 상태"),
        ("약정 단계 이후로 진행된 검토건은?", "📊 단계 추출"),
        ("정체된 검토건 있어?", "🚨 정체 조회"),
        ("AAA 등급 중앙정부 익스포져 위험가중치는?", "🔢 위험가중치"),
        ("LP출자한 펀드의 RWA는 어떻게 산정해?", "📖 규제 설명"),
        ("적합성 원칙은 어떻게 확인해?", "📖 규제 설명"),
        ("대체투자 분류 기준은?", "📖 규제 설명"),
    ]
    
    cols = st.columns(3)
    for idx, (q, label) in enumerate(example_questions):
        col = cols[idx % 3]
        with col:
            if st.button(f"{label}\n{q[:30]}{'...' if len(q) > 30 else ''}", 
                         key=f"ex_q_{idx}", 
                         use_container_width=True,
                         help=q):
                st.session_state.pending_question = q
                st.rerun()
    
    st.markdown("---")
    
    # 채팅 히스토리
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.info("위 예시 질문을 클릭하거나, 아래 입력창에 자유롭게 질문해 보세요.")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "meta" in msg:
                    with st.expander("🔍 답변 근거 (라우팅·SPARQL/RAG)", expanded=False):
                        st.caption(f"라우팅: `{msg['meta']['route']}`")
                        st.caption(f"컨텍스트: {msg['meta']['context_summary']}")
    
    # 입력창
    user_input = st.chat_input("질문을 입력하세요... (예: 'ABC펀드 검토건 어디까지 갔어?')")
    
    # 예시 버튼 클릭으로 들어온 질문 처리
    if st.session_state.pending_question and not user_input:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None
    
    if user_input:
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 답변 생성
        with st.spinner("🤔 답변 생성 중... (Gemma 4 e4b가 처리 중일 수 있습니다)"):
            try:
                start_time = time.time()
                result = rag_engine.answer_question(
                    user_input,
                    graph, chunks, alias, lookup,
                    user_instances=st.session_state.user_instances,
                    use_gemma=st.session_state.use_gemma,
                )
                elapsed = time.time() - start_time
                
                meta = {
                    "route": result["route"],
                    "context_summary": result["context_summary"],
                    "elapsed": f"{elapsed:.1f}초",
                }
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "meta": meta,
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ 답변 생성 중 오류가 발생했습니다: {str(e)}",
                    "meta": {"route": "error", "context_summary": str(e)},
                })
        
        st.rerun()


# ============================================
# 우측: 인스턴스 관리
# ============================================

def render_instance_panel(graph):
    st.subheader("📁 검토건 관리")
    
    # 등록된 사용자 인스턴스 목록
    user_invs = im.list_all_user_investments(st.session_state.user_instances)
    
    with st.expander(f"📋 직접 등록한 검토건 ({len(user_invs)}건)", expanded=len(user_invs) > 0):
        if not user_invs:
            st.caption("아래 폼에서 새 검토건을 등록해 보세요.")
        else:
            for idx, inv in enumerate(user_invs):
                with st.container(border=True):
                    st.markdown(f"**{inv['fund_name']}** ({inv['n_branches']}개 분기)")
                    st.caption(f"GP: {inv['gp_name']}")
                    if st.button("🗑️ 삭제", key=f"del_{idx}", use_container_width=True):
                        # 그래프에서 제거
                        record_to_remove = st.session_state.user_instances[idx]
                        im.remove_user_investment(graph, record_to_remove)
                        # 세션 리스트에서 제거
                        st.session_state.user_instances.pop(idx)
                        st.success(f"✅ '{inv['fund_name']}' 삭제됨")
                        time.sleep(0.5)
                        st.rerun()
            
            if st.button("🗑️ 등록한 모든 검토건 삭제", type="secondary", use_container_width=True):
                im.remove_all_user_investments(graph, st.session_state.user_instances)
                st.session_state.user_instances = []
                st.success("✅ 모든 등록 검토건 삭제됨")
                time.sleep(0.5)
                st.rerun()
    
    # 기존 데모 인스턴스 안내
    with st.expander("📋 기본 제공 검토건 (데모용 가상 인스턴스)", expanded=False):
        st.markdown("""
- **ABC펀드 출자·인수금융 동시 검토** (가상)
  - 분기 2개: LP출자 50억 (출자금 납입 완료) + 인수금융 100억 (예비검토 정체)
- **상장사 A 인수목적 펀드 1q2w 출자 및 인수금융 검토**
  - 분기 2개: LP출자 10억 + 인수금융 100억 (모두 예비검토 진행 중)
- **alldayproject 펀드 LP출자 검토** (비상장사 B 전환사채 인수목적)
  - 분기 1개: LP출자 40억 (사전협의 진행 중)
- **babymonster 신기술조합 LP출자** (Pre-IPO 비상장사 C CPS 인수목적)
  - 분기 1개: LP출자 10억 (약정체결완료)
        """)
    
    st.markdown("---")
    
    # 새 검토건 추가 폼
    st.markdown("### ➕ 새 검토건 추가")
    
    # 토글: 단계·상태 전체 / 축약
    use_full = st.toggle(
        "🔧 전체 단계·상태 옵션 표시",
        value=st.session_state.show_full_options,
        help="끄면 자주 쓰는 옵션만, 켜면 v1.10 온톨로지의 전체 단계·상태 표시",
    )
    st.session_state.show_full_options = use_full
    
    stages = im.STAGES_FULL if use_full else im.STAGES_SIMPLE
    states = im.STATES_FULL if use_full else im.STATES_SIMPLE
    
    with st.form("add_investment_form", clear_on_submit=True):
        # 펀드 기본 정보
        col1, col2 = st.columns(2)
        with col1:
            fund_name = st.text_input(
                "펀드명 *",
                placeholder="예: 새한벤처1호",
                max_chars=50,
            )
        with col2:
            gp_name = st.text_input(
                "GP명 (운용사) *",
                placeholder="예: 새한자산운용",
                max_chars=50,
            )
        
        st.markdown("**분기 정보** (1~3개 등록 가능)")
        
        n_branches = st.radio(
            "분기 개수",
            options=[1, 2, 3],
            horizontal=True,
            help="LP출자 단독은 1개, LP출자 + 인수금융 동시 진행은 2~3개",
        )
        
        branches_data = []
        for i in range(n_branches):
            with st.container(border=True):
                st.markdown(f"**분기 {i+1}**")
                
                bcol1, bcol2 = st.columns([2, 1])
                with bcol1:
                    product = st.selectbox(
                        f"상품 유형",
                        options=[p[0] for p in im.PRODUCT_TYPES],
                        format_func=lambda x: dict(im.PRODUCT_TYPES)[x],
                        key=f"branch_{i}_product",
                    )
                with bcol2:
                    amount = st.number_input(
                        f"금액 (억 원)",
                        min_value=1,
                        max_value=10000,
                        value=50,
                        step=10,
                        key=f"branch_{i}_amount",
                    )
                
                bcol3, bcol4 = st.columns(2)
                with bcol3:
                    stage = st.selectbox(
                        f"현재 단계",
                        options=[s[0] for s in stages],
                        format_func=lambda x: dict(stages).get(x, x),
                        key=f"branch_{i}_stage",
                    )
                with bcol4:
                    state = st.selectbox(
                        f"브랜치 상태",
                        options=[s[0] for s in states],
                        format_func=lambda x: {s[0]: s[1] for s in states}.get(x, x),
                        key=f"branch_{i}_state",
                    )
                
                branches_data.append({
                    "product_iri": product,
                    "amount": amount,
                    "stage_iri": stage,
                    "state_iri": state,
                })
        
        submitted = st.form_submit_button(
            "✅ 검토건 추가",
            type="primary",
            use_container_width=True,
        )
        
        if submitted:
            # 검증
            is_valid, err_msg = im.validate_input(fund_name, gp_name, branches_data)
            if not is_valid:
                st.error(f"❌ {err_msg}")
            else:
                try:
                    # 인스턴스 추가
                    record = im.add_user_investment(
                        graph, fund_name, gp_name, "RecipientFund", branches_data
                    )
                    st.session_state.user_instances.append(record)
                    
                    # 자동 안내 메시지를 챗봇에 추가
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": (
                            f"✅ **'{fund_name}' 검토건 등록 완료**\n\n"
                            f"- GP: {gp_name}\n"
                            f"- 분기: {len(branches_data)}개\n\n"
                            f"이제 **'{fund_name} 검토건은 어디까지 갔어?'** 또는  \n"
                            f"**'{fund_name}의 RWA는 어떻게 산정해?'** 같이 질문해 보세요."
                        ),
                        "meta": {
                            "route": "system_notification",
                            "context_summary": f"인스턴스 추가: {record['investment_iri']}",
                        },
                    })
                    
                    st.success(f"✅ '{fund_name}' 등록 완료! 챗봇에서 질문해 보세요.")
                    time.sleep(0.8)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 등록 중 오류: {str(e)}")
    
    # 챗봇 초기화 버튼
    st.markdown("---")
    if st.button("🔄 채팅 기록 초기화", use_container_width=True):
        st.session_state.messages = []
        st.success("채팅 기록이 초기화되었습니다.")
        time.sleep(0.3)
        st.rerun()


# ============================================
# 메인
# ============================================

def main():
    init_session_state()
    
    # 데이터 로드
    try:
        chunks, alias, lookup = load_static_data()
        graph = get_session_graph()
    except FileNotFoundError as e:
        st.error(f"❌ 데이터 파일을 찾을 수 없습니다: {e}")
        st.info(f"data/ 폴더에 다음 파일이 있는지 확인하세요:\n- investment_ontology_v1_10.ttl\n- regulations_chunks_v14.jsonl\n- alias_dictionary.json\n- risk_weight_lookup.json")
        return
    
    # 헤더
    render_header()
    
    # 시스템 상태 (간단 인디케이터)
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    with status_col1:
        st.metric("온톨로지 트리플", f"{len(graph):,}")
    with status_col2:
        st.metric("규제 청크", f"{len(chunks)}")
    with status_col3:
        st.metric("등록한 검토건", f"{len(st.session_state.user_instances)}건")
    with status_col4:
        if st.session_state.use_gemma:
            label = "✅ 연결됨"
            help_text = st.session_state.get("gemma_status", "")
        else:
            label = "⚠️ 시연 모드"
            help_text = st.session_state.get("gemma_status", "Gemma 미연결, 구조화된 raw 컨텍스트 표시")
        st.metric("LLM 상태", label, help=help_text)
    
    # Gemma 미연결 시 안내 (한 번만)
    if not st.session_state.use_gemma:
        st.info(
            f"💡 **시연 모드로 동작 중입니다.** "
            f"({st.session_state.get('gemma_status', 'LLM 미연결')})  \n"
            f"이 모드에서는 LLM이 답변을 다듬지 않고, "
            f"**SPARQL/Lookup/RAG 검색 결과를 구조화된 형태로 직접 표시**합니다. "
            f"답변의 사실은 동일하며, 자연어 표현만 거치지 않습니다.",
            icon="ℹ️",
        )
    
    st.markdown("---")
    
    # 좌우 분할
    chat_col, instance_col = st.columns([3, 2])
    
    with chat_col:
        render_chat_panel(graph, chunks, alias, lookup)
    
    with instance_col:
        render_instance_panel(graph)
    
    # 푸터
    st.markdown("---")
    st.caption(
        "🔬 **본 시스템은 사내 LLM 도입 검증을 위한 개념 검증(PoC) 프로토타입입니다.**  "
        "온톨로지·규제 청크·위험가중치 lookup은 모두 외부 공개 자료 기반의 가상 데이터이며, "
        "실제 회사 데이터를 포함하지 않습니다.  "
        "최종 업무 판단은 항상 담당자가 수행합니다."
    )


if __name__ == "__main__":
    main()
