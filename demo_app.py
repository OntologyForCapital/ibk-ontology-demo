"""LP출자 온톨로지 RAG 시스템 — Hugging Face Space 안내 랜딩.

기존 Streamlit 시연 앱은 HF Space(FastAPI+React)로 이전했고,
이 Streamlit Cloud 페이지는 그쪽으로 안내하는 랜딩 역할만 한다.
"""
import os
import streamlit as st

HF_URL = "https://huggingface.co/spaces/ForStream/ontology-prototype"
STANFORD_URL = "https://protege.stanford.edu/publications/ontology_development/ontology101.pdf"
YOUTUBE_URL = "https://youtu.be/W0MBC6in4Q4"

st.set_page_config(
    page_title="LP출자 온톨로지 RAG 시스템",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 LP출자 온톨로지 RAG 시스템")
st.caption("IBK캐피탈 투자금융부 · 사내 AI 경진대회 출품작 · 온톨로지 기반 폐쇄망 RAG 시스템")

st.info("ℹ️ 데모가 **Hugging Face Space**로 이전했습니다. 아래 버튼에서 바로 실행하세요.")

# ── 라이브 데모 ───────────────────────────────────────────────
st.markdown("### 🔗 라이브 데모")
st.link_button("👉  Hugging Face Space에서 실행", HF_URL, type="primary", use_container_width=True)

_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LP출자_온톨로지_제안.pdf")
if os.path.exists(_PDF):
    with open(_PDF, "rb") as _f:
        st.download_button(
            "📄  제안서 PDF 다운로드 (LP출자 온톨로지)",
            _f.read(),
            file_name="LP출자_온톨로지_제안.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

st.markdown(
    "→ **FastAPI + React 데모 콘솔**로 연결됩니다. 3개 탭(설명 / 테스트 / 데이터 관리)에서, "
    "같은 질문에 대해 **Python(결정론적) · Sonnet · Gemma** 세 방식의 답변을 나란히 비교할 수 있습니다."
)

st.divider()

# ── 관련 자료 ─────────────────────────────────────────────────
st.markdown("### 📚 관련 자료")

st.markdown(
    f"**[Ontology Development 101 — A Guide to Creating Your First Ontology (Stanford)]({STANFORD_URL})**  \n"
    "→ 스탠퍼드대 Noy & McGuinness의 **온톨로지 설계 입문 가이드 PDF**로 연결됩니다. "
    "이 프로젝트의 클래스·속성·인스턴스 설계 방법론의 기반이 된 표준 문헌입니다."
)

st.markdown(
    f"**[기업이 꼭 알아야 할 '온톨로지'의 모든 것 — 김학래 중앙대 교수 (YouTube)]({YOUTUBE_URL})**  \n"
    "→ 온톨로지의 개념과 기업 활용을 다룬 **한국어 강연 영상**으로 연결됩니다. "
    "온톨로지가 비즈니스에 왜 필요한지 이해하는 입문 영상입니다."
)
st.video(YOUTUBE_URL)

st.divider()

# ── 구성 참고 ─────────────────────────────────────────────────
st.markdown("### 🧩 구성 (참고)")
st.markdown(
    "- **온톨로지**: 64 클래스 / 트리플 3,712개 (LP출자 업무·규제 구조화)\n"
    "- **RAG**: 규제 원문 274 청크 + KoSimCSE 한국어 의미검색\n"
    "- **LLM**: 폐쇄망 로컬 Gemma + (데모) Sonnet API"
)

st.caption("사내 AI 경진대회 출품작 · 온톨로지 기반 폐쇄망 RAG 시스템")
