# LP출자 온톨로지 RAG 시스템

IBK캐피탈 투자금융부 LP출자 업무용 **온톨로지 기반 RAG 시스템** — 사내 AI 경진대회 출품작.
폐쇄망·로컬 LLM 환경을 가정해, 온톨로지(SPARQL)·규제 RAG·LLM을 결합한 프로토타입입니다.

> ℹ️ 기존 Streamlit 시연 앱은 **Hugging Face Space로 이전**했습니다. 아래에서 바로 실행할 수 있습니다.

## 🔗 라이브 데모

**👉 [Hugging Face Space에서 실행](https://huggingface.co/spaces/ForStream/ontology-prototype)**

→ FastAPI + React 데모 콘솔로 연결됩니다. 3개 탭(설명 / 테스트 / 데이터 관리)에서, 같은 질문에 대해
**Python(결정론적) · Sonnet · Gemma** 세 방식의 답변을 나란히 비교할 수 있습니다.

## 📚 관련 자료

- **[Ontology Development 101 — A Guide to Creating Your First Ontology (Stanford)](https://protege.stanford.edu/publications/ontology_development/ontology101.pdf)**
  → 스탠퍼드대 Noy & McGuinness의 **온톨로지 설계 입문 가이드 PDF**로 연결됩니다. 이 프로젝트의
  클래스·속성·인스턴스 설계 방법론의 기반이 된 표준 문헌입니다.

- **[기업이 꼭 알아야 할 '온톨로지'의 모든 것 — 김학래 중앙대 교수 (YouTube)](https://youtu.be/W0MBC6in4Q4)**
  → 온톨로지의 개념과 기업 활용을 다룬 **한국어 강연 영상**으로 연결됩니다. 온톨로지가 비즈니스에
  왜 필요한지 이해하는 입문 영상입니다.

## 구성 (참고)

- **온톨로지**: 64 클래스 / 트리플 3,712개 (LP출자 업무·규제 구조화)
- **RAG**: 규제 원문 274 청크 + KoSimCSE 한국어 의미검색
- **LLM**: 폐쇄망 로컬 Gemma + (데모) Sonnet API

---
사내 AI 경진대회 출품작 · 온톨로지 기반 폐쇄망 RAG 시스템
