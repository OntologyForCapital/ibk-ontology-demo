# LP출자 온톨로지 RAG 시스템 — Streamlit 시연 앱

사내 AI 경진대회 라이브 배포용 Streamlit 앱입니다. 기존 `app.py`는 보존하고, `demo_app.py`를 시연 콘솔로 사용합니다.

## 실행

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run demo_app.py
```

## 모델 모드

- Gemma 4 e4b 로컬: 로컬 Ollama 환경에서만 사용합니다.
- Sonnet 4.6 API: `ANTHROPIC_API_KEY`는 Streamlit Cloud secrets로 관리합니다. 1차 앱에서는 UI 자리만 제공합니다.
- 구조화 모드: LLM 없이 템플릿, lookup, RAG 근거 중심으로 표시합니다.

## 배포 참고

Streamlit Cloud는 CPU 환경이므로 Gemma 4 e4b 로컬 토글은 안내 메시지만 표시하고, 구조화 모드 또는 Sonnet API 모드를 권장합니다.
