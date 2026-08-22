# RAG Chatbot + PyTorch Intent Classifier

PyTorch 기반 Intent Classifier와 직접 구현한 NumPy VectorStore 및 Retriever를 결합하고, Sentence Transformer 임베딩과 Gemini API를 활용하여 구현한 TXT 문서 기반 RAG 챗봇 개인 프로젝트입니다.

이 프로젝트의 주요 목적은 LangChain이나 Vector DB와 같은 고수준 프레임워크에 의존하기보다, Intent Classification과 RAG가 실제로 어떤 구조로 연결되는지 Python 코드 수준에서 직접 이해하는 것입니다.

---

## 1. Project Overview

사용자의 질문을 먼저 Intent Classifier로 분류한 뒤, 문서 관련 질문인 경우에만 RAG 파이프라인을 실행합니다.

전체 구조는 다음과 같습니다.

```text
사용자 질문
    ↓
IntentPredictor
    ↓
Intent + Confidence
    ↓
Confidence Threshold 검사
    │
    ├─ 낮은 Confidence ──> Fallback
    │
    ├─ 일반 Intent ──────> 일반 응답
    │
    └─ document_query
         ↓
      Retriever
         ↓
   질문 Embedding
         ↓
     VectorStore
         ↓
 Cosine Similarity
         ↓
       Top-K
         ↓
   SearchResult
         ↓
  AnswerGenerator
         ↓
 Search Score Filtering
         ↓
 Duplicate Chunk 제거
         ↓
    Context 구성
         ↓
       Gemini
         ↓
     최종 답변
```

---

## 2. Main Features

### Intent Classification
* PyTorch 기반 Intent Classifier
* `intents.json` 기반 학습 데이터 처리
* Intent Classifier 학습 및 모델 저장
* 저장된 `intent_classifier.pt` 기반 추론
* Intent별 confidence 계산 및 threshold 기반 fallback
* 일반 Intent와 문서 검색 Intent 분리

### RAG (Retrieval-Augmented Generation)
* TXT 문서 로딩, 전처리, Chunk 분할 및 Overlap 설정
* Sentence Transformer 기반 문서/질문 임베딩
* NumPy 기반 in-memory VectorStore 및 Cosine Similarity 기반 Top-K 검색
* Retriever 구현 (source 및 chunk_id 관리)

### Answer Generation
* 검색 결과 score filtering 및 중복 Chunk 제거
* Context 제한 (최대 Chunk 수 및 전체 길이) 설정 후 Prompt 구성
* Gemini API 기반 자연어 답변 생성

### TXT Upload
* Google Colab `files.upload()` 기반 사용자 파일 업로드 (`.txt` 확장자 검사)
* 업로드 문서를 `data/uploaded_documents/`에 별도 저장
* 새 문서 업로드 후 VectorStore / Retriever / Chatbot 자동 재생성

---

## 3. Project Structure

```text
rag_intent_chatbot/
│
├─ data/
│  ├─ documents/
│  │  └─ sample.txt
│  │
│  ├─ uploaded_documents/
│  │  └─ 사용자 업로드 TXT 파일
│  │
│  └─ intents.json
│
├─ models/
│  └─ intent_classifier.pt
│
├─ src/
│  ├─ __init__.py
│  │
│  ├─ intent/
│  │  ├─ __init__.py
│  │  ├─ dataset.py
│  │  ├─ model.py
│  │  ├─ train.py
│  │  └─ predict.py
│  │
│  ├─ rag/
│  │  ├─ __init__.py
│  │  ├─ document_loader.py
│  │  ├─ text_preprocessor.py
│  │  ├─ chunker.py
│  │  ├─ embedder.py
│  │  ├─ vector_store.py
│  │  ├─ retriever.py
│  │  └─ answer_generator.py
│  │
│  └─ chatbot.py
│
├─ main.py
├─ requirements.txt
│
└─ 08_TXT_업로드_RAG_챗봇.ipynb
```

---

## 4. Module Description

* **`src/intent/dataset.py`**: `intents.json` 데이터를 PyTorch Dataset 형태로 처리합니다.
* **`src/intent/model.py`**: PyTorch `nn.Module`을 기반으로 Intent Classifier 신경망 구조를 정의합니다.
* **`src/intent/train.py`**: Intent Classifier를 학습하고 `models/intent_classifier.pt`에 파라미터를 저장합니다.
* **`src/intent/predict.py`**: 저장된 모델을 불러와 Intent와 confidence를 예측하는 `IntentPredictor`를 제공합니다.

```text
[Intent Train Flow]
intents.json → Dataset → IntentClassifier → Forward → Loss → Backpropagation → Optimizer → intent_classifier.pt

[Intent Predict Flow]
intent_classifier.pt + IntentClassifier 구조 → IntentPredictor → Intent + Confidence
```

* **`src/rag/document_loader.py`**: TXT 파일을 읽어 문서 객체로 변환합니다.
* **`src/rag/text_preprocessor.py`**: 문서의 텍스트를 검색 가능하도록 전처리합니다.
* **`src/rag/chunker.py`**: 문서를 나눕니다 (`CHUNK_SIZE = 500`, `CHUNK_OVERLAP = 100`).
* **`src/rag/embedder.py`**: Pretrained Sentence Transformer를 활용하여 문서와 질문을 동일한 임베딩 공간으로 변환합니다.
* **`src/rag/vector_store.py`**: NumPy 기반 in-memory VectorStore입니다. Cosine Similarity 계산을 직접 구현했습니다.
* **`src/rag/retriever.py`**: 질문 임베딩과 VectorStore 검색을 단일 파이프라인으로 연결합니다.
* **`src/rag/answer_generator.py`**: 검색 결과를 필터링/가공하여 Gemini Context로 변환 후 답변을 생성합니다.
* **`src/chatbot.py`**: IntentPredictor, Retriever, AnswerGenerator를 통합하여 질의 라우팅을 제어합니다.
* **`main.py`**: 프로젝트의 주요 객체를 생성하고 조립하는 메인 엔트리 포인트입니다.

---

## 5. Intent Classification and RAG Routing

모든 질문을 RAG로 전달하지 않고 Intent Classifier를 먼저 거칩니다.

```text
사용자 질문
 ↓
IntentPredictor.predict()
 ↓
Intent + Confidence
```

* **Low Confidence (`confidence < 0.60`)**: Fallback 처리
* **General Intent**: 지정된 일반 응답 반환
* **Document Query (`confidence >= 0.60`)**: RAG 파이프라인(Retriever → AnswerGenerator → Gemini) 실행

---

## 6. Two Different Scores

프로젝트에는 독립적인 두 종류의 점수가 존재합니다.

```text
사용자 질문 ──> Intent Confidence (Threshold 검사) ──> document_query ──> Retriever Score (Cosine Similarity)
```

1. **Intent Confidence**: 질문이 어떤 의도인지 분류할 때의 확신도 (`CONFIDENCE_THRESHOLD = 0.60`)
2. **Retriever Score**: 문서 Chunk와 질문 사이의 의미적 유사도 (Cosine Similarity)

---

## 7. Main Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `CONFIDENCE_THRESHOLD` | `0.60` | Intent 판단 최소 확신도 |
| `RETRIEVAL_TOP_K` | `3` | Retriever 반환 최대 Chunk 수 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `100` | 문맥 분할 및 중첩 크기 |
| `min_search_score` | `0.30` | Context 채택 최소 검색 유사도 |
| `max_context_chunks` | `3` | LLM Context 전달 최대 Chunk 수 |
| `max_context_length` | `1500` | LLM Context 최대 길이고 |
| `temperature` / `max_output_tokens` | `0.2` / `500` | Gemini 답변 생성 파라미터 |

---

## 8. Gemini Configuration

* **Python SDK**: `google-genai==2.18.0`
* **Model**: `gemini-3.5-flash-lite`
* **API Key**: Google Colab Secrets (`GEMINI_API_KEY`)로 안전하게 관리

```python
answer_generator = AnswerGenerator(
    client=gemini_client,
    model_name="gemini-3.5-flash-lite",
    min_search_score=0.30,
    max_context_chunks=3,
    max_context_length=1500,
    answer_preview_length=500,
    temperature=0.2,
    max_output_tokens=500,
)
```

---

## 9. TXT Upload Pipeline & Update

```text
사용자 PC ──> files.upload() ──> .txt 검사 ──> data/uploaded_documents/ 저장
    ↓
[재구성 파이프라인]
TXT 로드 → 전처리 → Chunk 생성 → Embedding 생성 → 새 VectorStore → 새 Retriever → 새 Chatbot
```

* 별도의 Vector DB 없이 In-Memory 방식이므로, 새 문서를 업로드하면 RAG 검색 인덱스를 메모리 상에서 새로 재구성합니다. (AI 모델 자체를 재학습하는 것이 아닙니다.)

---

## 10. Document Preparation vs Query Time

```text
[Preparation Time] TXT → Document → Preprocessing → Chunk → Chunk Embedding → VectorStore (미리 생성)
[Query Time]       Query → Query Embedding → Cosine Similarity 비교 → Top-K → SearchResult (실시간)
```

---

## 11. Testing

단계적 검증을 위해 모듈별 테스팅 체계를 분리했습니다.

1. **Retriever Test**: Intent 및 LLM을 제외하고 Cosine Similarity 검색 정확도 및 `source`/`chunk_id` 반환 검증
2. **AnswerGenerator Test**: Filtering, 중복 제거, Context 길이 제한 및 Gemini API 연동 동작 검증
3. **Full Chatbot Test**: Intent Routing, Fallback, Document Query, 종료(`exit`/`quit`) 제어 전체 흐름 검증

---

## 12. Implementation Scope

* **직접 구현한 요소**: PyTorch Intent Classifier, Custom Dataset & Pipeline, Confidence Fallback Routing, Chunking With Overlap, NumPy In-Memory VectorStore, Cosine Similarity Engine, Custom Retriever & Context Construction Engine.
* **외부 Pretrained/SDK 활용**: PyTorch, NumPy, Pretrained Sentence Transformer, Gemini API (`google-genai`), Google Colab `files.upload()`.

---

## 13. Environment & Running

### Requirements & Setup
Google Colab 환경에서 Google Drive를 마운트 후 실행합니다.

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/rag_intent_chatbot
```

```bash
pip install -r requirements.txt
```

### Running Notebook
`GEMINI_API_KEY`를 Colab Secret에 등록한 뒤 아래 노트북을 실행합니다.
* `08_TXT_업로드_RAG_챗봇.ipynb`

---

## 14. Current Limitations & Future Improvements

### Current Limitations
* In-Memory NumPy 구조로 인해 대규모 문서 처리 제한
* TXT 포맷만 직접 지원
* 단기/장기 대화 메모리(Conversation History) 미구현
* Web UI 미지원 (Colab 전용)

### Future Improvements
* PDF/DOCX Loader 추가 및 Incremental Indexing 도입
* External Vector Database (FAISS, Chroma 등) 연동
* FastAPI 백엔드 및 Web Frontend 구축
* Multi-session / Multi-user 대화 메모리 구현

---

## 15. Tech Stack

* **Language**: Python
* **Deep Learning**: PyTorch
* **Numerical Computing**: NumPy
* **Embedding**: Sentence Transformer
* **LLM**: Gemini API (`google-genai`)
* **Environment**: Google Colab

---

## 16. Project Summary

```text
PyTorch Intent Classifier
        ↓
Intent + Confidence
        ↓
Chatbot Routing
        ↓
Sentence Transformer
        ↓
NumPy VectorStore
        ↓
Cosine Similarity / Top-K
        ↓
Retriever
        ↓
AnswerGenerator
        ↓
Gemini
        ↓
Document-based Response
```
