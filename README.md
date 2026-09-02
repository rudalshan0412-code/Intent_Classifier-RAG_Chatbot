# RAG Chatbot + PyTorch Intent Classifier

PyTorch 기반 Intent Classifier와 직접 구현한 NumPy VectorStore 및 Retriever를 결합하고, Sentence Transformer 임베딩과 Gemini API를 활용한 TXT 문서 기반 RAG 챗봇 프로젝트입니다.

이 프로젝트는 LangChain, FAISS, Chroma와 같은 고수준 RAG 프레임워크를 바로 사용하는 대신, Intent Classification부터 문서 검색, Context 구성, LLM 답변 생성까지의 흐름을 직접 구현하며 RAG 구조를 이해하는 것을 목표로 진행했습니다.

---

## 1. Project Overview

사용자의 질문을 먼저 Intent Classifier로 분류합니다.

문서 관련 질문인 `document_query`로 판단된 경우에만 Retriever를 실행하고, 검색된 문서 Chunk를 Context로 구성하여 Gemini가 최종 답변을 생성합니다.

```text
User Question
    |
    v
IntentPredictor
    |
    v
Intent + Confidence
    |
    v
Confidence Threshold Check
    |
    +--------------------+---------------------+
    |                    |                     |
    v                    v                     v
Fallback            General Intent       document_query
                                              |
                                              v
                                          Retriever
                                              |
                                              v
                                       Query Embedding
                                              |
                                              v
                                         VectorStore
                                              |
                                              v
                                     Cosine Similarity
                                              |
                                              v
                                            Top-K
                                              |
                                              v
                                        SearchResult
                                              |
                                              v
                                      AnswerGenerator
                                              |
                                              v
                                       Score Filtering
                                              |
                                              v
                                    Duplicate Removal
                                              |
                                              v
                                      Context Building
                                              |
                                              v
                                           Gemini
                                              |
                                              v
                                       Final Response
```

---

## 2. Main Features

### Intent Classification

* PyTorch 기반 Intent Classifier 구현
* `intents.json` 기반 학습 데이터 처리
* Intent Classifier 학습 및 모델 저장
* 저장된 `intent_classifier.pt`를 이용한 추론
* Intent confidence 계산
* Confidence threshold 기반 fallback 처리
* 일반 Intent와 문서 검색 Intent 분리

### RAG

* TXT 문서 로딩
* 텍스트 전처리
* 문서 Chunk 분할
* Chunk overlap 적용
* Sentence Transformer 기반 임베딩
* NumPy 기반 in-memory VectorStore 구현
* Cosine Similarity 기반 검색
* Top-K 검색
* Retriever 구현
* 검색 결과의 `source`, `chunk_id`, `score` 관리

### Answer Generation

* Retriever 결과의 score filtering
* 중복 Chunk 제거
* Context에 사용할 Chunk 개수 제한
* Context 전체 길이 제한
* 검색 결과 기반 Context 구성
* Gemini API를 이용한 최종 답변 생성

### TXT Upload

* Google Colab `files.upload()`을 이용한 TXT 파일 업로드
* `.txt` 확장자 검사
* TXT가 아닌 파일 제외
* 여러 TXT 파일 동시 업로드
* 파일명 및 파일 크기 확인
* 업로드 문서를 `data/uploaded_documents/`에 별도 저장
* 업로드 문서만 대상으로 Retriever 구성
* 새 문서 업로드 후 VectorStore 및 Retriever 재생성
* 새 Retriever를 사용하는 Chatbot 재생성

---

## 3. Project Structure

```text
rag_intent_chatbot/
|
├── data/
│   ├── documents/
│   │   └── sample.txt
│   │
│   ├── uploaded_documents/
│   │   └── user_uploaded.txt
│   │
│   └── intents.json
│
├── models/
│   └── intent_classifier.pt
│
├── src/
│   ├── __init__.py
│   │
│   ├── intent/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── document_loader.py
│   │   ├── text_preprocessor.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   └── answer_generator.py
│   │
│   └── chatbot.py
│
├── main.py
├── requirements.txt
│
└── 08_TXT_업로드_RAG_챗봇.ipynb
```

---

## 4. Module Description

### `src/intent/dataset.py`

`intents.json`의 학습 데이터를 Intent Classifier가 사용할 수 있는 형태로 변환합니다.

---

### `src/intent/model.py`

PyTorch `nn.Module`을 기반으로 Intent Classifier 신경망 구조를 정의합니다.

---

### `src/intent/train.py`

Intent Classifier를 학습하고 학습된 모델을 저장합니다.

```text
intents.json
    |
    v
Dataset
    |
    v
IntentClassifier
    |
    v
Forward
    |
    v
Loss
    |
    v
Backpropagation
    |
    v
Optimizer
    |
    v
intent_classifier.pt
```

학습된 모델은 다음 경로에 저장됩니다.

```text
models/intent_classifier.pt
```

`train.py`는 모델 학습 시 사용하며, 챗봇 실행마다 다시 학습하지 않습니다.

---

### `src/intent/predict.py`

저장된 Intent Classifier를 불러와 실제 사용자 질문의 Intent와 confidence를 예측하는 `IntentPredictor`를 구현합니다.

```text
IntentClassifier Structure
        +
intent_classifier.pt
        |
        v
IntentPredictor
        |
        v
User Question
        |
        v
Intent + Confidence
```

---

### `src/rag/document_loader.py`

TXT 파일을 읽어 Document 객체로 변환합니다.

---

### `src/rag/text_preprocessor.py`

로드된 문서의 텍스트를 검색에 사용할 수 있도록 전처리합니다.

---

### `src/rag/chunker.py`

긴 문서를 일정한 크기의 여러 Chunk로 분할합니다.

현재 주요 설정값은 다음과 같습니다.

```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
```

Chunk 사이에 overlap을 적용하여 Chunk 경계에서 문맥이 완전히 끊어지는 문제를 줄였습니다.

---

### `src/rag/embedder.py`

Sentence Transformer를 이용하여 문서 Chunk와 사용자 질문을 Embedding Vector로 변환합니다.

```text
Document Chunk
    |
    v
Sentence Transformer
    |
    v
Chunk Embedding
```

```text
User Question
    |
    v
Sentence Transformer
    |
    v
Query Embedding
```

문서와 질문 모두 동일한 TextEmbedder를 사용하여 같은 embedding space에 표현됩니다.

Sentence Transformer 모델 자체를 직접 구현하거나 학습하지는 않았으며, pretrained model을 사용합니다.

---

### `src/rag/vector_store.py`

Chunk와 Chunk Embedding을 메모리에 저장하고 검색하는 NumPy 기반 VectorStore입니다.

FAISS, Chroma 또는 별도의 Vector Database를 사용하지 않고 Cosine Similarity 기반 검색 로직을 직접 구현했습니다.

```text
Query Embedding
        |
        v
Compare with Chunk Embeddings
        |
        v
Cosine Similarity
        |
        v
Similarity Score
        |
        v
Sort by Score
        |
        v
Top-K Results
```

VectorStore는 질문을 저장하는 것이 아니라 다음 정보를 관리합니다.

* Chunk
* Chunk Embedding
* Metadata

---

### `src/rag/retriever.py`

질문 임베딩과 VectorStore 검색 과정을 하나의 검색 흐름으로 연결합니다.

```text
User Question
    |
    v
TextEmbedder
    |
    v
Query Embedding
    |
    v
VectorStore
    |
    v
Cosine Similarity
    |
    v
Top-K
    |
    v
SearchResult
```

---

### `src/rag/answer_generator.py`

Retriever가 반환한 검색 결과를 필터링하고 Gemini에 전달할 Context를 구성합니다.

```text
SearchResult
    |
    v
Score Filtering
    |
    v
Duplicate Chunk Removal
    |
    v
Chunk Count Limit
    |
    v
Context Length Limit
    |
    v
Context Building
    |
    v
Gemini API
    |
    v
Final Answer
```

현재 주요 설정값은 다음과 같습니다.

```python
min_search_score = 0.30
max_context_chunks = 3
max_context_length = 1500

temperature = 0.2
max_output_tokens = 500
```

---

### `src/chatbot.py`

IntentPredictor, Retriever, AnswerGenerator를 연결하고 사용자 질문의 처리 경로를 결정합니다.

```text
Chatbot
|
├── IntentPredictor
├── Retriever
└── AnswerGenerator
```

사용자 질문의 Intent와 confidence를 확인한 뒤 다음 세 가지 경로 중 하나를 선택합니다.

```text
Low Confidence
-> Fallback

General Intent
-> General Response

document_query
-> Retriever
-> AnswerGenerator
-> Gemini
```

---

### `main.py`

프로젝트에 필요한 주요 객체를 생성하고 서로 연결하는 역할을 담당합니다.

```text
IntentPredictor

TextEmbedder
    |
    v
VectorStore
    |
    v
Retriever

Gemini Client
    |
    v
AnswerGenerator

IntentPredictor
    +
Retriever
    +
AnswerGenerator
    |
    v
Chatbot
```

---

## 5. Intent Classification and Routing

사용자의 모든 질문을 바로 RAG에 전달하지 않습니다.

먼저 Intent Classifier가 질문의 Intent와 confidence를 예측합니다.

```text
User Question
    |
    v
IntentPredictor.predict()
    |
    v
Intent + Confidence
```

현재 confidence threshold는 다음과 같습니다.

```python
CONFIDENCE_THRESHOLD = 0.60
```

### Low Confidence

```text
confidence < 0.60
-> Fallback
```

### General Intent

일반 Intent로 판단되면 해당 Intent에 맞는 일반 응답을 반환합니다.

### Document Query

질문이 `document_query`로 분류되고 confidence가 threshold 이상이면 RAG 검색을 실행합니다.

```text
document_query
    |
    v
Retriever
    |
    v
AnswerGenerator
    |
    v
Gemini
```

---

## 6. Intent Confidence and Retriever Score

이 프로젝트에는 서로 다른 두 종류의 score가 존재합니다.

### Intent Confidence

Intent Classifier가 특정 Intent라고 판단하는 확신 정도입니다.

예시:

```text
Intent: document_query
Confidence: 0.82
```

현재 기준:

```python
CONFIDENCE_THRESHOLD = 0.60
```

---

### Retriever Score

질문 Embedding과 문서 Chunk Embedding 사이의 Cosine Similarity 기반 검색 점수입니다.

예시:

```text
Chunk 1: 0.72
Chunk 2: 0.64
Chunk 3: 0.57
```

두 score는 서로 다른 단계에서 사용됩니다.

```text
User Question
    |
    v
Intent Confidence
    |
    v
Confidence Threshold Check
    |
    v
document_query
    |
    v
Retriever
    |
    v
Retriever Score
```

따라서 다음 두 값은 서로 다른 값입니다.

```text
Intent Confidence != Retriever Score
```

---

## 7. Main Configuration

```python
CONFIDENCE_THRESHOLD = 0.60

RETRIEVAL_TOP_K = 3

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

min_search_score = 0.30
max_context_chunks = 3
max_context_length = 1500

temperature = 0.2
max_output_tokens = 500
```

### `CONFIDENCE_THRESHOLD`

Intent Classifier의 예측을 실제 Intent로 받아들일 최소 confidence입니다.

### `RETRIEVAL_TOP_K`

질문과 가장 유사한 Chunk를 Retriever가 최대 몇 개 반환할지 결정합니다.

### `CHUNK_SIZE`

문서를 나눌 때 사용하는 Chunk 크기입니다.

### `CHUNK_OVERLAP`

연속된 Chunk 사이에 중복으로 포함할 텍스트 크기입니다.

### `min_search_score`

Retriever 결과 중 AnswerGenerator의 Context 후보로 사용할 최소 검색 score입니다.

### `max_context_chunks`

Gemini Context에 실제로 포함할 최대 Chunk 개수입니다.

### `max_context_length`

Gemini에 전달할 전체 Context 길이의 최대값입니다.

### `temperature`

Gemini 답변 생성의 무작위성을 조절합니다.

### `max_output_tokens`

Gemini가 생성하는 최종 답변의 최대 출력 token 수입니다.

---

## 8. Gemini Configuration

사용 중인 Python SDK:

```text
google-genai==2.18.0
```

프로젝트에서 설정한 Gemini 모델:

```text
gemini-3.5-flash-lite
```

Gemini API Key는 코드에 직접 작성하지 않고 Google Colab Secrets에 저장합니다.

Secret 이름:

```text
GEMINI_API_KEY
```

AnswerGenerator는 다음과 같은 형태로 생성합니다.

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

Chatbot은 다음과 같이 주요 객체를 연결하여 생성합니다.

```python
chatbot = Chatbot(
    intent_predictor=intent_predictor,
    retriever=retriever,
    answer_generator=answer_generator,
    confidence_threshold=0.60,
    retrieval_top_k=3,
    preview_length=200,
)
```

---

## 9. TXT Upload Pipeline

Google Colab의 `files.upload()`을 이용하여 사용자 PC에서 TXT 파일을 직접 업로드할 수 있습니다.

`files.upload()`의 반환값은 개념적으로 다음과 같습니다.

```python
{
    "file1.txt": b"...",
    "file2.txt": b"...",
}
```

여기서:

```text
key   -> filename
value -> file bytes
```

업로드된 파일은 다음 과정을 거칩니다.

```text
User PC
    |
    v
files.upload()
    |
    v
Filename + Bytes
    |
    v
.txt Extension Check
    |
    v
Write File
    |
    v
data/uploaded_documents/
```

현재 Document Loader가 TXT 파일을 대상으로 구현되어 있으므로 `.txt`가 아닌 파일은 검색 대상에서 제외합니다.

---

## 10. RAG Index Rebuild

새 TXT 파일을 디스크에 저장하더라도 기존 VectorStore 객체가 자동으로 변경되지는 않습니다.

```text
File System != In-memory VectorStore
```

따라서 새 문서를 업로드하면 현재 업로드된 TXT 문서를 다시 처리하여 검색 환경을 재구성합니다.

```text
Uploaded TXT Files
    |
    v
Load Documents
    |
    v
Preprocessing
    |
    v
Chunking
    |
    v
Embedding
    |
    v
New VectorStore
    |
    v
New Retriever
    |
    v
New Chatbot
```

이 과정은 모델 재학습이 아닙니다.

```text
Intent Classifier Retraining    -> No
Sentence Transformer Retraining -> No
Gemini Retraining               -> No

RAG Index Rebuild               -> Yes
```

---

## 11. Training and Inference

### Intent Classifier Training

```text
intents.json
    |
    v
Dataset
    |
    v
IntentClassifier
    |
    v
Forward
    |
    v
Loss
    |
    v
Backpropagation
    |
    v
Optimizer
    |
    v
intent_classifier.pt
```

`train.py`는 Intent Classifier를 학습할 때 사용합니다.

---

### Chatbot Inference

실제 챗봇을 사용할 때는 이미 저장된 모델을 불러옵니다.

```text
IntentClassifier Structure
        +
intent_classifier.pt
        |
        v
IntentPredictor
        |
        v
User Question
        |
        v
Intent + Confidence
```

따라서 챗봇을 실행할 때마다 Intent Classifier를 다시 학습하지 않습니다.

---

## 12. Document Preparation and Query Time

### Document Preparation

```text
TXT
 |
 v
Document
 |
 v
Preprocessing
 |
 v
Chunk
 |
 v
Chunk Embedding
 |
 v
VectorStore
 |
 v
Retriever Ready
```

문서 Embedding은 Retriever를 준비하는 과정에서 생성됩니다.

### Query Time

```text
User Question
    |
    v
Query Embedding
    |
    v
Compare with Chunk Embeddings
    |
    v
Cosine Similarity
    |
    v
Top-K
    |
    v
SearchResult
```

질문 Embedding은 사용자가 실제 질문을 입력했을 때 생성됩니다.

문서와 질문은 동일한 TextEmbedder를 사용합니다.

---

## 13. Testing

프로젝트의 각 기능을 분리하여 테스트한 뒤 전체 Chatbot을 통합 테스트했습니다.

### Retriever Test

Intent Classifier와 Gemini를 제외하고 검색 기능 자체를 확인합니다.

```text
Question
    |
    v
Retriever
    |
    v
Query Embedding
    |
    v
VectorStore
    |
    v
Top-K
    |
    v
SearchResult
```

주요 확인 항목:

* 검색 결과 생성 여부
* 질문과 검색 Chunk의 관련성
* Top-K 결과
* Retriever score
* `source`
* `chunk_id`

---

### AnswerGenerator Test

Retriever 결과가 주어졌을 때 최종 답변 생성 과정이 정상적으로 동작하는지 확인합니다.

주요 확인 항목:

* `min_search_score` 적용
* 낮은 score의 검색 결과 제거
* 중복 Chunk 제거
* Context 구성
* Context Chunk 개수 제한
* Context 길이 제한
* Gemini API 호출
* 검색된 Context 기반 답변 생성

---

### Full Chatbot Test

Intent Classification과 RAG를 연결한 전체 Chatbot을 테스트했습니다.

검증한 주요 경우:

* 일반 Intent
* Low-confidence fallback
* `document_query`
* 여러 TXT 문서 검색
* 여러 문서의 `source` 구분
* 반복 질문
* `exit`
* `quit`
* `종료`

Retriever, AnswerGenerator, 전체 Chatbot을 단계별로 테스트하여 검색 문제와 Intent routing 문제를 분리해 확인할 수 있도록 했습니다.

---

## 14. Implementation Scope

### 직접 구현 또는 구성한 부분

* PyTorch 기반 Intent Classifier 구조
* Intent Dataset 처리
* Intent Classifier 학습 루프
* IntentPredictor
* Confidence 기반 fallback routing
* TXT 문서 로딩 및 전처리 흐름
* Chunk 분할 및 overlap
* NumPy 기반 in-memory VectorStore
* Cosine Similarity 기반 검색
* Top-K 검색
* Retriever
* Intent Classifier와 RAG routing
* AnswerGenerator 처리 흐름
* Retriever score filtering
* 중복 Chunk 제거
* Context 구성
* `source`, `chunk_id` 관리
* 사용자 TXT 업로드 처리
* 업로드 문서 전용 폴더 관리
* 업로드 후 VectorStore 및 Retriever 재생성
* 새 Retriever와 Chatbot 연결
* 전체 모듈 조립 및 테스트

### 외부 라이브러리 및 Pretrained Model

* PyTorch
* NumPy
* Sentence Transformer
* Gemini API
* `google-genai`
* Google Colab `files.upload()`

Sentence Transformer 모델 자체를 직접 구현하거나 학습한 것은 아닙니다.

Gemini 또는 LLM 자체를 직접 구현하거나 학습한 것도 아닙니다.

이 프로젝트에서는 pretrained Sentence Transformer를 Embedding Model로 사용하고 Gemini를 최종 자연어 생성 모델로 활용합니다.

---

## 15. Development Environment

```text
Development Environment
- Google Colab

Project Root
- /content/drive/MyDrive/rag_intent_chatbot
```

---

## 16. Installation

Google Drive를 연결합니다.

```python
from google.colab import drive

drive.mount("/content/drive")
```

프로젝트 디렉터리로 이동합니다.

```python
%cd /content/drive/MyDrive/rag_intent_chatbot
```

필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

---

## 17. Gemini API Key

Gemini API 사용을 위해 Google Colab Secrets에 API Key를 등록해야 합니다.

Secret 이름:

```text
GEMINI_API_KEY
```

API Key는 코드나 GitHub 저장소에 직접 포함하지 않습니다.

---

## 18. How to Run

현재 프로젝트는 Google Colab을 중심으로 실행합니다.

```text
1. Mount Google Drive
2. Move to Project Directory
3. Install requirements.txt
4. Load GEMINI_API_KEY
5. Load intent_classifier.pt
6. Upload TXT Files
7. Load Documents
8. Preprocess Documents
9. Create Chunks
10. Generate Chunk Embeddings
11. Create VectorStore
12. Create Retriever
13. Create Gemini Client
14. Create AnswerGenerator
15. Create Chatbot
16. Enter User Question
17. Run Intent Classification
18. Run RAG if Intent is document_query
19. Generate Gemini Response
20. Exit with exit, quit, or 종료
```

TXT 파일 업로드를 포함한 최종 실행은 다음 Notebook을 중심으로 진행합니다.

```text
08_TXT_업로드_RAG_챗봇.ipynb
```

---

## 19. Current Limitations

현재 프로젝트는 RAG 구조를 직접 이해하기 위한 소규모 학습 프로젝트이며 다음과 같은 한계가 있습니다.

* Intent 학습 데이터의 규모와 표현 다양성이 제한적일 경우 일반화 성능이 떨어질 수 있습니다.
* 실제 문서 관련 질문이어도 Intent confidence가 threshold보다 낮으면 fallback으로 처리될 수 있습니다.
* NumPy 기반 VectorStore는 모든 Chunk Embedding과 직접 유사도를 계산하므로 대규모 문서 검색에는 적합하지 않습니다.
* 현재 TXT 파일만 직접 지원합니다.
* 새 문서를 추가하면 업로드된 문서 전체를 다시 처리하여 VectorStore와 Retriever를 재생성합니다.
* Incremental Indexing을 지원하지 않습니다.
* VectorStore는 메모리 기반이며 검색 인덱스를 별도로 영구 저장하지 않습니다.
* 검색 품질은 사용하는 pretrained Sentence Transformer의 표현 성능에 영향을 받습니다.
* 최종 답변 생성을 Gemini API에 의존하므로 인터넷 연결과 API Key가 필요합니다.
* Intent Classification 및 RAG 검색에 대한 체계적인 정량 평가 시스템은 구축하지 않았습니다.
* 반복 질문은 가능하지만 이전 대화를 Context로 사용하는 Multi-turn Conversation Memory는 구현하지 않았습니다.
* 일반 사용자를 위한 별도의 Web UI는 없습니다.
* Google Colab과 Google Drive 중심으로 실행됩니다.

---

## 20. Future Improvements

다음 항목은 현재 구현되어 있지 않으며, 프로젝트를 실제 서비스 형태로 확장할 경우 고려할 수 있는 개선 방향입니다.

* PDF / DOCX Document Loader
* FAISS 또는 Vector Database
* Incremental Indexing
* Intent Dataset 확장
* Intent Classifier 정량 평가
* RAG 검색 성능 정량 평가
* FastAPI Backend
* Web Frontend
* 사용자별 문서 저장
* 사용자 또는 세션별 Retriever
* Conversation History
* Docker
* Cloud Deployment

---

## 21. Tech Stack

| Category                | Technology           |
| ----------------------- | -------------------- |
| Language                | Python               |
| Deep Learning           | PyTorch              |
| Numerical Computing     | NumPy                |
| Embedding               | Sentence Transformer |
| LLM                     | Gemini               |
| Gemini SDK              | google-genai         |
| Development Environment | Google Colab         |
| Storage                 | Google Drive         |

---

## 22. What I Learned

이 프로젝트를 통해 다음 내용을 직접 구현하고 연결하면서 RAG 챗봇의 전체 흐름을 학습했습니다.

* 딥러닝 모델의 학습과 추론 분리
* PyTorch 기반 Intent Classification
* Confidence 기반 routing
* 문서 preprocessing 및 Chunking
* Chunk overlap
* 문서와 질문의 Embedding
* Cosine Similarity 기반 검색
* Top-K Retrieval
* in-memory VectorStore 구조
* Retriever의 역할
* 검색 결과의 score filtering
* RAG Context 구성
* LLM API 연동
* Intent Classifier와 RAG 통합
* Python 객체 간 의존 관계
* 문서 업로드 후 검색 인덱스 재구성
* 단계별 테스트를 이용한 오류 위치 분리

완성된 RAG Framework를 바로 사용하는 대신 VectorStore, Retriever, Routing 등의 핵심 구조를 직접 구현하여 문서가 Embedding Vector로 변환되고, 질문과 비교되며, 검색 결과가 LLM Context로 전달되는 과정을 코드 수준에서 이해하는 데 중점을 두었습니다.

---

## 23. Summary

```text
PyTorch Intent Classifier
        |
        v
Intent + Confidence
        |
        v
Chatbot Routing
        |
        v
Sentence Transformer
        |
        v
NumPy VectorStore
        |
        v
Cosine Similarity + Top-K
        |
        v
Retriever
        |
        v
AnswerGenerator
        |
        v
Gemini
        |
        v
Document-based Response
```

PyTorch 기반 Intent Classification과 NumPy 기반 RAG 검색 구조를 결합하여, 사용자가 업로드한 TXT 문서를 검색하고 Gemini가 검색된 Context를 기반으로 답변하는 챗봇을 구현했습니다.
