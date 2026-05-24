# 아키텍처 문서

> Fine-tuning CPU MVP — KoGPT2 기반 한국 민법(부동산/전세) Q&A 모델을 CPU 환경에서 LoRA로 파인튜닝하고, FastAPI 백엔드와 React 프론트엔드로 서빙하는 MVP 프로젝트.

---

## 1. 시스템 개요

본 프로젝트는 **GPU 없이 CPU만으로 한국어 LLM(KoGPT2)을 LoRA 파인튜닝**하고, 그 모델로 추론까지 수행할 수 있는 단일 사용자(MVP) 웹 애플리케이션이다. 핵심 가치는 다음과 같다.

- **로컬 완결성**: 모든 학습/추론이 로컬 CPU에서 수행되며, 외부 API 의존성이 없다.
- **저자원 친화**: LoRA(r=8), `batch_size=1` + `gradient_accumulation=32`, FP32 등 CPU 환경에 맞춘 설정.
- **얇은 레이어링**: 기존 CLI 학습/추론 스크립트(`main_train_cpu.py`, `inference_manager.py` 등)를 **그대로 두고**, FastAPI 레이어(`backend/app/`)가 이를 wrap 한다.
- **SSE 기반 실시간 로그**: 학습 프로세스의 stdout을 파일로 떨어뜨리고, 백엔드가 `tail -f` 방식으로 SSE 스트리밍.

---

## 2. 디렉터리 구조

```
fine_tuning_cpu_base_model/
├── backend/
│   ├── app/                         # FastAPI 애플리케이션 (얇은 서빙 레이어)
│   │   ├── main.py                  # FastAPI 엔트리·CORS·라우터 등록·정적 파일 마운트
│   │   ├── config.py                # 스토리지 경로, FRONTEND_DIST, CORS origins
│   │   ├── schemas.py               # Pydantic 스키마 (요청/응답 DTO)
│   │   ├── routes/                  # HTTP 엔드포인트
│   │   │   ├── datasets.py          # CSV 업로드/목록/삭제
│   │   │   ├── train.py             # 학습 시작/조회/중단/SSE 로그 스트림
│   │   │   ├── models.py            # 모델 목록/삭제
│   │   │   └── infer.py             # 추론 (모델 로더 캐싱)
│   │   └── services/
│   │       ├── job_store.py         # In-memory job 상태 (싱글톤, 락 보호)
│   │       └── train_runner.py      # subprocess로 main_train_cpu.py 실행
│   │
│   ├── main_train_cpu.py            # [기존 CLI] 학습 진입점
│   ├── main_inference_cpu.py        # [기존 CLI] 추론 진입점 (대화형/단일/벤치마크)
│   ├── model_manager_cpu.py         # CPUModelManager / InferenceModelManager
│   ├── data_loader.py               # CSV → Dataset → 토크나이즈
│   ├── trainer.py                   # TrainingManager (HF Trainer 래핑)
│   ├── inference_manager.py         # InferenceManager / ChatBot
│   ├── config_cpu.py                # 모델/LoRA/훈련/추론 dataclass 설정
│   ├── utils.py                     # 로깅·CSV 검증·진행률 등 공통 유틸
│   └── storage/                     # 런타임 산출물 (gitignored)
│       ├── datasets/  <dataset_id>.csv
│       ├── models/    <job_id>/      ← LoRA 어댑터, tokenizer, training_stats.json
│       └── logs/      <job_id>.log
│
├── frontend/                        # React + Vite + TS + Tailwind (Node 18)
│   ├── src/
│   │   ├── main.tsx                 # BrowserRouter + <App/>
│   │   ├── App.tsx                  # 헤더 네비 + Routes
│   │   ├── api/client.ts            # fetch 래퍼 + DTO 타입 + SSE 헬퍼
│   │   └── pages/
│   │       ├── Dashboard.tsx        # 최근 작업/모델 요약
│   │       ├── Train.tsx            # CSV 업로드 + 하이퍼파라미터 + 실시간 로그
│   │       ├── Evaluate.tsx         # 모델 선택 + 질문 입력 + 배치 응답
│   │       └── Models.tsx           # 모델 목록/삭제
│   └── (vite, tailwind, tsconfig)
│
└── docs/                            # 본 문서가 위치
```

---

## 3. 레이어 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│  Presentation (Browser)                                          │
│  React 18 + React Router + Tailwind 3.4                          │
│   Dashboard / Train / Evaluate / Models                          │
└──────────────────┬───────────────────────────────────────────────┘
                   │  fetch JSON · multipart upload · SSE
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  API Layer  (backend/app)                                        │
│  FastAPI 0.115 + uvicorn                                         │
│   routes/datasets · routes/train · routes/models · routes/infer  │
│   services/job_store · services/train_runner                     │
└──────────────────┬─────────────────────────────┬─────────────────┘
                   │ subprocess(Popen)            │ in-process import
                   ▼                              ▼
┌──────────────────────────────────────┐  ┌──────────────────────────┐
│  Training Process (별도 프로세스)     │  │  Inference (in-process)   │
│  main_train_cpu.py                    │  │  inference_manager        │
│   → CPUModelManager (HF + PEFT)       │  │   ← model_manager_cpu     │
│   → DataLoader (CSV → tokens)         │  │   ← config_cpu            │
│   → TrainingManager (HF Trainer)      │  │  모델은 _cache 딕셔너리     │
│  stdout → storage/logs/<job_id>.log   │  │  에 model_id별 캐시       │
└──────────────────┬───────────────────┘  └──────────┬───────────────┘
                   │                                  │
                   ▼                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│  Persistence (로컬 파일시스템)                                     │
│  storage/datasets/*.csv · storage/models/<job_id>/ · storage/logs│
└──────────────────────────────────────────────────────────────────┘
```

### 3.1 Presentation Layer (frontend/)
- **스택**: React 18, Vite 5, TypeScript 5, Tailwind 3.4, react-router-dom v6, Node 18 핀(`.nvmrc`).
- **상태 관리**: 별도 라이브러리 없이 `useState` + `useEffect` 만 사용. MVP 규모에 적합.
- **API 어댑터**: `src/api/client.ts` 가 모든 백엔드 호출과 SSE 구독을 단일 진입점으로 노출.
- **개발 모드**: Vite dev server(5173) → FastAPI(보통 8000) 로 직접 호출. CORS 는 `allow_origins=["http://localhost:5173", ...]` 로 허용.
- **프로덕션 모드**: `npm run build` → `frontend/dist/` 생성, FastAPI 가 존재 시 `StaticFiles` 로 `/` 에 마운트하여 **동일 origin 서빙**.

### 3.2 API Layer (backend/app/)
- **얇은 라우터**: 각 라우터는 Pydantic 스키마 검증과 HTTPException 매핑만 담당, 비즈니스 로직은 `services/` 로 위임.
- **CORS / 정적 파일**: `main.py` 가 라우터를 등록한 뒤 빌드된 프론트엔드가 있으면 같은 앱이 SPA 도 서빙.
- **공통 설정**: `config.py` 가 `STORAGE_ROOT` 하위에 `datasets/`, `models/`, `logs/` 를 부팅 시 보장.

### 3.3 Service Layer
- **`job_store.JobStore`**: 메모리 딕셔너리 + `threading.Lock`. 단일 사용자 가정으로 **동시에 1개의 running 잡만 허용** (`create()` 가 `RuntimeError` 로 거부).
- **`train_runner`**: `subprocess.Popen(start_new_session=True)` 로 학습 프로세스를 띄우고, 별도 데몬 스레드가 `proc.wait()` 후 status/exit_code 를 `job_store` 에 업데이트. 중단은 `os.killpg(SIGTERM)` 으로 프로세스 그룹 전체 종료.

### 3.4 Domain / Compute Layer (backend/*.py, 루트의 기존 스크립트들)
- **모델 라이프사이클** (`model_manager_cpu.py`):
  - 학습: `CPUModelManager` → `load_tokenizer()` → `load_base_model()` (`device_map={"":cpu}`, FP32, `low_cpu_mem_usage=True`) → `setup_lora()` (`r=8, alpha=16, target=[c_attn,c_proj]`).
  - 추론: `InferenceModelManager.load_finetuned_model()` 가 base model 로드 → `resize_token_embeddings` → `PeftModel.from_pretrained(base, adapter_path)`.
- **데이터 파이프라인** (`data_loader.py`):
  - `pd.read_csv(utf-8-sig)` → `question/answer` 컬럼 검증 → `format_prompt()` 로 `"질문: ...\n답변: ...</s>"` 형식 통일 → `Dataset.map(tokenize, batched=True)` → `labels = input_ids.copy()` (Causal LM).
- **트레이너** (`trainer.py`):
  - HF `TrainingArguments` + `DataCollatorForLanguageModeling(mlm=False)` + `Trainer` 조합.
  - CPU 특수성: `fp16=False`, `gradient_checkpointing=False`, `dataloader_num_workers=0`, `dataloader_pin_memory=False`.
- **추론** (`inference_manager.py`):
  - `InferenceManager.generate_response()` 가 `format_prompt(question)` → tokenize → `model.generate(...)` → 신규 토큰만 디코드 → 후처리.
  - 라우터 레벨에서 model_id 별 캐시(`_cache: dict`) 로 첫 호출 후 재로딩 비용 제거.

### 3.5 Persistence Layer
- **DB 없음** — 모든 상태는 파일시스템 + 메모리.
- **데이터셋**: `storage/datasets/<dataset_id>.csv` (업로드 시 12-hex UUID 부여).
- **모델 산출물**: `storage/models/<job_id>/` — `adapter_config.json` 의 존재로 LoRA 어댑터 디렉터리 식별.
- **학습 로그**: `storage/logs/<job_id>.log` — stdout/stderr 통합 캡처, SSE 스트림의 원천.

---

## 4. 데이터 플로우

### 4.1 학습 플로우
1. 사용자가 `Train` 페이지에서 CSV 업로드 → `POST /api/datasets/upload` → 컬럼 검증 후 `storage/datasets/<dataset_id>.csv` 저장.
2. 하이퍼파라미터 입력 후 **학습 시작** → `POST /api/train`.
3. `train_runner.start_training()`:
   - `job_store.create(job_id, req)` — 동시 실행 1개 제약 검사.
   - `subprocess.Popen(["python", "-u", "main_train_cpu.py", ...])`, `CUDA_VISIBLE_DEVICES=""`.
   - stdout → `storage/logs/<job_id>.log`.
   - 데몬 스레드 `_watch_process` 가 종료 코드에 따라 status를 `completed/failed/stopped` 로 업데이트.
4. 프론트엔드는 `streamJobLog(jobId)` 로 `GET /api/train/jobs/<id>/stream` 을 `EventSource` 구독. 백엔드는 0.5초 폴링으로 로그 파일을 읽어 `data:` 이벤트로 푸시, 잡 종료 시 `event: end` 송신.
5. 학습 종료 시 `main_train_cpu.py` 는 `<output_dir>` 에 LoRA 어댑터 + tokenizer + `training_stats.json` 저장.

### 4.2 추론 플로우
1. `Evaluate` 페이지에서 모델/질문 입력 → `POST /api/infer`.
2. `routes/infer._load(model_id)`:
   - `_cache` 히트 시 즉시 반환.
   - 미스 시 `create_loader(model_dir).load_finetuned_model()` 로 base 모델 + LoRA 어댑터 결합, `create_inference_manager(model, tokenizer)` 캐싱.
3. `inf.generate_response(question, ...)` → 응답 + `elapsed_ms` JSON 반환.

### 4.3 라이프사이클 플로우 (모델·잡 관리)
- `GET /api/models` 는 `storage/models/*/adapter_config.json` 존재 디렉터리만 노출.
- `DELETE /api/models/<id>` 는 `shutil.rmtree` — 캐시 무효화는 별도 구현되어 있지 않음(서버 재기동 필요).
- `GET /api/train/jobs` 는 `job_store` 의 메모리 상태 — 서버 재기동 시 잡 이력 휘발.

---

## 5. 주요 API 표

| Method | Path | 입력 | 출력 | 비고 |
|---|---|---|---|---|
| GET  | `/api/health` | — | `{status:"ok"}` | liveness |
| GET  | `/api/datasets` | — | `DatasetInfo[]` | |
| POST | `/api/datasets/upload` | `multipart file` | `DatasetInfo` | `question`,`answer` 컬럼 필수 |
| DELETE | `/api/datasets/{id}` | — | `{deleted}` | |
| POST | `/api/train` | `TrainRequest` | `JobInfo` | 동시 1개 제한 (409) |
| GET  | `/api/train/jobs` | — | `JobInfo[]` | |
| GET  | `/api/train/jobs/{id}` | — | `JobInfo` | |
| POST | `/api/train/jobs/{id}/stop` | — | `{stopped}` | SIGTERM 프로세스 그룹 |
| GET  | `/api/train/jobs/{id}/stream` | — | `text/event-stream` | SSE, `event:end` 로 종료 |
| GET  | `/api/models` | — | `ModelInfo[]` | adapter_config 보유 디렉터리만 |
| DELETE | `/api/models/{id}` | — | `{deleted}` | rmtree |
| POST | `/api/infer` | `InferRequest` | `InferResponse` | 모델 캐싱 |

---

## 6. 동시성 모델

- **싱글 사용자/싱글 학습잡 가정**: `JobStore.create()` 가 running 잡 존재 시 `RuntimeError` 던짐 → 409 매핑.
- **프로세스 격리**: 학습은 별도 프로세스(`Popen`), 추론은 FastAPI 워커 내부 in-process. PyTorch 의 `torch.set_num_threads(physical_cores)` 로 CPU 코어 활용.
- **SSE 백프레셔**: 파일 EOF 시 0.5s sleep 후 재읽기. 단일 사용자 기준으로 충분.
- **캐시 일관성**: `routes/infer._cache` 는 동기화 락이 없음 — 단일 워커(uvicorn 기본) 가정.

---

## 7. 설정·환경

- **Python**: 3.9 (Conda env `py39_pt`, torch 2.8 GPU 빌드를 CPU 모드로 강제 사용).
- **CPU 강제**: 학습 subprocess 의 `env={..."CUDA_VISIBLE_DEVICES":""}` + 코드 내 `os.environ.setdefault("CUDA_VISIBLE_DEVICES","")`.
- **Node**: 18 핀 (`.nvmrc`).
- **베이스 모델**: `skt/kogpt2-base-v2` (한국어 GPT-2, ~125M params).
- **LoRA 기본값**: `r=8, alpha=16, dropout=0.1, target=[c_attn, c_proj]`.
- **학습 기본값**: `max_steps=100, batch=1, grad_accum=32, lr=5e-5, fp16=False`.

---

## 8. 배포·실행

- **개발**:
  1. `cd backend && uvicorn app.main:app --reload --port 8000`
  2. `cd frontend && npm run dev` (Vite 5173).
- **단일 origin 배포**:
  1. `cd frontend && npm run build` → `frontend/dist/`.
  2. `uvicorn app.main:app` 만 띄우면 `FRONTEND_DIST` 가 존재하므로 `/` 에 SPA, `/api/*` 에 API 가 같이 노출.

---

## 9. 알려진 한계 및 향후 개선 포인트

| 영역 | 한계 | 개선 방향 |
|---|---|---|
| 잡 영속성 | `JobStore` 는 메모리, 재기동 시 휘발 | SQLite 또는 JSON 파일 기반 영속 스토어 |
| 멀티 사용자 | 동시 학습 1개, 인증/권한 없음 | 토큰 기반 인증 + 큐(Celery/RQ) |
| 모델 캐시 무효화 | 삭제 후 캐시 잔존 | `DELETE` 시 캐시 키 제거 훅 |
| 평가 메트릭 | 단일 질의응답만 지원, 메트릭 없음 | BLEU/ROUGE/perplexity 일괄 평가 잡 |
| 추론 성능 | CPU에서 토큰 생성 느림 | onnxruntime / ggml 변환 옵션 |
| 보안 | `model_id` 가 경로에 직접 매핑 | path traversal 방어 (이미 디렉터리 존재 검사 수준) |
