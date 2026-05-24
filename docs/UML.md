# UML 다이어그램

> 본 문서는 Fine-tuning CPU MVP 의 정적/동적 구조를 [Mermaid](https://mermaid.js.org/) 표기법으로 정리한다. GitHub·VSCode·IntelliJ 등 대부분의 마크다운 뷰어에서 직접 렌더링된다.

목차
- [1. 컴포넌트 다이어그램](#1-컴포넌트-다이어그램)
- [2. 클래스 다이어그램 — 백엔드 도메인](#2-클래스-다이어그램--백엔드-도메인)
- [3. 클래스 다이어그램 — API/서비스 레이어](#3-클래스-다이어그램--api서비스-레이어)
- [4. 클래스 다이어그램 — 프론트엔드](#4-클래스-다이어그램--프론트엔드)
- [5. 시퀀스 다이어그램 — 학습 시작 + SSE 로그](#5-시퀀스-다이어그램--학습-시작--sse-로그)
- [6. 시퀀스 다이어그램 — 추론](#6-시퀀스-다이어그램--추론)
- [7. 시퀀스 다이어그램 — 학습 중단](#7-시퀀스-다이어그램--학습-중단)
- [8. 상태 다이어그램 — Job 라이프사이클](#8-상태-다이어그램--job-라이프사이클)
- [9. 액티비티 다이어그램 — 데이터 처리 파이프라인](#9-액티비티-다이어그램--데이터-처리-파이프라인)
- [10. 배포 다이어그램](#10-배포-다이어그램)

---

## 1. 컴포넌트 다이어그램

```mermaid
flowchart LR
    subgraph Browser["Browser (SPA)"]
        UI["React App<br/>(Dashboard / Train / Evaluate / Models)"]
        APICLIENT["api/client.ts<br/>(fetch + EventSource)"]
        UI --> APICLIENT
    end

    subgraph FastAPI["FastAPI app (uvicorn, single worker)"]
        ROUTES["routes/<br/>datasets · train · models · infer"]
        SCHEMAS["schemas.py<br/>(Pydantic DTO)"]
        SERVICES["services/<br/>job_store · train_runner"]
        STATIC["StaticFiles ('/')<br/>frontend/dist"]
        ROUTES --> SCHEMAS
        ROUTES --> SERVICES
    end

    subgraph Domain["Domain / Compute (backend/*.py)"]
        MM["CPUModelManager<br/>InferenceModelManager"]
        DL["DataLoader"]
        TM["TrainingManager"]
        IM["InferenceManager"]
        CFG["config_cpu<br/>(dataclasses)"]
        MM --> CFG
        DL --> CFG
        TM --> CFG
        IM --> CFG
    end

    subgraph FS["Local Filesystem (storage/)"]
        DS["datasets/*.csv"]
        MD["models/[job_id]/<br/>(adapter + tokenizer)"]
        LG["logs/[job_id].log"]
    end

    subgraph TrainProc["Training Subprocess<br/>main_train_cpu.py"]
        TP["uses MM + DL + TM"]
    end

    APICLIENT -- "JSON / multipart" --> ROUTES
    APICLIENT -- "SSE" --> ROUTES
    APICLIENT -- "static" --> STATIC

    SERVICES -- "Popen(start_new_session)" --> TrainProc
    ROUTES -- "in-process import" --> MM
    ROUTES -- "in-process import" --> IM

    TP --> MM
    TP --> DL
    TP --> TM
    TP -- "stdout/stderr" --> LG
    TP -- "save_pretrained" --> MD

    ROUTES --> DS
    ROUTES --> MD
    ROUTES --> LG
```

---

## 2. 클래스 다이어그램 — 백엔드 도메인

```mermaid
classDiagram
    direction LR

    class ModelConfig {
        +str model_name = "skt/kogpt2-base-v2"
        +int max_length = 256
        +dtype torch_dtype = float32
        +bool low_cpu_mem_usage = true
    }
    class LoRAConfig {
        +int r = 8
        +int lora_alpha = 16
        +float lora_dropout = 0.1
        +List~str~ target_modules = [c_attn, c_proj]
    }
    class TrainingConfig {
        +str csv_path
        +str output_dir
        +int batch_size = 1
        +int gradient_accumulation_steps = 32
        +float learning_rate = 5e-5
        +int num_train_epochs = 1
        +int max_steps = 100
        +bool fp16 = false
    }
    class InferenceConfig {
        +int max_new_tokens = 256
        +float temperature = 0.8
        +float top_p = 0.9
        +float repetition_penalty = 1.1
    }

    class CPUModelManager {
        -torch.device device = cpu
        -AutoTokenizer tokenizer
        -PreTrainedModel model
        +load_tokenizer(name) AutoTokenizer
        +load_base_model(name) AutoModelForCausalLM
        +setup_lora(model) PeftModel
        +load_model_and_tokenizer() tuple
        +save_model(output_dir, tokenizer)
    }

    class InferenceModelManager {
        -str model_path
        -str base_model
        -torch.device device = cpu
        -AutoTokenizer tokenizer
        -PeftModel model
        +load_finetuned_model() tuple
    }

    class DataLoader {
        -AutoTokenizer tokenizer
        -int max_length = 256
        +load_csv_data(path) Dataset
        +tokenize_dataset(dataset) Dataset
        +get_data_statistics(dataset) Dict
        -_create_dataset(df) Dataset
    }

    class TrainingManager {
        -PreTrainedModel model
        -AutoTokenizer tokenizer
        -Trainer trainer
        +create_training_arguments(output_dir) TrainingArguments
        +create_data_collator() DataCollatorForLanguageModeling
        +create_trainer(dataset, output_dir) Trainer
        +train(dataset, output_dir) TrainResult
        +save_model(output_dir)
    }

    class CustomTrainer {
        +compute_loss(model, inputs, ...) Tensor
    }

    class InferenceManager {
        -PreTrainedModel model
        -AutoTokenizer tokenizer
        -dict generation_config
        +generate_response(question, **kwargs) str
        +generate_batch_responses(questions) List~str~
        +update_generation_config(**kwargs)
        +interactive_chat()
        -_post_process_response(text) str
    }

    class ChatBot {
        -InferenceManager inference_manager
        -list conversation_history
        +chat(question) str
        +save_conversation(filepath)
        +clear_history()
    }

    CustomTrainer --|> Trainer : extends
    TrainingManager o-- Trainer : uses
    CPUModelManager ..> ModelConfig : reads
    CPUModelManager ..> LoRAConfig : reads
    TrainingManager ..> TrainingConfig : reads
    InferenceManager ..> InferenceConfig : reads
    DataLoader ..> ModelConfig : reads
    DataLoader ..> TrainingConfig : reads
    InferenceModelManager ..> ModelConfig : reads
    ChatBot o-- InferenceManager
```

---

## 3. 클래스 다이어그램 — API/서비스 레이어

```mermaid
classDiagram
    direction LR

    class FastAPIApp {
        +include_router(datasets)
        +include_router(train)
        +include_router(models)
        +include_router(infer)
        +mount("/", StaticFiles)
        +health() dict
    }

    class TrainRequest {
        +str dataset_id
        +str model_name
        +int max_steps
        +int batch_size
        +float learning_rate
        +int epochs
    }
    class JobInfo {
        +str job_id
        +Literal status
        +str started_at
        +Optional finished_at
        +Optional exit_code
        +Optional output_dir
        +Optional request
    }
    class InferRequest {
        +str model_id
        +str question
        +int max_new_tokens
        +float temperature
    }
    class InferResponse {
        +str question
        +str answer
        +int elapsed_ms
    }
    class DatasetInfo {
        +str dataset_id
        +str filename
        +int rows
        +int size_bytes
    }
    class ModelInfo {
        +str model_id
        +str path
        +str created_at
        +int size_bytes
    }

    class JobStore {
        -Dict~str,dict~ _jobs
        -threading.Lock _lock
        +create(job_id, request) dict
        +get(job_id) Optional~dict~
        +list() list
        +update(job_id, **fields)
        +running_job() Optional~dict~
    }
    note for JobStore "단일 사용자 가정.\ncreate() 가 running 잡 존재 시 RuntimeError"

    class TrainRunner {
        <<module>>
        +start_training(req) str
        +stop_training(job_id) bool
        -_watch_process(job_id, proc, output_dir)
    }

    class DatasetsRouter {
        <<router /api/datasets>>
        +list_datasets() List~DatasetInfo~
        +upload_dataset(file) DatasetInfo
        +delete_dataset(id)
    }
    class TrainRouter {
        <<router /api/train>>
        +start(req) JobInfo
        +list_jobs() List~JobInfo~
        +get_job(id) JobInfo
        +stop(id)
        +stream_log(id) StreamingResponse
    }
    class ModelsRouter {
        <<router /api/models>>
        +list_models() List~ModelInfo~
        +delete_model(id)
    }
    class InferRouter {
        <<router /api/infer>>
        -dict _cache
        +infer(req) InferResponse
        -_load(model_id) InferenceManager
    }

    FastAPIApp o-- DatasetsRouter
    FastAPIApp o-- TrainRouter
    FastAPIApp o-- ModelsRouter
    FastAPIApp o-- InferRouter

    TrainRouter ..> TrainRequest
    TrainRouter ..> JobInfo
    TrainRouter ..> JobStore
    TrainRouter ..> TrainRunner
    TrainRunner ..> JobStore
    TrainRunner ..> TrainRequest

    InferRouter ..> InferRequest
    InferRouter ..> InferResponse
    DatasetsRouter ..> DatasetInfo
    ModelsRouter ..> ModelInfo
```

---

## 4. 클래스 다이어그램 — 프론트엔드

```mermaid
classDiagram
    direction LR

    class App {
        <<component>>
        +render() JSX
    }
    class Dashboard {
        <<component>>
        -JobInfo[] jobs
        -ModelInfo[] models
        -string error
    }
    class TrainPage {
        <<component>>
        -DatasetInfo[] datasets
        -string datasetId
        -number maxSteps
        -number batchSize
        -number lr
        -number epochs
        -JobInfo job
        -string[] logs
        +handleUpload(e)
        +start()
        +stop()
    }
    class Evaluate {
        <<component>>
        -ModelInfo[] models
        -string modelId
        -string question
        -string answer
        -number elapsed
        +ask()
    }
    class Models {
        <<component>>
        -ModelInfo[] models
        +refresh()
        +remove(id)
    }

    class ApiClient {
        <<module>>
        +listDatasets()
        +uploadDataset(file)
        +deleteDataset(id)
        +listModels()
        +deleteModel(id)
        +startTrain(req)
        +listJobs()
        +getJob(id)
        +stopJob(id)
        +infer(model_id, question)
    }
    class streamJobLog {
        <<function>>
        +(jobId, onLine, onEnd) cleanup
    }

    App ..> Dashboard
    App ..> TrainPage
    App ..> Evaluate
    App ..> Models

    Dashboard ..> ApiClient
    TrainPage ..> ApiClient
    TrainPage ..> streamJobLog
    Evaluate ..> ApiClient
    Models ..> ApiClient
```

---

## 5. 시퀀스 다이어그램 — 학습 시작 + SSE 로그

```mermaid
sequenceDiagram
    autonumber
    actor User as "사용자"
    participant UI as "TrainPage (React)"
    participant API as "TrainRouter"
    participant TR as "train_runner"
    participant JS as "JobStore"
    participant PROC as "main_train_cpu.py (subprocess)"
    participant FS as "storage/logs/[id].log"

    User->>UI: CSV 선택 + 하이퍼파라미터 입력 + 학습 시작 클릭
    UI->>API: POST /api/train (TrainRequest)
    API->>TR: start_training(req)
    TR->>JS: create(job_id, req)
    alt running 잡 존재
        JS-->>TR: RuntimeError
        TR-->>API: raise
        API-->>UI: 409 Conflict
    else 가능
        JS-->>TR: job dict (status=running)
        TR->>PROC: Popen(python -u main_train_cpu.py ...)<br/>env: CUDA_VISIBLE_DEVICES=""<br/>stdout→file, start_new_session
        TR->>JS: update(job_id, pid, log_path)
        TR-->>API: job_id
        API-->>UI: JobInfo(status=running)

        Note over TR,PROC: 데몬 스레드<br/>_watch_process 가 proc.wait() 대기

        UI->>API: GET /api/train/jobs/{id}/stream (EventSource)
        loop 0.5s 폴링
            API->>FS: readline()
            alt 라인 존재
                FS-->>API: log line
                API-->>UI: data: <line>
            else EOF & 잡 running
                API->>API: asyncio.sleep(0.5)
            end
        end

        PROC->>FS: stdout writes
        PROC->>PROC: train, save_pretrained
        PROC-->>TR: exit code
        TR->>JS: update(status=completed|failed|stopped, finished_at, exit_code, output_dir)
        API-->>UI: event: end<br/>data: <status>
        UI->>API: GET /api/train/jobs/{id}
        API-->>UI: JobInfo(최종 상태)
    end
```

---

## 6. 시퀀스 다이어그램 — 추론

```mermaid
sequenceDiagram
    autonumber
    actor User as "사용자"
    participant UI as "Evaluate"
    participant API as "InferRouter"
    participant CACHE as "_cache (dict)"
    participant LD as "InferenceModelManager"
    participant IM as "InferenceManager"
    participant MODEL as "PeftModel + Tokenizer"

    User->>UI: 모델 선택 + 질문 입력 + 답변 받기
    UI->>API: POST /api/infer<br/>{ model_id, question, ... }

    API->>CACHE: model_id 조회
    alt cache hit
        CACHE-->>API: InferenceManager
    else cache miss
        API->>LD: create_loader(model_dir)
        LD->>MODEL: AutoModelForCausalLM.from_pretrained(base)
        LD->>MODEL: resize_token_embeddings
        LD->>MODEL: PeftModel.from_pretrained(base, adapter)
        MODEL-->>LD: model, tokenizer
        LD-->>API: model, tokenizer
        API->>IM: create_inference_manager(model, tokenizer)
        IM-->>API: inf
        API->>CACHE: store(model_id, inf)
    end

    API->>IM: update_generation_config(max_new_tokens, temperature)
    API->>IM: generate_response(question)
    IM->>MODEL: tokenize(format_prompt(question))
    IM->>MODEL: model.generate(...)
    MODEL-->>IM: output ids
    IM->>IM: decode new tokens + post_process
    IM-->>API: answer

    API-->>UI: InferResponse(question, answer, elapsed_ms)
    UI-->>User: 답변 렌더링
```

---

## 7. 시퀀스 다이어그램 — 학습 중단

```mermaid
sequenceDiagram
    autonumber
    actor User as "사용자"
    participant UI as "TrainPage"
    participant API as "TrainRouter"
    participant TR as "train_runner"
    participant JS as "JobStore"
    participant PROC as "Training Subprocess"
    participant W as "_watch_process (daemon thread)"

    User->>UI: 중단 클릭
    UI->>API: POST /api/train/jobs/{id}/stop
    API->>TR: stop_training(job_id)
    TR->>JS: get(job_id)
    alt job 없음 / running 아님 / pid 없음
        JS-->>TR: None / invalid
        TR-->>API: false
        API-->>UI: 400 Bad Request
    else 정상
        TR->>PROC: os.killpg(os.getpgid(pid), SIGTERM)
        TR-->>API: true
        API-->>UI: { stopped: job_id }
        PROC-->>W: exit (음수 코드)
        W->>JS: update(status=stopped, exit_code, finished_at)
    end
```

---

## 8. 상태 다이어그램 — Job 라이프사이클

```mermaid
stateDiagram-v2
    [*] --> created : POST /api/train
    created --> running : Popen 성공 + pid 기록
    created --> [*] : Popen 실패<br/>(state는 메모리에만 남음)

    running --> completed : exit_code == 0
    running --> failed    : exit_code > 0
    running --> stopped   : SIGTERM (exit_code < 0)

    completed --> [*]
    failed --> [*]
    stopped --> [*]

    note right of running
      _watch_process 데몬 스레드가
      proc.wait() 후 status/exit_code/
      finished_at/output_dir 업데이트
    end note
```

---

## 9. 액티비티 다이어그램 — 데이터 처리 파이프라인

```mermaid
flowchart TD
    A(["CSV 업로드"]) --> B{"확장자<br/>.csv 검증"}
    B -- 실패 --> Z1(["400 반환"])
    B -- 성공 --> C["storage/datasets/[id].csv 저장"]
    C --> D["pd.read_csv utf-8-sig nrows=5"]
    D --> E{"question/answer<br/>컬럼 존재?"}
    E -- 없음 --> Z2["파일 삭제 + 400 반환"]
    E -- 있음 --> F(["DatasetInfo 반환"])

    F -.->|학습 시작 시| G["main_train_cpu.py"]
    G --> H["load_csv_data"]
    H --> I["dropna, 빈 문자열 제거"]
    I --> J["format_prompt 적용<br/>질문/답변 + EOS 토큰"]
    J --> K["Dataset.from_dict"]
    K --> L["tokenize batched<br/>truncation max_length=256"]
    L --> M["labels = input_ids.copy"]
    M --> N(["Trainer.train"])

    N --> O["save_pretrained<br/>adapter + tokenizer"]
    O --> P["training_stats.json 저장"]
    P --> Q(["storage/models/[job_id]/"])
```

---

## 10. 배포 다이어그램

```mermaid
flowchart TB
    subgraph Dev["로컬 개발 환경"]
        direction LR
        subgraph Node["Node 18 + Vite"]
            VITE["vite dev<br/>:5173"]
        end
        subgraph Py["Conda py39_pt"]
            UV["uvicorn app.main:app<br/>:8000"]
            STORAGE[("storage/<br/>datasets · models · logs")]
        end
        VITE -. "/api/* (CORS)" .-> UV
        UV --- STORAGE
    end

    subgraph Prod["단일 origin 배포"]
        direction LR
        subgraph BuiltSPA["frontend/dist (정적)"]
            STATIC["index.html<br/>+ assets/"]
        end
        subgraph PyProd["uvicorn (single worker)"]
            APP["FastAPI app<br/>:8000"]
        end
        STORAGE2[("storage/")]
        APP -- "mount('/')" --> STATIC
        APP --- STORAGE2
    end

    subgraph Sub["학습 시 spawn"]
        TRAIN["python -u<br/>main_train_cpu.py"]
    end

    UV -. "Popen<br/>start_new_session" .-> TRAIN
    APP -. "Popen" .-> TRAIN
    TRAIN --- STORAGE
    TRAIN --- STORAGE2
```
