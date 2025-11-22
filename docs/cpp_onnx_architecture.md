# C++ ONNX Runtime Backend Architecture

## Model Conversion Process (convert_to_onnx.py)

```mermaid
graph LR
    subgraph Input["🔹 INPUT"]
        MODEL_ID["HuggingFace Model ID<br/>openai/whisper-small"]
    end

    subgraph Conversion["⚙️ CONVERSION PROCESS"]
        LOAD["optimum<br/>ORTModelForSpeechSeq2Seq"]
        TRACE["ONNX Graph<br/>Tracing"]
        OPTIMIZE["Graph<br/>Optimization"]
    end

    subgraph Output["📦 OUTPUT FILES"]
        direction TB
        ENC["encoder_model.onnx<br/>336 MB"]
        DEC["decoder_model.onnx<br/>738 MB"]
        DECPAST["decoder_with_past.onnx<br/>684 MB"]
        CONFIGS["configs + vocab + tokens"]
    end

    MODEL_ID --> LOAD
    LOAD --> TRACE
    TRACE --> OPTIMIZE
    OPTIMIZE --> ENC
    OPTIMIZE --> DEC
    OPTIMIZE --> DECPAST
    OPTIMIZE --> CONFIGS

    style Input fill:#e1f5ff,stroke:#01579b,stroke-width:2px,color:#000
    style Conversion fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    style Output fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000
    
    style ENC fill:#bbdefb,stroke:#1976d2,stroke-width:2px,color:#000
    style DEC fill:#bbdefb,stroke:#1976d2,stroke-width:2px,color:#000
    style DECPAST fill:#bbdefb,stroke:#1976d2,stroke-width:2px,color:#000
```

## Conversion Details: Encoder Model

```mermaid
graph LR
    subgraph PyTorch Model
        PT_ENC[Whisper Encoder<br/>PyTorch weights]
    end
    
    subgraph ONNX Export
        DUMMY[Dummy Input<br/>mel-spec [1×80×3000]]
        TRACE[torch.onnx.export]
        OPS[ONNX Operators<br/>Conv1D, LayerNorm,<br/>MultiHeadAttention]
    end
    
    subgraph ONNX Model
        ONNX_ENC[encoder_model.onnx<br/>opset_version=14]
        INPUT_NODE[input_features<br/>float32[1,80,3000]]
        OUTPUT_NODE[last_hidden_state<br/>float32[1,1500,768]]
    end
    
    PT_ENC --> TRACE
    DUMMY --> TRACE
    TRACE --> OPS
    OPS --> ONNX_ENC
    ONNX_ENC --> INPUT_NODE
    ONNX_ENC --> OUTPUT_NODE
    
    style PT_ENC fill:#fce4ec,stroke:#880e4f
    style ONNX_ENC fill:#e1f5fe,stroke:#01579b
```

## Conversion Details: Decoder Models

```mermaid
graph TB
    subgraph PyTorch Decoder
        PT_DEC[Whisper Decoder<br/>with Cross-Attention]
        PT_CACHE[KV-Cache Mechanism]
    end
    
    subgraph Two ONNX Variants
        DEC_FIRST["decoder_model.onnx<br/>📥 First token decoding<br/>Creates initial KV-cache"]
        DEC_PAST["decoder_with_past_model.onnx<br/>⚡ Subsequent tokens<br/>Reuses past KV-cache"]
    end
    
    subgraph Inputs & Outputs
        IN1["Inputs:<br/>- input_ids [batch, seq_len]<br/>- encoder_hidden_states [1,1500,768]<br/>- past_key_values (optional)"]
        
        OUT1["Outputs:<br/>- logits [batch, seq_len, vocab_size]<br/>- present_key_values"]
    end
    
    PT_DEC --> DEC_FIRST
    PT_DEC --> DEC_PAST
    PT_CACHE --> DEC_PAST
    
    DEC_FIRST --> IN1
    DEC_PAST --> IN1
    IN1 --> OUT1
    
    style DEC_FIRST fill:#fff9c4,stroke:#f57f17
    style DEC_PAST fill:#c8e6c9,stroke:#2e7d32
```

## Model Size Comparison (whisper-small)

```mermaid
graph LR
    subgraph Original
        PT["PyTorch Model<br/>~484 MB<br/>✅ Full precision<br/>❌ Python only"]
    end
    
    subgraph ONNX
        ONNX_TOTAL["Total ONNX<br/>~1758 MB<br/>✅ Cross-platform<br/>✅ Optimized inference"]
        ENC_SIZE["encoder: 336 MB"]
        DEC_SIZE["decoder: 738 MB"]
        PAST_SIZE["decoder_past: 684 MB"]
    end
    
    PT --> ONNX_TOTAL
    ONNX_TOTAL --> ENC_SIZE
    ONNX_TOTAL --> DEC_SIZE
    ONNX_TOTAL --> PAST_SIZE
    
    style PT fill:#fce4ec,stroke:#c2185b
    style ONNX_TOTAL fill:#e1f5fe,stroke:#0277bd
```

## Why Three Models?

```mermaid
mindmap
  root((ONNX Models))
    encoder_model
      Processes mel-spectrogram
      One-time execution
      Output: context vectors
    decoder_model
      First token generation
      Creates initial KV-cache
      All 12 encoder layers
      All 12 decoder layers
    decoder_with_past_model
      Subsequent tokens
      Reuses past KV-cache
      Only decoder layers
      40% faster per token
      Memory efficient
```

## Mermaid Diagram: Data Flow

```mermaid
graph TB
    subgraph Client["Client Layer"]
        HTTP[HTTP Request<br/>POST /process_audio]
    end

    subgraph Python["Python Layer (FastAPI)"]
        API[FastAPI Endpoint]
        LOAD[librosa.load<br/>OPUS → PCM]
        MEL[transformers.AutoProcessor<br/>Audio → Mel-spectrogram<br/>80x3000]
        SAVE[Save to Binary<br/>/tmp/mel_spec.bin]
        TOK[transformers.AutoTokenizer<br/>Token IDs → UTF-8 Text]
    end

    subgraph CPP["C++ Layer (Persistent Process)"]
        PROC[C++ Process<br/>stdin/stdout IPC]
        READ[Read Mel Binary<br/>shape: 80×3000]
        ENC[ONNX Encoder<br/>encoder_model.onnx<br/>336 MB]
        DEC[ONNX Decoder<br/>decoder_model.onnx<br/>738 MB]
        DECPAST[ONNX Decoder w/Past<br/>decoder_with_past.onnx<br/>684 MB]
        SUPP[Suppress Tokens<br/>Filter non-speech]
    end

    subgraph ONNX["ONNX Runtime"]
        RUNTIME[CPU Execution Provider<br/>Multi-threading]
        OPT[Graph Optimizations<br/>Constant Folding]
    end

    HTTP --> API
    API --> LOAD
    LOAD --> MEL
    MEL --> SAVE
    SAVE --> PROC
    
    PROC --> READ
    READ --> ENC
    
    ENC --> RUNTIME
    RUNTIME --> DEC
    DEC --> SUPP
    SUPP --> DECPAST
    
    DECPAST -->|Token IDs| PROC
    PROC -->|TOKENS:50258,50263,...| TOK
    TOK -->|Транскрипция| API
    API -->|JSON Response| HTTP

    style Python fill:#3776ab,stroke:#fff,color:#fff
    style CPP fill:#f34b7d,stroke:#fff,color:#fff
    style ONNX fill:#00d4aa,stroke:#fff,color:#000
    style Client fill:#61dafb,stroke:#fff,color:#000
```

## Sequence Diagram: Request Processing

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant librosa
    participant transformers
    participant C++ Process
    participant ONNX Runtime

    Client->>FastAPI: POST /process_audio (OPUS file)
    FastAPI->>librosa: load(audio_bytes, sr=16000)
    librosa-->>FastAPI: audio_array [N samples]
    
    FastAPI->>transformers: processor(audio_array)
    transformers-->>FastAPI: mel_spec [80×3000]
    
    FastAPI->>C++ Process: MEL /tmp/mel_spec.bin\n
    
    C++ Process->>C++ Process: Read binary mel-spec
    C++ Process->>ONNX Runtime: run_encoder(mel_spec)
    ONNX Runtime-->>C++ Process: encoder_hidden_states [1×1500×768]
    
    loop Autoregressive Decoding (max 448 tokens)
        C++ Process->>ONNX Runtime: run_decoder(input_ids, encoder_states, past_kv)
        ONNX Runtime-->>C++ Process: logits, updated_past_kv
        C++ Process->>C++ Process: Apply suppress_tokens
        C++ Process->>C++ Process: Select next token (argmax)
        
        alt Token is EOS (50257)
            C++ Process->>C++ Process: Break loop
        end
    end
    
    C++ Process->>FastAPI: TOKENS:50258,50263,12345,...\n
    
    FastAPI->>transformers: tokenizer.decode(token_ids)
    transformers-->>FastAPI: transcription_text
    
    FastAPI->>Client: {"text": "...", "processing_time_ms": 1624}
```

## Component Architecture

```mermaid
graph LR
    subgraph Docker Container
        subgraph Python Process
            UVICORN[uvicorn<br/>FastAPI Server]
            MODEL[WhisperCppPersistentModel]
        end
        
        subgraph C++ Binary
            MAIN[main.cpp<br/>Interactive Mode]
            WHISPER[WhisperONNXModel]
            ENCODER_LIB[encoder_model.onnx]
            DECODER_LIB[decoder_model.onnx]
            DECODER_PAST[decoder_with_past.onnx]
        end
        
        UVICORN --> MODEL
        MODEL <-->|stdin/stdout| MAIN
        MAIN --> WHISPER
        WHISPER --> ENCODER_LIB
        WHISPER --> DECODER_LIB
        WHISPER --> DECODER_PAST
    end
    
    CLIENT[Client] -->|HTTP| UVICORN
    
    style Python Process fill:#3776ab,stroke:#fff,color:#fff
    style C++ Binary fill:#f34b7d,stroke:#fff,color:#fff
```

## Key Optimizations

```mermaid
mindmap
  root((C++ ONNX<br/>Optimizations))
    Persistent Process
      No model reload overhead
      ~200ms saved per request
      Warm-up on startup
    KV-Cache
      decoder_with_past_model.onnx
      Only compute new token
      ~40% faster decoding
    Mel-spec in Python
      Use transformers (proven)
      Match training pipeline
      Better accuracy
    Token Decoding in Python
      UTF-8 handling
      Special tokens
      Consistent with HF
    Multi-threading
      ONNX Runtime threads=4
      Parallel ops
      CPU optimization
```

## Performance Comparison

| Metric | Python (transformers) | C++ (ONNX) | Improvement |
|--------|----------------------|------------|-------------|
| **Latency** | 2618 ms | 1625 ms | **38% faster** ⚡ |
| **WER** | 0.4933 | 0.4895 | **0.77% better** ✅ |
| **CER** | 0.2355 | 0.2180 | **7.4% better** ✅ |
| **Memory** | ~4 GB | ~2 GB | **50% less** 💾 |
| **Throughput** | 23 req/min | 37 req/min | **60% more** 📈 |


