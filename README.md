# Memory-Agnostic Input Processing for LLM Agent Memory Systems

This repository contains the implementation and experiments for **Memory-Agnostic Input Processing**, a graduation project that analyzes how input preprocessing affects the performance and cost of LLM-based agent memory systems.

Instead of modifying the internal algorithm of each memory backend, this project focuses on transforming the input dialogue before memory construction. The main goal is to investigate whether memory performance can be improved by changing the input unit and information preservation strategy.

---

이 저장소는 **Memory-Agnostic Input Processing**의 구현과 실험을 포함합니다. 이 프로젝트는 졸업프로젝트로, 입력 전처리가 LLM 기반 Agent Memory System의 성능과 비용에 어떤 영향을 미치는지 분석합니다.

각 memory backend의 내부 알고리즘을 수정하는 대신, 본 프로젝트는 memory construction 이전에 입력 대화를 변환하는 방식에 집중합니다. 주요 목표는 입력 단위와 정보 보존 전략을 바꾸는 것만으로 memory 성능을 개선할 수 있는지 확인하는 것입니다.


## Project Summary

Personal AI Agents need long-term memory to store and retrieve user preferences, past conversations, plans, relationships, schedules, and task records. However, directly putting all previous conversations into the context window is inefficient and often ineffective.

This project evaluates whether memory performance can be changed **without modifying the memory backend itself**.

The core question is:

> Can memory performance be improved without modifying the memory backend?

To answer this, the project applies memory-agnostic input processing modules to existing memory backends and compares the resulting performance and cost.

---

Personal AI Agent는 사용자의 선호, 과거 대화, 계획, 관계, 일정, 작업 기록 등을 저장하고 검색하기 위해 장기 기억이 필요합니다. 그러나 모든 과거 대화를 context window에 직접 넣는 방식은 비효율적이며, 실제로도 효과적이지 않은 경우가 많습니다.

본 프로젝트는 **memory backend 자체를 수정하지 않고도** memory 성능이 달라질 수 있는지 평가합니다.

핵심 질문은 다음과 같습니다.

> Memory backend를 수정하지 않고도 memory 성능을 개선할 수 있는가?

이를 확인하기 위해, 본 프로젝트는 기존 memory backend에 memory-agnostic input processing module을 적용하고, 그 결과 나타나는 성능과 비용 변화를 비교합니다.


## Main Idea

The project studies two types of preprocessing:

### 1. LLM-based Filtering

- Removes non-informative parts of each utterance.
- Removes social filler, conversational routines, and redundant confirmations.
- Preserves informative content without summarization or paraphrasing.
- Goal: reduce noise before memory construction.

### 2. Attention + Similarity-based Chunking

- Detects topic boundaries between dialogue turns.
- Uses attention change and semantic similarity between adjacent turns.
- Generates topic-level chunks before memory construction.
- Goal: construct more meaningful memory input units.

The key insight is:

> Simply reducing information is not enough.  
> It is more important to preserve information in the right semantic unit.

## Repository Structure

```text
MemoryAgnostic/
├── configs/                 # YAML configuration files for experiments
│   ├── amem_default.yaml
│   ├── amem_32bfilter.yaml
│   ├── amem_8bfilter.yaml
│   ├── simplemem_default.yaml
│   ├── simplemem_32bfilter.yaml
│   ├── simplemem_8bfilter.yaml
│   ├── test_amem.yaml
│   └── test_simplemem.yaml
│
├── core/                    # Core modules
│   ├── chunker/             # Chunking modules
│   ├── compressor/          # Compression interface
│   ├── filter/              # Filtering modules
│   └── memory/              # Memory backend wrappers
│
├── data/                    # Dataset loaders and schema definitions
├── results/                 # Experiment outputs
├── scripts/                 # Utility scripts
├── factory.py               # Module factory for filters, chunkers, and memory backends
└── main.py                  # Main experiment runner
```

## Supported Modules

### Filtering

| Module | Description |
|---|---|
| `NoFilter` | Keeps the original dialogue unchanged. |
| `LLMFilter` | Removes social filler, redundant confirmations, and non-informative utterance parts. |

### Chunking

| Module | Description |
|---|---|
| `NoChunker` | Uses the original dialogue without topic-based chunking. |
| `FixedSizeChunker` | Splits dialogue into fixed-size chunks. |
| `AttentionSimilarityChunker` | Detects topic boundaries using attention change and semantic similarity. |

### Memory Backends

| Backend | Description |
|---|---|
| `SimpleMemBackend` | Wrapper for SimpleMem-style memory construction and retrieval. |
| `AMemBackend` | Wrapper for A-Mem-style memory construction and retrieval. |

## Environment

The experiments were conducted with the following environment:

```text
GPU: RTX 3090
CUDA: 12.2
Python: 3.10
vLLM: 0.8.5
```

The filtering model used in the main experiment:

```text
Qwen3-32B
```

The inference models used for memory backends:

```text
SimpleMem: Qwen3-14B, GPT-4.1-mini
A-Mem: GPT-4o-mini
```

## Installation

Clone the repository:

```bash
git clone https://github.com/delta9043/MemoryAgnostic.git
cd MemoryAgnostic
```

Create a Python environment:

```bash
conda create -n memoryagnostic python=3.10
conda activate memoryagnostic
```

Install dependencies according to your local environment.
> Note: dependency versions may depend on the local CUDA and model-serving environment.

## Running Experiments

The main experiment runner uses YAML configuration files.

### SimpleMem baseline

```bash
python main.py --config configs/simplemem_default.yaml
```

### SimpleMem with filtered data

```bash
python main.py --config configs/simplemem_32bfilter.yaml
```

### A-Mem baseline

```bash
python main.py --config configs/amem_default.yaml
```

### A-Mem with filtered data

```bash
python main.py --config configs/amem_32bfilter.yaml
```

## Running LLM-based Filtering

To generate filtered dialogue data:

```bash
python run_filter.py \
  --model_path /path/to/Qwen3-32B \
  --output data/filtered_data/filtered_32b.json
```

Example output format:

```json
[
  {
    "sample_id": "conv-26",
    "turns": [
      {
        "turn_id": "D1:1",
        "speaker": "A",
        "content": "filtered content",
        "timestamp": "...",
        "session_id": "..."
      }
    ]
  }
]
```

## Key Findings

1. **Filtering alone was unstable**
   - It improved some categories but degraded Temporal QA.
   - It did not reduce construction cost.

2. **Topic-based chunking was more consistent**
   - It improved performance across different memory backends.
   - It reduced irrelevant turn mixing by constructing topic-level input units.

3. **Input unit design matters**
   - Memory performance depends not only on the memory backend, but also on how the input dialogue is segmented and preserved before memory construction.

## Limitations

- Experiments were conducted on LoCoMo10, a small subset of the full benchmark.
- LLM-based filtering may produce unstable outputs, which can cause memory entry generation failures.
- Chunking can separate temporal expressions from their relevant context.
- More experiments are needed with larger datasets and additional memory backends.

## Future Work

- Evaluate on the full LoCoMo benchmark.
- Add temporal-aware chunking to preserve relative and absolute time information.
- Compare additional memory backends such as Mem0 and LightMem.
- Analyze memory construction cost in terms of token usage and API calls.
- Improve robustness of LLM-based filtering output parsing.

## References

- Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
- A-Mem: Agentic Memory for LLM Agents
- SimpleMem: Efficient Lifelong Memory for LLM Agents
- LightMem: Lightweight and Efficient Memory-Augmented Generation
- Evaluating Very Long-Term Conversational Memory of LLM Agents
- LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression

## Author

**Moon Sanghyeok**  
Department of Applied Mathematics  
Kyung Hee University  
Email: delta9043@khu.ac.kr
