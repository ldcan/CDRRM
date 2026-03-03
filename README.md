# CDRRM: Contrast-Driven Rubric Generation for Reliable and Interpretable Reward Modeling

## Overview

CDRRM (Comprehensive Discriminative Rubric-based Reward Model) is a framework for:
1. **Rubric Synthesis**: Generating structured, verifiable evaluation rubrics from pairwise data
2. **SFT Data Preparation**: Creating supervised fine-tuning datasets for rubric generation and judge models
3. **Model Evaluation**: Evaluating judge models on various benchmarks
4. **Model Training**: Training rubric generation and judge models

## Repository Structure

```
cdrrm/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── prompts.py                  # Centralized prompt templates
├── utils.py                    # Shared utility functions (I/O, logging)
├── llm_client.py               # Async OpenAI-compatible LLM client
│
├── rubric_synthesis/           # Rubric generation pipeline
│   ├── diagnosis_generation.py # StepA: Generate structured diagnoses
│   ├── rubric_generation.py    # StepB: Generate discriminative rubrics
│   ├── run_pipeline.sh         # Main pipeline script
│   └── data/                   # Input data directory
│
├── sft_data_synthesis/         # SFT data preparation pipeline
│   ├── filter_inconsistent_rubrics.py  # Filter inconsistent rubrics
│   ├── prepare_sft_data.py     # Prepare rubric generation SFT data
│   ├── judge_synthesis.py      # Synthesize judge SFT data via LLM API
│   ├── run_rubric_generation.sh  # Generate rubric SFT data
│   └── run_judge_synthesis.sh    # Generate judge SFT data
│
├── eval/                       # Evaluation pipeline
│   ├── generate_rubrics.py     # Generate rubrics using SFT model
│   ├── evaluate.py             # Evaluate judge models on benchmarks
│   ├── run_generate_rubrics.sh # Script to generate rubrics
│   ├── run_eval.sh             # Script to run evaluation
│   └── eval_dataset/           # Benchmark datasets
│
└── train/                      # Training scripts
    └── train.sh                # SFT training script
```

## Installation

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd cdrrm

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Important Preparation
Before running API-based pipelines, set environment variables:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="your_api_base_url"   # optional
export MODEL_NAME="your_model_name"
```
You can also set the API key and base URL directly in `llm_client.py` (environment variables are recommended).

### 1. Rubric Synthesis

Following the Contrast-then-Synthesis paradigm, generate discriminative rubrics from pairwise data:

```bash
cd rubric_synthesis

# Run the pipeline
bash run_pipeline.sh
```

**Output:**
- `output/diagnosis.jsonl`: Structured diagnoses for each response
- `output/rubric_pair.jsonl`: Discriminative rubrics for each pair

### 2. Prepare SFT Training Data

#### 2.1 Rubric Generation SFT Data

Prepare training data for rubric generation models:

```bash
cd sft_data_synthesis

bash run_rubric_generation.sh 
```

**Output:**
- `output/rubric_generation_sft_data/train.jsonl`: Training data
We also provide rubric-generator training data at:
`https://www.modelscope.cn/datasets/ldccdrrm/cdrrm-rubrics`
You can use it directly and skip this step.

#### 2.2 Synthesize Judge SFT Data

Synthesize judge SFT data using LLM API:

```bash
# You need rubrics with formatted_rubric first (from the eval generation step).
cd sft_data_synthesis

bash run_judge_synthesis.sh
```

**Output:**
- `output/judge_synthesis/sft_data_pairwise.jsonl`: Judge SFT training data
- `output/judge_synthesis/results_all_pairwise.jsonl`: All evaluation results
- `output/judge_synthesis/results_filtered_pairwise.jsonl`: Filtered results (valid predictions only)
- `output/judge_synthesis/metrics_pairwise.json`: Evaluation metrics

### 3. Generate Rubrics for Evaluation

Use a fine-tuned rubric generation model to generate rubrics for evaluation:

```bash
cd eval

# Set model path and benchmark
export SFT_MODEL_PATH="/path/to/rubric-generator-model"

# Generate rubrics for a benchmark
bash run_generate_rubrics.sh \
    --sft_model_path /path/to/model \
    --benchmark rewardbench \
    --output_file ./rubrics/rubrics.jsonl \
    --batch_size 256 \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.95 \
    --max_tokens 8192 \
    --seed 42 
```

### 4. Evaluate Judge Models

Evaluate judge models on benchmarks:

```bash
cd eval

# Run evaluation
bash run_eval.sh \
    --model_path /path/to/model \
    --benchmark rewardbench \
    --prompt_type rubric_judge \
    --rubrics_file ./rubrics/rubrics_rewardbench.jsonl \
    --output_root ./eval_results \
    --batch_size 256 \
    --tensor_parallel_size 8 \
    --gpu_memory_utilization 0.95 \
    --max_tokens 8192 \
    --seed 42 \
    --shuffle 
```

**Supported benchmarks:**
- `rewardbench`: RewardBench dataset
- `rmbench`: RM-Bench dataset
- `rmb`: RMB dataset

`run_eval.sh` currently supports: `rewardbench`, `rmbench`, `rmb`.
`openrubrics` is supported by `run_generate_rubrics.sh`.

**Supported prompt types:**
- `direct_judge`: Direct pairwise judgment without rubric
- `rubric_judge`: Pairwise judgment with rubric guidance

### 5. Train Models

Train SFT models using the prepared data:

```bash
cd train

# Run training
bash train.sh 
```
