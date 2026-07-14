# Model Fine-Tuning Evaluation for Sinhala Q&A
================================================

This document evaluates the feasibility, methodology, and compute requirements for fine-tuning a small open-source Large Language Model (LLM) on Sinhala Q&A datasets. This acts as the Phase 3 Stretch/Differentiation component of the Sinhala Voice Assistant project, assessing how to transition from API-driven generation (Gemini Flash) to a self-hosted or adapted open model.

---

## 1. Rationale: RAG vs. Fine-Tuning for Low-Resource Languages

While Retrieval-Augmented Generation (RAG) grounds answers in specific source documents, low-resource languages like Sinhala face distinct challenges:
- **Tokenization Efficiency:** Standard multilingual models have highly inefficient tokenizers for Sinhala, representing a single Sinhala character using up to 3–6 tokens. This blows up the context length and increases latency.
- **Syntactic & Cultural Nuances:** Standard models often translate answers literally from English, missing idiomatic Sinhala phrasing, appropriate level of politeness (honorifics), and proper grammar.
- **Resource Constraints:** Base models have weak innate understanding of Sinhala facts and vocabulary, leading to high failure rates when generating long-form Sinhala answers even when retrieved context is present.

Fine-tuning a base open LLM helps the model **learn the formatting, grammar, and vocabulary structure of Sinhala**, while RAG continues to supply the **grounded runtime facts** to prevent hallucination.

---

## 2. Model Candidates Analysis

Evaluating candidates for Sinhala language model adaptation:

| Model | Base Size | Sinhala Tokenizer Efficiency | Pre-training Sinhala Data | Pros & Cons |
|---|---|---|---|---|
| **Gemma 2 (9B / 2B)** | 9B or 2B | **High** (256k vocab, includes excellent Sinhala coverage) | Moderate | **Pros:** Very efficient tokenization; high reasoning capability for its size.<br>**Cons:** Needs a GPU with at least 8GB VRAM (2B) or 18GB (9B) for inference. |
| **Llama 3 / 3.1 (8B)** | 8B | **Low** (128k vocab, poor Sinhala representation) | Very Low | **Pros:** Robust general instruction following.<br>**Cons:** Poor tokenization efficiency increases latency and memory usage in Sinhala. |
| **SinLlama (Research)** | ~7B | **Medium** | High (focused on Sinhala corpus) | **Pros:** Domain-adapted specifically for Sinhala.<br>**Cons:** Built on older model architectures (Llama-2 derivatives); weaker instruction-following and reasoning. |
| **Qwen 2.5 (7B / 1.5B)**| 7B or 1.5B | **High** (151k vocab, great multilingual representation) | High | **Pros:** Excellent performance on multilingual benchmarks; lightweight variants run well on CPU. |

### Recommendation
**Qwen-2.5-7B-Instruct** or **Gemma-2-9B-It** are the recommended base candidates due to their superior tokenizer efficiency (avoiding context explosion) and robust multilingual instruction-following architectures.

---

## 3. Training Data Requirements

To teach the model conversational Sinhala and Q&A styles, we require a mixed dataset of:
1. **Instruction-Tuning Datasets (General Sinhala):**
   - **Sinhala Alpaca:** Translated version of Stanford's Alpaca dataset (approx. 52,000 instructions translated to Sinhala).
   - **Multilingual SQuAD (Sinhala subset):** Grounded question-answering pairs in Sinhala.
2. **Domain-Specific Corpora (Grounded Q&A):**
   - Synthetically generated Q&A pairs (using Gemini Flash) derived from the **NSINA News** and **Sinhala Wikipedia** datasets.
   - Example format: `{"instruction": "සන්දර්භය ඇසුරින් ප්‍රශ්නයට පිළිතුරු සපයන්න.", "input": "සන්දර්භය: ...\nප්‍රශ්නය: ...", "output": "පිළිතුර: ..."}`

---

## 4. Fine-Tuning Methodology: 4-Bit QLoRA

To train on free or low-cost infrastructure (like a single NVIDIA T4 GPU with 16GB VRAM on Google Colab or Kaggle), we must use **Quantized Low-Rank Adaptation (QLoRA)**.

### Technical Parameters:
- **Quantization:** 4-bit NormalFloat (NF4) double quantization to load the base model in ~5.5 GB VRAM.
- **LoRA Hyperparameters:**
  - Rank ($r$): 16 (captures sufficient linguistic capacity).
  - Alpha ($\alpha$): 32.
  - Target Modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Training Configurations:**
  - Sequence Length: 1024 tokens.
  - Learning Rate: $2\times 10^{-4}$ (cosine schedule).
  - Batch Size: 2 (gradient accumulation steps = 4; effective batch size = 8).
  - Optimizer: `paged_adamw_32bit` (handles VRAM spikes smoothly).

---

## 5. Implementation Steps in Python (Unsloth/PEFT)

Using the highly optimized `unsloth` library to speed up training and decrease memory overhead on free GPUs:

```python
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Load Model & Tokenizer
max_seq_length = 1024
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    torch_dtype=torch.float16,
)

# 2. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# 3. Load & Format Dataset
dataset = load_dataset("json", data_files="sinhala_qa_dataset.json", split="train")

def format_prompt(examples):
    # Format according to Qwen/Gemma chat templates
    formatted = []
    for prompt, response in zip(examples["instruction"], examples["output"]):
        formatted.append(f"<|im_start|>system\nඔබ සිංහල භාෂාවෙන් පිළිතුරු දෙන සහායකයෙකි.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>")
    return {"text": formatted}

dataset = dataset.map(format_prompt, batched=True)

# 4. Train model
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs",
    ),
)
trainer.train()
```

---

## 6. Evaluation Framework

To evaluate the success of the fine-tuned model against the baseline API:
1. **Linguistic Fluency (Perplexity & Human Rating):**
   - Check if Sinhala syntax, grammar, and sentence endings (e.g. verbs like `පවතී`, `ඇත`, `වේ`) are natural.
2. **Faithfulness (Retrieval-Grounded Accuracy):**
   - Feed the model retrieved Sinhala chunks and verify if the generated answer stays 100% grounded in the context without introducing hallucinated facts.
3. **Latency Benchmarking:**
   - Measure execution time. Deploying the quantized 4-bit model locally on a CPU or lightweight GPU must maintain a response latency of under 4 seconds.

---

## 7. Strategic Recommendations

1. **Keep RAG at the Core:** Fine-tuning should not replace RAG. RAG provides the dynamic knowledge base (changing news and facts), while the fine-tuned model provides the Sinhala grammatical skeleton and conversational fluency.
2. **Start with Qwen 2.5 1.5B/7B:** Run initial experiments on a free Google Colab notebook using the Qwen 2.5 architecture due to its outstanding cost-to-performance ratio in low-resource setups.
3. **Save and Share Adapters:** Export the resulting fine-tuning weights as LoRA adapters (approx. 50–100MB), which can be dynamically loaded onto base models at startup to keep deployment storage footprint minimal.
