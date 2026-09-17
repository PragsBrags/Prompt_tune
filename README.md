# Tamang Translation Pipeline

This project fine-tunes, indexes, and evaluates a multilingual causal language model for Tamang, Nepali, and English translation. It combines Unsloth, PEFT/LoRA, TRL, Transformers, Chroma, SentenceTransformers, Hugging Face `datasets`, `evaluate`, and Weights & Biases (W&B).

Runtime behavior is controlled by [`configs/config.yaml`](configs/config.yaml). That file is now an exhaustive field reference: every active value has a preceding comment that describes its purpose, allowed values, and important constraints.

## Repository layout

```text
configs/config.yaml        Runtime configuration
dataset/train_split.csv    Training CSV
dataset/eval_split.csv     Evaluation CSV
src/cli.py                 Hydra entry point and run-mode dispatcher
src/data/                  CSV loading and training-message conversion
src/training/              LoRA/SFT training and model export
src/retrieval/             Chroma index construction and retrieval
src/prompting/             Prompt builders and output extraction
src/inference/             Model loading and batched generation
src/evaluation/            Evaluation loop, prediction CSV, and metrics
src/tracking/              Local JSON experiment logs
```

## Requirements and installation

You need a Python environment supported by `requirements.txt`, a CUDA-capable GPU suitable for the selected model, and Hugging Face access/cached assets for the model and metrics. Log in to W&B only for `wandb.mode: online`; use `offline` or `disabled` otherwise.

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The Docker image uses the expected working directory automatically:

```powershell
docker build -t tamang-translation .
docker run --gpus all --rm tamang-translation
```

## Run from `src`

For local runs, start in `src`:

```powershell
cd src
python cli.py
```

The configured paths such as `../dataset/eval_split.csv` and `../output/...` are relative to this directory. Therefore they resolve to the repository's `dataset` and `output` directories. This matches the Dockerfile's `/app/src` working directory.

Hydra loads `../configs/config.yaml`. Override any field for one command without editing the file:

```powershell
python cli.py run.mode=train model.source=base wandb.mode=offline
python cli.py run.mode=evaluate prompt.strategy=rag_few_shot
```

## Data format and preparation

The included CSVs have headers including:

```text
sentence_id,translation_tmg,relevant_sentences,translation_en,...
```

Every `source_column` and `target_column` named by a direction must exist in the matching CSV. For each configured direction, the loader maps those columns into the normalized fields `source`, `target`, `source_language`, and `target_language`. It then concatenates every configured direction. The default training configuration creates Tamang → English, English → Tamang, Nepali → Tamang, and Tamang → Nepali examples from one CSV.

`train_data.max_samples` caps each direction before concatenation, not the final combined dataset. `shuffle: true` uses `run.seed` for deterministic ordering.

## Modes

Set `run.mode` to one of `train`, `index`, or `evaluate`.

### Train

Training loads a base Hugging Face model or a local merged model with Unsloth. It injects LoRA adapters into `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. Training data is converted to a system/user translation prompt plus an assistant completion and passed to `SFTTrainer`.

The LoRA rank, alpha, dropout, optimization parameters, and trainer output path are under `training`. `completion_loss: true` limits loss to assistant-completion tokens. After a successful run the code saves:

- adapter and tokenizer: `run.adapter_dir`;
- merged 16-bit model and tokenizer: `run.merged_dir`;
- Q4_K_M GGUF export: `run.output_dir`.

`training.output_dir` belongs to `SFTTrainer` and is separate from final export directories. The implementation always creates a LoRA adapter; `training.method` is currently a descriptive W&B tag, not an algorithm switch.

```powershell
cd src
python cli.py run.mode=train model.source=base wandb.mode=offline
```

To evaluate the resulting merged export, point `model.model_path` at `run.merged_dir` and use `model.source=merged`.

### Index

Index mode loads the concatenated training directions, embeds each source as `passage: <source>`, and upserts it into a persistent Chroma collection. Metadata stores the paired target and both language labels. RAG later filters candidates to the requested source/target language pair.

Build an index before using `rag_few_shot`:

```powershell
cd src
python cli.py run.mode=index rag.rebuild=true
```

`rag.index_path`, `rag.collection_name`, and `rag.embedding_model` must stay identical between indexing and RAG evaluation. Set `rag.rebuild: true` after changing the corpus, directions, or embedding model; otherwise stale records may remain in the collection.

### Evaluate

Evaluation loads the selected model, builds one prompt per normalized evaluation row, generates in batches, writes a CSV, computes metrics, and logs results. Generation is deterministic (`do_sample=False`) and currently limited to 128 new tokens. `model.max_seq_length` controls the training path; evaluation uses tokenizer truncation and the generator's fixed output limit.

Available `prompt.strategy` values:

| Strategy | What it does |
| --- | --- |
| `zero_shot` | Direct translation instruction. |
| `few_shot` | Inserts hand-authored examples from `prompt.examples[prompt.direction]`. |
| `rag_few_shot` | Retrieves language-matched Chroma examples and adds the best `rag.top_k` to the prompt. |
| `cot_translation` | Requests visible linguistic analysis and extracts the final `Translation:` line. |
| `back_translation` | Generates forward and reverse translations, then asks the model to review/correct the forward output. |

`few_shot` needs a populated examples mapping for the selected `prompt.direction`. Blank example values are still passed to the prompt, so do not choose one of the placeholder mappings until it is filled in.

```powershell
cd src
python cli.py run.mode=evaluate model.source=merged model.model_path=../output/merged/Qwen
```

## Evaluation CSV and metrics

Once all evaluation batches complete, `src/evaluation/runner.py` writes:

```text
<eval_data.model_output>/<safe-model-name>_<model.source>.csv
```

With the current configuration, the file is:

```text
../output/eval_results/Qwen__Qwen3.5-4B_merged.csv
```

`/` in a Hugging Face model ID is changed to `__`, so it cannot accidentally create a subdirectory. The file contains aligned rows in this format:

```csv
source,prediction,reference
<input sentence>,<model translation>,<reference translation>
```

The file is opened in write mode, so a second evaluation with the same model name and source overwrites it. The evaluator then attempts BLEU, METEOR, TER, chrF, chrF++, COMET, and multilingual BERTScore. A metric error is handled independently and recorded as `None`; it does not prevent the other metrics from running. The prediction CSV is written before metric calculation.

## Tracking and outputs

Every mode initializes W&B with the resolved Hydra configuration and tags for model source, training method, and prompt strategy.

With the recommended `src` working directory, local JSON logs are written to:

- `src/train_experiments/run_<timestamp>.json` for a training run, including the training configuration under `lora_parameters`;
- `src/run_experiments/run_<timestamp>.json` for an evaluation run;
- `src/run_experiments/evalrun_<timestamp>.json` for evaluation columns, strategy, model source, and scores.

An evaluation error during model loading or generation occurs before the CSV write, so no CSV will exist. A metrics-only error occurs after the CSV write and leaves the CSV intact.

## Configuration relationships

- `model.source: base` loads `model.name`; `model.source: merged` loads `model.model_path`.
- After training, set `model.model_path` to `run.merged_dir` to evaluate the merged model.
- Data-column names must match CSV headers exactly.
- `rag.top_k` must be no greater than `rag.candidate_k`.
- Use the same RAG path, collection name, and embedding model for index and retrieval.
- Keep artifact directories writable and separate if trainer checkpoints, adapters, merged models, GGUF exports, and prediction files should not mix.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Dataset path not found | Run from `src`, or update every relative path for your current working directory. |
| RAG collection not found | Run index mode first and confirm the RAG path/name/embedding settings match. |
| RAG yields too few examples | Verify matching language-label pairs were indexed; reduce `top_k` or `candidate_k` if needed. |
| CUDA out of memory | Lower training/evaluation batch size, training sequence length, or LoRA rank; enable 4-bit evaluation loading. |
| Prediction CSV is absent | Confirm `run.mode=evaluate`; errors before all generation batches finish prevent the write. |
| W&B fails | Use `wandb.mode=offline` or `disabled`, or authenticate with `wandb login`. |
