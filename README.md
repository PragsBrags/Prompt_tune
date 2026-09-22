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

Every `source_column` and `target_column` named by a direction must exist in the matching CSV. For each configured direction, the loader maps those columns into the normalized fields `source`, `target`, `source_language`, and `target_language`. Training and indexing concatenate every configured direction; evaluation keeps directions separate. The default training and evaluation configurations include Tamang ↔ English, Tamang ↔ Nepali, and English ↔ Nepali examples from one CSV.

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

To evaluate the resulting merged export, use `model.source=merged`. `model.model_path` automatically resolves to that model's `run.merged_dir`.

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
| `few_shot` | Inserts hand-authored examples from the `prompt.examples` entry matching the row's source and target languages. |
| `rag_few_shot` | Retrieves language-matched Chroma examples and adds the best `rag.top_k` to the prompt. |
| `cot_translation` | Requests visible linguistic analysis and extracts the final `Translation:` line. |
| `back_translation` | Generates forward and reverse translations, then asks the model to review/correct the forward output. |

`few_shot` automatically maps English, Nepali, and Tamang pairs to the corresponding `en_np`, `en_tmg`, `np_en`, `np_tmg`, `tmg_np`, or `tmg_en` examples entry. No prompt-direction override is needed. Every configured mapping must be populated before using the strategy.

```powershell
cd src
python cli.py run.mode=evaluate model.source=merged
```

## Evaluation CSV and metrics

Evaluation processes each entry in `eval_data.directions` independently. The default configuration contains all six English, Nepali, and Tamang directions. It loads the model once, then writes per-direction artifacts:

```text
<eval_data.model_output>/<safe-model-name>_<model.source>_<strategy>_<source-language>_to_<target-language>.csv
<eval_data.model_output>/<safe-model-name>_<model.source>_<strategy>_<source-language>_to_<target-language>.metrics.json
```

For example, the Tamang-to-English output is:

```text
../output/eval_results/Qwen__Qwen3.5-4B_merged_zero_shot_tamang_to_english.csv
```

`/` in a Hugging Face model ID is changed to `__`, so it cannot accidentally create a subdirectory. The file contains aligned rows in this format:

```csv
source,prediction,reference
<input sentence>,<model translation>,<reference translation>
```

The evaluator writes a prediction CSV and a score JSON for every direction, and logs each metric to a matching W&B namespace such as `eval/tamang_to_english/bleu`. It also writes one local `evalrun_..._<direction>.json` log per direction. A repeat run with the same model, strategy, and direction overwrites only that direction's prediction and score artifacts. The evaluator attempts BLEU, METEOR, TER, chrF, chrF++, COMET, and multilingual BERTScore independently; a metric error is recorded as `null` and does not prevent the others from running.

## Tracking and outputs

Every mode initializes W&B with the resolved Hydra configuration and tags for model source, training method, and prompt strategy.

With the recommended `src` working directory, local JSON logs are written to:

- `src/train_experiments/run_<timestamp>.json` for a training run, including the training configuration under `lora_parameters`;
- `src/run_experiments/run_<timestamp>.json` for an evaluation run;
- `src/run_experiments/evalrun_<timestamp>_<direction>.json` for each direction's columns, strategy, model source, artifact paths, and scores.

An error during model loading or generation prevents artifacts for the affected direction. Individual metric failures are caught and recorded as `null`, so they do not prevent that direction's CSV and score JSON from being saved.

## Configuration relationships

- `model.source: base` loads `model.name`; `model.source: merged` loads the model-specific `model.model_path`.
- Changing `model.name` automatically updates `run.output_dir`, `run.adapter_dir`, `run.merged_dir`, `model.model_path`, and `training.output_dir`.
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
