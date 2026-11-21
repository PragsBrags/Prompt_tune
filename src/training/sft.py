# from unsloth import FastLanguageModel
# from trl import SFTTrainer, SFTConfig
# from transformers import TrainingArguments
# import torch
# import wandb
# from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
# from data.data_loader import load_train_and_validation_data
# from training.format_data import train_message


# # ── FIX: Ministral's tokenizer returns tensors from apply_chat_template, but TRL
# #    expects plain Python lists (its `if prompt_ids and ...` check crashes on a tensor).
# def _to_list(x):
#     if torch.is_tensor(x):
#         x = x.tolist()
#         if len(x) == 1 and isinstance(x[0], list):   # (1, seq) -> (seq,)
#             x = x[0]
#     return x


# def make_list_returning(tok):
#     orig = tok.apply_chat_template

#     def wrapped(*args, **kwargs):
#         out = orig(*args, **kwargs)
#         if hasattr(out, "items"):                    # dict / BatchEncoding
#             return {k: _to_list(v) for k, v in out.items()}
#         return _to_list(out)

#     tok.apply_chat_template = wrapped
#     return tok


# def load_model(cfg_model, cfg):
#     model_Name = None

#     if cfg_model.source == "base":
#         model_Name = cfg_model.name
#     elif cfg_model.source == "merged":
#         model_Name = cfg_model.model_path

#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name=model_Name,
#         max_seq_length=cfg_model.max_seq_length,
#         dtype=None,
#         attn_implementation="eager"
#     )

#     model = FastLanguageModel.get_peft_model(
#         model,
#         r=cfg.training.lora_rank,
#         target_modules=[
#             'q_proj', 'k_proj', 'v_proj', 'o_proj',
#             'gate_proj', 'up_proj', 'down_proj',
#         ],  # which layers to inject LoRA into
#         lora_alpha=cfg.training.lora_alpha,
#         lora_dropout=cfg.training.lora_dropout,
#         bias="none",
#         use_gradient_checkpointing='unsloth',
#     )

#     return model, tokenizer

# def train_model(cfg):
#     model, tokenizer = load_model(cfg.model,cfg)

#     MINISTRAL_TEMPLATE = (
#     "{{ bos_token }}"
#     "{% for m in messages %}"
#     "{% if m['role'] == 'system' %}[SYSTEM_PROMPT]{{ m['content'] }}[/SYSTEM_PROMPT]"
#     "{% elif m['role'] == 'user' %}[INST]{{ m['content'] }}[/INST]"
#     "{% elif m['role'] == 'assistant' %}{{ m['content'] }}{{ eos_token }}"
#     "{% endif %}"
#     "{% endfor %}"
#     )

#     # for text-only SFT, use the inner tokenizer rather than the multimodal processor
#     inner = getattr(tokenizer, "tokenizer", tokenizer)
#     inner.chat_template = MINISTRAL_TEMPLATE
#     tokenizer.chat_template = MINISTRAL_TEMPLATE   # keep the processor in sync

#     make_list_returning(inner)
#     tokenizer = inner   # pass this to SFTTrainer as processing_class

#     # ── FIX: make apply_chat_template return lists (patch the processor's inner tokenizer too)
#     print("tokenizer type:", type(tokenizer))
#     make_list_returning(tokenizer)
#     if hasattr(tokenizer, "tokenizer"):
#         make_list_returning(tokenizer.tokenizer)
#     try:
#         _t = tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=True)
#         print("apply_chat_template ->", type(_t), _t[:8] if isinstance(_t, list) else _t)
#     except Exception as e:
#         print("apply_chat_template self-test failed:", repr(e))

#     data, valid_data = load_train_and_validation_data(
#         cfg.train_data,
#         cfg.run.seed,
#     )
#     dataset = train_message(data)
#     valid_dataset = train_message(valid_data)

#     trainer = SFTTrainer(
#         model=model,
#         train_dataset=dataset,
#         eval_dataset=valid_dataset,
#         processing_class=tokenizer,
#         args=SFTConfig(
#             report_to="wandb",
#             run_name=wandb.run.name,
#             eval_strategy="steps",
#             eval_steps=cfg.training.eval_steps,
            
#             completion_only_loss=cfg.training.completion_loss,
#             seed=cfg.run.seed,
#             data_seed=cfg.run.seed,
#             max_length=cfg.model.max_seq_length,
#             learning_rate=cfg.training.learning_rate,
#             per_device_train_batch_size=cfg.training.batch_size,
#             gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
#             warmup_steps=cfg.training.warmup_steps,
#             logging_steps=cfg.training.logging_steps,
#             output_dir=cfg.training.output_dir,
#             optim=cfg.training.optim,
#             num_train_epochs=cfg.training.num_train_epochs,
#         ),

#     )

#     trainer.train(resume_from_checkpoint=True)

#     return model, tokenizer

# Llama Qwen Gemma

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import wandb

from data.data_loader import load_train_and_validation_data
from training.format_data import train_message
from training.prompt_format import stop_token


def load_model(cfg_model, cfg):
    model_name = None
    if cfg_model.source == "base":
        model_name = cfg_model.name
    elif cfg_model.source == "merged":
        model_name = cfg_model.model_path

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=cfg_model.max_seq_length,
        dtype=None,
        attn_implementation="eager" 
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.training.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # Llama base: the header / <|eot_id|> embeddings are barely trained. If the model still
        # does not stop after fixing the data, add "embed_tokens" and "lm_head" to target_modules.
        lora_alpha=cfg.training.lora_alpha,
        lora_dropout=cfg.training.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer


def check_training_data(trainer, tokenizer, family):
    """Fail fast if the model would be trained on the wrong text."""
    ex = trainer.train_dataset[0]
    ids = list(ex["input_ids"])
    text = tokenizer.decode(ids)
    stop = stop_token(family)

    print("DECODED:", repr(text))
    print("LAST 5 :", tokenizer.convert_ids_to_tokens(ids[-5:]))
    mask = ex.get("completion_mask")
    if mask is not None:
        print("COMPLETION MASK (last 5):", list(mask)[-5:], "| trained tokens:", sum(mask))

    assert "[{'type'" not in text, "List-style content leaked into the text; content must be plain strings."
    assert stop in text, f"Stop token {stop} missing from the training text."
    stop_id = tokenizer.convert_tokens_to_ids(stop)
    assert stop_id is not None and stop_id != tokenizer.unk_token_id, f"{stop} is not a real token in this vocab."
    assert stop_id in ids[-3:], f"{stop} is not at the end of the example."
    bos = tokenizer.bos_token_id
    if bos is not None:
        assert ids.count(bos) == 1, f"Expected exactly one BOS, found {ids.count(bos)}."
    print("Data check passed.")


def train_model(cfg):
    model, tokenizer = load_model(cfg.model, cfg)

    # Unsloth may return a processor; SFT on text only needs the inner tokenizer
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if tokenizer.pad_token is None:
        # Prefer a dedicated pad token so padding never collides with a stop token
        pad = "<|finetune_right_pad_id|>"
        if tokenizer.convert_tokens_to_ids(pad) not in (None, tokenizer.unk_token_id):
            tokenizer.pad_token = pad
        else:
            tokenizer.pad_token = tokenizer.eos_token

    family = cfg.model.family          # "llama3" | "qwen" | "gemma4"

    data, valid_data = load_train_and_validation_data(cfg.train_data, cfg.run.seed)
    dataset = train_message(data, family)
    valid_dataset = train_message(valid_data, family)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            report_to="wandb",
            run_name=wandb.run.name,
            eval_strategy="steps",
            eval_steps=cfg.training.eval_steps,
            completion_only_loss=True,      # loss on translation + stop token only
            seed=cfg.run.seed,
            data_seed=cfg.run.seed,
            max_length=cfg.model.max_seq_length,
            learning_rate=cfg.training.learning_rate,
            per_device_train_batch_size=cfg.training.batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            warmup_steps=cfg.training.warmup_steps,
            logging_steps=cfg.training.logging_steps,
            output_dir=cfg.training.output_dir,
            optim=cfg.training.optim,
            num_train_epochs=cfg.training.num_train_epochs,
        ),
    )

    check_training_data(trainer, tokenizer, family)

    trainer.train()         
    return model, tokenizer