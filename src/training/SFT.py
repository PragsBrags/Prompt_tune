from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
import wandb

from data.data_loader import load_translation_data
from training.format_data import train_message


def load_model(cfg_model, cfg):
    model_Name = None

    if cfg_model.source == "base":
        model_Name = cfg_model.name
    elif cfg_model.source == "merged":
        model_Name = cfg_model.model_path

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_Name,
        max_seq_length=cfg_model.max_seq_length,
        dtype=None,
        load_in_4bit=cfg_model.quantization.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.training.lora_rank,
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ],  # which layers to inject LoRA into
        lora_alpha=cfg.training.lora_alpha,
        lora_dropout=cfg.training.lora_dropout,
        bias="none",
        use_gradient_checkpointing='unsloth',
    )

    return model, tokenizer

def train_model(cfg):
    model, tokenizer = load_model(cfg.model,cfg)
    data = load_translation_data(cfg.train_data, cfg.run.seed)
    dataset = train_message(data)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        args=SFTConfig(
            report_to="wandb",
            run_name=wandb.run.name,
            eval_strategy="steps",
            eval_steps=cfg.training.eval_steps,
            
            completion_only_loss=cfg.training.completion_loss,
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

    trainer.train()

    return model, tokenizer

