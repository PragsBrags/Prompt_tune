from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

from data.data_loader import load_translation_data
from training.format_data import train_message

def load_model(cfg_model):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg_model.name,
        max_seq_length=cfg_model.max_seq_length,
        torch_dtype="auto",
        load_in_4bit=cfg_model.quantization.load_in_4bit,
        device_map="auto",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg_model.lora_rank,
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ],  # which layers to inject LoRA into
        lora_alpha=cfg_model.lora_alpha,
        lora_dropout=cfg_model.lora_dropout,
        bias="none",
        use_gradient_checkpointing='unsloth',
    )

    return model, tokenizer

def train_model(cfg):
    model, tokenizer = load_model(cfg.model)
    data = load_translation_data(cfg.data, cfg.run.seed)
    dataset = train_message(cfg.data, data, tokenizer)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            dataset_text_field='text',
            seed=cfg.run.seed,
            data_seed=cfg.run.seed,
            max_length=cfg.model.max_seq_length,
            per_device_train_batch_size=cfg.training.batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            warmup_steps=cfg.training.warmup_steps,
            max_steps=cfg.training.max_steps,
            logging_steps=cfg.training.logging_steps,
            output_dir=cfg.training.output_dir,
            optim=cfg.training.optim,
            num_train_epochs=cfg.training.num_train_epochs,
        ),

    )

    trainer.train()

    return model, tokenizer

