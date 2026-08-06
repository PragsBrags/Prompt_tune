from unsloth import FastLanguageModel

def save_model(model,tokenizer,cfg_run):
    model.save_pretrained_gguf(cfg_run.output_dir, tokenizer, quantization_method="q4_k_m")