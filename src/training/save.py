from unsloth import FastLanguageModel

def save_model(model,tokenizer,cfg_run):

    model.save_pretrained(cfg_run.adapter_dir)
    tokenizer.save_pretrained(cfg_run.adapter_dir)

    model.save_pretrained_merged(
    cfg_run.merged_dir,
    tokenizer,
    save_method="merged_16bit",
    )

    model.save_pretrained_gguf(
        cfg_run.output_dir, 
        tokenizer, 
        quantization_method="q4_k_m"
    )