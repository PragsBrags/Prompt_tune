from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_cfg):
    model_Name = None
    
    if model_cfg.source == "base":
        model_Name = model_cfg.name
    elif model_cfg.source == "merged":
        model_Name = model_cfg.model_path
    
    # quantization_config = None
    
    # if model_cfg.quantization.enabled:
    #     quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=model_cfg.quantization.load_in_4bit,
    #     bnb_4bit_compute_dtype=torch.float16, 
    #     bnb_4bit_quant_type="nf4",             
    #     bnb_4bit_use_double_quant=True,
    #     llm_int8_enable_fp32_cpu_offload=True,
        # )

    tokenizer = AutoTokenizer.from_pretrained(model_Name,
        padding_side="left",                                          
    )
    if tokenizer.chat_template is None:
        # Llama keeps its template in tokenizer_config.json (there is no chat_template.jinja file)
        tokenizer.chat_template = AutoTokenizer.from_pretrained(model_Name).chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
    model_Name,
    dtype=torch.bfloat16,
    device_map="auto"
    )

    return tokenizer, model