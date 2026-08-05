from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def load_model(model_cfg):

    quantization_config = None
    
    if model_cfg.quantization.enabled:
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg.quantization.load_in_4bit,
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_quant_type="nf4",             
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name)
    model = AutoModelForCausalLM.from_pretrained(
    model_cfg.name,
    quantization_config=quantization_config,
    torch_dtype="auto",
    device_map="auto"
    )

    return tokenizer, model