from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch


def _uses_conditional_generation(config) -> bool:
    """Return whether a checkpoint declares a conditional-generation model.

    This is deliberately based on checkpoint metadata rather than a list of
    model names.  Gemma 4, Ministral 3, and Qwen 3.5 publish conditional
    generation architectures, while Llama publishes a causal-LM architecture.
    """
    architectures = getattr(config, "architectures", None) or ()
    return any(name.endswith("ForConditionalGeneration") for name in architectures)


def load_model(model_cfg):
    model_Name = None
    
    if model_cfg.source == "base":
        model_Name = model_cfg.name
    elif model_cfg.source == "merged":
        model_Name = model_cfg.model_path
    
    quantization_config = None
    
    if model_cfg.quantization.enabled:
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=model_cfg.quantization.load_in_4bit,
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_quant_type="nf4",             
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
        )

    config = AutoConfig.from_pretrained(model_Name)
    loader_kwargs = {
        "quantization_config": quantization_config,
        "torch_dtype": "auto",
        "device_map": "auto",
    }

    if _uses_conditional_generation(config):
        # A processor owns the checkpoint's tokenizer plus any optional
        # modality preprocessing. It remains valid for this text-only pipeline.
        processor = AutoProcessor.from_pretrained(model_Name)
        tokenizer = processor.tokenizer
        tokenizer.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(model_Name, **loader_kwargs)
        return processor, model

    tokenizer = AutoTokenizer.from_pretrained(model_Name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_Name, **loader_kwargs)
    return tokenizer, model
