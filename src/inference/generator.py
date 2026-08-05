import torch

def translate(model,tokenizer,message,generation_cfg):
    prompt = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=generation_cfg.thinking,
    )
    
    inputs = tokenizer(
        prompt,
        return_tensors='pt'
        ).to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )
    
    generated = tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    return generated