import torch

def translate(model,tokenizer,message_batch,cfg_model):
    prompts =[ 
        tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=cfg_model.thinking,
        )
        for messages in message_batch
    ]
    
    inputs = tokenizer(
        text = prompts,
        return_tensors='pt',
        padding = True,
        truncation = True,
        ).to(model.device)
    
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,  # might be low for CoT translation
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_length = inputs.input_ids.shape[1]

    generated_tokens = output[:, input_length:]
    
    generated = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return [text.strip() for text in generated]