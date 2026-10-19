import torch


def _tokenizer_from_processor(processor):
    """Return a tokenizer for either a tokenizer or an AutoProcessor."""
    return getattr(processor, "tokenizer", processor)


def _scalar_token_id(token_id, name):
    """Normalize optional scalar token ids without relying on tensor truthiness."""
    if token_id is None:
        return None
    if isinstance(token_id, torch.Tensor):
        if token_id.numel() != 1:
            raise ValueError(f"{name} must be one token id, not a tensor of shape {tuple(token_id.shape)}.")
        return token_id.item()
    return token_id


def translate(model, processor, message_batch, cfg_model):
    """Render, tokenize, generate, and decode a batch of text-only chats."""
    prompts =[ 
        processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=cfg_model.thinking,
        )
        for messages in message_batch
    ]
    
    inputs = processor(
        text = prompts,
        return_tensors='pt',
        padding = True,
        truncation = True,
        ).to(model.device)

    tokenizer = _tokenizer_from_processor(processor)
    pad_token_id = _scalar_token_id(tokenizer.pad_token_id, "pad_token_id")
    
    with torch.inference_mode():
        try:
            output = model.generate(
                **inputs,
                max_new_tokens=256,  # might be low for CoT translation
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        except RuntimeError as error:
            attention_mask = inputs.get("attention_mask")
            raise RuntimeError(
                "Generation failed "
                f"(model={model.__class__.__name__}, "
                f"input_ids_shape={tuple(inputs.input_ids.shape)}, "
                f"attention_mask_shape={None if attention_mask is None else tuple(attention_mask.shape)}, "
                f"pad_token_id={pad_token_id!r})."
            ) from error

    input_length = inputs.input_ids.shape[1]

    generated_tokens = output[:, input_length:]
    
    generated = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return [text.strip() for text in generated]
