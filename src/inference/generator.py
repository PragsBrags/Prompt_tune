import torch


def translate(
    model,
    tokenizer,
    message_batch,
    cfg_model,
    strategy,
    target_languages,
):
    prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=cfg_model.thinking,
        )
        for messages in message_batch
    ]

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=strategy.max_new_tokens,
            num_beams=strategy.num_beams,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = output[:, input_length:]

    decoded = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return [
        strategy.extractor(text.strip(), target_language)
        for text, target_language in zip(decoded, target_languages)
    ]