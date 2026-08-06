
def to_message(cfg_data, data, tokenizer):
    resp = data[cfg_data.target_column]

    messages = [
    {
        "role": "system",
        "content": "You are a professional English-to-Nepali translator."
    },
    {
        "role": "user",
        "content": data[cfg_data.source_column]
    },
    {
        "role": "assistant",
        "content": resp
    }
    ]

    return {
        "text" : tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }

def train_message(cfg_data, data, tokenizer):
    train_data = data.map(
        to_message, 
        fn_kwargs={"cfg_data": cfg_data, "tokenizer": tokenizer},
        remove_columns=data.column_names,
        )

    return train_data