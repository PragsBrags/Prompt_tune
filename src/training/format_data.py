
def to_message(data, tokenizer):
    resp = data["response"]

    messages = [
    {
        "role": "system",
        "content": "You are a professional English-to-Nepali translator."
    },
    {
        "role": "user",
        "content": data["instruction"]
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

def train_message(data, tokenizer):
    train_data = data.map(
        to_message, 
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=data.column_names,
        )

    return train_data