
def to_message(data, tokenizer):
    messages = [
    {
        "role": "system",
        "content": (
            "You are a professional multilingual translator."
            "Return only the translation and no further explanation"
        ),
    },
    {
        "role": "user",
        "content": (
            f"Translate from {data['source_language']}"
            f"to {data['target_language']}. \n\n"
            f"{data['source']}"
        ),
    },
    {
        "role": "assistant",
        "content": data["target"]
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