
def to_message(data):
    messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are a professional multilingual translator."
                    "Return only the translation and no further explanation"
                ),
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"Translate from {data['source_language']} "
                    f"to {data['target_language']}.\n\n"
                    f"{data['source']}"
                        ),
                    }
                ],
            },
    ]
    completion = [
        {"role": "assistant", 
         "content":[
             {
                 "type": "text",
                 "text": data["target"],
             }
         ],
        },
    ]

    return {
            "prompt": messages,
            "completion": completion
            }

def train_message(data):
    train_data = data.map(
        to_message, 
        remove_columns=data.column_names,
        )

    return train_data