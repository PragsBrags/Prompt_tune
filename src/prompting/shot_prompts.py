def build_messages_zero(source_text: str, source_lang, target_lang):
    return [
        {"role": "system", "content":(
                                "You are a professional multilingual translator."
                                "Return only the translation and no further explanation"
                                )
                            },
        {"role": "user", "content": f"Translate from {source_lang} "
                                    f"to {target_lang}.\n\n"
                                    f"{source_text}"
                                    }
    ]

def build_messages_3(examples, source_lang, target_lang, source_text):

    i = 1
    shot_examples = []

    for i, example in enumerate(examples, start = 1):
        shot_examples.append(
            f"Example{i}\n"
            f"{source_lang}: {example.source}\n"
            f"{target_lang}: {example.target}"
        )

    shot_examples = "\n".join(shot_examples)

    return [
        {"role": "system", 
        "content":(
                f"You are a language translator that translates "
                f"{source_lang} to {target_lang} without any explanation. "
                f"You will only provide the translated text"
            ),
        }

        {"role": "user",
        "content": f"""Translate {source_lang} to {target_lang}.

            {shot_examples}

            {source_lang}: {source_text}
            """
        }
    ]