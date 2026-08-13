FEW_SHOT_EXAMPLES = {
    "Nepali": [
        ("Good morning.", "शुभ प्रभात।"),
        ("How are you?", "तपाईंलाई कस्तो छ?"),
        ("Thank you very much.", "धेरै धेरै धन्यवाद।"),
    ],
    "Maithili": [
        ("Good morning.", "सुप्रभात।"),
        ("How are you?", "अहाँ कोना छी?"),
        ("Thank you very much.", "बहुत बहुत धन्यवाद।"),
    ],
}


def build_messages_zero(source_text: str, target_lang: str):
    return [
        {"role": "system", "content": f"You are a language translator that translates English to {target_lang} without any explanation. You will only provide the translated text"},
        {"role": "user", "content": f"Translate the following English sentence into {target_lang}.{source_text}"}
    ]


def build_messages_3(source_text: str, target_lang: str):
    examples = FEW_SHOT_EXAMPLES[target_lang]
    examples_block = "\n\n".join(
        f"Example {i + 1}\nEnglish: {en}\n{target_lang}: {translated}"
        for i, (en, translated) in enumerate(examples)
    )
    return [
        {"role": "system", "content": f"You are a language translator that translates English to {target_lang} without any explanation. You will only provide the translated text"},
        {
            "role": "user",
            "content": f"""Translate English to {target_lang}.

{examples_block}

Now translate:

English: {source_text}
{target_lang}:
"""
        }
    ]
