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


def build_messages_zero(source_text: str, target_lang: str, source_lang: str = "English"):
    return [
        {
            "role": "system",
            "content": (
                f"You are a language translator that translates {source_lang} to "
                f"{target_lang} without any explanation. You will only provide the "
                "translated text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Translate the following {source_lang} sentence into {target_lang}.\n\n"
                f"{source_text}"
            ),
        },
    ]


def build_messages_3(source_text: str, target_lang: str, source_lang: str = "English"):
    examples = FEW_SHOT_EXAMPLES[target_lang]
    examples_block = "\n\n".join(
        f"Example {i + 1}\nEnglish: {en}\n{target_lang}: {translated}"
        for i, (en, translated) in enumerate(examples)
    )
    return [
        {
            "role": "system",
            "content": (
                f"You are a language translator that translates {source_lang} to "
                f"{target_lang} without any explanation. You will only provide the "
                "translated text."
            ),
        },
        {
            "role": "user",
            "content": f"""Translate {source_lang} to {target_lang}.

{examples_block}

Now translate:

{source_lang}: {source_text}
{target_lang}:
"""
        }
    ]


def build_messages_decomposed_translation(
    source_text: str,
    source_lang: str,
    target_lang: str,
):
    """Build a translation prompt that preserves grammar before lexical fluency."""
    return [
        {
            "role": "system",
            "content": (
                f"You are an expert {source_lang}-to-{target_lang} translator. "
                "Before translating, silently identify tense, polarity, sentence type, "
                "and subject-object structure. Preserve the corresponding grammatical "
                f"markers in {target_lang}. Return only the {target_lang} translation."
            ),
        },
        {
            "role": "user",
            "content": f"Translate this {source_lang} sentence into {target_lang}:\n\n{source_text}",
        },
    ]


def build_messages_back_translation(
    translated_text: str,
    source_lang: str,
    target_lang: str,
):
    """Build the reverse pass used to check a forward translation."""
    return [
        {
            "role": "system",
            "content": (
                f"You are a language translator that translates {target_lang} to "
                f"{source_lang}. Return only the translated text."
            ),
        },
        {
            "role": "user",
            "content": f"Translate this {target_lang} sentence into {source_lang}:\n\n{translated_text}",
        },
    ]


def build_messages_consistency_review(
    source_text: str,
    translated_text: str,
    back_translated_text: str,
    source_lang: str,
    target_lang: str,
):
    """Ask the model to correct a translation when its reverse pass changes meaning."""
    return [
        {
            "role": "system",
            "content": (
                f"You verify {source_lang}-to-{target_lang} translations. Compare the "
                f"original {source_lang} sentence with its back-translation, including "
                "meaning and grammatical features. If they differ, correct the target "
                f"translation. Return only the final {target_lang} translation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original {source_lang}: {source_text}\n"
                f"Candidate {target_lang}: {translated_text}\n"
                f"Back-translation {source_lang}: {back_translated_text}"
            ),
        },
    ]
