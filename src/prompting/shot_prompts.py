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

def build_messages_cot_translation(
    source_text: str,
    source_lang: str,
    target_lang: str,
):
    """Build a translation prompt using explicit (visible) chain-of-thought,
    following the decomposition pattern used in MAPS / CoD-style MT prompting."""
    return [
        {
            "role": "system",
            "content": (
                f"You are an expert {source_lang}-to-{target_lang} translator. "
                "Think through the translation step by step, writing out each step. "
                "Do not skip steps."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Translate this {source_lang} sentence into {target_lang}:\n\n"
                f"{source_text}\n\n"
                "Follow this exact format:\n"
                "1. Tense: <identify the tense>\n"
                "2. Polarity: <affirmative/negative>\n"
                "3. Sentence type: <declarative/interrogative/imperative/etc.>\n"
                "4. Subject-object structure: <briefly describe>\n"
                "5. Key terms: <list 2-4 important words/phrases and their "
                f"{target_lang} equivalents>\n"
                f"6. Draft translation: <a first-pass {target_lang} translation>\n"
                "7. Check: <verify the draft preserves the tense, polarity, and "
                "sentence type identified above; note any fix needed>\n"
                "Translation: <the final corrected translation only, on its own line>"
            ),
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
