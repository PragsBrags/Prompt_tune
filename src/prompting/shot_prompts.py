def build_messages_zero(source_text: str, source_lang, target_lang):
    return [
        {"role": "system", "content":"You are a professional multilingual translator."
                            "Return only the translation and no further explanation"},
        {"role": "user", "content": f"Translate from {source_lang} "
                                    f"to {target_lang}.\n\n"
                                    f"{source_text}"
                                    }
    ]


NATIVE_NAMES = {
    "English": "English",
    "Hindi": "हिंदी",
    "Nepali": "नेपाली",
}


PIVOT_MAP = {
    ("English", "Hindi"): "Nepali",
    ("English", "Nepali"): "Hindi",
    ("Hindi", "English"): "Nepali",
    ("Hindi", "Nepali"): "English",
    ("Nepali", "English"): "Hindi",
    ("Nepali", "Hindi"): "English",
}


FEW_SHOT_EXAMPLES = {
    ("English", "Hindi"): [
        ("Good morning.", "शुभ प्रभात।"),
        ("How are you?", "आप कैसे हैं?"),
        ("Thank you.", "धन्यवाद।"),
    ],
    ("English", "Nepali"): [
        ("Good morning.", "शुभ प्रभात।"),
        ("How are you?", "तपाईंलाई कस्तो छ?"),
        ("Thank you.", "धन्यवाद।"),
    ],
    ("Hindi", "English"): [
        ("नमस्ते।", "Hello."),
        ("आप कैसे हैं?", "How are you?"),
        ("धन्यवाद।", "Thank you."),
    ],
    ("Nepali", "English"): [
        ("नमस्ते।", "Hello."),
        ("तपाईंलाई कस्तो छ?", "How are you?"),
        ("धन्यवाद।", "Thank you."),
    ],
    ("Hindi", "Nepali"): [
        ("नमस्ते।", "नमस्ते।"),
        ("आप कैसे हैं?", "तपाईंलाई कस्तो छ?"),
        ("धन्यवाद।", "धन्यवाद।"),
    ],
    ("Nepali", "Hindi"): [
        ("नमस्ते।", "नमस्ते।"),
        ("तपाईंलाई कस्तो छ?", "आप कैसे हैं?"),
        ("धन्यवाद।", "धन्यवाद।"),
    ],
}


LANGUAGE_SPECIFIC_INSTRUCTIONS = {
    ("English", "Hindi"): (
        "Use natural Hindi, correct gender and number agreement, and "
        "appropriate formal honorifics. Preserve proper names and meaning."
    ),
    ("English", "Nepali"): (
        "Use natural Nepali, appropriate honorifics, and correct grammar. "
        "Preserve proper names and meaning."
    ),
    ("Hindi", "English"): (
        "Use clear, natural English. Preserve the original meaning, tone, "
        "honorifics, and named entities."
    ),
    ("Nepali", "English"): (
        "Use natural, fluent English. Preserve the original meaning, tone, "
        "honorifics, and named entities."
    ),
    ("Hindi", "Nepali"): (
        "Use fluent Nepali and preserve Hindi honorifics and cultural meaning. "
        "Do not produce Hindi-sounding Nepali."
    ),
    ("Nepali", "Hindi"): (
        "Use fluent Hindi with correct gender agreement and honorifics. "
        "Preserve the original meaning and named entities."
    ),
}


def build_messages_language_specific(
    source_text: str,
    source_lang: str,
    target_lang: str,
):
    instruction = LANGUAGE_SPECIFIC_INSTRUCTIONS.get(
        (source_lang, target_lang),
        f"Translate accurately from {source_lang} to {target_lang}.",
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a professional multilingual translator. "
                f"{instruction} Return only the translation, without explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Translate from {source_lang} to {target_lang}:\n\n"
                f"{source_text}"
            ),
        },
    ]


def build_messages_few_shot(source_text, source_lang, target_lang):
    examples = FEW_SHOT_EXAMPLES.get((source_lang, target_lang), [])
    source_name = NATIVE_NAMES[source_lang]
    target_name = NATIVE_NAMES[target_lang]
    example_text = "\n\n".join(
        f"{source_name}: {source}\n{target_name}: {target}"
        for source, target in examples
    )

    return [
        {
            "role": "system",
            "content": (
                f"You translate {source_lang} to {target_lang}. "
                "Follow the examples and return only the translation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{example_text}\n\n"
                f"Now translate:\n{source_name}: {source_text}\n"
                f"{target_name}:"
            ),
        },
    ]


def build_messages_pivot(source_text, source_lang, target_lang):
    pivot_lang = PIVOT_MAP[(source_lang, target_lang)]
    source_name = NATIVE_NAMES[source_lang]
    pivot_name = NATIVE_NAMES[pivot_lang]
    target_name = NATIVE_NAMES[target_lang]

    return [
        {
            "role": "system",
            "content": (
                f"Translate from {source_lang} to {target_lang} through "
                f"{pivot_lang}. First produce an intermediate {pivot_lang} "
                "translation, then produce the final translation.\n\n"
                "Use exactly this format:\n"
                f"{pivot_name}: <intermediate translation>\n"
                f"{target_name}: <final translation>"
            ),
        },
        {
            "role": "user",
            "content": f"{source_name}: {source_text}\n{pivot_name}:",
        },
    ]


def build_messages_self_refinement(source_text, source_lang, target_lang):
    instruction = LANGUAGE_SPECIFIC_INSTRUCTIONS.get(
        (source_lang, target_lang),
        f"Translate accurately from {source_lang} to {target_lang}.",
    )

    return [
        {
            "role": "system",
            "content": (
                f"You are an expert translator for {source_lang} to "
                f"{target_lang}. {instruction}\n\n"
                "Use this self-refinement process:\n"
                "DRAFT: write an initial translation.\n"
                "CRITIQUE: check meaning, grammar, tone, and fluency.\n"
                "FINAL_TRANSLATION: write the corrected translation.\n\n"
                "Return the final translation after FINAL_TRANSLATION."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Translate from {source_lang} to {target_lang}:\n\n"
                f"{source_text}"
            ),
        },
    ]


def build_messages_3(source_text: str, source_lang=None, target_lang=None):
    return [
        {"role": "system", "content":"You are a language translator that translates English to Nepali without any explanation. You will only provide the translated text"},

        {
            "role": "user",
            "content": f"""Translate English to Nepali.

            Example 1
            English: Good morning.
            Nepali: शुभ प्रभात।

            Example 2
            English: How are you?
            Nepali: तपाईंलाई कस्तो छ?

            Example 3
            English: Thank you very much.
            Nepali: धेरै धेरै धन्यवाद।

            Now translate:

            English: {source_text}
            Nepali:
            """
        }
    ]