LANGUAGE_CODES = {
    "english": "en",
    "en": "en",
    "nepali": "np",
    "np": "np",
    "tamang": "tmg",
    "tmg": "tmg",
}


def few_shot_direction_key(source_lang: str, target_lang: str) -> str:
    """Return the examples-map key for a configured language direction.

    The key is derived from the row's language metadata instead of a global
    setting, so a mixed-direction evaluation always receives matching shots.
    """
    try:
        source_code = LANGUAGE_CODES[source_lang.strip().lower()]
        target_code = LANGUAGE_CODES[target_lang.strip().lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError(
            "Cannot select few-shot examples for "
            f"{source_lang!r} -> {target_lang!r}. Supported languages are "
            "English, Nepali, and Tamang."
        ) from error
    return f"{source_code}_{target_code}"


def get_few_shot_examples(examples, source_lang: str, target_lang: str):
    """Fetch demonstrations matching a source/target language pair."""
    direction_key = few_shot_direction_key(source_lang, target_lang)
    if direction_key not in examples:
        raise ValueError(
            f"Missing prompt.examples.{direction_key} for "
            f"{source_lang} -> {target_lang}."
        )
    return examples[direction_key]


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
        },

        {"role": "user",
        "content":( 
            f"Translate {source_lang} to {target_lang}."
            f"{shot_examples}"
            f"{source_lang}: {source_text}"
            ),
        }
    ]


def build_messages_rag(examples, source_lang, target_lang, source_text):
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a professional translator from {source_lang} "
                f"to {target_lang}. Return only the translation."
            ),
        }
    ]

    for example in examples:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": (
                        f"Translate from {source_lang} to {target_lang}:\n\n"
                        f"{example.source}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": example.target,
                },
            ]
        )

    messages.append(
        {
            "role": "user",
            "content": (
                f"Translate from {source_lang} to {target_lang}:\n\n"
                f"{source_text}"
            ),
        }
    )

    return messages


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


def extract_final_translation(model_output: str) -> str:
    """Return the final translation from a visible CoT translation response."""
    for line in reversed(model_output.strip().splitlines()):
        if line.strip().lower().startswith("translation:"):
            return line.split(":", 1)[1].strip()

    # Fallback when the model did not follow the requested response format.
    lines = [line.strip() for line in model_output.strip().splitlines() if line.strip()]
    return lines[-1] if lines else model_output.strip()


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
    """Ask the model to correct a translation when its reverse pass changes meaning,
    while tolerating harmless paraphrase drift from the back-translation step."""
    return [
        {
            "role": "system",
            "content": (
                f"You verify {source_lang}-to-{target_lang} translations using a "
                "back-translation check. Back-translations often differ in surface "
                "wording even when the original translation is correct — that is "
                "expected and NOT an error. Only flag a real problem if the "
                "back-translation reveals a genuine difference in meaning, tense, "
                "polarity, or sentence type versus the original."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original {source_lang}: {source_text}\n"
                f"Candidate {target_lang}: {translated_text}\n"
                f"Back-translation {source_lang}: {back_translated_text}\n\n"
                "Follow this format:\n"
                "1. Discrepancies found: <list any real meaning/tense/polarity/"
                "sentence-type differences, or 'none'>\n"
                "2. Verdict: <'keep as-is' or 'needs correction'>\n"
                f"Translation: <the final {target_lang} translation, corrected only "
                "if needed, on its own line>"
            ),
        },
    ]
