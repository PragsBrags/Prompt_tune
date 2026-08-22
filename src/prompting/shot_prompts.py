def build_messages_zero(source_text: str, source_lang, target_lang):
    return [
        {"role": "system", "content":"You are a professional multilingual translator."
                            "Return only the translation and no further explanation"},
        {"role": "user", "content": f"Translate from {source_lang} "
                                    f"to {target_lang}.\n\n"
                                    f"{source_text}"
                                    }
    ]

def build_messages_3(source_text:str):
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