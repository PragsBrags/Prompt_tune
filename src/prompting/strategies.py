from dataclasses import dataclass
from typing import Callable

from prompting.shot_prompts import (
    NATIVE_NAMES,
    build_messages_3,
    build_messages_few_shot,
    build_messages_language_specific,
    build_messages_pivot,
    build_messages_self_refinement,
    build_messages_zero,
)


def extract_standard(text, target_lang):
    delimiter = f"{NATIVE_NAMES[target_lang]}:"
    if text.startswith(delimiter):
        text = text[len(delimiter):].strip()
    return text.splitlines()[0].strip() if text else ""


def extract_pivot(text, target_lang):
    delimiter = f"{NATIVE_NAMES[target_lang]}:"
    if delimiter in text:
        text = text.split(delimiter)[-1].strip()
    return text.splitlines()[0].strip() if text else ""


def extract_self_refinement(text, target_lang):
    for marker in (
        "FINAL_TRANSLATION:",
        "FINAL TRANSLATION:",
        "**FINAL_TRANSLATION:**",
    ):
        if marker in text:
            return text.split(marker)[-1].strip().splitlines()[0]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


@dataclass(frozen=True)
class PromptStrategy:
    builder: Callable
    max_new_tokens: int
    num_beams: int
    extractor: Callable


PROMPT_STRATEGIES = {
    "zero_shot": PromptStrategy(
        builder=build_messages_zero,
        max_new_tokens=128,
        num_beams=1,
        extractor=extract_standard,
    ),
    "3_shot": PromptStrategy(
        builder=build_messages_3,
        max_new_tokens=128,
        num_beams=1,
        extractor=extract_standard,
    ),
    "few_shot": PromptStrategy(
        builder=build_messages_few_shot,
        max_new_tokens=128,
        num_beams=1,
        extractor=extract_standard,
    ),
    "language_specific": PromptStrategy(
        builder=build_messages_language_specific,
        max_new_tokens=128,
        num_beams=1,
        extractor=extract_standard,
    ),
    "pivot_prompting": PromptStrategy(
        builder=build_messages_pivot,
        max_new_tokens=256,
        num_beams=1,
        extractor=extract_pivot,
    ),
    "self_refinement": PromptStrategy(
        builder=build_messages_self_refinement,
        max_new_tokens=512,
        num_beams=1,
        extractor=extract_self_refinement,
    ),
}


def get_prompt_strategy(name: str) -> PromptStrategy:
    if name not in PROMPT_STRATEGIES:
        available = ", ".join(PROMPT_STRATEGIES)
        raise ValueError(
            f"Unknown prompting strategy '{name}'. Available: {available}"
        )
    return PROMPT_STRATEGIES[name]