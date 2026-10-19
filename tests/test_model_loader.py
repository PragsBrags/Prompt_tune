"""Regression tests for architecture-aware inference loading."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from inference.model_loader import _uses_conditional_generation, load_model


def _model_config(architectures):
    return SimpleNamespace(architectures=architectures)


def _cfg():
    return SimpleNamespace(
        source="base",
        name="example/model",
        model_path="unused",
        quantization=SimpleNamespace(enabled=False, load_in_4bit=False),
    )


class ModelLoaderTests(unittest.TestCase):
    def test_conditional_generation_is_selected_from_architecture_metadata(self):
        self.assertTrue(_uses_conditional_generation(_model_config(["Gemma4ForConditionalGeneration"])))
        self.assertTrue(_uses_conditional_generation(_model_config(["Mistral3ForConditionalGeneration"])))
        self.assertTrue(_uses_conditional_generation(_model_config(["Qwen3_5ForConditionalGeneration"])))
        self.assertFalse(_uses_conditional_generation(_model_config(["LlamaForCausalLM"])))

    @patch("inference.model_loader.AutoModelForImageTextToText")
    @patch("inference.model_loader.AutoProcessor")
    @patch("inference.model_loader.AutoConfig")
    def test_conditional_checkpoint_uses_processor_and_conditional_model(
        self, auto_config, auto_processor, conditional_model
    ):
        auto_config.from_pretrained.return_value = _model_config(["Mistral3ForConditionalGeneration"])
        processor = Mock()
        processor.tokenizer = Mock()
        auto_processor.from_pretrained.return_value = processor

        loaded_processor, _ = load_model(_cfg())

        self.assertIs(loaded_processor, processor)
        self.assertEqual(processor.tokenizer.padding_side, "left")
        conditional_model.from_pretrained.assert_called_once()

    @patch("inference.model_loader.AutoModelForCausalLM")
    @patch("inference.model_loader.AutoTokenizer")
    @patch("inference.model_loader.AutoConfig")
    def test_causal_checkpoint_keeps_tokenizer_and_causal_model(self, auto_config, auto_tokenizer, causal_model):
        auto_config.from_pretrained.return_value = _model_config(["LlamaForCausalLM"])

        tokenizer, _ = load_model(_cfg())

        self.assertIs(tokenizer, auto_tokenizer.from_pretrained.return_value)
        auto_tokenizer.from_pretrained.assert_called_once_with("example/model", padding_side="left")
        causal_model.from_pretrained.assert_called_once()
