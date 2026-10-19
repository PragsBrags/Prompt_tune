"""Regression tests for tensor-safe generation helpers."""

import torch
import unittest

from inference.generator import _scalar_token_id


class GeneratorTests(unittest.TestCase):
    def test_scalar_token_id_accepts_python_and_singleton_tensor_values(self):
        self.assertEqual(_scalar_token_id(0, "pad_token_id"), 0)
        self.assertEqual(_scalar_token_id(torch.tensor([11]), "pad_token_id"), 11)
        self.assertIsNone(_scalar_token_id(None, "pad_token_id"))

    def test_scalar_token_id_rejects_multi_element_tensor_without_boolean_evaluation(self):
        with self.assertRaisesRegex(ValueError, "must be one token id"):
            _scalar_token_id(torch.tensor([1, 2]), "pad_token_id")
