"""Shared fixtures for precipitate tests."""

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from pydantic import BaseModel


class SampleOutput(BaseModel):
    """Sample Pydantic model for testing."""

    name: str
    value: int


@pytest.fixture
def sample_output_type() -> type[SampleOutput]:
    """Return a sample Pydantic model type for testing."""
    return SampleOutput


@pytest.fixture
def sample_outputs() -> list[SampleOutput]:
    """Return sample outputs for testing."""
    return [
        SampleOutput(name="Alice", value=42),
        SampleOutput(name="Bob", value=17),
    ]


@pytest.fixture
def sample_inputs() -> list[str]:
    """Return sample inputs for testing."""
    return [
        "Extract info from: Alice scored 42 points",
        "Extract info from: Bob scored 17 points",
    ]


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer with realistic behavior."""
    tokenizer = MagicMock()

    # Set up token properties
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1

    def apply_chat_template(
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> str:
        """Simulate chat template application."""
        result = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            result += f"<{role}>{content}</{role}>"
        if add_generation_prompt:
            result += "<assistant>"
        return result

    tokenizer.apply_chat_template = apply_chat_template

    def tokenize_call(
        text: str,
        truncation: bool = True,
        max_length: int = 2048,
        return_tensors: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        """Simulate tokenization."""
        # Simple tokenization: each character is a token
        tokens = list(range(len(text)))
        attention_mask = [1] * len(tokens)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([tokens]),
                "attention_mask": torch.tensor([attention_mask]),
            }
        return {
            "input_ids": tokens,
            "attention_mask": attention_mask,
        }

    tokenizer.side_effect = tokenize_call
    tokenizer.__call__ = tokenize_call

    def decode(token_ids: torch.Tensor | list[int], skip_special_tokens: bool = True) -> str:
        """Simulate decoding."""
        return '{"name": "Test", "value": 123}'

    tokenizer.decode = decode

    return tokenizer


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock model with realistic behavior."""
    model = MagicMock()

    # Set up config
    model.config = MagicMock()
    model.config.pad_token_id = 0
    model.config.vocab_size = 32000

    # Mock forward pass
    def forward_call(**kwargs: Any) -> MagicMock:  # noqa: ANN401
        input_tensor = kwargs.get("input_ids", torch.zeros(1, 10))
        batch_size, seq_len = input_tensor.shape[0], input_tensor.shape[1]
        vocab_size = 32000

        output = MagicMock()
        output.logits = torch.randn(batch_size, seq_len, vocab_size)
        return output

    model.side_effect = forward_call
    model.__call__ = forward_call

    # Mock generate
    def generate_call(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 512,
        **kwargs: Any,  # noqa: ANN401
    ) -> torch.Tensor:
        batch_size, _ = input_ids.shape
        # Generate some fake tokens
        new_tokens = torch.randint(0, 32000, (batch_size, max_new_tokens))
        return torch.cat([input_ids, new_tokens], dim=1)

    model.generate = generate_call

    # Mock eval mode
    model.eval = MagicMock(return_value=model)

    # Mock to() method
    model.to = MagicMock(return_value=model)

    return model


@pytest.fixture
def mock_trainer_class() -> MagicMock:
    """Create a mock Trainer class."""
    trainer_instance = MagicMock()
    trainer_instance.train = MagicMock()

    trainer_class = MagicMock(return_value=trainer_instance)
    return trainer_class
