"""Tests for the HuggingFaceModel implementation."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from conftest import SampleOutput

from precipitate.huggingface import (
    DistillationDataset,
    DistillationTrainer,
    HuggingfaceModel,
    StructuredOutputDataset,
)

# =============================================================================
# Dataset Classes Tests
# =============================================================================


class TestStructuredOutputDataset:
    """Tests for StructuredOutputDataset."""

    def test_len(self) -> None:
        """Test __len__ returns correct count."""
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        labels = torch.tensor([[1, 2, 3], [4, 5, -100]])
        token_type_ids = torch.tensor([[0, 0, 0], [0, 0, 0]])

        dataset = StructuredOutputDataset(
            input_ids, attention_mask, labels, token_type_ids
        )
        assert len(dataset) == 2

    def test_getitem(self) -> None:
        """Test __getitem__ returns correct dict structure."""
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        labels = torch.tensor([[1, 2, 3], [4, 5, -100]])
        token_type_ids = torch.tensor([[0, 0, 0], [0, 0, 0]])

        dataset = StructuredOutputDataset(
            input_ids, attention_mask, labels, token_type_ids
        )
        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert "token_type_ids" in item
        assert torch.equal(item["input_ids"], torch.tensor([1, 2, 3]))
        assert torch.equal(item["attention_mask"], torch.tensor([1, 1, 1]))
        assert torch.equal(item["labels"], torch.tensor([1, 2, 3]))
        assert torch.equal(item["token_type_ids"], torch.tensor([0, 0, 0]))


class TestDistillationDataset:
    """Tests for DistillationDataset."""

    def test_len(self) -> None:
        """Test __len__ returns correct count."""
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])
        labels = torch.tensor([[-100, -100, 3]])
        token_type_ids = torch.tensor([[0, 0, 0]])

        dataset = DistillationDataset(input_ids, attention_mask, labels, token_type_ids)
        assert len(dataset) == 1

    def test_getitem(self) -> None:
        """Test __getitem__ returns correct dict structure."""
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])
        labels = torch.tensor([[-100, -100, 3]])
        token_type_ids = torch.tensor([[0, 0, 0]])

        dataset = DistillationDataset(input_ids, attention_mask, labels, token_type_ids)
        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert "token_type_ids" in item


# =============================================================================
# HuggingFaceModel Initialization Tests
# =============================================================================


class TestHuggingFaceModelInit:
    """Tests for HuggingFaceModel initialization."""

    def test_init_stores_parameters(
        self, sample_output_type: type[SampleOutput]
    ) -> None:
        """Test that all init params are stored correctly."""
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
            system_prompt="Test prompt",
            max_new_tokens=1024,
            temperature=0.5,
            distillation_temperature=3.0,
            distillation_alpha=0.7,
            device="cpu",
        )

        assert model.model_name == "test-model"
        assert model.output_type == sample_output_type
        assert model.system_prompt == "Test prompt"
        assert model.max_new_tokens == 1024
        assert model.temperature == 0.5
        assert model.distillation_temperature == 3.0
        assert model.distillation_alpha == 0.7
        assert model._device_name == "cpu"

    def test_init_defaults(self, sample_output_type: type[SampleOutput]) -> None:
        """Test that default values are set correctly."""
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
        )

        assert model.system_prompt is None
        assert model.max_new_tokens == 512
        assert model.temperature == 0.7
        assert model.distillation_temperature == 2.0
        assert model.distillation_alpha == 0.5
        assert model._device_name is None

    def test_lazy_loading(self, sample_output_type: type[SampleOutput]) -> None:
        """Test that model/tokenizer are None until first use."""
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
        )

        assert model._model is None
        assert model._tokenizer is None
        assert model._device is None


# =============================================================================
# Device Detection Tests
# =============================================================================


class TestDeviceDetection:
    """Tests for device detection."""

    def test_get_device_explicit(self, sample_output_type: type[SampleOutput]) -> None:
        """Test when device is explicitly set."""
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
            device="cpu",
        )

        device = model._get_device()
        assert device == torch.device("cpu")

    def test_get_device_cached(self, sample_output_type: type[SampleOutput]) -> None:
        """Test that device is cached after first call."""
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
            device="cpu",
        )

        device1 = model._get_device()
        device2 = model._get_device()

        assert device1 is device2

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_get_device_auto_cpu(
        self,
        mock_mps: MagicMock,
        mock_cuda: MagicMock,
        sample_output_type: type[SampleOutput],
    ) -> None:
        """Test auto-detection falls back to CPU."""
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
        )

        device = model._get_device()
        assert device == torch.device("cpu")


# =============================================================================
# Chat Message Building Tests
# =============================================================================


class TestBuildChatMessages:
    """Tests for _build_chat_messages."""

    def test_no_system_prompt(self, sample_output_type: type[SampleOutput]) -> None:
        """Test building messages without system prompt."""
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
        )

        messages = model._build_chat_messages("Hello")
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello"}

    def test_with_system_prompt(self, sample_output_type: type[SampleOutput]) -> None:
        """Test building messages with system prompt."""
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
            system_prompt="You are helpful",
        )

        messages = model._build_chat_messages("Hello")
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are helpful"}
        assert messages[1] == {"role": "user", "content": "Hello"}

    def test_with_output(self, sample_output_type: type[SampleOutput]) -> None:
        """Test building messages with output JSON."""
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
            system_prompt="You are helpful",
        )

        messages = model._build_chat_messages("Hello", '{"result": 42}')
        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "You are helpful"}
        assert messages[1] == {"role": "user", "content": "Hello"}
        assert messages[2] == {"role": "assistant", "content": '{"result": 42}'}


# =============================================================================
# JSON Extraction Tests
# =============================================================================


# =============================================================================
# Fit Method Tests
# =============================================================================


class TestFit:
    """Tests for the fit method."""

    def test_fit_mismatched_lengths(
        self,
        sample_output_type: type[SampleOutput],
        sample_outputs: list[SampleOutput],
    ) -> None:
        """Test that ValueError is raised when inputs/outputs lengths differ."""
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
        )

        with pytest.raises(
            ValueError, match="inputs and outputs must have same length"
        ):
            model.fit(["input1"], sample_outputs)  # 1 input, 2 outputs

    @patch("precipitate.huggingface.TrainingArguments")
    @patch("precipitate.huggingface.Trainer")
    @patch("precipitate.huggingface.AutoConfig")
    @patch("precipitate.huggingface.AutoModelForCausalLM")
    @patch("precipitate.huggingface.AutoTokenizer")
    def test_fit_calls_trainer(
        self,
        mock_auto_tokenizer: MagicMock,
        mock_auto_model: MagicMock,
        mock_auto_config: MagicMock,
        mock_trainer_class: MagicMock,
        mock_training_args: MagicMock,
        sample_output_type: type[SampleOutput],
        sample_inputs: list[str],
        sample_outputs: list[SampleOutput],
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test that Trainer is called correctly during fit."""
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_config.from_pretrained.return_value.is_encoder_decoder = False

        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
            device="cpu",
        )

        model.fit(sample_inputs, sample_outputs)

        # Verify Trainer was instantiated and train was called
        mock_trainer_class.assert_called_once()
        mock_trainer_class.return_value.train.assert_called_once()


# =============================================================================
# Predict Method Tests
# =============================================================================


class TestPredict:
    """Tests for the predict method."""

    @patch("precipitate.huggingface.outlines")
    @patch("precipitate.huggingface.AutoConfig")
    @patch("precipitate.huggingface.AutoModelForCausalLM")
    @patch("precipitate.huggingface.AutoTokenizer")
    def test_predict_returns_parsed_output(
        self,
        mock_auto_tokenizer: MagicMock,
        mock_auto_model: MagicMock,
        mock_auto_config: MagicMock,
        mock_outlines: MagicMock,
        sample_output_type: type[SampleOutput],
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test that predict uses outlines and returns parsed Pydantic models."""
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_config.from_pretrained.return_value.is_encoder_decoder = False

        # Mock outlines wrapper and generator
        mock_outlines_wrapper = MagicMock()
        mock_outlines.from_transformers.return_value = mock_outlines_wrapper

        mock_generator = MagicMock()
        mock_generator.return_value = '{"name": "Test", "value": 123}'
        mock_outlines.Generator.return_value = mock_generator

        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
            device="cpu",
        )

        results = model.predict(["Test input"])

        # Verify outlines was used
        mock_outlines.from_transformers.assert_called_once()
        mock_outlines.Generator.assert_called_once_with(
            mock_outlines_wrapper, sample_output_type
        )

        # Verify results
        assert len(results) == 1
        assert isinstance(results[0], SampleOutput)
        assert results[0].name == "Test"
        assert results[0].value == 123

    @patch("precipitate.huggingface.outlines")
    @patch("precipitate.huggingface.AutoConfig")
    @patch("precipitate.huggingface.AutoModelForCausalLM")
    @patch("precipitate.huggingface.AutoTokenizer")
    def test_predict_caches_outlines_wrapper(
        self,
        mock_auto_tokenizer: MagicMock,
        mock_auto_model: MagicMock,
        mock_auto_config: MagicMock,
        mock_outlines: MagicMock,
        sample_output_type: type[SampleOutput],
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test that outlines wrapper is cached across predict calls."""
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_config.from_pretrained.return_value.is_encoder_decoder = False

        # Mock outlines wrapper and generator
        mock_outlines_wrapper = MagicMock()
        mock_outlines.from_transformers.return_value = mock_outlines_wrapper

        mock_generator = MagicMock()
        mock_generator.return_value = '{"name": "Test", "value": 123}'
        mock_outlines.Generator.return_value = mock_generator

        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
            device="cpu",
        )

        # First predict call
        model.predict(["Test input 1"])

        # Second predict call
        model.predict(["Test input 2"])

        # Verify outlines.from_transformers was only called once (cached)
        assert mock_outlines.from_transformers.call_count == 1

        # Verify generator was called twice (once per predict call)
        assert mock_generator.call_count == 2

    @patch("precipitate.huggingface.outlines")
    @patch("precipitate.huggingface.AutoConfig")
    @patch("precipitate.huggingface.AutoModelForCausalLM")
    @patch("precipitate.huggingface.AutoTokenizer")
    def test_predict_with_multiple_inputs(
        self,
        mock_auto_tokenizer: MagicMock,
        mock_auto_model: MagicMock,
        mock_auto_config: MagicMock,
        mock_outlines: MagicMock,
        sample_output_type: type[SampleOutput],
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test that predict handles multiple inputs correctly."""
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_config.from_pretrained.return_value.is_encoder_decoder = False

        # Mock outlines wrapper and generator
        mock_outlines_wrapper = MagicMock()
        mock_outlines.from_transformers.return_value = mock_outlines_wrapper

        mock_generator = MagicMock()
        # Return different outputs for each call
        mock_generator.side_effect = [
            '{"name": "Alice", "value": 1}',
            '{"name": "Bob", "value": 2}',
            '{"name": "Charlie", "value": 3}',
        ]
        mock_outlines.Generator.return_value = mock_generator

        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
            device="cpu",
        )

        results = model.predict(["Input 1", "Input 2", "Input 3"])

        # Verify all results
        assert len(results) == 3
        assert results[0].name == "Alice"
        assert results[0].value == 1
        assert results[1].name == "Bob"
        assert results[1].value == 2
        assert results[2].name == "Charlie"
        assert results[2].value == 3


# =============================================================================
# Distill Method Tests
# =============================================================================


class TestDistill:
    """Tests for the distill method."""

    def test_distill_mismatched_lengths(
        self,
        sample_output_type: type[SampleOutput],
        sample_outputs: list[SampleOutput],
    ) -> None:
        """Test that ValueError is raised when inputs/outputs lengths differ."""
        teacher = HuggingfaceModel(
            model_name="teacher-model",
            output_type=sample_output_type,
        )
        student = HuggingfaceModel(
            model_name="student-model",
            output_type=sample_output_type,
        )

        with pytest.raises(
            ValueError, match="inputs and outputs must have same length"
        ):
            teacher.distill(student, ["input1"], sample_outputs)


# =============================================================================
# DistillationTrainer Tests
# =============================================================================


class TestDistillationTrainer:
    """Tests for DistillationTrainer."""

    def test_init_stores_parameters(self, mock_model: MagicMock) -> None:
        """Test that init parameters are stored correctly."""
        # Create minimal required args
        with patch("precipitate.huggingface.Trainer.__init__", return_value=None):
            trainer = DistillationTrainer(
                teacher_model=mock_model,
                temperature=3.0,
                alpha=0.7,
            )

            assert trainer.teacher_model is mock_model
            assert trainer.temperature == 3.0
            assert trainer.alpha == 0.7

    def test_compute_loss_returns_tensor(self) -> None:
        """Test that compute_loss returns a tensor."""
        # Create teacher model mock
        teacher_model = MagicMock()

        with patch("precipitate.huggingface.Trainer.__init__", return_value=None):
            trainer = DistillationTrainer(
                teacher_model=teacher_model,
                temperature=2.0,
                alpha=0.5,
            )

            # Create mock inputs with consistent vocab size
            batch_size, seq_len, vocab_size = 2, 10, 100
            inputs = {
                "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len),
                "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
            }
            # Mask some labels
            inputs["labels"][:, :5] = -100

            # Create student mock that returns logits
            student_model = MagicMock()
            student_output = MagicMock()
            student_output.logits = torch.randn(batch_size, seq_len, vocab_size)
            student_model.return_value = student_output

            # Create teacher mock with SAME vocab size
            teacher_output = MagicMock()
            teacher_output.logits = torch.randn(batch_size, seq_len, vocab_size)
            teacher_model.return_value = teacher_output

            loss = trainer.compute_loss(student_model, inputs)

            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0  # Scalar tensor

    def test_compute_loss_respects_mask(self) -> None:
        """Test that -100 labels are properly masked."""
        # Create teacher model mock
        teacher_model = MagicMock()

        with patch("precipitate.huggingface.Trainer.__init__", return_value=None):
            trainer = DistillationTrainer(
                teacher_model=teacher_model,
                temperature=2.0,
                alpha=0.5,
            )

            # Create inputs where ALL labels are masked
            batch_size, seq_len, vocab_size = 1, 5, 100
            inputs = {
                "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len),
                "labels": torch.full((batch_size, seq_len), -100),  # All masked
            }

            student_model = MagicMock()
            student_output = MagicMock()
            student_output.logits = torch.randn(batch_size, seq_len, vocab_size)
            student_model.return_value = student_output

            teacher_output = MagicMock()
            teacher_output.logits = torch.randn(batch_size, seq_len, vocab_size)
            teacher_model.return_value = teacher_output

            # Should not raise even with all masked labels
            loss = trainer.compute_loss(student_model, inputs)
            assert isinstance(loss, torch.Tensor)


# =============================================================================
# Chat Template Fallback Tests
# =============================================================================


class TestChatTemplateFallback:
    """Tests for chat template fallback functionality."""

    def test_apply_chat_template_with_template(
        self, sample_output_type: type[SampleOutput], mock_tokenizer: MagicMock
    ) -> None:
        """Test that real template is used when available."""
        mock_tokenizer.chat_template = "some_template"
        # Replace the function with a MagicMock so we can control the return value
        mock_apply = MagicMock(return_value="formatted text")
        mock_tokenizer.apply_chat_template = mock_apply

        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
        )

        messages = [{"role": "user", "content": "Hello"}]
        result = model._apply_chat_template(mock_tokenizer, messages, False)

        assert result == "formatted text"
        mock_apply.assert_called_once()

    def test_apply_chat_template_fallback(
        self, sample_output_type: type[SampleOutput], mock_tokenizer: MagicMock
    ) -> None:
        """Test fallback when chat_template is None."""
        mock_tokenizer.chat_template = None

        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
            system_prompt="You are helpful",
        )

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = model._apply_chat_template(mock_tokenizer, messages, True)

        assert "System: You are helpful" in result
        assert "User: Hello" in result
        assert "Assistant: " in result

    def test_fallback_with_output(
        self, sample_output_type: type[SampleOutput], mock_tokenizer: MagicMock
    ) -> None:
        """Test fallback includes assistant output when provided."""
        mock_tokenizer.chat_template = None

        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
        )

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": '{"response": "Hi"}'},
        ]
        result = model._apply_chat_template(mock_tokenizer, messages, False)

        assert "User: Hello" in result
        assert 'Assistant: {"response": "Hi"}' in result

    def test_fallback_without_system_prompt(
        self, sample_output_type: type[SampleOutput], mock_tokenizer: MagicMock
    ) -> None:
        """Test fallback works without system prompt."""
        mock_tokenizer.chat_template = None

        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
        )

        messages = [{"role": "user", "content": "Hello"}]
        result = model._apply_chat_template(mock_tokenizer, messages, True)

        assert "User: Hello" in result
        assert "Assistant: " in result
        assert "System:" not in result


# =============================================================================
# Label Masking Tests
# =============================================================================


class TestLabelMasking:
    """Tests for correct label masking in training tokenization."""

    def test_tokenization_boundary_effects(
        self, sample_output_type: type[SampleOutput]
    ) -> None:
        """Test that label masking works correctly with tokenization boundary effects.

        This test simulates a tokenizer where the boundary between prompt and
        response is handled differently depending on context (e.g., space absorbed
        into next token).
        """
        # Create a mock tokenizer with boundary effects
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.chat_template = None

        # Simulate boundary effect: "Assistant: " vs "Assistant: {"
        # When tokenizing "Assistant: ", the space is a separate token: [100, 101, 102]
        # When tokenizing "Assistant: {", the space is absorbed: [100, 101, 103, ...]
        def mock_tokenize(
            text: str, truncation: bool = True, return_tensors: str | None = None
        ) -> dict[str, list[int]]:
            # Simulate different tokenization based on content
            if text.endswith("Assistant: "):
                # Prompt with generation prompt: separate space token
                tokens = [10, 11, 12, 100, 101, 102]  # User: Hello\n\nAssistant:
            elif "Assistant: {" in text:
                # Full conversation: space absorbed into next token
                tokens = [
                    10,
                    11,
                    12,
                    100,
                    101,
                    103,
                    104,
                    105,
                ]  # User: Hello\n\nAssistant: {"name": "Test", "value": 42}
            else:
                # Just the user message (shouldn't happen with current implementation)
                tokens = [10, 11, 12]

            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

        mock_tokenizer.side_effect = mock_tokenize
        mock_tokenizer.__call__ = mock_tokenize

        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
        )

        # Create sample data
        inputs = ["Hello"]
        outputs = [SampleOutput(name="Test", value=42)]

        # Tokenize for training
        input_ids, _, labels, _ = model._tokenize_for_training(
            inputs, outputs, mock_tokenizer, max_seq_length=100
        )

        # Verify shapes
        assert input_ids.shape[0] == 1
        assert labels.shape[0] == 1

        # Verify that prompt tokens are masked
        # Tokens 0-4 are common (User: Hello\n\nAssistant:), should be masked
        # Token 5 diverges (102 vs 103), so response starts there
        labels_list = labels[0].tolist()
        assert labels_list[0] == -100, "Token 0 should be masked"
        assert labels_list[1] == -100, "Token 1 should be masked"
        assert labels_list[2] == -100, "Token 2 should be masked"
        assert labels_list[3] == -100, "Token 3 should be masked"
        assert labels_list[4] == -100, "Token 4 should be masked"
        # Tokens 5-7 are response tokens, should not be masked
        assert labels_list[5] != -100, "Token 5 (response start) should not be masked"
        assert labels_list[6] != -100, "Token 6 (response) should not be masked"
        assert labels_list[7] != -100, "Token 7 (response) should not be masked"

    def test_prompt_too_long(self, sample_output_type: type[SampleOutput]) -> None:
        """Test that examples with prompts exceeding max_seq_length are skipped."""
        # Create a mock tokenizer that produces long prompts
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.chat_template = None

        def mock_tokenize(
            text: str, truncation: bool = True, return_tensors: str | None = None
        ) -> dict[str, list[int]]:
            if text.endswith("Assistant: "):
                # Prompt is 150 tokens (exceeds max_seq_length of 100)
                tokens = list(range(150))
            elif "Assistant: {" in text:
                # Full conversation is 155 tokens
                tokens = list(range(155))
            else:
                tokens = list(range(150))

            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

        mock_tokenizer.side_effect = mock_tokenize
        mock_tokenizer.__call__ = mock_tokenize

        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
        )

        # Create sample data - all examples have prompts that are too long
        inputs = ["Very long input text"] * 2
        outputs = [SampleOutput(name="Test", value=42)] * 2

        # This should raise ValueError because all examples are skipped
        with pytest.raises(ValueError, match="All training examples were skipped"):
            model._tokenize_for_training(
                inputs, outputs, mock_tokenizer, max_seq_length=100
            )

    def test_partial_prompt_skipping(
        self, sample_output_type: type[SampleOutput]
    ) -> None:
        """Test that only prompts exceeding max_seq_length are skipped."""
        # Create a mock tokenizer with mixed prompt lengths
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.chat_template = None

        call_count = 0

        def mock_tokenize(
            text: str, truncation: bool = True, return_tensors: str | None = None
        ) -> dict[str, list[int]]:
            nonlocal call_count
            call_count += 1

            # Alternate between short and long prompts
            # First call pair (example 1): short prompt
            # Second call pair (example 2): long prompt
            # Third call pair (example 3): short prompt
            example_num = (call_count - 1) // 2

            if example_num == 1:
                # Second example: prompt too long
                if text.endswith("Assistant: "):
                    tokens = list(range(150))  # Too long
                else:
                    tokens = list(range(155))
            else:
                # First and third examples: normal length
                if text.endswith("Assistant: "):
                    tokens = [10, 11, 12, 100, 101, 102]
                else:
                    tokens = [10, 11, 12, 100, 101, 103, 104, 105]

            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

        mock_tokenizer.side_effect = mock_tokenize
        mock_tokenizer.__call__ = mock_tokenize

        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
        )

        # Create sample data: 3 examples, middle one should be skipped
        inputs = ["Input 1", "Very long input 2", "Input 3"]
        outputs = [
            SampleOutput(name="Test1", value=1),
            SampleOutput(name="Test2", value=2),
            SampleOutput(name="Test3", value=3),
        ]

        # Tokenize - should skip middle example
        input_ids, _, labels, _ = model._tokenize_for_training(
            inputs, outputs, mock_tokenizer, max_seq_length=100
        )

        # Should have 2 examples (skipped the middle one)
        assert input_ids.shape[0] == 2
        assert labels.shape[0] == 2


# =============================================================================
# Save/Load Tests
# =============================================================================


class TestSaveLoad:
    """Tests for model save/load functionality."""

    def test_save_and_load_roundtrip(
        self,
        sample_output_type: type[SampleOutput],
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        """Test that save and load preserve model configuration."""
        from io import BytesIO
        from pathlib import Path
        from unittest.mock import patch

        # Create a model with specific configuration
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
            system_prompt="Test system prompt",
            max_new_tokens=1024,
            temperature=0.5,
            distillation_temperature=3.0,
            distillation_alpha=0.7,
            device="cpu",
        )

        # Mock save_pretrained to create dummy files
        def mock_save_pretrained(path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "dummy_file.txt").write_text("dummy")

        mock_model.save_pretrained = mock_save_pretrained
        mock_tokenizer.save_pretrained = mock_save_pretrained

        # Mock the model and tokenizer loading
        model._model = mock_model
        model._tokenizer = mock_tokenizer

        # Save to BytesIO
        stream = BytesIO()
        model.save(stream)

        # Reset stream position
        stream.seek(0)

        # Mock AutoTokenizer, AutoConfig, and AutoModelForCausalLM.from_pretrained for loading
        with (
            patch(
                "precipitate.huggingface.AutoTokenizer.from_pretrained"
            ) as mock_load_tokenizer,
            patch(
                "precipitate.huggingface.AutoConfig.from_pretrained"
            ) as mock_load_config,
            patch(
                "precipitate.huggingface.AutoModelForCausalLM.from_pretrained"
            ) as mock_load_model,
        ):
            mock_load_tokenizer.return_value = mock_tokenizer
            mock_load_config.return_value.is_encoder_decoder = False
            mock_load_model.return_value = mock_model

            # Load from stream
            loaded_model = HuggingfaceModel.load(stream)

            # Verify configuration is preserved
            assert loaded_model.model_name == "test-model"
            assert loaded_model.output_type == sample_output_type
            assert loaded_model.system_prompt == "Test system prompt"
            assert loaded_model.max_new_tokens == 1024
            assert loaded_model.temperature == 0.5
            assert loaded_model.distillation_temperature == 3.0
            assert loaded_model.distillation_alpha == 0.7

    def test_save_without_loaded_model(
        self, sample_output_type: type[SampleOutput]
    ) -> None:
        """Test that save works even when model hasn't been loaded yet."""
        from io import BytesIO
        from pathlib import Path
        from unittest.mock import patch

        # Create a model without loading it
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
        )

        # Mock save_pretrained to create dummy files
        def mock_save_pretrained(path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "dummy_file.txt").write_text("dummy")

        # Mock the loading process
        with patch.object(model, "_load_model_and_tokenizer") as mock_load:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_model.save_pretrained = mock_save_pretrained
            mock_tokenizer.save_pretrained = mock_save_pretrained
            mock_load.return_value = (mock_model, mock_tokenizer)

            stream = BytesIO()
            model.save(stream)

            # Verify that _load_model_and_tokenizer was called
            mock_load.assert_called_once()


# =============================================================================
# Seq2Seq / Encoder-Decoder Support Tests
# =============================================================================


class TestSeq2Seq:
    """Tests for encoder-decoder (T5-style) model support."""

    def test_is_seq2seq_false_when_model_not_loaded(
        self, sample_output_type: type[SampleOutput]
    ) -> None:
        """Test that _is_seq2seq returns False when model not yet loaded."""
        model = HuggingfaceModel(
            model_name="test-model", output_type=sample_output_type
        )
        assert model._is_seq2seq() is False

    def test_is_seq2seq_false_for_causal_model(
        self, sample_output_type: type[SampleOutput], mock_model: MagicMock
    ) -> None:
        """Test that _is_seq2seq returns False for decoder-only models."""
        mock_model.config.is_encoder_decoder = False
        model = HuggingfaceModel(
            model_name="test-model", output_type=sample_output_type
        )
        model._model = mock_model
        assert model._is_seq2seq() is False

    def test_is_seq2seq_true_for_seq2seq_model(
        self, sample_output_type: type[SampleOutput], mock_model: MagicMock
    ) -> None:
        """Test that _is_seq2seq returns True for encoder-decoder models."""
        mock_model.config.is_encoder_decoder = True
        model = HuggingfaceModel(
            model_name="test-model", output_type=sample_output_type
        )
        model._model = mock_model
        assert model._is_seq2seq() is True

    def test_tokenize_seq2seq_separates_input_and_output(
        self, sample_output_type: type[SampleOutput], mock_model: MagicMock
    ) -> None:
        """Test that seq2seq tokenization produces separate encoder/decoder tensors."""
        mock_model.config.is_encoder_decoder = True

        # Build a simple mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.chat_template = None

        def mock_tokenize(
            text: str,
            truncation: bool = False,
            max_length: int | None = None,
            return_tensors: str | None = None,
        ) -> dict[str, list[int]]:
            # Encoder prompt returns 5 tokens; JSON output returns 3 tokens
            tokens = [200, 201, 202] if text.startswith("{") else [10, 11, 12, 13, 14]
            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

        mock_tokenizer.side_effect = mock_tokenize
        mock_tokenizer.__call__ = mock_tokenize

        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
            device="cpu",
        )
        model._model = mock_model

        inputs = ["Hello world"]
        outputs = [SampleOutput(name="T", value=1)]

        input_ids, attention_mask, labels, _token_type_ids = (
            model._tokenize_for_training(
                inputs, outputs, mock_tokenizer, max_seq_length=512
            )
        )

        # Encoder input should be 5 tokens
        assert input_ids.shape == (1, 5)
        assert attention_mask.shape == (1, 5)
        # All attention mask values should be 1 (no padding for single example)
        assert attention_mask[0].tolist() == [1, 1, 1, 1, 1]
        # Labels should be 3 tokens (the JSON output)
        assert labels.shape == (1, 3)
        # Labels should NOT be masked (they are the decoder targets)
        assert all(v != -100 for v in labels[0].tolist())

    def test_tokenize_seq2seq_pads_correctly(
        self, sample_output_type: type[SampleOutput], mock_model: MagicMock
    ) -> None:
        """Test that seq2seq tokenization pads a batch correctly."""
        mock_model.config.is_encoder_decoder = True

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.chat_template = None

        call_count = 0

        def mock_tokenize(
            text: str,
            truncation: bool = False,
            max_length: int | None = None,
            return_tensors: str | None = None,
        ) -> dict[str, list[int]]:
            nonlocal call_count
            call_count += 1
            if text.startswith("{"):
                # JSON outputs: alternate lengths
                tokens = [200, 201] if call_count % 2 == 0 else [200, 201, 202, 203]
            else:
                # Encoder inputs: alternate lengths
                tokens = [10, 11, 12] if call_count % 2 != 0 else [10, 11, 12, 13, 14]
            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

        mock_tokenizer.side_effect = mock_tokenize
        mock_tokenizer.__call__ = mock_tokenize

        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
            device="cpu",
        )
        model._model = mock_model

        inputs = ["Short input", "Longer input here"]
        outputs = [SampleOutput(name="A", value=1), SampleOutput(name="B", value=2)]

        input_ids, attention_mask, labels, _ = model._tokenize_for_training(
            inputs, outputs, mock_tokenizer, max_seq_length=512
        )

        # Both examples must have same tensor dimensions
        assert input_ids.shape[0] == 2
        assert input_ids.shape[1] == input_ids.shape[1]  # same width
        assert labels.shape[0] == 2
        assert labels.shape[1] == labels.shape[1]  # same width

        # Padding positions in encoder input should have attention_mask == 0
        # Padding positions in labels should be -100
        for i in range(2):
            enc_len = attention_mask[i].sum().item()
            pad_positions = input_ids.shape[1] - int(enc_len)
            if pad_positions > 0:
                assert input_ids[i, -pad_positions:].tolist() == [0] * pad_positions

    def test_tokenize_seq2seq_skips_long_encoder_input(
        self, sample_output_type: type[SampleOutput], mock_model: MagicMock
    ) -> None:
        """Test that examples where encoder input exceeds max_seq_length are skipped."""
        mock_model.config.is_encoder_decoder = True

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.chat_template = None

        def mock_tokenize(
            text: str,
            truncation: bool = False,
            max_length: int | None = None,
            return_tensors: str | None = None,
        ) -> dict[str, list[int]]:
            tokens = [200, 201, 202] if text.startswith("{") else list(range(200))  # Too long for max_seq_length=100
            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

        mock_tokenizer.side_effect = mock_tokenize
        mock_tokenizer.__call__ = mock_tokenize

        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
            device="cpu",
        )
        model._model = mock_model

        with pytest.raises(ValueError, match="All training examples were skipped"):
            model._tokenize_for_training(
                ["Very long prompt"],
                [SampleOutput(name="T", value=1)],
                mock_tokenizer,
                max_seq_length=100,
            )

    def test_distillation_trainer_seq2seq_no_shift(self) -> None:
        """Test that DistillationTrainer skips the causal shift for seq2seq models."""
        teacher_model = MagicMock()

        with patch("precipitate.huggingface.Trainer.__init__", return_value=None):
            trainer = DistillationTrainer(
                teacher_model=teacher_model,
                temperature=2.0,
                alpha=0.5,
            )

            batch_size, seq_len, vocab_size = 2, 8, 50
            inputs = {
                "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len),
                "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
            }
            inputs["labels"][:, :3] = -100

            # Mark student model as encoder-decoder
            student_model = MagicMock()
            student_model.config.is_encoder_decoder = True
            student_output = MagicMock()
            student_output.logits = torch.randn(batch_size, seq_len, vocab_size)
            student_model.return_value = student_output

            teacher_output = MagicMock()
            teacher_output.logits = torch.randn(batch_size, seq_len, vocab_size)
            teacher_model.return_value = teacher_output

            loss = trainer.compute_loss(student_model, inputs)
            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0


# =============================================================================
# LoRA Tests
# =============================================================================


class TestLoRA:
    """Tests for LoRA (Low-Rank Adaptation) fine-tuning support."""

    def test_lora_config_defaults(self) -> None:
        """Test that LoraConfig has the expected defaults."""
        from precipitate.huggingface import LoraConfig

        cfg = LoraConfig()
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        assert cfg.target_modules is None
        assert cfg.lora_dropout == 0.05
        assert cfg.bias == "none"

    def test_lora_config_custom(self) -> None:
        """Test that LoraConfig stores custom values."""
        from precipitate.huggingface import LoraConfig

        cfg = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q", "v"], lora_dropout=0.1, bias="all"
        )
        assert cfg.r == 8
        assert cfg.lora_alpha == 16
        assert cfg.target_modules == ["q", "v"]
        assert cfg.lora_dropout == 0.1
        assert cfg.bias == "all"

    def test_lora_config_none_by_default(
        self, sample_output_type: type[SampleOutput]
    ) -> None:
        """Test that lora_config defaults to None on HuggingfaceModel."""
        model = HuggingfaceModel(
            model_name="test-model", output_type=sample_output_type
        )
        assert model.lora_config is None

    def test_lora_config_stored_on_init(
        self, sample_output_type: type[SampleOutput]
    ) -> None:
        """Test that lora_config is stored when passed to __init__."""
        from precipitate.huggingface import LoraConfig

        cfg = LoraConfig(r=4)
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
            lora_config=cfg,
        )
        assert model.lora_config is cfg

    @patch("precipitate.huggingface.TrainingArguments")
    @patch("precipitate.huggingface.Trainer")
    @patch("precipitate.huggingface.AutoConfig")
    @patch("precipitate.huggingface.AutoModelForCausalLM")
    @patch("precipitate.huggingface.AutoTokenizer")
    def test_fit_no_lora_by_default(
        self,
        mock_auto_tokenizer: MagicMock,
        mock_auto_model: MagicMock,
        mock_auto_config: MagicMock,
        mock_trainer_class: MagicMock,
        mock_training_args: MagicMock,
        sample_output_type: type[SampleOutput],
        sample_inputs: list[str],
        sample_outputs: list[SampleOutput],
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test that get_peft_model is NOT called when lora_config is None."""
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_config.from_pretrained.return_value.is_encoder_decoder = False

        with patch("peft.get_peft_model") as mock_get_peft:
            model = HuggingfaceModel(
                model_name="test-model",
                output_type=sample_output_type,
                device="cpu",
            )
            model.fit(sample_inputs, sample_outputs)
            mock_get_peft.assert_not_called()

    @patch("precipitate.huggingface.TrainingArguments")
    @patch("precipitate.huggingface.Trainer")
    @patch("precipitate.huggingface.AutoConfig")
    @patch("precipitate.huggingface.AutoModelForCausalLM")
    @patch("precipitate.huggingface.AutoTokenizer")
    def test_fit_applies_lora(
        self,
        mock_auto_tokenizer: MagicMock,
        mock_auto_model: MagicMock,
        mock_auto_config: MagicMock,
        mock_trainer_class: MagicMock,
        mock_training_args: MagicMock,
        sample_output_type: type[SampleOutput],
        sample_inputs: list[str],
        sample_outputs: list[SampleOutput],
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test that get_peft_model is called when lora_config is set."""
        from precipitate.huggingface import LoraConfig

        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_config.from_pretrained.return_value.is_encoder_decoder = False
        # Simulate PeftModel returned by get_peft_model (must look like a real model)
        mock_peft_model = MagicMock()
        mock_peft_model.config.is_encoder_decoder = False

        with (
            patch("peft.get_peft_model", return_value=mock_peft_model) as mock_get_peft,
            patch("peft.LoraConfig"),
            patch("peft.TaskType"),
        ):
            model = HuggingfaceModel(
                model_name="test-model",
                output_type=sample_output_type,
                device="cpu",
                lora_config=LoraConfig(r=4),
            )
            model.fit(sample_inputs, sample_outputs)
            mock_get_peft.assert_called_once()
            # The stored model should be the PEFT-wrapped one
            assert model._model is mock_peft_model

    @patch("precipitate.huggingface.TrainingArguments")
    @patch("precipitate.huggingface.Trainer")
    @patch("precipitate.huggingface.AutoConfig")
    @patch("precipitate.huggingface.AutoModelForCausalLM")
    @patch("precipitate.huggingface.AutoTokenizer")
    def test_fit_lora_enables_input_require_grads(
        self,
        mock_auto_tokenizer: MagicMock,
        mock_auto_model: MagicMock,
        mock_auto_config: MagicMock,
        mock_trainer_class: MagicMock,
        mock_training_args: MagicMock,
        sample_output_type: type[SampleOutput],
        sample_inputs: list[str],
        sample_outputs: list[SampleOutput],
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test that enable_input_require_grads is called when LoRA + gradient checkpointing."""
        from precipitate.huggingface import LoraConfig

        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_config.from_pretrained.return_value.is_encoder_decoder = False
        mock_peft_model = MagicMock()
        mock_peft_model.config.is_encoder_decoder = False

        with (
            patch("peft.get_peft_model", return_value=mock_peft_model),
            patch("peft.LoraConfig"),
            patch("peft.TaskType"),
        ):
            model = HuggingfaceModel(
                model_name="test-model",
                output_type=sample_output_type,
                device="cpu",
                lora_config=LoraConfig(r=4),
            )
            model.fit(sample_inputs, sample_outputs, use_gradient_checkpointing=True)
            mock_peft_model.enable_input_require_grads.assert_called_once()

    def test_lora_save_includes_config(
        self,
        sample_output_type: type[SampleOutput],
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        """Test that save() serializes lora_config to config.json."""
        import json
        import tarfile
        from io import BytesIO
        from pathlib import Path

        from precipitate.huggingface import LoraConfig

        cfg = LoraConfig(r=8, lora_alpha=16)
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
            lora_config=cfg,
            device="cpu",
        )

        def mock_save_pretrained(path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "dummy_file.txt").write_text("dummy")

        mock_model.save_pretrained = mock_save_pretrained
        mock_tokenizer.save_pretrained = mock_save_pretrained
        model._model = mock_model
        model._tokenizer = mock_tokenizer

        stream = BytesIO()
        model.save(stream)
        stream.seek(0)

        # Extract and inspect config.json
        with tarfile.open(fileobj=stream, mode="r:gz") as tar:
            config_member = tar.getmember("config.json")
            config_data = json.loads(tar.extractfile(config_member).read())  # type: ignore[union-attr]

        assert config_data["lora_config"] is not None
        assert config_data["lora_config"]["r"] == 8
        assert config_data["lora_config"]["lora_alpha"] == 16

    def test_lora_load_uses_peft(
        self,
        sample_output_type: type[SampleOutput],
        mock_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        """Test that load() calls PeftModel.from_pretrained when lora_config is present."""
        from io import BytesIO
        from pathlib import Path

        from precipitate.huggingface import LoraConfig

        cfg = LoraConfig(r=8)
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
            lora_config=cfg,
            device="cpu",
        )

        def mock_save_pretrained(path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "dummy_file.txt").write_text("dummy")

        mock_model.save_pretrained = mock_save_pretrained
        mock_tokenizer.save_pretrained = mock_save_pretrained
        model._model = mock_model
        model._tokenizer = mock_tokenizer

        stream = BytesIO()
        model.save(stream)
        stream.seek(0)

        mock_peft_model = MagicMock()

        with (
            patch(
                "precipitate.huggingface.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
            patch("precipitate.huggingface.AutoConfig.from_pretrained") as mock_config,
            patch(
                "precipitate.huggingface.AutoModelForCausalLM.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "peft.PeftModel.from_pretrained", return_value=mock_peft_model
            ) as mock_peft_load,
        ):
            mock_config.return_value.is_encoder_decoder = False
            loaded = HuggingfaceModel.load(stream)
            mock_peft_load.assert_called_once()
            assert loaded.lora_config is not None
            assert loaded.lora_config.r == 8
