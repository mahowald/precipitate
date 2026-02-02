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
            learning_rate=1e-4,
            num_epochs=5,
            batch_size=16,
            device="cpu",
        )

        assert model.model_name == "test-model"
        assert model.output_type == sample_output_type
        assert model.system_prompt == "Test prompt"
        assert model.learning_rate == 1e-4
        assert model.num_epochs == 5
        assert model.batch_size == 16
        assert model._device_name == "cpu"

    def test_init_defaults(self, sample_output_type: type[SampleOutput]) -> None:
        """Test that default values are set correctly."""
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
        )

        assert model.system_prompt is None
        assert model.learning_rate == 5e-5
        assert model.num_epochs == 3
        assert model.batch_size == 4
        assert model.gradient_accumulation_steps == 1
        assert model.warmup_ratio == 0.1
        assert model.weight_decay == 0.01
        assert model.max_seq_length == 2048
        assert model.max_new_tokens == 512
        assert model.temperature == 0.7
        assert model.distillation_temperature == 2.0
        assert model.distillation_alpha == 0.5
        assert model.output_dir == "./hf_model_output"
        # Memory optimization defaults
        assert model.use_gradient_checkpointing is True
        assert model.use_8bit_optimizer is False
        assert model.optimizer_type == "adamw_torch_fused"
        assert model.max_grad_norm == 1.0
        assert model.per_device_eval_batch_size == 8  # batch_size * 2

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


class TestExtractAndParseJson:
    """Tests for _extract_and_parse_json."""

    def test_clean_json(self, sample_output_type: type[SampleOutput]) -> None:
        """Test parsing clean JSON."""
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
        )

        result = model._extract_and_parse_json('{"name": "Alice", "value": 42}')
        assert result.name == "Alice"
        assert result.value == 42

    def test_json_with_prefix(self, sample_output_type: type[SampleOutput]) -> None:
        """Test extracting JSON with text before it."""
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
        )

        result = model._extract_and_parse_json(
            'Here is the result: {"name": "Bob", "value": 17}'
        )
        assert result.name == "Bob"
        assert result.value == 17

    def test_json_with_suffix(self, sample_output_type: type[SampleOutput]) -> None:
        """Test extracting JSON with text after it."""
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
        )

        result = model._extract_and_parse_json(
            '{"name": "Charlie", "value": 99} Hope that helps!'
        )
        assert result.name == "Charlie"
        assert result.value == 99

    def test_json_extraction_failure(
        self, sample_output_type: type[SampleOutput]
    ) -> None:
        """Test ValueError is raised for invalid JSON."""
        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
        )

        with pytest.raises(ValueError, match="Could not parse output"):
            model._extract_and_parse_json("This has no JSON at all")


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
    @patch("precipitate.huggingface.AutoModelForCausalLM")
    @patch("precipitate.huggingface.AutoTokenizer")
    def test_fit_calls_trainer(
        self,
        mock_auto_tokenizer: MagicMock,
        mock_auto_model: MagicMock,
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

    @patch("precipitate.huggingface.AutoModelForCausalLM")
    @patch("precipitate.huggingface.AutoTokenizer")
    def test_predict_returns_parsed_output(
        self,
        mock_auto_tokenizer: MagicMock,
        mock_auto_model: MagicMock,
        sample_output_type: type[SampleOutput],
        mock_tokenizer: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test that predict returns parsed Pydantic models."""
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model

        # Override decode to return valid JSON
        mock_tokenizer.decode = lambda x, **kwargs: '{"name": "Test", "value": 123}'

        model = HuggingfaceModel(
            model_name="test-model",
            output_type=sample_output_type,
            device="cpu",
        )

        results = model.predict(["Test input"])

        assert len(results) == 1
        assert isinstance(results[0], SampleOutput)
        assert results[0].name == "Test"
        assert results[0].value == 123


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
