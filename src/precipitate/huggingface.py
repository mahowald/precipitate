"""HuggingFace Transformers implementation of the Model protocol."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T", bound=BaseModel)


class StructuredOutputDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for structured output fine-tuning."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.token_type_ids = token_type_ids

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "token_type_ids": self.token_type_ids[idx],
        }


class DistillationDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset for distillation training (without pre-computed teacher logits)."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.token_type_ids = token_type_ids

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "token_type_ids": self.token_type_ids[idx],
        }


class DistillationTrainer(Trainer):  # type: ignore[misc]
    """Custom Trainer that implements knowledge distillation loss."""

    def __init__(
        self,
        teacher_model: PreTrainedModel,
        temperature: float = 2.0,
        alpha: float = 0.5,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha  # Weight for hard labels vs soft labels

    def compute_loss(  # type: ignore[override]
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute distillation loss combining KL divergence and cross-entropy."""
        labels = inputs.pop("labels")

        # Student forward pass
        outputs = model(**inputs)
        student_logits = outputs.logits

        # Teacher forward pass (frozen)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # Shift for causal LM: predict next token
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Hard label loss (cross-entropy on non-masked positions)
        hard_loss = F.cross_entropy(
            shift_student_logits.view(-1, shift_student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Soft label loss (KL divergence)
        # Create mask for non-ignored positions
        mask = (shift_labels != -100).float()

        # Temperature-scaled soft probabilities
        teacher_probs = F.softmax(shift_teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(
            shift_student_logits / self.temperature, dim=-1
        )

        # KL divergence: sum over vocab dimension
        kl_div = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1)

        # Apply mask and average
        if mask.sum() > 0:
            soft_loss = (kl_div * mask).sum() / mask.sum()
        else:
            soft_loss = torch.tensor(0.0, device=kl_div.device)
        soft_loss = soft_loss * (self.temperature**2)  # Scale by T^2

        # Combined loss
        loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return (loss, outputs) if return_outputs else loss


class HuggingfaceModel(Generic[T]):
    """HuggingFace Transformers implementation of the Model protocol."""

    def __init__(
        self,
        model_name: str,
        output_type: type[T],
        *,
        # Prompt configuration
        system_prompt: str | None = None,
        # Training hyperparameters
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_seq_length: int = 2048,
        # Generation parameters
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        # Distillation parameters
        distillation_temperature: float = 2.0,
        distillation_alpha: float = 0.5,
        # Memory optimization parameters
        use_gradient_checkpointing: bool = True,
        use_8bit_optimizer: bool = False,
        optimizer_type: str = "adamw_torch_fused",
        max_grad_norm: float = 1.0,
        per_device_eval_batch_size: int | None = None,
        # Device configuration
        device: str | None = None,
        # Output directory for checkpoints
        output_dir: str = "./hf_model_output",
    ) -> None:
        self.model_name = model_name
        self.output_type = output_type
        self.system_prompt = system_prompt

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_seq_length = max_seq_length

        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Distillation parameters
        self.distillation_temperature = distillation_temperature
        self.distillation_alpha = distillation_alpha

        # Memory optimization parameters
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_8bit_optimizer = use_8bit_optimizer
        self.optimizer_type = optimizer_type
        self.max_grad_norm = max_grad_norm
        self.per_device_eval_batch_size = (
            per_device_eval_batch_size if per_device_eval_batch_size is not None else batch_size * 2
        )

        # Device configuration
        self._device_name = device
        self._device: torch.device | None = None

        # Output directory
        self.output_dir = output_dir

        # Lazily loaded model and tokenizer
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: PreTrainedModel | None = None

    def _get_device(self) -> torch.device:
        """Auto-detect best available device."""
        if self._device is not None:
            return self._device

        if self._device_name is not None:
            self._device = torch.device(self._device_name)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        return self._device

    def _load_model_and_tokenizer(
        self,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load model and tokenizer if not already loaded."""
        if self._model is None or self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            device = self._get_device()

            # Use bfloat16 for GPU, float32 for CPU
            dtype = torch.bfloat16 if device.type != "cpu" else torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=dtype,
            )
            model.to(device)  # type: ignore[arg-type]
            self._model = cast(PreTrainedModel, model)

            # Enable gradient checkpointing for memory efficiency
            if self.use_gradient_checkpointing:
                self._model.gradient_checkpointing_enable()

            # Ensure pad token is set
            tok = cast(PreTrainedTokenizerBase, self._tokenizer)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
                self._model.config.pad_token_id = tok.pad_token_id

        return self._model, cast(PreTrainedTokenizerBase, self._tokenizer)

    def _build_chat_messages(
        self, input_text: str, output_json: str | None = None
    ) -> list[dict[str, str]]:
        """Build chat messages for the model."""
        messages: list[dict[str, str]] = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({"role": "user", "content": input_text})

        if output_json is not None:
            messages.append({"role": "assistant", "content": output_json})

        return messages

    def _apply_chat_template(
        self,
        tokenizer: PreTrainedTokenizerBase,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        """Apply chat template with fallback for models without templates.

        Args:
            tokenizer: The tokenizer to use
            messages: List of message dicts with 'role' and 'content'
            add_generation_prompt: If True, add assistant generation prompt

        Returns:
            Formatted text string
        """
        # Try to use the model's chat template if available
        if tokenizer.chat_template is not None:
            return cast(
                str,
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                ),
            )

        # Fallback: construct prompt manually
        parts = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                parts.append(f"System: {content}\n\n")
            elif role == "user":
                parts.append(f"User: {content}\n\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")

        # Add generation prompt if needed
        if add_generation_prompt:
            parts.append("Assistant: ")

        return "".join(parts)

    def _tokenize_for_training(
        self,
        inputs: Sequence[str],
        outputs: Sequence[T],
        tokenizer: PreTrainedTokenizerBase,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize inputs and outputs for training with masked labels."""
        all_input_ids: list[list[int]] = []
        all_labels: list[list[int]] = []

        for input_text, output in zip(inputs, outputs, strict=True):
            output_json = output.model_dump_json()

            # Build full conversation
            full_messages = self._build_chat_messages(input_text, output_json)

            # Tokenize full conversation
            full_text = self._apply_chat_template(
                tokenizer, full_messages, add_generation_prompt=False
            )
            full_encoding = tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors=None,
            )

            # Tokenize without the assistant response to find the boundary
            prompt_messages = self._build_chat_messages(input_text, None)
            prompt_text = self._apply_chat_template(
                tokenizer, prompt_messages, add_generation_prompt=True
            )
            prompt_encoding = tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors=None,
            )

            # Create labels: -100 for prompt tokens, actual ids for output tokens
            input_ids: list[int] = full_encoding["input_ids"]  # type: ignore[assignment]
            prompt_length = len(prompt_encoding["input_ids"])  # type: ignore[arg-type]

            labels: list[int] = [-100] * prompt_length + input_ids[prompt_length:]

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # Pad to same length
        max_len = max(len(ids) for ids in all_input_ids)
        pad_token_id = tokenizer.pad_token_id or 0

        padded_input_ids: list[list[int]] = []
        padded_attention_mask: list[list[int]] = []
        padded_labels: list[list[int]] = []
        padded_token_type_ids: list[list[int]] = []

        for ids, lbls in zip(all_input_ids, all_labels, strict=True):
            padding_length = max_len - len(ids)
            padded_input_ids.append(ids + [pad_token_id] * padding_length)  # type: ignore[arg-type]
            padded_attention_mask.append([1] * len(ids) + [0] * padding_length)
            padded_labels.append(lbls + [-100] * padding_length)
            # Generate token_type_ids as all 0s (for text-only fine-tuning)
            padded_token_type_ids.append([0] * max_len)

        return (
            torch.tensor(padded_input_ids),
            torch.tensor(padded_attention_mask),
            torch.tensor(padded_labels),
            torch.tensor(padded_token_type_ids),
        )

    def fit(self, inputs: list[str], outputs: list[T]) -> None:
        """Fine-tune the model on input-output pairs.

        Training uses masked loss - only the output JSON tokens contribute to the loss.
        """
        if len(inputs) != len(outputs):
            msg = "inputs and outputs must have same length"
            raise ValueError(msg)

        model, tokenizer = self._load_model_and_tokenizer()

        # Tokenize with masked labels
        input_ids, attention_mask, labels, token_type_ids = self._tokenize_for_training(
            inputs, outputs, tokenizer
        )

        # Create dataset
        dataset = StructuredOutputDataset(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            token_type_ids=token_type_ids,
        )

        # Calculate warmup_steps from warmup_ratio with gradient accumulation
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        num_update_steps_per_epoch = max(
            1, (len(dataset) + effective_batch_size - 1) // effective_batch_size
        )
        total_steps = num_update_steps_per_epoch * self.num_epochs
        warmup_steps = int(self.warmup_ratio * total_steps)

        # Determine optimizer for memory efficiency
        optim = "paged_adamw_8bit" if self.use_8bit_optimizer else self.optimizer_type

        # Configure training
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            logging_steps=10,
            save_strategy="epoch",
            bf16=self._get_device().type == "cuda",
            dataloader_pin_memory=False,  # For MPS compatibility
            optim=optim,
            gradient_checkpointing=self.use_gradient_checkpointing,
        )

        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

    def predict(self, inputs: list[str]) -> list[T]:
        """Generate predictions for the given inputs.

        Returns a list of Pydantic model instances parsed from the generated JSON.
        """
        model, tokenizer = self._load_model_and_tokenizer()
        model.eval()

        results: list[T] = []

        for input_text in inputs:
            # Build prompt messages
            messages = self._build_chat_messages(input_text, None)
            prompt = self._apply_chat_template(
                tokenizer, messages, add_generation_prompt=True
            )

            # Tokenize
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length,
            )
            device = self._get_device()
            input_ids_tensor: torch.Tensor = encoded["input_ids"].to(device)  # type: ignore[union-attr]
            attention_mask_tensor: torch.Tensor = encoded["attention_mask"].to(device)  # type: ignore[union-attr]

            # Generate
            with torch.no_grad():
                generated = model.generate(  # type: ignore[operator]
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask_tensor,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=self.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode only the new tokens
            input_length = input_ids_tensor.shape[1]
            new_tokens = generated[0, input_length:]
            output_text = str(tokenizer.decode(new_tokens, skip_special_tokens=True))

            # Parse as Pydantic model
            try:
                parsed = self.output_type.model_validate_json(output_text)
            except Exception:
                # Try to extract JSON from the output
                parsed = self._extract_and_parse_json(output_text)

            results.append(parsed)

        return results

    def _extract_and_parse_json(self, text: str) -> T:
        """Attempt to extract and parse JSON from text that may have extra content."""
        # Try to find JSON object in the text
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            try:
                return self.output_type.model_validate_json(json_match.group())
            except Exception:
                pass

        msg = f"Could not parse output as {self.output_type.__name__}: {text}"
        raise ValueError(msg)

    def distill(
        self, student: HuggingfaceModel[T], inputs: list[str], outputs: list[T]
    ) -> None:
        """Distill knowledge from this model (teacher) to the student model.

        Uses KL divergence between teacher and student logits, combined with
        hard label cross-entropy loss. Teacher logits are computed on-the-fly
        during training to minimize memory usage.
        """
        if len(inputs) != len(outputs):
            msg = "inputs and outputs must have same length"
            raise ValueError(msg)

        teacher_model, _ = self._load_model_and_tokenizer()
        student_model, student_tokenizer = student._load_model_and_tokenizer()

        teacher_model.eval()  # Teacher is frozen

        # Tokenize with student tokenizer
        input_ids, attention_mask, labels, token_type_ids = (
            student._tokenize_for_training(inputs, outputs, student_tokenizer)
        )

        # Create dataset
        dataset = DistillationDataset(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            token_type_ids=token_type_ids,
        )

        # Calculate warmup_steps from warmup_ratio with gradient accumulation
        effective_batch_size = student.batch_size * student.gradient_accumulation_steps
        num_update_steps_per_epoch = max(
            1, (len(dataset) + effective_batch_size - 1) // effective_batch_size
        )
        total_steps = num_update_steps_per_epoch * student.num_epochs
        warmup_steps = int(student.warmup_ratio * total_steps)

        # Determine optimizer for memory efficiency
        optim = "paged_adamw_8bit" if student.use_8bit_optimizer else student.optimizer_type

        # Configure training
        training_args = TrainingArguments(
            output_dir=student.output_dir,
            num_train_epochs=student.num_epochs,
            per_device_train_batch_size=student.batch_size,
            per_device_eval_batch_size=student.per_device_eval_batch_size,
            gradient_accumulation_steps=student.gradient_accumulation_steps,
            learning_rate=student.learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=student.weight_decay,
            max_grad_norm=student.max_grad_norm,
            logging_steps=10,
            save_strategy="epoch",
            bf16=student._get_device().type == "cuda",
            dataloader_pin_memory=False,  # For MPS compatibility
            optim=optim,
            gradient_checkpointing=student.use_gradient_checkpointing,
            remove_unused_columns=False,
        )

        # Train with distillation
        trainer = DistillationTrainer(
            teacher_model=teacher_model,
            temperature=student.distillation_temperature,
            alpha=student.distillation_alpha,
            model=student_model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()
