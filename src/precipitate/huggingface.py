"""HuggingFace Transformers implementation of the Model protocol."""

from __future__ import annotations

import importlib
import json
import re
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Generic, TypeVar, cast

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
        # Generation parameters
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        # Distillation parameters
        distillation_temperature: float = 2.0,
        distillation_alpha: float = 0.5,
        # Device configuration
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.output_type = output_type
        self.system_prompt = system_prompt

        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Distillation parameters
        self.distillation_temperature = distillation_temperature
        self.distillation_alpha = distillation_alpha

        # Device configuration
        self._device_name = device
        self._device: torch.device | None = None

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
        max_seq_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize inputs and outputs for training with masked labels.

        Only trains on the assistant response tokens - prompt tokens are masked.
        Skips examples where the prompt alone exceeds max_seq_length.
        """
        all_input_ids: list[list[int]] = []
        all_labels: list[list[int]] = []
        skipped_count = 0

        for input_text, output in zip(inputs, outputs, strict=True):
            output_json = output.model_dump_json()

            # Build full conversation
            full_messages = self._build_chat_messages(input_text, output_json)

            # Tokenize full conversation WITHOUT truncation
            full_text = self._apply_chat_template(
                tokenizer, full_messages, add_generation_prompt=False
            )
            full_encoding = tokenizer(
                full_text,
                truncation=False,
                return_tensors=None,
            )

            # Tokenize prompt WITHOUT truncation to find the boundary
            prompt_messages = self._build_chat_messages(input_text, None)
            prompt_text = self._apply_chat_template(
                tokenizer, prompt_messages, add_generation_prompt=True
            )
            prompt_encoding = tokenizer(
                prompt_text,
                truncation=False,
                return_tensors=None,
            )

            # Find where token sequences diverge
            full_tokens: list[int] = full_encoding["input_ids"]  # type: ignore[assignment]
            prompt_tokens: list[int] = prompt_encoding["input_ids"]  # type: ignore[assignment]

            divergence_idx = 0
            for i in range(min(len(full_tokens), len(prompt_tokens))):
                if full_tokens[i] != prompt_tokens[i]:
                    break
                divergence_idx = i + 1

            # Skip examples where prompt alone is too long
            if divergence_idx >= max_seq_length:
                skipped_count += 1
                print(
                    f"WARNING: Skipping example {skipped_count}: "
                    f"prompt length ({divergence_idx}) exceeds "
                    f"max_seq_length ({max_seq_length})"
                )
                continue

            # Apply truncation to full conversation
            if len(full_tokens) > max_seq_length:
                input_ids = full_tokens[:max_seq_length]
            else:
                input_ids = full_tokens

            # Create labels: mask prompt, keep response (up to max_length)
            labels = ([-100] * divergence_idx + input_ids[divergence_idx:])[
                :max_seq_length
            ]

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        if skipped_count > 0:
            print(
                f"WARNING: Skipped {skipped_count} example(s) due to prompt length "
                f"exceeding max_seq_length. Consider increasing max_seq_length or "
                f"using shorter prompts."
            )

        # Check if all examples were skipped
        if not all_input_ids:
            msg = (
                "All training examples were skipped because prompts exceed "
                f"max_seq_length ({max_seq_length}). "
                "Increase max_seq_length or use shorter prompts."
            )
            raise ValueError(msg)

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

    def fit(
        self,
        inputs: list[str],
        outputs: list[T],
        *,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_seq_length: int = 2048,
        use_gradient_checkpointing: bool = True,
        use_8bit_optimizer: bool = False,
        optimizer_type: str = "adamw_torch_fused",
        max_grad_norm: float = 1.0,
        per_device_eval_batch_size: int | None = None,
        output_dir: str = "./hf_model_output",
    ) -> None:
        """Fine-tune the model on input-output pairs.

        Training uses masked loss - only the output JSON tokens contribute to the loss.

        Args:
            inputs: List of input strings
            outputs: List of Pydantic model instances representing expected outputs
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            batch_size: Training batch size
            gradient_accumulation_steps: Number of steps to accumulate gradients
            warmup_ratio: Ratio of total steps for learning rate warmup
            weight_decay: Weight decay for regularization
            max_seq_length: Maximum sequence length for tokenization
            use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency
            use_8bit_optimizer: Use 8-bit Adam optimizer (requires bitsandbytes)
            optimizer_type: Optimizer type to use
            max_grad_norm: Maximum gradient norm for clipping
            per_device_eval_batch_size: Eval batch size (defaults to batch_size * 2)
            output_dir: Directory for saving checkpoints
        """
        if len(inputs) != len(outputs):
            msg = "inputs and outputs must have same length"
            raise ValueError(msg)

        model, tokenizer = self._load_model_and_tokenizer()

        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Tokenize with masked labels
        input_ids, attention_mask, labels, token_type_ids = self._tokenize_for_training(
            inputs, outputs, tokenizer, max_seq_length
        )

        # Create dataset
        dataset = StructuredOutputDataset(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            token_type_ids=token_type_ids,
        )

        # Calculate warmup_steps from warmup_ratio with gradient accumulation
        effective_batch_size = batch_size * gradient_accumulation_steps
        num_update_steps_per_epoch = max(
            1, (len(dataset) + effective_batch_size - 1) // effective_batch_size
        )
        total_steps = num_update_steps_per_epoch * num_epochs
        warmup_steps = int(warmup_ratio * total_steps)

        # Determine eval batch size
        eval_batch_size = (
            per_device_eval_batch_size
            if per_device_eval_batch_size is not None
            else batch_size * 2
        )

        # Determine optimizer for memory efficiency
        optim = "paged_adamw_8bit" if use_8bit_optimizer else optimizer_type

        # Configure training
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            logging_steps=10,
            save_strategy="epoch",
            bf16=self._get_device().type == "cuda",
            dataloader_pin_memory=False,  # For MPS compatibility
            optim=optim,
            gradient_checkpointing=use_gradient_checkpointing,
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
                max_length=2048,  # Default max length for inference
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
        self,
        student: HuggingfaceModel[T],
        inputs: list[str],
        outputs: list[T],
        *,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        max_seq_length: int = 2048,
        use_gradient_checkpointing: bool = True,
        use_8bit_optimizer: bool = False,
        optimizer_type: str = "adamw_torch_fused",
        max_grad_norm: float = 1.0,
        per_device_eval_batch_size: int | None = None,
        output_dir: str = "./hf_model_output",
    ) -> None:
        """Distill knowledge from this model (teacher) to the student model.

        Uses KL divergence between teacher and student logits, combined with
        hard label cross-entropy loss. Teacher logits are computed on-the-fly
        during training to minimize memory usage.

        Args:
            student: Student model to train
            inputs: List of input strings
            outputs: List of Pydantic model instances representing expected outputs
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            batch_size: Training batch size
            gradient_accumulation_steps: Number of steps to accumulate gradients
            warmup_ratio: Ratio of total steps for learning rate warmup
            weight_decay: Weight decay for regularization
            max_seq_length: Maximum sequence length for tokenization
            use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency
            use_8bit_optimizer: Use 8-bit Adam optimizer (requires bitsandbytes)
            optimizer_type: Optimizer type to use
            max_grad_norm: Maximum gradient norm for clipping
            per_device_eval_batch_size: Eval batch size (defaults to batch_size * 2)
            output_dir: Directory for saving checkpoints
        """
        if len(inputs) != len(outputs):
            msg = "inputs and outputs must have same length"
            raise ValueError(msg)

        teacher_model, _ = self._load_model_and_tokenizer()
        student_model, student_tokenizer = student._load_model_and_tokenizer()

        teacher_model.eval()  # Teacher is frozen

        # Enable gradient checkpointing on student if requested
        if use_gradient_checkpointing:
            student_model.gradient_checkpointing_enable()

        # Tokenize with student tokenizer
        input_ids, attention_mask, labels, token_type_ids = (
            student._tokenize_for_training(
                inputs, outputs, student_tokenizer, max_seq_length
            )
        )

        # Create dataset
        dataset = DistillationDataset(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            token_type_ids=token_type_ids,
        )

        # Calculate warmup_steps from warmup_ratio with gradient accumulation
        effective_batch_size = batch_size * gradient_accumulation_steps
        num_update_steps_per_epoch = max(
            1, (len(dataset) + effective_batch_size - 1) // effective_batch_size
        )
        total_steps = num_update_steps_per_epoch * num_epochs
        warmup_steps = int(warmup_ratio * total_steps)

        # Determine eval batch size
        eval_batch_size = (
            per_device_eval_batch_size
            if per_device_eval_batch_size is not None
            else batch_size * 2
        )

        # Determine optimizer for memory efficiency
        optim = "paged_adamw_8bit" if use_8bit_optimizer else optimizer_type

        # Configure training
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            logging_steps=10,
            save_strategy="epoch",
            bf16=student._get_device().type == "cuda",
            dataloader_pin_memory=False,  # For MPS compatibility
            optim=optim,
            gradient_checkpointing=use_gradient_checkpointing,
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

    def save(self, stream: BinaryIO) -> None:
        """Save the model, tokenizer, and configuration to a binary stream.

        Args:
            stream: Binary stream to write the model to
        """
        # Load model and tokenizer if not already loaded
        model, tokenizer = self._load_model_and_tokenizer()

        # Create temporary directory for saving
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save model and tokenizer
            model_dir = temp_path / "model"
            tokenizer_dir = temp_path / "tokenizer"
            model.save_pretrained(str(model_dir))
            tokenizer.save_pretrained(str(tokenizer_dir))

            # Save configuration
            config = {
                "model_name": self.model_name,
                "output_type": f"{self.output_type.__module__}.{self.output_type.__name__}",
                "system_prompt": self.system_prompt,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "distillation_temperature": self.distillation_temperature,
                "distillation_alpha": self.distillation_alpha,
                "device": self._device_name,
            }
            config_file = temp_path / "config.json"
            config_file.write_text(json.dumps(config, indent=2))

            # Create tar archive and write to stream
            with tarfile.open(fileobj=stream, mode="w:gz") as tar:
                tar.add(model_dir, arcname="model")
                tar.add(tokenizer_dir, arcname="tokenizer")
                tar.add(config_file, arcname="config.json")

    @classmethod
    def load(cls, stream: BinaryIO) -> HuggingfaceModel[BaseModel]:  # type: ignore[type-arg]
        """Load a model from a binary stream.

        Args:
            stream: Binary stream containing the saved model

        Returns:
            Loaded HuggingfaceModel instance
        """
        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Extract tar archive
            with tarfile.open(fileobj=stream, mode="r:gz") as tar:
                tar.extractall(temp_path)

            # Load configuration
            config_file = temp_path / "config.json"
            config = json.loads(config_file.read_text())

            # Reconstruct output_type class
            module_name, class_name = config["output_type"].rsplit(".", 1)
            module = importlib.import_module(module_name)
            output_type = getattr(module, class_name)

            # Create model instance
            model = cls(
                model_name=config["model_name"],
                output_type=output_type,
                system_prompt=config.get("system_prompt"),
                max_new_tokens=config.get("max_new_tokens", 512),
                temperature=config.get("temperature", 0.7),
                distillation_temperature=config.get("distillation_temperature", 2.0),
                distillation_alpha=config.get("distillation_alpha", 0.5),
                device=config.get("device"),
            )

            # Load model and tokenizer from saved files
            model_dir = temp_path / "model"
            tokenizer_dir = temp_path / "tokenizer"

            loaded_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
            device = model._get_device()
            dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
            loaded_model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                dtype=dtype,
            )
            loaded_model.to(device)  # type: ignore[arg-type]

            # Set loaded model and tokenizer
            model._model = loaded_model  # type: ignore[assignment]
            model._tokenizer = loaded_tokenizer

            # Ensure pad token is set
            if loaded_tokenizer.pad_token is None:
                loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
                loaded_model.config.pad_token_id = loaded_tokenizer.pad_token_id

            return model  # type: ignore[return-value]
