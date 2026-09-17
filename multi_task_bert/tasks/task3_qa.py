"""
Task 3: Extractive Question Answering (QA) using BERT
Features:
- Sliding window chunking with doc_stride and overflow mapping
- Exact character-to-token offset alignment
- Unanswerable / out-of-span mapping to CLS (position 0)
- Native PyTorch loop with Mixed Precision (BF16/FP16) & Gradient Accumulation
- SQuAD standard Exact Match (EM) and Token-level F1 evaluation
- Hub integration with auto-generated Question-Answering Model Card.
"""

import os
import re
import string
import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForQuestionAnswering
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
import numpy as np

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODULE_DIR = Path(__file__).resolve().parent.parent
for p in [str(ROOT_DIR), str(MODULE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from multi_task_bert.common.trainer_utils import (
        set_seed,
        get_device_and_amp_dtype,
        build_optimizer_with_weight_decay,
        build_scheduler,
        save_training_checkpoint,
        check_hub_connectivity
    )
    from multi_task_bert.common.hub_utils import (
        push_model_and_card_to_hub,
        generate_model_card
    )
except ImportError:
    from common.trainer_utils import (
        set_seed,
        get_device_and_amp_dtype,
        build_optimizer_with_weight_decay,
        build_scheduler,
        save_training_checkpoint,
        check_hub_connectivity
    )
    from common.hub_utils import (
        push_model_and_card_to_hub,
        generate_model_card
    )


class QADataset(Dataset):
    """PyTorch Dataset for QA features."""
    def __init__(self, features: List[Dict[str, Any]]):
        self.features = features

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        feat = self.features[idx]
        item = {
            "input_ids": torch.tensor(feat["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(feat["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(feat["token_type_ids"], dtype=torch.long),
            "start_positions": torch.tensor(feat["start_positions"], dtype=torch.long),
            "end_positions": torch.tensor(feat["end_positions"], dtype=torch.long)
        }
        return item

    def __len__(self) -> int:
        return len(self.features)


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace (SQuAD official metric)."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_qa_f1(pred: str, ground_truth: str) -> float:
    """Computes word token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(pred).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)

    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)


def compute_qa_exact_match(pred: str, ground_truth: str) -> float:
    """Computes exact match (1.0 or 0.0)."""
    return float(normalize_answer(pred) == normalize_answer(ground_truth))


def prepare_train_qa_features(
    examples: List[Dict[str, Any]],
    tokenizer: BertTokenizerFast,
    max_seq_length: int = 384,
    doc_stride: int = 128
) -> List[Dict[str, Any]]:
    """
    Tokenizes question-context pairs with sliding-window overflow and maps
    character start/end answer positions to token start/end positions.
    """
    questions = [q.strip() for q in [ex["question"] for ex in examples]]
    contexts = [ex["context"] for ex in examples]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    features = []
    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        example = examples[sample_idx]
        answer = example["answers"]

        input_ids = tokenized_examples["input_ids"][i]
        attention_mask = tokenized_examples["attention_mask"][i]
        token_type_ids = tokenized_examples["token_type_ids"][i]
        sequence_ids = tokenized_examples.sequence_ids(i)

        # CLS index is 0 in BERT
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # If no answers are given, set the cls_index as target
        if len(answer["answer_start"]) == 0:
            start_pos = cls_index
            end_pos = cls_index
        else:
            char_start = answer["answer_start"][0]
            char_end = char_start + len(answer["text"][0])

            # Find boundary of context in current chunk (sequence_id == 1)
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Check if answer span is entirely within current context slice
            if not (offsets[token_start_index][0] <= char_start and offsets[token_end_index][1] >= char_end):
                start_pos = cls_index
                end_pos = cls_index
            else:
                # Move token_start_index forward to match start
                curr_start = token_start_index
                while curr_start <= token_end_index and offsets[curr_start][0] <= char_start:
                    curr_start += 1
                start_pos = curr_start - 1

                # Move token_end_index backward to match end
                curr_end = token_end_index
                while curr_end >= token_start_index and offsets[curr_end][1] >= char_end:
                    curr_end -= 1
                end_pos = curr_end + 1

        features.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "start_positions": start_pos,
            "end_positions": end_pos,
            "offsets": offsets,
            "example_id": example.get("id", str(sample_idx)),
            "context": example["context"],
            "ground_truth_answer": answer["text"][0] if len(answer["text"]) > 0 else ""
        })

    return features


def get_mock_qa_data() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generates synthetic SQuAD-like QA pairs for isolated dry-runs and offline environments."""
    train_data = [
        {
            "id": "1",
            "question": "Where is Google headquarters located?",
            "context": "Google LLC is an American multinational technology company. Google headquarters is located in Mountain View, California.",
            "answers": {"text": ["Mountain View, California"], "answer_start": [87]}
        },
        {
            "id": "2",
            "question": "Who is the CEO of Alphabet?",
            "context": "Sundar Pichai was appointed CEO of Google in August 2015 and later became the CEO of Alphabet Inc.",
            "answers": {"text": ["Sundar Pichai"], "answer_start": [0]}
        },
        {
            "id": "3",
            "question": "What architecture does BERT use?",
            "context": "BERT is a language representation model which stands for Bidirectional Encoder Representations from Transformers.",
            "answers": {"text": ["Transformers"], "answer_start": [97]}
        },
        {
            "id": "4",
            "question": "When was BERT introduced?",
            "context": "BERT was introduced in October 2018 by researchers at Google AI Language.",
            "answers": {"text": ["October 2018"], "answer_start": [23]}
        }
    ]
    val_data = [
        {
            "id": "v1",
            "question": "Where is the Eiffel Tower situated?",
            "context": "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France.",
            "answers": {"text": ["Paris, France"], "answer_start": [81]}
        },
        {
            "id": "v2",
            "question": "What does NLP stand for?",
            "context": "Natural Language Processing (NLP) is a subfield of computer science and artificial intelligence.",
            "answers": {"text": ["Natural Language Processing"], "answer_start": [0]}
        }
    ]
    return train_data, val_data


def evaluate_qa(
    model: nn.Module,
    val_features: List[Dict[str, Any]],
    device: torch.device,
    amp_dtype: torch.dtype,
    batch_size: int = 16,
    max_answer_length: int = 30
) -> Dict[str, float]:
    """
    Evaluates Extractive QA with Exact Match (EM) and Token F1.
    Decodes best span (start, end) where start <= end and span_len <= max_answer_length.
    """
    model.eval()
    val_dataset = QADataset(val_features)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    all_start_logits = []
    all_end_logits = []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                loss = outputs.loss
                total_loss += loss.item()

            all_start_logits.append(outputs.start_logits.float().cpu().numpy())
            all_end_logits.append(outputs.end_logits.float().cpu().numpy())

    avg_loss = total_loss / max(len(val_loader), 1)
    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    # Decode spans
    em_scores = []
    f1_scores = []

    for i, feat in enumerate(val_features):
        start_logits = all_start_logits[i]
        end_logits = all_end_logits[i]
        offsets = feat["offsets"]
        context = feat["context"]
        ground_truth = feat["ground_truth_answer"]

        # Find best span
        best_score = -float("inf")
        best_span = (0, 0)

        # Get top-20 start and end indices
        start_top_k = np.argsort(start_logits)[-20:]
        end_top_k = np.argsort(end_logits)[-20:]

        for s_idx in start_top_k:
            for e_idx in end_top_k:
                if s_idx <= e_idx and (e_idx - s_idx + 1) <= max_answer_length:
                    # Ignore special tokens with (0,0) offset if not CLS
                    if s_idx > 0 and offsets[s_idx] == (0, 0):
                        continue
                    if e_idx > 0 and offsets[e_idx] == (0, 0):
                        continue
                    score = start_logits[s_idx] + end_logits[e_idx]
                    if score > best_score:
                        best_score = score
                        best_span = (s_idx, e_idx)

        # Extract text from span
        pred_start_token, pred_end_token = best_span
        if pred_start_token == 0 or pred_end_token == 0:
            pred_text = ""
        else:
            char_start = offsets[pred_start_token][0]
            char_end = offsets[pred_end_token][1]
            pred_text = context[char_start:char_end]

        em_scores.append(compute_qa_exact_match(pred_text, ground_truth))
        f1_scores.append(compute_qa_f1(pred_text, ground_truth))

    return {
        "val_loss": float(avg_loss),
        "exact_match": float(np.mean(em_scores) * 100.0) if em_scores else 0.0,
        "f1": float(np.mean(f1_scores) * 100.0) if f1_scores else 0.0
    }


def train_qa(
    model_name: str = "bert-base-uncased",
    output_dir: str = "./checkpoints/qa",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None
) -> Tuple[nn.Module, BertTokenizerFast, Dict[str, float]]:
    """
    Executes production-grade fine-tuning for Question Answering.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device, amp_dtype = get_device_and_amp_dtype()

    print(f"\n[QA Pipeline] Initializing BertTokenizerFast for {model_name}...")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    # Attempt to load SQuAD or fallback to mock
    if check_hub_connectivity():
        try:
            from datasets import load_dataset
            print("[QA Pipeline] Loading SQuAD v1.1 dataset from Hugging Face...")
            raw_dataset = load_dataset("squad")
            train_examples = [ex for ex in raw_dataset["train"]]
            val_examples = [ex for ex in raw_dataset["validation"]]
            dataset_name = "squad"
        except Exception as e:
            print(f"[QA Pipeline] Warning: Unable to load remote SQuAD dataset ({e}). Using mock dataset.")
            train_examples, val_examples = get_mock_qa_data()
            dataset_name = "squad_synthetic"
    else:
        print("[QA Pipeline] Offline / local environment detected. Using mock SQuAD benchmark dataset.")
        train_examples, val_examples = get_mock_qa_data()
        dataset_name = "squad_synthetic"

    print(f"[QA Pipeline] Extracting sliding window features (doc_stride=128, max_len=384)...")
    train_features = prepare_train_qa_features(train_examples, tokenizer)
    val_features = prepare_train_qa_features(val_examples, tokenizer)

    train_dataset = QADataset(train_features)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"[QA Pipeline] Loading {model_name} for Extractive Question Answering...")
    model = BertForQuestionAnswering.from_pretrained(model_name).to(device)

    # Optimizer & Scheduler
    optimizer = build_optimizer_with_weight_decay(model, learning_rate=learning_rate, weight_decay=weight_decay)
    total_training_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    warmup_steps = int(0.1 * total_training_steps)
    scheduler = build_scheduler(
        optimizer,
        scheduler_type="cosine",
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    use_grad_scaler = (device.type == "cuda" and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    best_val_f1 = 0.0
    best_metrics = {}

    print(f"[QA Pipeline] Starting training: {epochs} epochs, {len(train_dataset)} features, device: {device}, amp: {amp_dtype}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                loss = outputs.loss / gradient_accumulation_steps

            if use_grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            train_loss += loss.item() * gradient_accumulation_steps

            if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                if use_grad_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                if use_grad_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({
                "loss": f"{train_loss / step:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })

        val_metrics = evaluate_qa(model, val_features, device, amp_dtype, batch_size=batch_size)
        print(f"\n[Epoch {epoch} Results] Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Val Loss: {val_metrics['val_loss']:.4f} | EM: {val_metrics['exact_match']:.2f}% | "
              f"F1: {val_metrics['f1']:.2f}%")

        if val_metrics["f1"] >= best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_metrics = val_metrics
            save_training_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=os.path.join(output_dir, "best_checkpoint"),
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=val_metrics
            )
            print(f"[*] New best checkpoint saved with Val F1: {best_val_f1:.2f}%")

    # Generate Model Card
    hyperparams = {
        "base_model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_seq_length": 384,
        "doc_stride": 128,
        "precision": str(amp_dtype)
    }
    card_content = generate_model_card(
        model_name_or_repo=hub_repo_id or f"bert-qa-{dataset_name}",
        task="question-answering",
        dataset_name=dataset_name,
        metrics=best_metrics,
        hyperparameters=hyperparams,
        pipeline_tag="question-answering"
    )
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card_content)

    if push_to_hub and hub_repo_id:
        print(f"[Hub Integration] Uploading fine-tuned QA model to {hub_repo_id}...")
        push_model_and_card_to_hub(
            repo_id=hub_repo_id,
            local_dir=os.path.join(output_dir, "best_checkpoint"),
            readme_content=card_content
        )

    return model, tokenizer, best_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BERT Question Answering Fine-Tuning")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/qa")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--repo_id", type=str, default=None)
    args = parser.parse_args()

    train_qa(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.repo_id
    )
