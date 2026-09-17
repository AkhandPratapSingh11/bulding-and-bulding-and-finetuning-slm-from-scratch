"""
Task 2: Named Entity Recognition (NER) / Token Classification
Using BERT with WordPiece Subword Alignment, Class-Imbalance Weighting,
Mixed Precision (BF16/FP16), Gradient Accumulation, and Hub Integration.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForTokenClassification
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report

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

# Standard CoNLL-2003 entity mapping
CONLL_LABELS = [
    "O",        # 0
    "B-PER",    # 1
    "I-PER",    # 2
    "B-ORG",    # 3
    "I-ORG",    # 4
    "B-LOC",    # 5
    "I-LOC",    # 6
    "B-MISC",   # 7
    "I-MISC"    # 8
]
LABEL2ID = {lbl: i for i, lbl in enumerate(CONLL_LABELS)}
ID2LABEL = {i: lbl for i, lbl in enumerate(CONLL_LABELS)}


class NERTokenDataset(Dataset):
    """
    Fast PyTorch dataset storing pre-tokenized and subword-aligned inputs.
    """
    def __init__(self, encodings: Dict[str, Any], labels: List[List[int]]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)


def align_labels_with_tokens(
    labels: List[List[int]],
    word_ids_list: List[List[Optional[int]]],
    label_all_tokens: bool = False
) -> List[List[int]]:
    """
    Align word-level ground-truth NER labels with WordPiece subword tokens.
    
    Rule:
    - Special tokens (e.g. [CLS], [SEP], [PAD]) have word_id is None -> label = -100 (ignored in loss)
    - First subword of a word -> assigned the word's label
    - Subsequent subwords of the same word:
      If label_all_tokens is False: label = -100 (standard recommendation in Devlin et al.)
      If label_all_tokens is True: mapped to 'I-' label or retained.
    """
    aligned_labels = []
    for i, word_ids in enumerate(word_ids_list):
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # Special token ([CLS], [SEP], [PAD])
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word
                label_ids.append(labels[i][word_idx])
            else:
                # Subsequent subword of the same word
                label_ids.append(labels[i][word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)
    return aligned_labels


def compute_ner_class_weights(aligned_labels: List[List[int]], num_classes: int = 9) -> torch.Tensor:
    """
    Computes inverse frequency weights to counteract the severe 'O' class imbalance in NER.
    Tokens marked as -100 are excluded from frequency calculation.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for seq in aligned_labels:
        for lbl in seq:
            if 0 <= lbl < num_classes:
                counts[lbl] += 1
                
    total = np.sum(counts)
    # Smooth inverse class frequencies
    weights = np.zeros(num_classes, dtype=np.float32)
    for i in range(num_classes):
        if counts[i] > 0:
            # Dampen extreme ratios using square root
            weights[i] = np.sqrt(total / (counts[i] * num_classes))
        else:
            weights[i] = 1.0
            
    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def get_mock_ner_data() -> Tuple[List[List[str]], List[List[int]], List[List[str]], List[List[int]]]:
    """Generates synthetic CoNLL-2003 formatted data for isolated testing and offline environments."""
    train_tokens = [
        ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."],
        ["Peter", "Blackburn", "reported", "from", "Brussels", "on", "Friday", "."],
        ["Google", "Headquarters", "is", "located", "in", "Mountain", "View", ",", "California", "."],
        ["Sundar", "Pichai", "delivered", "the", "keynote", "address", "at", "Google", "I/O", "."],
        ["Barack", "Obama", "visited", "Paris", "in", "the", "summer", "of", "2015", "."],
        ["The", "United", "Nations", "summit", "will", "take", "place", "in", "Geneva", "."]
    ]
    train_labels = [
        [3, 0, 7, 0, 0, 0, 7, 0, 0],       # B-ORG, O, B-MISC, O, O, O, B-MISC, O, O
        [1, 2, 0, 0, 5, 0, 0, 0],          # B-PER, I-PER, O, O, B-LOC, O, O, O
        [3, 4, 0, 0, 0, 5, 6, 0, 5, 0],    # B-ORG, I-ORG, O, O, O, B-LOC, I-LOC, O, B-LOC, O
        [1, 2, 0, 0, 0, 0, 0, 3, 7, 0],    # B-PER, I-PER, O, O, O, O, O, B-ORG, B-MISC, O
        [1, 2, 0, 5, 0, 0, 0, 0, 0, 0],    # B-PER, I-PER, O, B-LOC, O, O, O, O, O, O
        [0, 3, 4, 0, 0, 0, 0, 0, 5, 0]     # O, B-ORG, I-ORG, O, O, O, O, O, B-LOC, O
    ]
    val_tokens = [
        ["Apple", "unveiled", "new", "iPhone", "models", "in", "Cupertino", "."],
        ["Angela", "Merkel", "spoke", "in", "Berlin", "today", "."]
    ]
    val_labels = [
        [3, 0, 0, 7, 0, 0, 5, 0],          # B-ORG, O, O, B-MISC, O, O, B-LOC, O
        [1, 2, 0, 0, 5, 0, 0]              # B-PER, I-PER, O, O, B-LOC, O, O
    ]
    return train_tokens, train_labels, val_tokens, val_labels


def evaluate_ner(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    loss_fn: nn.Module
) -> Dict[str, float]:
    """
    Evaluates NER predictions excluding -100 subword masks.
    Computes Token-level Accuracy, Precision, Recall, F1, and Non-O Entity F1.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                logits = outputs.logits
                loss = loss_fn(logits.view(-1, len(CONLL_LABELS)), labels.view(-1))

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)

            # Mask out -100 tokens
            mask = labels != -100
            valid_preds = preds[mask].cpu().numpy().tolist()
            valid_targets = labels[mask].cpu().numpy().tolist()

            all_preds.extend(valid_preds)
            all_targets.extend(valid_targets)

    avg_loss = total_loss / max(len(val_loader), 1)

    if len(all_targets) == 0:
        return {"val_loss": avg_loss, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "entity_f1": 0.0}

    # Micro/macro metrics across all tokens
    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    accuracy = np.mean(all_preds_arr == all_targets_arr)
    
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets_arr, all_preds_arr, average="weighted", zero_division=0
    )

    # Focus specifically on entity tokens (excluding 'O' = 0)
    entity_mask = all_targets_arr != 0
    if np.any(entity_mask):
        _, _, entity_f1, _ = precision_recall_fscore_support(
            all_targets_arr[entity_mask],
            all_preds_arr[entity_mask],
            average="weighted",
            zero_division=0
        )
    else:
        entity_f1 = f1

    return {
        "val_loss": float(avg_loss),
        "accuracy": float(accuracy),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "entity_f1": float(entity_f1)
    }


def train_ner(
    model_name: str = "bert-base-uncased",
    output_dir: str = "./checkpoints/ner",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    use_class_weights: bool = True,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None
) -> Tuple[nn.Module, BertTokenizerFast, Dict[str, float]]:
    """
    Executes production-grade fine-tuning for Token Classification / NER.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device, amp_dtype = get_device_and_amp_dtype()

    print(f"\n[NER Pipeline] Initializing BertTokenizerFast for {model_name}...")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    # Attempt to load CoNLL-2003 or fallback to synthetic mock data
    if check_hub_connectivity():
        try:
            from datasets import load_dataset
            print("[NER Pipeline] Loading CoNLL-2003 dataset from Hugging Face...")
            raw_dataset = load_dataset("conll2003")
            train_words = raw_dataset["train"]["tokens"]
            train_ner_tags = raw_dataset["train"]["ner_tags"]
            val_words = raw_dataset["validation"]["tokens"]
            val_ner_tags = raw_dataset["validation"]["ner_tags"]
            dataset_name = "conll2003"
        except Exception as e:
            print(f"[NER Pipeline] Warning: Unable to load remote CoNLL dataset ({e}). Using mock dataset.")
            train_words, train_ner_tags, val_words, val_ner_tags = get_mock_ner_data()
            dataset_name = "conll2003_synthetic"
    else:
        print("[NER Pipeline] Offline / local environment detected. Using mock CoNLL benchmark dataset.")
        train_words, train_ner_tags, val_words, val_ner_tags = get_mock_ner_data()
        dataset_name = "conll2003_synthetic"

    # Tokenize with word-level alignment
    print("[NER Pipeline] Tokenizing and aligning subwords with word_ids()...")
    train_encodings = tokenizer(
        train_words,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors=None
    )
    val_encodings = tokenizer(
        val_words,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors=None
    )

    # Align labels
    train_word_ids = [train_encodings.word_ids(batch_index=i) for i in range(len(train_words))]
    val_word_ids = [val_encodings.word_ids(batch_index=i) for i in range(len(val_words))]

    aligned_train_labels = align_labels_with_tokens(train_ner_tags, train_word_ids)
    aligned_val_labels = align_labels_with_tokens(val_ner_tags, val_word_ids)

    train_dataset = NERTokenDataset(train_encodings, aligned_train_labels)
    val_dataset = NERTokenDataset(val_encodings, aligned_val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[NER Pipeline] Loading {model_name} for Token Classification with {len(CONLL_LABELS)} labels...")
    model = BertForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(CONLL_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    ).to(device)

    # Setup loss function with class weighting
    if use_class_weights:
        weights = compute_ner_class_weights(aligned_train_labels, num_classes=len(CONLL_LABELS)).to(device)
        print(f"[NER Pipeline] Computed inverse class weights: {np.round(weights.float().cpu().numpy(), 3)}")
        loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

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

    print(f"[NER Pipeline] Starting training: {epochs} epochs, {len(train_dataset)} samples, device: {device}, amp: {amp_dtype}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                logits = outputs.logits
                loss = loss_fn(logits.view(-1, len(CONLL_LABELS)), labels.view(-1))
                loss = loss / gradient_accumulation_steps

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

        # Evaluate at epoch end
        val_metrics = evaluate_ner(model, val_loader, device, amp_dtype, loss_fn)
        print(f"\n[Epoch {epoch} Results] Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Val Loss: {val_metrics['val_loss']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
              f"Entity F1: {val_metrics['entity_f1']:.4f} | Acc: {val_metrics['accuracy']:.4f}")

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
            print(f"[*] New best checkpoint saved with Val F1: {best_val_f1:.4f}")

    # Generate Model Card
    hyperparams = {
        "base_model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "precision": str(amp_dtype),
        "class_weighting": use_class_weights
    }
    card_content = generate_model_card(
        model_name_or_repo=hub_repo_id or f"bert-ner-{dataset_name}",
        task="token-classification",
        dataset_name=dataset_name,
        metrics=best_metrics,
        hyperparameters=hyperparams,
        pipeline_tag="token-classification"
    )
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card_content)

    if push_to_hub and hub_repo_id:
        print(f"[Hub Integration] Uploading fine-tuned NER model to {hub_repo_id}...")
        push_model_and_card_to_hub(
            repo_id=hub_repo_id,
            local_dir=os.path.join(output_dir, "best_checkpoint"),
            readme_content=card_content
        )

    return model, tokenizer, best_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BERT Named Entity Recognition (Token Classification)")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/ner")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--repo_id", type=str, default=None)
    args = parser.parse_args()

    train_ner(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.repo_id
    )
