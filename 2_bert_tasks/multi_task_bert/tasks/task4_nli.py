"""
Task 4: Sentence Pair Classification / Natural Language Inference (NLI)
Using BERT with Segment Embeddings (token_type_ids), Mixed Precision (BF16/FP16),
Gradient Accumulation, Differential Weight Decay, and Hugging Face Hub Integration.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

# MNLI Label Mapping
NLI_LABELS = ["entailment", "neutral", "contradiction"]
LABEL2ID = {lbl: i for i, lbl in enumerate(NLI_LABELS)}
ID2LABEL = {i: lbl for i, lbl in enumerate(NLI_LABELS)}


class NLIDataset(Dataset):
    """PyTorch Dataset for Premise-Hypothesis sentence pairs."""
    def __init__(self, encodings: Dict[str, Any], labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)


def get_mock_nli_data() -> Tuple[List[str], List[str], List[int], List[str], List[str], List[int]]:
    """Generates synthetic Premise-Hypothesis sentence pairs for isolated testing and offline environments."""
    train_premises = [
        "A soccer game with multiple males playing.",
        "An older man sitting with a drink.",
        "Two women are embracing while holding to-go packages.",
        "A man in a green shirt is reading a book under a tree.",
        "A black dog is running through the grass after a ball.",
        "Children are swimming in a pool on a hot summer day."
    ]
    train_hypotheses = [
        "Some men are playing a sport.",              # Entailment (0)
        "An old man is asleep in his bed.",           # Contradiction (2)
        "The women are sisters saying goodbye.",       # Neutral (1)
        "A person is outdoors with a book.",          # Entailment (0)
        "A cat is sleeping on a sofa inside.",        # Contradiction (2)
        "The kids are enjoying the water.",           # Entailment (0)
    ]
    train_labels = [0, 2, 1, 0, 2, 0]

    val_premises = [
        "A chef is preparing food in a busy restaurant kitchen.",
        "A musician is playing an acoustic guitar on stage."
    ]
    val_hypotheses = [
        "Someone is cooking in a kitchen.",           # Entailment (0)
        "The stage is empty and dark."                # Contradiction (2)
    ]
    val_labels = [0, 2]

    return train_premises, train_hypotheses, train_labels, val_premises, val_hypotheses, val_labels


def evaluate_nli(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    loss_fn: nn.Module
) -> Dict[str, float]:
    """
    Evaluates NLI classification accuracy, precision, recall, and macro F1.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                logits = outputs.logits
                loss = loss_fn(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(val_loader), 1)
    acc = accuracy_score(all_targets, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    return {
        "val_loss": float(avg_loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }


def train_nli(
    model_name: str = "bert-base-uncased",
    output_dir: str = "./checkpoints/nli",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None
) -> Tuple[nn.Module, BertTokenizerFast, Dict[str, float]]:
    """
    Executes production-grade fine-tuning for Natural Language Inference (Sentence Pair Classification).
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device, amp_dtype = get_device_and_amp_dtype()

    print(f"\n[NLI Pipeline] Initializing BertTokenizerFast for {model_name}...")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    # Attempt to load GLUE MNLI or fallback to mock
    if check_hub_connectivity():
        try:
            from datasets import load_dataset
            print("[NLI Pipeline] Loading GLUE MNLI dataset from Hugging Face...")
            raw_dataset = load_dataset("glue", "mnli")
            train_premises = raw_dataset["train"]["premise"]
            train_hypotheses = raw_dataset["train"]["hypothesis"]
            train_labels = raw_dataset["train"]["label"]

            val_premises = raw_dataset["validation_matched"]["premise"]
            val_hypotheses = raw_dataset["validation_matched"]["hypothesis"]
            val_labels = raw_dataset["validation_matched"]["label"]
            dataset_name = "glue_mnli"
        except Exception as e:
            print(f"[NLI Pipeline] Warning: Unable to load remote MNLI dataset ({e}). Using mock dataset.")
            train_premises, train_hypotheses, train_labels, val_premises, val_hypotheses, val_labels = get_mock_nli_data()
            dataset_name = "mnli_synthetic"
    else:
        print("[NLI Pipeline] Offline / local environment detected. Using mock MNLI benchmark dataset.")
        train_premises, train_hypotheses, train_labels, val_premises, val_hypotheses, val_labels = get_mock_nli_data()
        dataset_name = "mnli_synthetic"

    # Tokenize sentence pairs with segment embeddings
    print("[NLI Pipeline] Tokenizing sentence pairs with truncation='longest_first'...")
    train_encodings = tokenizer(
        train_premises,
        train_hypotheses,
        truncation=True,
        padding=True,
        max_length=128,
        return_token_type_ids=True,
        return_tensors=None
    )
    val_encodings = tokenizer(
        val_premises,
        val_hypotheses,
        truncation=True,
        padding=True,
        max_length=128,
        return_token_type_ids=True,
        return_tensors=None
    )

    train_dataset = NLIDataset(train_encodings, train_labels)
    val_dataset = NLIDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[NLI Pipeline] Loading {model_name} for 3-class NLI classification...")
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    ).to(device)

    # Compute class weights if slight imbalance exists
    class_counts = np.bincount(train_labels, minlength=3).astype(np.float32)
    weights = np.sum(class_counts) / (len(class_counts) * np.maximum(class_counts, 1.0))
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

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

    best_val_acc = 0.0
    best_metrics = {}

    print(f"[NLI Pipeline] Starting training: {epochs} epochs, {len(train_dataset)} pairs, device: {device}, amp: {amp_dtype}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                logits = outputs.logits
                loss = loss_fn(logits, labels)
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

        val_metrics = evaluate_nli(model, val_loader, device, amp_dtype, loss_fn)
        print(f"\n[Epoch {epoch} Results] Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Val Loss: {val_metrics['val_loss']:.4f} | Accuracy: {val_metrics['accuracy']:.4f} | "
              f"Macro F1: {val_metrics['f1']:.4f}")

        if val_metrics["accuracy"] >= best_val_acc:
            best_val_acc = val_metrics["accuracy"]
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
            print(f"[*] New best checkpoint saved with Val Accuracy: {best_val_acc:.4f}")

    # Generate Model Card
    hyperparams = {
        "base_model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "precision": str(amp_dtype),
        "sentence_pair_encoding": True
    }
    card_content = generate_model_card(
        model_name_or_repo=hub_repo_id or f"bert-nli-{dataset_name}",
        task="sentence-similarity",
        dataset_name=dataset_name,
        metrics=best_metrics,
        hyperparameters=hyperparams,
        pipeline_tag="text-classification"
    )
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card_content)

    if push_to_hub and hub_repo_id:
        print(f"[Hub Integration] Uploading fine-tuned NLI model to {hub_repo_id}...")
        push_model_and_card_to_hub(
            repo_id=hub_repo_id,
            local_dir=os.path.join(output_dir, "best_checkpoint"),
            readme_content=card_content
        )

    return model, tokenizer, best_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BERT Natural Language Inference (NLI) Fine-Tuning")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/nli")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--repo_id", type=str, default=None)
    args = parser.parse_args()

    train_nli(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.repo_id
    )
