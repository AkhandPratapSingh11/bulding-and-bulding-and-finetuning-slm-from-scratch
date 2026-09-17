"""
Task 1: Text Classification & Sentiment Analysis
================================================
- Dataset: stanfordnlp/imdb (or glue/sst2)
- Architecture: BertForSequenceClassification
- Optimizations: Mixed precision (BF16/FP16), gradient accumulation, cosine warmup, class weighting
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm.auto import tqdm

from common.trainer_utils import (
    set_seed,
    setup_device_and_autocast,
    build_optimizer_with_weight_decay,
    build_scheduler,
)
from common.hub_utils import generate_model_card, push_model_and_card_to_hub


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def compute_class_weights(labels: list, num_classes: int = 2) -> torch.Tensor:
    """Computes inverse class frequency weights to counteract class imbalances."""
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float)


def train_text_classifier(
    base_model: str = "bert-base-uncased",
    dataset_name: str = "stanfordnlp/imdb",
    num_classes: int = 2,
    max_length: int = 256,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    epochs: int = 2,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    sample_size: int = 1000,
    output_dir: str = "./checkpoints/classification",
    push_to_hub: bool = False,
    repo_name: str = "bert-base-imdb-sentiment",
    seed: int = 42,
):
    set_seed(42)
    set_seed(seed)
    device, autocast_dtype, use_grad_scaler = setup_device_and_autocast()
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    print(f"\n[Task 1] Loading tokenizer & model: {base_model}")
    tokenizer = BertTokenizerFast.from_pretrained(base_model)
    model = BertForSequenceClassification.from_pretrained(base_model, num_labels=num_classes).to(device)

    # 1. Load Dataset
    print(f"[Task 1] Loading dataset: {dataset_name}")
    raw_dataset = load_dataset(dataset_name)
    train_texts, train_labels, test_texts, test_labels = [], [], [], []
    from common.trainer_utils import check_hub_connectivity
    if check_hub_connectivity():
        try:
            print(f"[Task 1] Loading dataset: {dataset_name}")
            raw_dataset = load_dataset(dataset_name)
            train_data = raw_dataset["train"].shuffle(seed=seed).select(range(min(sample_size, len(raw_dataset["train"]))))
            test_data = raw_dataset["test"].shuffle(seed=seed).select(range(min(sample_size // 2, len(raw_dataset["test"]))))
            train_texts, train_labels = train_data["text"], train_data["label"]
            test_texts, test_labels = test_data["text"], test_data["label"]
        except Exception as e:
            print(f"[Task 1] Warning: Could not fetch online dataset ({e}). Using mock dataset.")
            raw_dataset = None
    else:
        raw_dataset = None

    train_data = raw_dataset["train"].shuffle(seed=42).select(range(min(sample_size, len(raw_dataset["train"]))))
    test_data = raw_dataset["test"].shuffle(seed=42).select(range(min(sample_size // 2, len(raw_dataset["test"]))))
    if not train_texts:
        print("[Task 1] Offline / local environment detected. Using mock IMDB benchmark dataset.")
        train_texts = [
            "This movie was an absolute masterpiece! Loved the direction and acting.",
            "Terrible film, completely boring and a waste of two hours.",
            "Outstanding cinematic work with unforgettable emotional depth.",
            "Horrible script and awful acting. Do not recommend.",
            "A breathtaking journey that kept me on the edge of my seat.",
            "Worst movie I have ever seen in my life."
        ]
        train_labels = [1, 0, 1, 0, 1, 0]
        test_texts = [
            "Brilliant acting and stunning visuals throughout!",
            "Completely unwatchable and utterly predictable."
        ]
        test_labels = [1, 0]

    train_texts, train_labels = train_data["text"], train_data["label"]
    test_texts, test_labels = test_data["text"], test_data["label"]

    # Class imbalance handling
    class_weights = compute_class_weights(train_labels, num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    train_ds = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    test_ds = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Optimizer and Scheduler
    optimizer = build_optimizer_with_weight_decay(model, learning_rate=learning_rate, weight_decay=weight_decay)
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_ratio=warmup_ratio, scheduler_type="cosine")

    print(f"[Task 1] Training: {epochs} epochs | Effective Batch Size: {batch_size * grad_accum_steps} | Steps: {total_steps}")
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=autocast_dtype):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss = loss / grad_accum_steps

            if use_grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                if use_grad_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

        print(f"Epoch {epoch+1} Complete | Average Loss: {running_loss / len(train_loader):.4f}")

    # Evaluation
    print("\n[Task 1] Running Evaluation on Held-out Test Set...")
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=autocast_dtype):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            preds.extend(batch_preds)
            targets.extend(labels.cpu().numpy())

    acc = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average="weighted")
    metrics = {
        "Accuracy": acc,
        "Weighted F1": f1,
        "Weighted Precision": precision,
        "Weighted Recall": recall,
    }

    print("\n" + "=" * 60)
    print("TASK 1 EVALUATION RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k:<25}: {v:.4f}")
    print("\n" + classification_report(targets, preds, digits=4))

    # Model Card & Hub Push
    hyperparameters = {
        "Base Model": base_model,
        "Learning Rate": learning_rate,
        "Batch Size (Micro)": batch_size,
        "Gradient Accumulation": grad_accum_steps,
        "Effective Batch Size": batch_size * grad_accum_steps,
        "Epochs": epochs,
        "Precision": "BFloat16" if autocast_dtype == torch.bfloat16 else "Float16",
        "Optimizer": "AdamW (differential weight decay)",
        "Scheduler": "Cosine Annealing with Warmup",
        "Warmup Ratio": warmup_ratio,
        "Max Length": max_length,
    }

    usage_snippet = f"""from transformers import pipeline

# Load pipeline from Hub
classifier = pipeline("text-classification", model="{repo_name}")

# Run inference
result = classifier("This movie was an absolute masterpiece with gripping performances!")
print(result)
# Output: [{'label': 'LABEL_1', 'score': 0.998}]
# Output: [{{'label': 'LABEL_1', 'score': 0.998}}]
"""

    model_card = generate_model_card(
        task_name="text-classification",
        repo_id=repo_name,
        base_model=base_model,
        dataset_name=dataset_name,
        dataset_description="50,000 highly polarized movie reviews from the Stanford IMDB benchmark for binary sentiment analysis.",
        intended_use="Binary sentiment classification (positive/negative sentiment analysis on consumer reviews and social text).",
        hyperparameters=hyperparameters,
        metrics=metrics,
        usage_snippet=usage_snippet,
        output_filepath=f"model_cards/{repo_name}_README.md",
    )

    # Save checkpoint
    from common.trainer_utils import save_training_checkpoint
    save_training_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=os.path.join(output_dir, "best_checkpoint"),
        epoch=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card)

    if push_to_hub:
        push_model_and_card_to_hub(
            model=model,
            tokenizer=tokenizer,
            repo_name=repo_name,
            model_card_content=model_card,
        )

    return model, tokenizer, metrics


def train_classification(
    model_name: str = "bert-base-uncased",
    output_dir: str = "./checkpoints/classification",
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 2,
    seed: int = 42,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None
):
    return train_text_classifier(
        base_model=model_name,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_accum_steps=gradient_accumulation_steps,
        seed=seed,
        push_to_hub=push_to_hub,
        repo_name=hub_repo_id or "bert-text-classification"
    )
