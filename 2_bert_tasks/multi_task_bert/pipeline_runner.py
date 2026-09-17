"""
Pipeline Runner: Unified CLI Orchestrator for Multi-Task BERT Fine-Tuning
Supports:
1. Text Classification / Sentiment Analysis
2. Token Classification / Named Entity Recognition (NER)
3. Extractive Question Answering (QA)
4. Sentence Pair Classification / Natural Language Inference (NLI)

Usage:
    python multi_task_bert/pipeline_runner.py --task classification --epochs 3 --batch_size 16
    python multi_task_bert/pipeline_runner.py --task ner --epochs 3
    python multi_task_bert/pipeline_runner.py --task qa --epochs 3
    python multi_task_bert/pipeline_runner.py --task nli --epochs 3
    python multi_task_bert/pipeline_runner.py --task all --epochs 2
"""

import os
import sys
import argparse
import time
import torch
from typing import Dict, Any

# Ensure parent directory is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from multi_task_bert.common.trainer_utils import set_seed, get_device_and_amp_dtype
from multi_task_bert.tasks.task1_classification import train_classification
from multi_task_bert.tasks.task2_ner import train_ner
from multi_task_bert.tasks.task3_qa import train_qa
from multi_task_bert.tasks.task4_nli import train_nli


def parse_args():
    parser = argparse.ArgumentParser(
        description="Production Multi-Task BERT Fine-Tuning Orchestrator"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "ner", "qa", "nli", "all"],
        default="all",
        help="Task to execute: classification, ner, qa, nli, or all (default: all)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Hugging Face pretrained BERT checkpoint (defaults: bert-base-uncased, or bert-base-cased for NER)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Per-device batch size (default: 16)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Peak learning rate for AdamW (default: 3e-5)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay parameter (default: 0.01)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of update steps to accumulate before backward/update pass (default: 1)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(PARENT_DIR, "checkpoints"),
        help="Root directory for saving checkpoints and generated model cards (default: 2_bert_tasks/checkpoints)"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to automatically push the fine-tuned model and card to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub_username",
        type=str,
        default=None,
        help="Hugging Face Hub username or organization namespace (e.g. AkhandPratapSingh11)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility across torch, numpy, and python (default: 42)"
    )
    return parser.parse_args()


def print_summary_banner(results: Dict[str, Dict[str, float]], elapsed_time: float):
    print("\n" + "=" * 80)
    print("                 MULTI-TASK BERT FINE-TUNING SUMMARY REPORT                 ")
    print("=" * 80)
    print(f"Total Execution Time: {elapsed_time:.2f} seconds\n")
    print(f"{'Task Name':<30} | {'Primary Metric':<25} | {'Score / Value':<15}")
    print("-" * 80)

    for task_name, metrics in results.items():
        if "accuracy" in metrics:
            metric_str = "Accuracy"
            val = f"{metrics['accuracy'] * 100:.2f}%"
        elif "entity_f1" in metrics:
            metric_str = "Entity F1 (Weighted)"
            val = f"{metrics['entity_f1'] * 100:.2f}%"
        elif "exact_match" in metrics:
            metric_str = f"EM: {metrics['exact_match']:.2f}% | F1"
            val = f"{metrics['f1']:.2f}%"
        elif "f1" in metrics:
            metric_str = "F1 Score"
            val = f"{metrics['f1'] * 100:.2f}%"
        else:
            metric_str = "Val Loss"
            val = f"{metrics.get('val_loss', 0.0):.4f}"

        print(f"{task_name:<30} | {metric_str:<25} | {val:<15}")

    print("=" * 80 + "\n")


def main():
    args = parse_args()
    set_seed(args.seed)
    device, amp_dtype = get_device_and_amp_dtype()

    print("\n" + "#" * 80)
    print(f"  BERT Multi-Task Production Fine-Tuning Pipeline")
    print(f"  Hardware: {device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Mixed Precision: {amp_dtype}")
    print(f"  Task Selected: {args.task}")
    print(f"  Push to Hub: {args.push_to_hub}")
    print("#" * 80 + "\n")

    start_time = time.time()
    results = {}

    tasks_to_run = []
    if args.task in ["classification", "all"]:
        tasks_to_run.append("classification")
    if args.task in ["ner", "all"]:
        tasks_to_run.append("ner")
    if args.task in ["qa", "all"]:
        tasks_to_run.append("qa")
    if args.task in ["nli", "all"]:
        tasks_to_run.append("nli")

    for task in tasks_to_run:
        print(f"\n>>> Running Pipeline: {task.upper()} ...")
        task_out = os.path.join(args.output_dir, task)

        if task == "classification":
            model_name = args.model_name or "bert-base-uncased"
            repo_id = f"{args.hub_username}/bert-text-classification" if (args.push_to_hub and args.hub_username) else None
            _, _, metrics = train_classification(
                model_name=model_name,
                output_dir=task_out,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                seed=args.seed,
                push_to_hub=args.push_to_hub,
                hub_repo_id=repo_id
            )
            results["1. Text Classification"] = metrics

        elif task == "ner":
            model_name = args.model_name or "bert-base-uncased"
            repo_id = f"{args.hub_username}/bert-ner-token-classification" if (args.push_to_hub and args.hub_username) else None
            _, _, metrics = train_ner(
                model_name=model_name,
                output_dir=task_out,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                seed=args.seed,
                push_to_hub=args.push_to_hub,
                hub_repo_id=repo_id
            )
            results["2. Token Classification (NER)"] = metrics

        elif task == "qa":
            model_name = args.model_name or "bert-base-uncased"
            repo_id = f"{args.hub_username}/bert-extractive-qa" if (args.push_to_hub and args.hub_username) else None
            _, _, metrics = train_qa(
                model_name=model_name,
                output_dir=task_out,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                seed=args.seed,
                push_to_hub=args.push_to_hub,
                hub_repo_id=repo_id
            )
            results["3. Question Answering (QA)"] = metrics

        elif task == "nli":
            model_name = args.model_name or "bert-base-uncased"
            repo_id = f"{args.hub_username}/bert-nli-sentence-pair" if (args.push_to_hub and args.hub_username) else None
            _, _, metrics = train_nli(
                model_name=model_name,
                output_dir=task_out,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                seed=args.seed,
                push_to_hub=args.push_to_hub,
                hub_repo_id=repo_id
            )
            results["4. Sentence Pair NLI"] = metrics

    elapsed_time = time.time() - start_time
    print_summary_banner(results, elapsed_time)


if __name__ == "__main__":
    main()
