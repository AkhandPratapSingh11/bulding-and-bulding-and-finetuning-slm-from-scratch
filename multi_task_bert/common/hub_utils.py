"""
Hugging Face Hub Integration and Dynamic Model Card Generator
=============================================================
Provides automated hub publishing and standardized Model Card (README.md) generation.
Hugging Face Hub and Dynamic Model Card Utilities
==================================================
Handles automated model uploading, authentication, and dynamic generation of
production-grade Model Cards (README.md) with YAML metadata frontmatter.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login, HfApi, create_repo, upload_file, upload_folder
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder, login


def get_authenticated_api(token: Optional[str] = None):
    """
    Loads Hugging Face token from environment, logs in, and returns username and token.
    """
    if not token:
        load_dotenv(find_dotenv(usecwd=True))
        load_dotenv(".env")
        load_dotenv("SLM_Experiment/.env")
        token = os.getenv("HUGGINGFACE_FULL_ACCESS_TOKEN") or os.getenv("HUGGINGFACE_WRITE_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    if not token:
        raise ValueError(
            "No Hugging Face token found! Please set HUGGINGFACE_WRITE_TOKEN in .env or pass token explicitly."
        )

    login(token=token, add_to_git_credential=True)
    api = HfApi()
    user_info = api.whoami()
    username = user_info["name"]
    print(f"[Hub Auth] Logged in as: {username} (Token: {user_info.get('type')})")
    return username, token, api


def generate_model_card(
    task_name: str,
    repo_id: str,
    base_model: str,
    dataset_name: str,
    dataset_description: str,
    intended_use: str,
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, Any],
    usage_snippet: str,
    task_name: Optional[str] = None,
    repo_id: Optional[str] = None,
    base_model: Optional[str] = None,
    dataset_name: str = "dataset",
    dataset_description: Optional[str] = None,
    intended_use: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    usage_snippet: Optional[str] = None,
    output_filepath: Optional[str] = None,
    model_name_or_repo: Optional[str] = None,
    task: Optional[str] = None,
    pipeline_tag: Optional[str] = None,
) -> str:
    """
    Dynamically generates a production-grade Hugging Face Model Card with YAML frontmatter.
    Accepts both positional and keyword arguments from all tasks.
    """
    final_repo = repo_id or model_name_or_repo or "bert-fine-tuned"
    final_task = task_name or task or "text-classification"
    hyperparameters = hyperparameters or {}
    metrics = metrics or {}
    final_base_model = base_model or hyperparameters.get("base_model", "bert-base-uncased")

    # Map task names to official Hugging Face pipeline tags
    pipeline_tags = {
        "text-classification": "text-classification",
        "token-classification": "token-classification",
        "question-answering": "question-answering",
        "sentence-similarity": "text-classification",
    }
    pipeline_tag = pipeline_tags.get(task_name.lower().replace(" ", "-"), "feature-extraction")
    tag = pipeline_tag or pipeline_tags.get(final_task.lower().replace(" ", "-"), "text-classification")

    if not intended_use:
        intended_use = f"This model is fine-tuned for {final_task} downstream applications in production and research."

    if not dataset_description:
        dataset_description = f"Fine-tuned on the `{dataset_name}` benchmark with standardized train/validation splits."

    if not usage_snippet:
        if tag == "token-classification":
            usage_snippet = 'res = nlp_pipe("Sundar Pichai is the CEO of Google in Mountain View, California.")\nprint(res)'
        elif tag == "question-answering":
            usage_snippet = 'res = nlp_pipe(question="Where is Google located?", context="Google is headquartered in Mountain View, California.")\nprint(res)'
        else:
            usage_snippet = 'res = nlp_pipe("The results of this fine-tuning experiment are exceptionally promising!")\nprint(res)'

    # Format hyperparameters into markdown table rows
    hp_rows = "\n".join([f"| **`{k}`** | `{v}` |" for k, v in hyperparameters.items()])
    hp_rows = "\n".join([f"| **`{k}`** | `{v}` |" for k, v in hyperparameters.items()]) if hyperparameters else "| None | - |"

    # Format metrics into markdown table rows
    metric_rows = "\n".join([f"| **`{k}`** | `{v if isinstance(v, str) else round(v, 4)}` |" for k, v in metrics.items()])
    metric_rows = "\n".join([f"| **`{k}`** | `{v if isinstance(v, str) else round(v, 4)}` |" for k, v in metrics.items()]) if metrics else "| None | - |"

    metric_keys_yaml = "\n".join([f"- {k.lower()}" for k in metrics.keys()]) if metrics else "- f1\n- accuracy"

    card_content = f"""---
language:
- en
license: apache-2.0
base_model: {base_model}
base_model: {final_base_model}
tags:
- bert
- fine-tuned
- {pipeline_tag}
- {tag}
datasets:
- {dataset_name}
metrics:
{chr(10).join([f"- {k.lower()}" for k in metrics.keys()])}
pipeline_tag: {pipeline_tag}
{metric_keys_yaml}
pipeline_tag: {tag}
---

# 🚀 Fine-Tuned BERT for {task_name.title()}
# 🚀 Fine-Tuned BERT for {final_task.title()}

This model is an optimized, fine-tuned checkpoint of **`{base_model}`** specialized for **{task_name.title()}**, trained on the industry-standard **`{dataset_name}`** benchmark.
This model is an optimized, fine-tuned checkpoint of **`{final_base_model}`** specialized for **{final_task.title()}**, trained on the **`{dataset_name}`** dataset.

---

## 📌 Model Description & Intended Use
- **Primary Task**: {task_name.title()}
- **Base Architecture**: `{base_model}` (12 Transformer Encoder Layers, 768 Hidden Dim, 12 Attention Heads)
- **Primary Task**: {final_task.title()}
- **Base Architecture**: `{final_base_model}` (12 Transformer Encoder Layers, 768 Hidden Dim, 12 Attention Heads)
- **Intended Use**: {intended_use}

---

## 📊 Dataset Details
- **Dataset Name**: `{dataset_name}`
- **Overview**: {dataset_description}

---

## ⚙️ Training Hyperparameters
The model was fine-tuned using modern optimization best practices (Mixed Precision, Gradient Clipping, Differential Weight Decay):
The model was fine-tuned using modern optimization best practices (Mixed Precision, Gradient Accumulation, Differential Weight Decay):

| Hyperparameter | Value |
| :--- | :--- |
{hp_rows}

---

## 📈 Evaluation Results & Benchmarks
Evaluation results on the held-out test split:
Evaluation results on the held-out validation/test split:

| Metric | Result |
| :--- | :--- |
{metric_rows}

---

## 💻 Quickstart & Inference Usage

### Method 1: Using Hugging Face `pipeline` (Recommended)
```python
from transformers import pipeline

# Load pipeline directly from Hugging Face Hub
nlp_pipe = pipeline("{pipeline_tag}", model="{repo_id}")
nlp_pipe = pipeline("{tag}", model="{final_repo}")

{usage_snippet}
```

### Method 2: Native PyTorch Inference
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForSequenceClassification.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{final_repo}")
model = AutoModel.from_pretrained("{final_repo}")
model.eval()

# Encode sample input
inputs = tokenizer("Sample input text", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
```

---

## 📄 License
This model checkpoint is released under the **Apache-2.0** license.
"""

    if output_filepath:
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(card_content)
        print(f"[Model Card] Saved Model Card to: {output_filepath}")

    return card_content


def push_model_and_card_to_hub(
    model,
    tokenizer,
    repo_name: str,
    model_card_content: str,
    token: Optional[str] = None,
    private: bool = False,
    *args,
    **kwargs
) -> str:
    """
    Creates repository, pushes model weights, tokenizer, and README.md model card.
    Polymorphic Hub Uploader:
    Signature A:
        push_model_and_card_to_hub(model, tokenizer, repo_name, model_card_content, token=None, private=False)
    Signature B:
        push_model_and_card_to_hub(repo_id="username/model", local_dir="./checkpoint", readme_content="...", token=None)
    """
    username, token, api = get_authenticated_api(token)
    repo_id = f"{username}/{repo_name}"
    token = kwargs.get("token", None)
    private = kwargs.get("private", False)

    print(f"[Hub Push] Preparing repository: https://huggingface.co/{repo_id}")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private, token=token)
    # Detect calling convention
    if len(args) >= 4 and hasattr(args[0], "save_pretrained"):
        model, tokenizer, repo_name, model_card_content = args[0], args[1], args[2], args[3]
        username, token, api = get_authenticated_api(token)
        repo_id = f"{username}/{repo_name}" if "/" not in repo_name else repo_name

    # Push model weights and configuration
    print(f"[Hub Push] Uploading model weights...")
    model.push_to_hub(repo_id, token=token)
        print(f"[Hub Push] Preparing repository: https://huggingface.co/{repo_id}")
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private, token=token)

    # Push tokenizer
    print(f"[Hub Push] Uploading tokenizer...")
    tokenizer.push_to_hub(repo_id, token=token)
        print(f"[Hub Push] Uploading model weights...")
        model.push_to_hub(repo_id, token=token)

    # Upload README.md Model Card
    temp_card_path = f"/tmp/{repo_name}_README.md"
    with open(temp_card_path, "w", encoding="utf-8") as f:
        f.write(model_card_content)
        print(f"[Hub Push] Uploading tokenizer...")
        tokenizer.push_to_hub(repo_id, token=token)

    print(f"[Hub Push] Uploading Model Card (README.md)...")
    upload_file(
        path_or_fileobj=temp_card_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token,
        commit_message="Add dynamically generated Model Card"
    )
        temp_card_path = f"/tmp/{repo_name.replace('/', '_')}_README.md"
        with open(temp_card_path, "w", encoding="utf-8") as f:
            f.write(model_card_content)

    if os.path.exists(temp_card_path):
        os.remove(temp_card_path)
        upload_file(
            path_or_fileobj=temp_card_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
            commit_message="Add dynamically generated Model Card"
        )
        if os.path.exists(temp_card_path):
            os.remove(temp_card_path)

    full_url = f"https://huggingface.co/{repo_id}"
    print(f"[Hub Push] Successfully published to: {full_url}")
    return full_url
        full_url = f"https://huggingface.co/{repo_id}"
        print(f"[Hub Push] Successfully published to: {full_url}")
        return full_url

    else:
        # Signature B (repo_id, local_dir, readme_content)
        repo_id = kwargs.get("repo_id") or (args[0] if len(args) > 0 else None)
        local_dir = kwargs.get("local_dir") or (args[1] if len(args) > 1 else None)
        readme_content = kwargs.get("readme_content") or kwargs.get("model_card_content") or (args[2] if len(args) > 2 else None)

        username, token, api = get_authenticated_api(token)
        if "/" not in repo_id:
            repo_id = f"{username}/{repo_id}"

        print(f"[Hub Push] Preparing repository: https://huggingface.co/{repo_id}")
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private, token=token)

        # Write README.md into local_dir if provided
        if readme_content and local_dir and os.path.exists(local_dir):
            with open(os.path.join(local_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write(readme_content)

        print(f"[Hub Push] Uploading folder {local_dir} to {repo_id}...")
        upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            token=token,
            commit_message="Upload fine-tuned weights and model card"
        )
        full_url = f"https://huggingface.co/{repo_id}"
        print(f"[Hub Push] Successfully published to: {full_url}")
        return full_url
