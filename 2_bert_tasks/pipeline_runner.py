#!/usr/bin/env python3
"""
CLI entrypoint wrapper for BERT Multi-Task Suite.
Enables running directly via:
    python 2_bert_tasks/pipeline_runner.py --task all --epochs 3
"""
import sys
from pathlib import Path

# Add multi_task_bert to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR / "multi_task_bert"))

from multi_task_bert.pipeline_runner import main

if __name__ == "__main__":
    main()

