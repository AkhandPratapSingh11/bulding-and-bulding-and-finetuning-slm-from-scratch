from multi_task_bert.tasks.task1_classification import train_classification
from multi_task_bert.tasks.task2_ner import train_ner
from multi_task_bert.tasks.task3_qa import train_qa
from multi_task_bert.tasks.task4_nli import train_nli

__all__ = [
    "train_classification",
    "train_ner",
    "train_qa",
    "train_nli"
]

