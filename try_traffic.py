import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import mteb

from datasets import load_dataset
"""
第三类

tasks = mteb.get_tasks(tasks=["TrafficIT2AnyRetrieval"])
model_name = "TIGER-Lab/VLM2Vec-LoRA"

model = mteb.get_model(model_name=model_name)

evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, split_corpus=False, split_results=False, overwrite_results=True, save_predictions=True)
"""


tasks = mteb.get_tasks(tasks=["TrafficIT2AnyRetrieval"])
model_name = "Qwen/Qwen3-Embedding-8B"

model = mteb.get_model(model_name=model_name)

evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, split_corpus=False, split_results=False, text_only=True, overwrite_results=True, save_predictions=True)