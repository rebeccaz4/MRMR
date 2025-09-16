import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import mteb

from datasets import load_dataset



tasks = mteb.get_tasks(tasks=["NegationI2TRetrieval"])
# model_name = "TIGER-Lab/VLM2Vec-LoRA"
model_name = "Qwen/Qwen3-Embedding-8B"

model = mteb.get_model(model_name=model_name)

evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, split_corpus=True, split_results=False, text_only=True, overwrite_results=True, save_predictions=True)

