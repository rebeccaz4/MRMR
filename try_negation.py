import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import mteb

from datasets import load_dataset



# tasks = mteb.get_tasks(tasks=["NegationI2TRetrieval"])
tasks = mteb.get_tasks(tasks=["NegationRetrieval"])

# model_name = "TIGER-Lab/VLM2Vec-LoRA"
model_name = "Qwen/Qwen3-Embedding-8B"
# model_name = "nvidia/NV-Embed-v2"

model = mteb.get_model(model_name=model_name)

evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, split_corpus=True, split_results=False, overwrite_results=True, save_predictions=True)

