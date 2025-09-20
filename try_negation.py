import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  #在跑nv-embed-v2的时候不能限制环境

import mteb

from datasets import load_dataset

"""
import torch, os
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
"""

tasks = mteb.get_tasks(tasks=["NegationI2TRetrieval"])
# tasks = mteb.get_tasks(tasks=["NegationRetrieval"])

# mutimodal model
# model_name = "TIGER-Lab/VLM2Vec-LoRA"
model_name = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"
# model_name = "royokong/e5-v"
# model_name = "BAAI/bge-visualized-m3"
# model_name = "vidore/colpali-v1.3"
# model_name = "nvidia/MM-Embed"

# text model
# model_name = "Qwen/Qwen3-Embedding-8B"
# model_name = "nvidia/NV-Embed-v2"
# model_name = "BAAI/bge-m3"

# model_name = "OpenSearch-AI/Ops-MM-embedding-v1-7B"

model = mteb.get_model(model_name=model_name)

encode_kwargs = {
    "batch_size": 8,  # Smallest possible batch size
    "show_progress_bar": True,
    "convert_to_tensor": True,
}

evaluation = mteb.MTEB(tasks=tasks)
# results = evaluation.run(model, encode_kwargs=encode_kwargs, split_corpus=True, split_results=False, overwrite_results=True, save_predictions=True)
results = evaluation.run(model, encode_kwargs=encode_kwargs, split_corpus=True, split_results=False, text_vision=False, overwrite_results=True, save_predictions=True)
