import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import mteb

from datasets import load_dataset



tasks = mteb.get_tasks(tasks=["NegationI2TRetrieval"])
# tasks = mteb.get_tasks(tasks=["NegationRetrieval"])

# mutimodal model
# model_name = "TIGER-Lab/VLM2Vec-LoRA"
# model_name = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"
model_name = "royokong/e5-v"

# text model
# model_name = "Qwen/Qwen3-Embedding-8B"
# model_name = "nvidia/NV-Embed-v2"

model = mteb.get_model(model_name=model_name)

encode_kwargs = {
    "batch_size": 8,  # Smallest possible batch size
    "show_progress_bar": True,
    "convert_to_tensor": True,
}

evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, encode_kwargs=encode_kwargs, split_corpus=True, split_results=False, overwrite_results=True, save_predictions=True)

