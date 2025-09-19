import mteb

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" # not working

# model_name = "nvidia/NV-Embed-v2"
# model_name = "Qwen/Qwen3-Embedding-8B"
# model_name = "TIGER-Lab/VLM2Vec-LoRA"
# model_name = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"
# model_name = "OpenSearch-AI/Ops-MM-embedding-v1-7B"
# model_name = "laion/CLIP-ViT-g-14-laion2B-s34B-b88K"
# model_name = "Salesforce/blip2-opt-2.7b" # not working
# model_name = "microsoft/LLM2CLIP-Openai-L-14-224"
model_name = "google/siglip-large-patch16-384"

# # Debug: Check GPU availability
# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"Number of GPUs: {torch.cuda.device_count()}")
# print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# # Set current device to 1
# # torch.cuda.set_device(0)
# print(f"Current device: {torch.cuda.current_device()}")

encode_kwargs = {
    "batch_size": 1,  # Smallest possible batch size
    "show_progress_bar": True,
    "convert_to_tensor": True,
}

model = mteb.get_model(model_name, device="cuda:0") # if the model is not implemented in MTEB it will be eq. to SentenceTransformer(model_name)
# tasks = mteb.get_tasks(tasks=["NegationI2TRetrieval"])
tasks = mteb.get_tasks(tasks=["CUB200I2IRetrieval"])
# tasks = mteb.get_tasks(tasks=["VisualNewsI2TRetrieval"])

evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, encode_kwargs=encode_kwargs, split_corpus=True, split_results=False, text_vision=True, overwrite_results=True, save_predictions=True)
