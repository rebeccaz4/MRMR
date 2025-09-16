import mteb

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3" # not working

model_name = "nvidia/NV-Embed-v2"
# model_name = "BAAI/bge-m3"

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

model = mteb.get_model(model_name) # if the model is not implemented in MTEB it will be eq. to SentenceTransformer(model_name)
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, encode_kwargs=encode_kwargs, corpus_chunk_size=1000)