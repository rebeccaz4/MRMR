import os
# Set GPU before importing any CUDA-related modules
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import mteb

# Additional GPU setting methods for redundancy
if torch.cuda.is_available():
    # Since we set CUDA_VISIBLE_DEVICES="3", GPU 3 will now be device 0 in PyTorch's view
    torch.cuda.set_device(0)  
    print(f"Using GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name()}")
    print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("CUDA not available!")

benchmark = mteb.get_benchmark("MRMR_text")
print(type(benchmark))
print(benchmark.name)
model_name = "Qwen/Qwen3-Embedding-8B"

model = mteb.get_model(model_name=model_name)

# Clear any cached memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Use very small batch size and chunking to avoid memory issues
encode_kwargs = {
    "batch_size": 1,  # Smallest possible batch size
    "show_progress_bar": True,
    "convert_to_tensor": True
}

evaluation = mteb.MTEB(tasks=benchmark)
# results = evaluation.run(model, output_folder="/home/siyue/Projects/rbenchmark_try")
results = evaluation.run(
        model,
        encode_kwargs=encode_kwargs,
        text_vision=False,
        is_clip=False,
        overwrite_results=True,
        save_predictions=True,
        text_length="original",
        output_folder="/home/siyue/Projects/results_benchmark_text",
    )
