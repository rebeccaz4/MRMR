import os
import torch
import mteb


benchmark = mteb.get_benchmark("MRMR_multimodal")
model_name = "OpenSearch-AI/Ops-MM-embedding-v1-7B"
model = mteb.get_model(model_name=model_name)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

encode_kwargs = {
    "batch_size": 1,  
    "show_progress_bar": True,
    "convert_to_tensor": True
}

evaluation = mteb.MTEB(tasks=benchmark)
results = evaluation.run(
        model,
        encode_kwargs=encode_kwargs,
        text_vision=False,
        is_clip=False,
        overwrite_results=True,
        save_predictions=True,
        text_length="original",
    )
