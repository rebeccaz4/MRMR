import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import mteb

from datasets import load_dataset


category_map = {
    "Accounting": "Business", "Finance": "Business", "Manage": "Business", "Marketing": "Business", "Economics": "Business", "Marketing": "Business", 
    "Electronics": "Engineering", "Computer_Science": "Engineering", "Mechanical_Engineering": "Engineering", "Materials": "Engineering", "Architecture_and_Engineering": "Engineering",
    "Math": "Math",
    "Physics": "Physics", "Energy_and_Power": "Physics"
    }

"""
import torch, os
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
"""

def main():

    tasks = mteb.get_tasks(tasks=["TheoremAny2AnyRetrieval"])
    # tasks = mteb.get_tasks(tasks=["TheoremRetrieval"])

    # mutimodal model
    # model_name = "TIGER-Lab/VLM2Vec-Full"
    # model_name = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"
    # model_name = "royokong/e5-v"
    # model_name = "BAAI/bge-visualized-m3"
    model_name = "vidore/colpali-v1.3"
    # model_name = "nvidia/MM-Embed"
    # model_name = "OpenSearch-AI/Ops-MM-embedding-v1-7B"

    # text model
    # model_name = "Qwen/Qwen3-Embedding-8B"
    # model_name = "nvidia/NV-Embed-v2"
    # model_name = "BAAI/bge-m3"  # bge 的 prompt 要在 beg_models.py 里改

    # model_name = "OpenSearch-AI/Ops-MM-embedding-v1-7B"

    # CLIP model
    # model_name = "QuanSun/EVA02-CLIP-L-14"
    # model_name = "jinaai/jina-clip-v2"
    # model_name = "google/siglip-base-patch16-384"
    # model_name = "laion/CLIP-ViT-g-14-laion2B-s34B-b88K"

    model = mteb.get_model(model_name=model_name, device="cuda:1")

    encode_kwargs = {
        "batch_size": 4,  # Smallest possible batch size
        "show_progress_bar": True,
        "convert_to_tensor": True,
    }

    evaluation = mteb.MTEB(tasks=tasks)
    # results = evaluation.run(model, encode_kwargs=encode_kwargs, split_corpus=False, split_results=True, category_map=category_map, overwrite_results=True, save_predictions=True, output_folder="/home/siyue/Projects/results_theorem_bge-m3")

    results = evaluation.run(
        model,
        encode_kwargs=encode_kwargs,
        split_corpus=False,
        split_results=True,
        text_vision=True,
        is_clip=False,
        overwrite_results=True,
        save_predictions=True,
        category_map=category_map,
        text_length="original",
        output_folder="/home/siyue/Projects/results_theorem_colpali",
    )


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Windows / spawn 模式多进程必须加
    main()
