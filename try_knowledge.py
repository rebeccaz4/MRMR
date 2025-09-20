import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import mteb

from datasets import load_dataset

category_map = {
    "Art": "Art", "Art_Theory": "Art", "Design": "Art", "Music": "Art",
    "Sociology": "Humanities", "Literature": "Humanities", "History": "Humanities", "Psychology": "Humanities",
    "Clinical_Medicine": "Medicine", "Diagnostics_and_Laboratory_Medicine": "Medicine", "Basic_Medical_Science": "Medicine", "Pharmacy": "Medicine",
    "Biology": "Science", "Chemistry": "Science", "Geography": "Science", "Agriculture": "Science",
    }

tasks = mteb.get_tasks(tasks=["KnowledgeAny2AnyRetrieval"])
# model_name = "TIGER-Lab/VLM2Vec-Full"
model_name = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"

model = mteb.get_model(model_name=model_name)

encode_kwargs = {
    "batch_size": 1,
    "show_progress_bar": True,
    "convert_to_tensor": True,
}


evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(
    model, 
    encode_kwargs=encode_kwargs, 
    split_corpus=False, 
    split_results=True, 
    category_map=category_map, 
    text_vision=False,
    overwrite_results=True, 
    ave_predictions=True,
    output_folder="/home/siyue/Projects/exp_results"
    )