import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import mteb

from datasets import load_dataset

tasks = mteb.get_tasks(tasks=["DesignI2AnyRetrieval"])
model_name = "TIGER-Lab/VLM2Vec-LoRA"

model = mteb.get_model(model_name=model_name)

evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, overwrite_results=True)