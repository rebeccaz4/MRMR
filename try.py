import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import mteb

tasks = mteb.get_tasks(tasks=["VidoreShiftProjectRetrieval"])
model_name = "TIGER-Lab/VLM2Vec-LoRA"

model = mteb.get_model(model_name=model_name)

evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model)