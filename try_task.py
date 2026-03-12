import mteb
from datasets import load_dataset

def main():
    tasks = mteb.get_tasks(tasks=["KnowledgeAny2AnyRetrieval"])
    model_name = "OpenSearch-AI/Ops-MM-embedding-v1-7B"
    model = mteb.get_model(model_name=model_name, device="cuda:1")

    encode_kwargs = {
        "batch_size": 1,  
        "show_progress_bar": True,
        "convert_to_tensor": True,
    }

    evaluation = mteb.MTEB(tasks=tasks)
   
    results = evaluation.run(
        model,
        encode_kwargs=encode_kwargs,
        text_vision=False,
        is_clip=False,
        overwrite_results=True,
        save_predictions=True,
    )


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  
    main()
