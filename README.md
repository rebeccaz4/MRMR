# MRMR

**A REALISTIC AND EXPERT-LEVEL MULTIDISCIPLINARY BENCHMARK FOR REASONING-INTENSIVE MULTIMODAL RETRIEVAL.**

![Hugging Face](https://huggingface.co/MMB-25)


## 📘 Introduction

MRMR includes 1,502 expert-annotated examples, covering 23 domains across 6 disciplines. It is specifically designed to assess multimodal retrieval models in expert-level, reasoning-intensive tasks. Notably, we originally introduce the Contradiction Retrieval task in the multimodal setting, which requires retrieving documents that conflict with the user query and features deeper logical reasoning.
Our benchmark extends MTEB by adding 5 new tasks. Additionally, to obtain more detailed and accurate test results, we have introduced three parameters: split_corpus, split_results, and category_map. When split_corpus is set to true, we limit each query to be tested only against its corresponding four corpora. When split_results is true and category_map is not empty, we can not only obtain the overall test results for the task but also get the results for specific subcategories within the task.


## ⚙️ Installation

You can install mteb simply using pip. For more on installation please see the [documentation](https://embeddings-benchmark.github.io/mteb/installation/).

```bash
cd mteb
pip install -e .
```

## 💡 Example Usage

Below we present a simple use-case example. For more information, see the [documentation](https://embeddings-benchmark.github.io/mteb/).

```python
import mteb

benchmark = mteb.get_benchmark("MRMR_multimodal")

model_name = "OpenSearch-AI/Ops-MM-embedding-v1-7B"

model = mteb.get_model(model_name=model_name)

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
    )
