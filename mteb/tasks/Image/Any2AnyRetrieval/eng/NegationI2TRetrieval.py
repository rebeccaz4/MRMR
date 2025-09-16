from __future__ import annotations

from datasets import load_dataset, Dataset

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

import json

# query是image，corpus是text
def _load_data(
    path: str,
    splits: list[str],
    cache_dir: str | None = None,
    revision: str | None = None,
    text_only: bool = False,
):
    corpus = {}
    query = {}
    qrels = {}

    # 如果需要纯文本模式，则直接在函数里读取 captions.json
    captions_map = {}
    if text_only:
        caption_path = "/home/siyue/Projects/mmb_pipeline/AllCaption/all_captions_negation.json"
        with open(caption_path, "r", encoding="utf-8") as f:
            cap_list = json.load(f)
        for item in cap_list:
            subset = item["subset"]          # "query"
            item_id = item["item_id"]
            cap = item["captions"]["image"]["caption"]
            captions_map[f"{subset}-{item_id}"] = cap

    for split in splits:
        # ---- query ----
        query_ds = load_dataset(
            path,
            "query",
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )

        def map_query(x):
            new_text = x["text"]
            if text_only:
                cap_key = f"query-{x['id']}"
                if cap_key in captions_map:
                    new_text = f"<{captions_map[cap_key]}>"
            return {
                "id": f"query-{split}-{x['id']}",
                "text": new_text,
                "image": None if text_only else x["image"],
                "modality": "text" if text_only else x["modality"],
            }

        query[split] = query_ds.map(map_query)

        # ---- corpus ----
        corpus_ds = load_dataset(
            path,
            "corpus",
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )
        
        def map_corpus(x):
             return {
                        "id": f"corpus-{split}-{x['id']}",   # 假设 corpus 的 id 字段叫 "id"
                        "text": x["text"],
                        "image": x["image"],                 # 保留 image 字段
                        "modality": x["modality"],           # 原样保留
                    } 

        corpus[split] = corpus_ds.map(map_corpus)

        # ---- qrels ----
        qrels_ds = load_dataset(
            path,
            "qrels",
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )
        qrels[split] = {}
        for row in qrels_ds:
            qid = f"query-{split}-{row['query-id']}"
            did = f"corpus-{split}-{row['corpus-id']}"
            qrels[split].setdefault(qid, {})[did] = int(row["score"])

    return corpus, query, qrels



class NegationI2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NegationI2TRetrieval",
        description="Retrieval of textual rule descriptions for design-related images.",
        reference="https://huggingface.co/datasets/MMB-25/negation",
        dataset={
            "path": "MMB-25/negation",
            "revision": "main",  
        },
        type="Any2AnyRetrieval",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        task_subtypes=["Image Text Retrieval"],
        dialect=[],
        # modalities=["image", "text"],
        modalities=["text"],
        sample_creation="created",
        bibtex_citation=r"""
@misc{design_dataset2024,
  title={MMB-25: Design Rule Violation Retrieval Dataset},
  author={Your Name or Org},
  year={2024},
  howpublished={\url{https://huggingface.co/datasets/MMB-25/design}},
}
""",
        prompt={"query": "Given the image caption, retrieve the text has contradictory information to the query."},
        descriptive_stats={
            "n_samples": {"test":28},  # 请填入真实样本数
            "avg_character_length": {
                "test": {
                    "average_document_length": 300.0,  # 乱填的
                    "average_query_length": 0.0,       # image 无长度
                    "num_documents": 700,            # 请替换为真实值
                    "num_queries": 28,              # 请替换为真实值
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )
    


    def load_data(self, **kwargs):
        text_only = kwargs.get("text_only", False)
        print(text_only)
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
            text_only=text_only,
        )
        self.data_loaded = True
        
#load-data是因为hugging face上的数据格式和MTEB要的有差距
        
