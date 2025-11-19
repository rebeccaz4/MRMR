from __future__ import annotations

from datasets import load_dataset, Dataset

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

import json

# query是单图，corpus是多图

def _load_data(
    path: str,
    splits: list[str],
    instruction: str = None,
    cache_dir: str | None = None,
    revision: str | None = None,
    text_vision: bool = False,
    is_clip: bool = False,
):
    corpus = {}
    query = {}
    qrels = {}

    # 如果需要纯文本模式，则直接在函数里读取 captions.json
    captions_map = {}
    if text_vision:
        caption_path = "/home/siyue/Projects/mmb_pipeline/AllCaption/all_captions_traffic.json"
        with open(caption_path, "r", encoding="utf-8") as f:
            cap_list = json.load(f)
        for item in cap_list:
            subset = item["subset"]          # "corpus" 或 "query"
            item_id = item["item_id"]
            if subset == "query":
                # query 保持原来方式
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
            if text_vision and "<image>" in new_text:
                cap_key = f"query-{x['id']}"
                if cap_key in captions_map:
                    new_text = new_text.replace("<image>", f"<{captions_map[cap_key]}>")
            if is_clip:
                new_text = f"{instruction}\n{new_text}"
            return {
                "id": f"query-{split}-{x['id']}",
                "text": new_text,
                "image": None if text_vision else x["image"],
                "modality": "text" if text_vision else x["modality"],
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
                "id": f"corpus-{split}-{x['id']}",
                "text": None if text_vision else x["text"],
                "image": x["vision"] if text_vision else x["image"],
                "modality": "image" if text_vision else x["modality"],
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
            
    print(
        f"[{split}] query 中 id 数量: {len(set(query[split]['id']))}, "
        f"qrels 中 query-id 数量: {len(set(qrels[split].keys()))}"
    )

    return corpus, query, qrels



class TrafficIT2AnyRetrieval(AbsTaskAny2AnyRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.split_results = False
        
    metadata = TaskMetadata(
        name="TrafficIT2AnyRetrieval",
        description="Retrieval of textual rule descriptions for design-related images.",
        reference="https://huggingface.co/datasets/MRMRbenchmark/traffic",
        dataset={
            "path": "MRMRbenchmark/traffic",
            "revision": "main",  
        },
        type="Any2AnyRetrieval",
        category="it2it",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        task_subtypes=["Image Text Retrieval"],
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@misc{zhang2025mrmrrealisticexpertlevelmultidisciplinary,
      title={MRMR: A Realistic and Expert-Level Multidisciplinary Benchmark for Reasoning-Intensive Multimodal Retrieval}, 
      author={Siyue Zhang and Yuan Gao and Xiao Zhou and Yilun Zhao and Tingyu Song and Arman Cohan and Anh Tuan Luu and Chen Zhao},
      year={2025},
      eprint={2510.09510},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2510.09510}, 
}
""",
        prompt={"query": "Given a traffic case, retrieve the driving rule documents that it violates."},
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
        text_vision = kwargs.get("text_vision", False)
        print(text_vision)
        
        is_clip = kwargs.get("is_clip", False)
        print(is_clip)
        
        if text_vision:
            self.metadata.prompt["query"] = (
                "Given a traffic case description, retrieve the driving rule documents that it violates."
            )
        else:
            self.metadata.prompt["query"] = (
                "Given a traffic case, retrieve the driving rule documents that it violates."
            )
            
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
            text_vision=text_vision,
            is_clip=is_clip,
            instruction=self.metadata.prompt["query"],
        )
        self.data_loaded = True
        
#load-data是因为hugging face上的数据格式和MTEB要的有差距