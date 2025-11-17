from __future__ import annotations

from datasets import load_dataset
from typing import Dict, List, Union

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

import json

from typing import List, Dict, Union
from datasets import load_dataset

# query单图，corpus多图
def _load_data(
    path: str,
    splits: List[str],
    cache_dir: str | None = None,
    revision: str | None = None,
) -> tuple[
    Dict[str, Dict[str, Dict[str, str]]],          # corpus[split][doc_id] = {"text": ...}
    Dict[str, Dict[str, Union[str, List[str]]]],   # queries[split][query_id] = str | list[str]
    Dict[str, Dict[str, Dict[str, int]]]           # qrels[split][query_id][doc_id] = int
]:

    corpus_all:  Dict[str, Dict[str, Dict[str, str]]]        = {}
    queries_all: Dict[str, Dict[str, Union[str, List[str]]]] = {}
    qrels_all:   Dict[str, Dict[str, Dict[str, int]]]        = {}
    
    captions_map = {}

    caption_path = "/home/siyue/Projects/mmb_pipeline/AllCaption/all_captions_traffic.json"
    with open(caption_path, "r", encoding="utf-8") as f:
        cap_list = json.load(f)
    for item in cap_list:
        subset = item["subset"]          # "corpus" 或 "query"
        item_id = item["item_id"]
        if subset == "query":
            # query：只有单张图片
            cap = item["captions"]["image"]["caption"]
            captions_map[f"{subset}-{item_id}"] = cap
        elif subset == "corpus":
            # corpus：多图，需要遍历所有图片键
            for img_key, img_info in item["captions"].items():
                cap = img_info["caption"]
                # key 形如 corpus-<id>-<img_key>
                captions_map[f"{subset}-{item_id}-{img_key}"] = cap



    for split in splits:
        split_corpus: Dict[str, Dict[str, str]] = {}
        split_queries: Dict[str, Union[str, List[str]]] = {}
        split_qrels: Dict[str, Dict[str, int]] = {}

        # ---- queries ----
        query_ds = load_dataset(
            path, name="query", split=split,
            cache_dir=cache_dir, revision=revision
        )
        
        def map_query(x):
            new_text = x["text"]
            if "<image>" in new_text:
                cap_key = f"query-{x['id']}"
                if cap_key in captions_map:
                    new_text = new_text.replace("<image>", f"<{captions_map[cap_key]}>")
            return new_text
        
        for row in query_ds:
            qid = f"query-{split}-{row['id']}"
            text = map_query(row)
            split_queries[qid] = f"<{text}>"

        # ---- corpus ----
        corpus_ds = load_dataset(
            path, name="corpus", split=split,
            cache_dir=cache_dir, revision=revision
        )
        
        def map_corpus(x):
            new_text = x["text"]
            for key, cap in captions_map.items():
                if key.startswith(f"corpus-{x['id']}-"):
                    img_key = key.split("-")[-1]
                    placeholder = f"<{img_key.replace('_', ' ')}>"
                    if placeholder in new_text:
                        new_text = new_text.replace(placeholder, f"<{cap}>")
            return new_text
        
        
        for row in corpus_ds:
            did = f"corpus-{split}-{row['id']}"
            text = map_corpus(row)
            split_corpus[did] = {"text": text}
           
        # ---- qrels ----
        qrels_ds = load_dataset(
            path, name="qrels", split=split,
            cache_dir=cache_dir, revision=revision
        )
        for row in qrels_ds:
            qid = f"query-{split}-{row['query-id']}"
            did = f"corpus-{split}-{row['corpus-id']}"
            split_qrels.setdefault(qid, {})[did] = int(row["score"])

        # 将每个 split 的结果存入总字典
        corpus_all[split]  = split_corpus
        queries_all[split] = split_queries
        qrels_all[split]   = split_qrels

    return corpus_all, queries_all, qrels_all




class TrafficRetrieval(AbsTaskRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.split_results = False

    metadata = TaskMetadata(
        name="TrafficRetrieval",
        description="The dataset consists 200 images and 800 discriptions.",
        reference="https://huggingface.co/datasets/MMB-25/traffic",
        dataset={
            "path": "MMB-25/traffic",
            "revision": "main",
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2017-01-01", "2019-12-31"),  # best guess,
        domains=["Medical", "Written"],
        task_subtypes=["Article retrieval"],
        license="openrail",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
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
        prompt={
            "query": "Given a traffic case description, retrieve the driving rule documents that it violates."
        },
    )
    
    def load_data(self, **kwargs):
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )
        self.data_loaded = True