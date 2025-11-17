from __future__ import annotations

from datasets import load_dataset
from typing import Dict, List, Union

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

import json

from typing import List, Dict, Union
from datasets import load_dataset


def _load_data(

    path: str,
    splits: List[str],
    cache_dir: str | None = None,
    revision: str | None = None,
) -> tuple[
    Dict[str, Dict[str, Dict[str, str]]],          # corpus[split][doc_id] = {"text": ...}
    # Dict[str, Dict[str, Union[str, List[str]]]],   # queries[split][query_id] = str | list[str]
    Dict[str, Dict[str, Dict[str, str]]],          # queries[split][query_id] = {"text": ..., "id": ..., "category": ...}
    Dict[str, Dict[str, Dict[str, int]]]           # qrels[split][query_id][doc_id] = int
]:
    corpus_all:  Dict[str, Dict[str, Dict[str, str]]]        = {}
    # queries_all: Dict[str, Dict[str, Union[str, List[str]]]] = {}
    queries_all: Dict[str, Dict[str, Dict[str, str]]] = {}
    qrels_all:   Dict[str, Dict[str, Dict[str, int]]]        = {}
    
    captions_map = {}
    caption_path = "/home/siyue/Projects/mmb_pipeline/AllCaption/all_captions_theorem.json"
    with open(caption_path, "r", encoding="utf-8") as f:
        cap_list = json.load(f)
    for item in cap_list:
        subset = item["subset"]          # "corpus" 或 "query"
        item_id = item["item_id"]
        for img_key, img_info in item["captions"].items():
            cap = img_info["caption"]
            captions_map[f"{subset}-{item_id}-{img_key}"] = cap

    for split in splits:
        split_corpus: Dict[str, Dict[str, str]] = {}
        # split_queries: Dict[str, Union[str, List[str]]] = {}
        split_queries: Dict[str, Dict[str, str]] = {}
        split_qrels: Dict[str, Dict[str, int]] = {}

        # ---- queries ----
        query_ds = load_dataset(
            path, name="query", split=split,
            cache_dir=cache_dir, revision=revision
        )
        
        def map_query(x):
            new_text = x["text"]
            for key, cap in captions_map.items():
                if key.startswith(f"query-{x['id']}-"):
                    img_key = key.split("-")[-1]
                    placeholder = f"<{img_key.replace('_', ' ')}>"
                    if placeholder in new_text:
                        new_text = new_text.replace(placeholder, f"<{cap}>")
            return new_text
        
        for row in query_ds:
            qid = f"query-{split}-{row['id']}"
            text = map_query(row)
            # split_queries[qid] = f"<{text}>"  # OLD: just a string
            # NEW: store as dict for category support
            split_queries[qid] = {
                "text": f"<{text}>",
                "id": qid,
                "category": row["category"]
            }

        # ---- corpus ----
        corpus_ds = load_dataset(
            path, name="corpus", split=split,
            cache_dir=cache_dir, revision=revision
        )

        pin_p_ds = load_dataset(
            path, name="pin_p", split=split,
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
        
        for row in pin_p_ds:
            did = f"corpus-{split}-{row['id']}"
            text = map_corpus(row)
            split_corpus[did] = {"text": text}
            
           
        # ---- qrels ----
        qrels_ds = load_dataset(
            path, name="qrels", split=split,
            cache_dir=cache_dir, revision=revision
        )
        for row in qrels_ds:
            qid = f"query-{split}-{row['query_id']}"
            did = f"corpus-{split}-{row['corpus_id']}"
            split_qrels.setdefault(qid, {})[did] = int(row["score"])

        # 将每个 split 的结果存入总字典
        corpus_all[split]  = split_corpus
        queries_all[split] = split_queries
        qrels_all[split]   = split_qrels

    return corpus_all, queries_all, qrels_all

category_map = {
    "Accounting": "Business", "Finance": "Business", "Manage": "Business", "Marketing": "Business", "Economics": "Business", "Marketing": "Business", 
    "Electronics": "Engineering", "Computer_Science": "Engineering", "Mechanical_Engineering": "Engineering", "Materials": "Engineering", "Architecture_and_Engineering": "Engineering",
    "Math": "Math",
    "Physics": "Physics", "Energy_and_Power": "Physics"
    }



class TheoremRetrieval(AbsTaskRetrieval):
    # Helper: get list of query texts for encoding
    # Usage: texts = [q["text"] for q in self.queries[split].values()]
    # This avoids passing dicts to the model.encode method, which expects strings.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.split_results = True
        self.category_map = category_map

    metadata = TaskMetadata(
        name="TheoremRetrieval",
        description="The dataset consists 200 images and 800 discriptions.",
        reference="https://huggingface.co/datasets/MMB-25/theorem",
        dataset={
            "path": "MMB-25/theorem",
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
@article{BenAbacha-BMC-2019,
  author = {Asma, Ben Abacha and Dina, Demner{-}Fushman},
  journal = {{BMC} Bioinform.},
  number = {1},
  pages = {511:1--511:23},
  title = {A Question-Entailment Approach to Question Answering},
  url = {https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4},
  volume = {20},
  year = {2019},
}
""",
        prompt={
            "query": "Retrieve relevant theorems that are involved in solving the problem."
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