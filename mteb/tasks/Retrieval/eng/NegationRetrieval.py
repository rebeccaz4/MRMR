from __future__ import annotations

from datasets import load_dataset
from typing import Dict, List, Union

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

import json

from typing import List, Dict, Union
from datasets import load_dataset
import json

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
    """
    加载 NegationRetrieval 所需的纯文本数据。
    返回:
        corpus  : {split: {doc_id: {"text": str}}}
        queries : {split: {query_id: str 或 list[str]}}
        qrels   : {split: {query_id: {doc_id: int}}}
    """
    corpus_all:  Dict[str, Dict[str, Dict[str, str]]]        = {}
    queries_all: Dict[str, Dict[str, Union[str, List[str]]]] = {}
    qrels_all:   Dict[str, Dict[str, Dict[str, int]]]        = {}

    # 读取 captions 文件，建立 {query-<item_id>: caption} 映射
    caption_path = "/home/siyue/Projects/mmb_pipeline/AllCaption/all_captions_negation.json"
    with open(caption_path, "r", encoding="utf-8") as f:
        cap_list = json.load(f)
    captions_map = {
        f"query-{item['item_id']}": item["captions"]["image"]["caption"]
        for item in cap_list
    }

    for split in splits:
        split_corpus: Dict[str, Dict[str, str]] = {}
        split_queries: Dict[str, Union[str, List[str]]] = {}
        split_qrels: Dict[str, Dict[str, int]] = {}

        # ---- queries ----
        query_ds = load_dataset(
            path, name="query", split=split,
            cache_dir=cache_dir, revision=revision
        )
        for row in query_ds:
            qid = f"query-{split}-{row['id']}"
            cap_key = f"query-{row['id']}"
            text = captions_map.get(cap_key, "")
            split_queries[qid] = f"<{text}>"

        # ---- corpus ----
        corpus_ds = load_dataset(
            path, name="corpus", split=split,
            cache_dir=cache_dir, revision=revision
        )
        for row in corpus_ds:
            did = f"corpus-{split}-{row['id']}"
            split_corpus[did] = {"text": row["text"]}

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



class NegationRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NegationRetrieval",
        description="The dataset consists 200 images and 800 discriptions.",
        reference="https://example.com",
        dataset={
            "path": "MMB-25/negation",
            "revision": "main",
        },
        type="Retrieval",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
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
            "query": "Given an image caption, retrieve descriptions that have contradictory information with the image caption."
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