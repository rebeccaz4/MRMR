from __future__ import annotations

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
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

class MedicalQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MedicalQARetrieval",
        description="The dataset consists 2048 medical question and answer pairs.",
        reference="https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4",
        dataset={
            "path": "mteb/medical_qa",
            "revision": "ae763399273d8b20506b80cf6f6f9a31a6a2b238",
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
        license="cc0-1.0",
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