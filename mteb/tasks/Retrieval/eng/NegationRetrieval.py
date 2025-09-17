from __future__ import annotations

from datasets import load_dataset, Dataset

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

import json

# query是image，corpus是text，所以只用转化query就可以了
def _load_data(
    path: str,
    splits: list[str],
    cache_dir: str | None = None,
    revision: str | None = None,
):
    """
       仅加载纯文本版本的 corpus / query / qrels。
       query 的 text 来自 all_captions_negation.json
    """
    corpus, query, qrels = {}, {}, {}

    # 读取 captions.json，建立 {query-<item_id>: caption} 映射
    caption_path = "/home/siyue/Projects/mmb_pipeline/AllCaption/all_captions_negation.json"
    with open(caption_path, "r", encoding="utf-8") as f:
        cap_list = json.load(f)
    captions_map = {
        f"query-{item['item_id']}": item["captions"]["image"]["caption"]
        for item in cap_list
    }

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
            cap_key = f"query-{x['id']}"
            # 如果在 captions_map 中找不到对应 caption，可选择用空字符串或原始文本
            new_text = captions_map.get(cap_key, "")
            return {
                "id": f"query-{split}-{x['id']}",
                "text": f"<{new_text}>",  # 保持原始格式
                # "image": None,
                # "modality": "text",
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
                "text": x["text"],
                # "image": None,       
                # "modality": "text",  
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

class NegationRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="NegationRetrieval",
        description="The dataset consists 200 images and 800 discriptions.",
        reference="https://example.com",
        dataset={
            "path": "MMB-25/negation",
            "revision": " ",
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
            "query": "Given a description of an image, retrieve the contradictory document."
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