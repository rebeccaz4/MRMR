from __future__ import annotations

from datasets import Dataset, load_dataset

from mteb.abstasks.Image.AbsTaskAny2AnyMultiChoice import AbsTaskAny2AnyMultiChoice
from mteb.abstasks.TaskMetadata import TaskMetadata

import json

# query是image，corpus是text, text_vision表示query用caption、corpus用vision
def _load_data(
    path: str,
    splits: list[str],
    instruction: str = None,
    cache_dir: str | None = None,
    revision: str | None = None,
    text_vision: bool = False,
    is_clip: bool = False,
):
    import json
    from datasets import load_dataset, Dataset

    corpus = {}
    query = {}
    qrels = {}
    relevant_docs = {}

    captions_map = {}
    if text_vision:
        caption_path = "/home/siyue/Projects/mmb_pipeline/AllCaption/all_captions_negation.json"
        with open(caption_path, "r", encoding="utf-8") as f:
            cap_list = json.load(f)
        for item in cap_list:
            subset = item["subset"]          # "query"
            item_id = item["item_id"]
            cap = item["captions"]["image"]["caption"]
            captions_map[f"{subset}-{item_id}"] = cap

    for split in splits:
        # ---- load query ----
        query_ds = load_dataset(
            path,
            "query",
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )

        def map_query(x):
            new_text = x["text"]
            if text_vision:
                x["modality"] = "text"
                cap_key = f"query-{x['id']}"
                if cap_key in captions_map:
                    new_text = f"<{captions_map[cap_key]}>"
            if is_clip:
                new_text = f"{instruction}"
                x["modality"] = "image,text"

            return {
                "id": f"query-{split}-{x['id']}",
                "text": new_text,
                "image": None if text_vision else x["image"],
                "modality": x["modality"],
            }

        query[split] = query_ds.map(map_query)

        # ---- load corpus ----
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

        # ---- load qrels ----
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
            qrels[split][qid] = {}
            qrels[split][qid][did] = int(row["score"])
            for i in range(2, 5):
                qrels[split][qid][f"corpus-{split}-{row['query-id']}_{i}"] = 0
    print("type(query)", type(query))
    print("type(corpus)", type(corpus))
    return corpus, query, qrels


class NegationI2TMultiChoice(AbsTaskAny2AnyMultiChoice):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.split_corpus = False
        self.split_results = False
        self.category_map = None

    metadata = TaskMetadata(
        name="NegationI2TMultiChoice",
        description="NegationI2TMultchoice",
        reference="https://arxiv.org/pdf/2406.16860",
        dataset={
            "path": "MMB-25/negation",
            "revision": "main",
        },
        type="Any2AnyMultiChoice",
        category="i2t",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2024-01-01", "2024-06-24"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        modalities=["image", "text"],
        sample_creation="found",
        bibtex_citation=r"""
@article{tong2024cambrian,
  author = {Tong, Shengbang and Brown, Ellis and Wu, Penghao and Woo, Sanghyun and Middepogu, Manoj and Akula, Sai Charitha and Yang, Jihan and Yang, Shusheng and Iyer, Adithya and Pan, Xichen and others},
  journal = {arXiv preprint arXiv:2406.16860},
  title = {Cambrian-1: A fully open, vision-centric exploration of multimodal llms},
  year = {2024},
}
""",
        prompt={"query": "Given an image caption, retrieve descriptions that have contradictory information with the image."},
        descriptive_stats={
            "n_samples": {"test": 419},
            "avg_character_length": {
                "test": {
                    "average_document_length": 0.0,
                    "average_query_length": 0.0,
                    "num_documents": 17,
                    "num_queries": 402,
                    "average_relevant_docs_per_query": 1,
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
                "Given an image caption, retrieve descriptions that have contradictory information with the image caption."
            )
        else:
            self.metadata.prompt["query"] = (
                "Given an image, retrieve descriptions that have contradictory information with the image."
            )
        print(self.metadata.prompt["query"])
        
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

