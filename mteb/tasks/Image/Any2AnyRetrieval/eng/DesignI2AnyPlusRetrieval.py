from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


def _load_data(
    path: str,
    splits: list[str],
    cache_dir: str | None = None,
    revision: str | None = None,
):
    corpus = {}
    query = {}
    qrels = {}

    for split in splits:
        query_ds = load_dataset(
            path,
            "query",  # 注意是 "query" 不是 "queries"
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )
        query_ds = query_ds.map(
            lambda x: {
                "id": f"query-{split}-{x['id']}",  # 假设query的id字段叫"id"
                "text": x["text"],               # 假设文本字段叫"text"
                "image": x["image"],
                "modality": x["modality"],
            }
        )
        query[split] = query_ds

        corpus_ds = load_dataset(
            path,
            "corpus",
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )
        corpus_ds = corpus_ds.map(
            lambda x: {
                "id": f"corpus-{split}-{x['id']}",  # 假设corpus的id字段也叫"id"
                "text": x["text"],
                "image": x["image"],                # 保留image字段
                "modality": x["modality"],                  #如果是image，text不合法，会被直接跳过报错
            },
        )
        corpus[split] = corpus_ds

        qrels_ds = load_dataset(
            path,
            "qrels",
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )
        qrels[split] = {}
        for row in qrels_ds:
            qid = f"query-{split}-{row['query-id']}"   # qrels里字段名可能是query_id
            did = f"corpus-{split}-{row['corpus-id']}" # corpus_id
            if qid not in qrels[split]:
                qrels[split][qid] = {}
            qrels[split][qid][did] = int(row["score"])

    return corpus, query, qrels



class DesignI2AnyPlusRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="DesignI2AnyRetrieval",
        description="Retrieval of textual rule descriptions for design-related images.",
        reference="https://huggingface.co/datasets/MMB-25/design",
        dataset={
            "path": "MMB-25/design",
            "revision": "main",  
        },
        type="Any2AnyRetrieval",
        category="i2it",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        task_subtypes=["Image Text Retrieval"],
        dialect=[],
        modalities=["image", "text"],
        sample_creation="created",
        bibtex_citation=r"""
@misc{design_dataset2024,
  title={MMB-25: Design Rule Violation Retrieval Dataset},
  author={Your Name or Org},
  year={2024},
  howpublished={\url{https://huggingface.co/datasets/MMB-25/design}},
}
""",
        prompt={"query": " "},
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
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir=kwargs.get("cache_dir", None),
            revision=self.metadata_dict["dataset"]["revision"],
        )
        self.data_loaded = True
#load-data是因为hugging face上的数据格式和MTEB要的有差距
        
