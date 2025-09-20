from __future__ import annotations

from datasets import load_dataset, concatenate_datasets
import json

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


def _load_data(
    path: str,
    splits: list[str],
    cache_dir: str | None = None,
    revision: str | None = None,
    text_vision: bool = False,
):
    from datasets import concatenate_datasets

    corpus = {}
    query = {}
    qrels = {}

    # 构建 captions_map
    captions_map = {}
    if text_vision:
        caption_path = "/home/siyue/Projects/mmb_pipeline/AllCaption/all_captions_knowledge.json"
        with open(caption_path, "r", encoding="utf-8") as f:
            cap_list = json.load(f)
        for item in cap_list:
            subset = item["subset"]          # "corpus" 或 "query"
            item_id = item["item_id"]
            for img_key, img_info in item["captions"].items():
                cap = img_info["caption"]
                captions_map[f"{subset}-{item_id}-{img_key}"] = cap  # key 加上 img_key

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
            if text_vision:
                # 遍历 captions_map 中属于当前 query 的 captions
                for key, cap in captions_map.items():
                    # key 格式: "query-{item_id}-{img_key}"
                    if key.startswith(f"query-{x['id']}-"):
                        img_key = key.split("-")[-1]  # 取 img_key，例如 "image_1"
                        placeholder = f"<{img_key.replace('_', ' ')}>"  # "<image 1>"
                        if placeholder in new_text:
                            new_text = new_text.replace(placeholder, f"<{cap}>")
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

        pin_p_ds = load_dataset(
            path,
            "pin_p",
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

        combined_ds = concatenate_datasets([corpus_ds, pin_p_ds])
        # combined_ds = concatenate_datasets([corpus_ds])

        corpus[split] = combined_ds.map(map_corpus)

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
            qid = f"query-{split}-{row['query_id']}"
            did = f"corpus-{split}-{row['corpus_id']}"
            qrels[split].setdefault(qid, {})[did] = int(row["score"])

    return corpus, query, qrels





class KnowledgeAny2AnyRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="KnowledgeAny2AnyRetrieval",
        description="Retrieval of knowledge to solve questions.",
        reference="https://huggingface.co/datasets/MMB-25/knowledge",
        dataset={
            "path": "MMB-25/knowledge",
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
@misc{design_dataset2024,
  title={MMB-25: Design Rule Violation Retrieval Dataset},
  author={Your Name or Org},
  year={2024},
  howpublished={\url{https://huggingface.co/datasets/MMB-25/design}},
}
""",   
        prompt={"query": "Retrieve relevant documents that help answer the question."},
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
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir="/data/siyue/",
            revision=self.metadata_dict["dataset"]["revision"],
            text_vision=text_vision,
        )
        
        # # ============================================================================
        # # DEBUG MODE: REMOVE THIS SECTION FOR PRODUCTION!
        # # This limits the dataset to a small subset for quick testing/debugging
        # # TODO: Remove or comment out this entire section before final evaluation
        # # ============================================================================
        # max_queries = 5   # DEBUG: Only use first 10 queries
        # max_corpus = 1000   # DEBUG: Only use first 100 corpus items
        
        # for split in self.metadata_dict["eval_splits"]:
        #     # DEBUG: Limit queries to top 10
        #     if len(self.queries[split]) > max_queries:
        #         # Get the first 10 queries
        #         limited_queries = self.queries[split].select(range(max_queries))
        #         self.queries[split] = limited_queries
                
        #         # Update relevant_docs to only include the limited queries
        #         limited_qrels = {}
        #         for i, query_data in enumerate(limited_queries):
        #             query_id = query_data["id"]
        #             if query_id in self.relevant_docs[split]:
        #                 limited_qrels[query_id] = self.relevant_docs[split][query_id]
        #         self.relevant_docs[split] = limited_qrels
            
        #     # DEBUG: Limit corpus to top 100
        #     if len(self.corpus[split]) > max_corpus:
        #         limited_corpus = self.corpus[split].select(range(max_corpus))
        #         self.corpus[split] = limited_corpus
                
        #         # Update relevant_docs to only include corpus items that still exist
        #         corpus_ids = set(item["id"] for item in limited_corpus)
        #         filtered_qrels = {}
        #         for query_id, doc_scores in self.relevant_docs[split].items():
        #             filtered_doc_scores = {doc_id: score for doc_id, score in doc_scores.items() 
        #                                  if doc_id in corpus_ids}
        #             if filtered_doc_scores:  # Only keep queries that have at least one relevant doc
        #                 filtered_qrels[query_id] = filtered_doc_scores
        #         self.relevant_docs[split] = filtered_qrels
        # # ============================================================================
        # # END DEBUG SECTION
        # # ============================================================================
        
        self.data_loaded = True
#load-data是因为hugging face上的数据格式和MTEB要的有差距