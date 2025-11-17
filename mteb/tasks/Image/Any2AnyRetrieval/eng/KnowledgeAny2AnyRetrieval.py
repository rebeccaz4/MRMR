from __future__ import annotations

from datasets import load_dataset, concatenate_datasets
import json
from PIL import Image

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


def resize_image(image, max_size=2000):
    """
    Resize image proportionally if width or height is larger than max_size.
    
    Args:
        image: PIL Image object
        max_size: Maximum allowed dimension (width or height)
    
    Returns:
        PIL Image object (resized if necessary)
    """
    if image is None:
        return None
    
    # Get current dimensions
    width, height = image.size
    
    # Check if resizing is needed
    if width <= max_size and height <= max_size:
        return image
    
    # Calculate scaling factor
    if width > height:
        scale_factor = max_size / width
    else:
        scale_factor = max_size / height
    
    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image


def _load_data(
    path: str,
    splits: list[str],
    instruction: str = None,
    cache_dir: str | None = None,
    revision: str | None = None,
    text_vision: bool = False,
    is_clip: bool = False,
    text_length: str = "original",
):
    """
    Load dataset splits and optionally replace/expand text using external JSONs.

    Args:
        path: HF dataset path
        splits: list of split names
        instruction: instruction string for CLIP-style prompting
        cache_dir, revision: passed to load_dataset
        text_vision: whether dataset is text-vision (affects modality and image/text fields)
        is_clip: if True, prepend instruction to texts
        text_length: one of "original", "2b", "72b" deciding where to take the text from
    """
    import os
    import json
    from datasets import concatenate_datasets

    corpus = {}
    query = {}
    qrels = {}

    # ----------------- helpers for expansion JSONs -----------------
    def _build_exp_map(p: str | None):
        """Load expansion JSON and build a lookup map keyed by "{subset}-{item_id}" and "{subset}-{row_id}".
        Returns an empty dict if file missing or load fails.
        """
        if not p:
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] failed to load expansion file {p}: {e}")
            return {}

        exp_map = {}
        # Normalize into a list of items
        if isinstance(data, dict):
            # single-item dict (one record) or dict of many records
            if "expansion" in data and ("item_id" in data or "row_id" in data):
                items = [data]
            else:
                # try to use dict values
                try:
                    items = list(data.values())
                except Exception:
                    items = [data]
        elif isinstance(data, list):
            items = data
        else:
            items = []

        for item in items:
            if not isinstance(item, dict):
                continue
            subset = item.get("subset")
            expansion = item.get("expansion") or {}
            item_id = item.get("item_id")
            if subset:
                if item_id is not None:
                    exp_map[f"{subset}-{item_id}"] = expansion
        return exp_map

    def _get_expansion_for_row(exp_map: dict, subset: str, x: dict):
        """Try multiple keys to find expansion for a dataset row x."""
        if not exp_map:
            return None
        # try explicit item_id fields
        item_id = x["id"]
        if item_id is not None:
            key = f"{subset}-{item_id}"
            if key in exp_map:
                return exp_map[key]
        return None

    # build expansion maps if requested or if path provided
    text2b_path = "/home/siyue/Projects/mmb_pipeline/Expansion/results/knowledge/qwen2-vl-2b_expansion_results.json"
    text72b_path = "/home/siyue/Projects/mmb_pipeline/Expansion/results/knowledge/qwen2.5-vl-72b_expansion_results.json"
    exp_map_2b = {}
    exp_map_72b = {}
    
    # default candidate filenames inside cache_dir or CWD if not provided
    if text_length == "2b":
        cand = text2b_path 
        exp_map_2b = _build_exp_map(cand)
    elif text_length == "72b":
        cand2 = text72b_path
        exp_map_72b = _build_exp_map(cand2)

    # ----------------- captions map (unchanged) -----------------
    captions_map = {}
    if text_vision:
        caption_path = "/home/siyue/Projects/mmb_pipeline/AllCaption/all_captions_knowledge.json"
        try:
            with open(caption_path, "r", encoding="utf-8") as f:
                cap_list = json.load(f)
            for item in cap_list:
                subset = item["subset"]
                item_id = item["item_id"]
                for img_key, img_info in item["captions"].items():
                    cap = img_info["caption"]
                    captions_map[f"{subset}-{item_id}-{img_key}"] = cap
        except Exception as e:
            print(f"[WARN] failed to load captions_map {caption_path}: {e}")

    # ----------------- main loop over splits -----------------
    for split in splits:
        # ---- query ----
        query_ds = load_dataset(
            path, "query", split=split,
            cache_dir=cache_dir, revision=revision,
        )

        def map_query(x):
            # choose text according to text_length
            if text_length == "2b":
                exp = _get_expansion_for_row(exp_map_2b, "query", x)
                if exp:
                    new_text = exp.get("expanded_content") 
                else:
                    print("exp is none.")
            elif text_length == "72b":

                exp = _get_expansion_for_row(exp_map_72b, "query", x)
                if exp:
                    new_text = exp.get("expanded_content") 
                else:
                    print("exp is none")
            else:
                new_text = x["text"]

            # caption replacement if text_vision
            if text_vision and new_text is not None:
                for key, cap in captions_map.items():
                    if key.startswith(f"query-{x['id']}-"):
                        img_key = key.split("-")[-1]
                        placeholder = f"<{img_key.replace('_', ' ')}>"
                        if placeholder in new_text:
                            new_text = new_text.replace(placeholder, f"<{cap}>")

            if is_clip:
                new_text = f"{instruction}\n{new_text or ''}"

            query_image = None if text_vision else resize_image(x.get("image"))

            return {
                "id": f"query-{split}-{x['id']}",
                "text": new_text,
                "image": query_image,
                "modality": "text" if text_vision else x["modality"],
                "category": x["category"],
            }

        query[split] = query_ds.map(map_query)

        # ---- corpus ----
        corpus_ds = load_dataset(path, "corpus", split=split,
                                 cache_dir=cache_dir, revision=revision)

        pin_p_ds = load_dataset(path, "pin_p", split=split,
                                cache_dir=cache_dir, revision=revision)

        def map_corpus(x):
            image_to_use = x["vision"] if text_vision else x["image"]
            resized_image = resize_image(image_to_use)
            
            return {
                "id": f"corpus-{split}-{x['id']}",
                "text": None if text_vision else x["text"],
                "image": resized_image,
                "modality": "image" if text_vision else x["modality"],
            }

        corpus_ds_mapped = corpus_ds.map(map_corpus)
        # pin_p_ds_mapped = pin_p_ds.map(map_corpus)

        def map_pin_p(x, idx):
            image_to_use = x["vision"] if text_vision else x["image"]
            resized_image = resize_image(image_to_use)
            
            return {
                "id": f"redo-{split}-{idx}",
                "text": None if text_vision else x["text"],
                "image": resized_image,
                "modality": "image" if text_vision else x["modality"],
            }

        pin_p_ds_mapped = pin_p_ds.map(map_pin_p, with_indices=True)

        combined_ds = concatenate_datasets([corpus_ds_mapped, pin_p_ds_mapped])
        corpus[split] = combined_ds

        # ---- qrels ----
        qrels_ds = load_dataset(path, "qrels", split=split,
                                cache_dir=cache_dir, revision=revision)
        qrels[split] = {}
        for row in qrels_ds:
            qid = f"query-{split}-{row['query_id']}"
            did = f"corpus-{split}-{row['corpus_id']}"
            qrels[split].setdefault(qid, {})[did] = int(row["score"])

    return corpus, query, qrels

category_map = {
    "Art": "Art", "Art_Theory": "Art", "Design": "Art", "Music": "Art",
    "Sociology": "Humanities", "Literature": "Humanities", "History": "Humanities", "Psychology": "Humanities",
    "Clinical_Medicine": "Medicine", "Diagnostics_and_Laboratory_Medicine": "Medicine", "Basic_Medical_Science": "Medicine", "Pharmacy": "Medicine",
    "Biology": "Science", "Chemistry": "Science", "Geography": "Science", "Agriculture": "Science",
    }



class KnowledgeAny2AnyRetrieval(AbsTaskAny2AnyRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) # so important, otherwise the hf_subsets will be None
        self.split_results = True
        self.category_map = category_map
        
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
        kwargs.update(self.kwargs)

        text_vision = kwargs.get("text_vision", False)
        print(text_vision)
        
        is_clip = kwargs.get("is_clip", False)
        print(is_clip)
        
        text_length = kwargs.get("text_length", "original")
        print(text_length)
        
        self.corpus, self.queries, self.relevant_docs = _load_data(
            path=self.metadata_dict["dataset"]["path"],
            splits=self.metadata_dict["eval_splits"],
            cache_dir="/data/siyue/",
            revision=self.metadata_dict["dataset"]["revision"],
            text_vision=text_vision,
            is_clip=is_clip,
            instruction=self.metadata.prompt["query"],
            text_length=text_length,
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