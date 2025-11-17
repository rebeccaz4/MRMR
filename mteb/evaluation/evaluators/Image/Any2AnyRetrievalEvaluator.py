from __future__ import annotations

import heapq
import io
import json
import logging
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np
import pytrec_eval
import torch
from datasets import Dataset
from PIL import Image
from torch.utils.data import DataLoader

from mteb.encoder_interface import Encoder, PromptType
from mteb.requires_package import requires_image_dependencies

from ..Evaluator import Evaluator
from ..utils import (
    confidence_scores,
    cos_sim,
    dot_score,
    download,
    hole,
    mrr,
    nAUC,
    recall_cap,
    top_k_accuracy,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def get_default_transform():
    requires_image_dependencies()
    from torchvision import transforms

    return transforms.Compose([transforms.PILToTensor()])


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, image_column_name: str = "image", transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_column_name = image_column_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx][self.image_column_name]
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        else:
            # Assume the image is already in a usable format (e.g., PIL Image)
            image = image
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


def custom_collate_fn(batch):
    return batch


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/search/dense/exact_search.py#L12
class Any2AnyDenseRetrievalExactSearch:
    def __init__(
        self,
        model: Encoder,
        encode_kwargs: dict[str, Any] = {},
        corpus_chunk_size: int = 20000,
        previous_results: str | None = None,
        transform=None,
        **kwargs: Any,
    ):
        # Model is class that provides get_text_embeddings() and get_image_embeddings()
        self.model = model
        self.encode_kwargs = encode_kwargs
        if transform is None:
            self.transform = get_default_transform()

        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 128

        self.score_functions = {"cos_sim": cos_sim, "dot": dot_score}
        self.score_function_desc = {
            "cos_sim": "Cosine Similarity",
            "dot": "Dot Product",
        }
        self.corpus_chunk_size = corpus_chunk_size
        self.previous_results = previous_results
        self.batch_size = encode_kwargs.get("batch_size")
        self.show_progress_bar = encode_kwargs.get("show_progress_bar")
        self.save_corpus_embeddings = kwargs.get("save_corpus_embeddings", False)
        self.corpus_embeddings = defaultdict(list)
        self.results = {}

        if self.previous_results is not None:
            self.previous_results = self.load_results_file()
    
    def search(
            self,
            corpus: Dataset,
            queries: Dataset,
            top_k: int,
            score_function: str,
            task_name: str,
            return_sorted: bool = False,
            **kwargs,
        ) -> dict[str, dict[str, float]]:
        
        if hasattr(self.model, "similarity"):
            score_function = self.model.similarity
            logger.info("Scoring Function: from model")
        else:
            if score_function not in self.score_functions:
                raise ValueError(
                    f"score function: {score_function} must be either (cos_sim) for cosine similarity or (dot) for dot product"
                )
            logger.info(
                f"Scoring Function: {self.score_function_desc[score_function]} ({score_function})"
            )
            score_function = self.score_functions[score_function]
       
        logger.warning("Encoding Queries.")
        queries = list(queries)  # 确保 queries 是 list of dict
        query_ids = [item["id"] for item in queries]
        self.results = {qid: {} for qid in query_ids}

        all_query_embeddings = []
        # 直接逐条处理
        for idx, item in enumerate(queries):
            modality = item.get("modality")
            try:
                if modality == "text":
                    emb = self.model.get_text_embeddings(
                        texts=[item["text"]],
                        task_name=task_name,
                        prompt_type=PromptType.query,
                        **self.encode_kwargs,
                    )

                elif modality == "image":
                    dataset = ImageDataset(
                        [item],
                        image_column_name="image",
                        transform=self.transform
                    )
                    dataloader = DataLoader(
                        dataset,
                        batch_size=1,
                        shuffle=False,
                        collate_fn=custom_collate_fn,
                        num_workers=0,
                    )
                    emb = self.model.get_image_embeddings(
                        images=dataloader,
                        task_name=task_name,
                        prompt_type=PromptType.query,
                        **self.encode_kwargs,
                    )

                elif modality == "image,text":
                    dataset = ImageDataset(
                        [item],
                        image_column_name="image",
                        transform=self.transform
                    )
                    dataloader = DataLoader(
                        dataset,
                        batch_size=1,
                        shuffle=False,
                        collate_fn=custom_collate_fn,
                        num_workers=0,
                    )
                    emb = self.model.get_fused_embeddings(
                        texts=[item["text"]],
                        images=dataloader,
                        task_name=task_name,
                        prompt_type=PromptType.query,
                        **self.encode_kwargs,
                    )

                else:
                    logger.warning(f"Unsupported modality: {modality}, skipping.")
                    continue

                # 将 embedding 添加到总列表
                all_query_embeddings.append(emb)
                # print(emb.shape)

            except Exception as e:
                logger.warning(f"Failed to encode item at index {idx}: {e}")
                continue

        # 拼接所有 embedding
        if all_query_embeddings:
            query_embeddings = torch.cat(all_query_embeddings, dim=0)
        else:
            query_embeddings = torch.empty((0,))
        
        logger.info("Preparing Corpus...")
               
        corpus_ids = list(corpus["id"])

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")

        result_heaps = {qid: [] for qid in query_ids}

        for chunk_start in range(0, len(corpus), self.corpus_chunk_size):
            chunk = corpus.select(
                range(
                    chunk_start, min(chunk_start + self.corpus_chunk_size, len(corpus))
                )
            )
            chunk_ids = corpus_ids[chunk_start : chunk_start + self.corpus_chunk_size]

            sub_corpus_embeddings = []
            valid_chunk_ids = []
            
            # check the modality one by one
            for i, item in enumerate(chunk):
                modality = item.get("modality")
                try:
                    if modality == "text":
                        embedding = self.model.get_text_embeddings(
                            texts=[item["text"]],
                            task_name=task_name,
                            prompt_type=PromptType.passage,
                            **self.encode_kwargs,
                        )
                    elif modality == "image":
                        dataset = ImageDataset(
                            [item], image_column_name="image", transform=self.transform
                        )
                        dataloader = DataLoader(
                            dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=custom_collate_fn,
                            num_workers=0,
                        )
                        embedding = self.model.get_image_embeddings(
                            images=dataloader,
                            task_name=task_name,
                            prompt_type=PromptType.passage,
                            **self.encode_kwargs,
                        )
                    elif modality == "image,text":
                        dataset = ImageDataset(
                            [item], image_column_name="image", transform=self.transform
                        )
                        dataloader = DataLoader(
                            dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=custom_collate_fn,
                            num_workers=0,
                        )
                        embedding = self.model.get_fused_embeddings(
                            texts=[item["text"]],
                            images=dataloader,
                            task_name=task_name,
                            prompt_type=PromptType.passage,
                            **self.encode_kwargs,
                        )
                    else:
                        logger.warning(f"Unsupported modality: {modality}, skipping.")
                        continue

                    sub_corpus_embeddings.append(embedding)
                    valid_chunk_ids.append(chunk_ids[i])
                except Exception as e:
                    logger.warning(f"Failed to encode item at index {chunk_start + i}: {e}")
                    continue

            if not sub_corpus_embeddings:
                continue

            sub_corpus_embeddings = torch.cat(sub_corpus_embeddings, dim=0)

            # cos_scores = score_function(query_embeddings, sub_corpus_embeddings)
            
            from tqdm import tqdm

            def score_function_chunked(q_embeds, c_embeds, batch_size=1):
                """
                按 query 维度分批算相似度矩阵，避免一次性占用过多显存，并显示进度条。
                """
                all_scores = []
                total = q_embeds.size(0)
                # tqdm 负责显示进度条
                for start in tqdm(range(0, total, batch_size), desc="Scoring", ncols=80):
                    end = start + batch_size
                    q_batch = q_embeds[start:end]
                    scores = score_function(q_batch, c_embeds)
                    all_scores.append(scores)
                return torch.cat(all_scores, dim=0)


            # 使用：
            cos_scores = score_function_chunked(query_embeddings, sub_corpus_embeddings, batch_size=256)

            cos_scores[torch.isnan(cos_scores)] = -1

            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores,
                min(top_k, cos_scores.size(1)),
                dim=1,
                largest=True,
                sorted=return_sorted,
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]
                ):
                    corpus_id = valid_chunk_ids[sub_corpus_id]
                    if len(result_heaps[query_id]) < top_k:
                        heapq.heappush(result_heaps[query_id], (score, corpus_id))
                    else:
                        heapq.heappushpop(result_heaps[query_id], (score, corpus_id))
                        
        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score

        return self.results
        

    """
    最初的mteb原始版本
    
    
    def search(
        self,
        corpus: Dataset,  # solve memoery issues
        queries: Dataset,  # solve memoery issues
        top_k: int,
        score_function: str,
        task_name: str,
        return_sorted: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        if hasattr(self.model, "similarity"):
            score_function = self.model.similarity
            logger.info("Scoring Function: from model")
        else:
            if score_function not in self.score_functions:
                raise ValueError(
                    f"score function: {score_function} must be either (cos_sim) for cosine similarity or (dot) for dot product"
                )
            logger.info(
                f"Scoring Function: {self.score_function_desc[score_function]} ({score_function})"
            )
            score_function = self.score_functions[score_function]

        logger.info("Encoding Queries.")
        query_ids = list(queries["id"])
        self.results = {qid: {} for qid in query_ids}

        q_modality = queries[0]["modality"]

        if q_modality == "text":
            query_texts = queries["text"]
            query_embeddings = self.model.get_text_embeddings(
                texts=query_texts,
                task_name=task_name,
                prompt_type=PromptType.query,
                **self.encode_kwargs,
            )
        else:
            queries_dataset = ImageDataset(
                queries, image_column_name="image", transform=self.transform
            )
            query_image_dataloader = DataLoader(
                queries_dataset,
                batch_size=self.encode_kwargs["batch_size"],
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=min(math.floor(os.cpu_count() / 2), 16),
            )
            if q_modality == "image":
                query_embeddings = self.model.get_image_embeddings(
                    images=query_image_dataloader,
                    task_name=task_name,
                    prompt_type=PromptType.query,
                    **self.encode_kwargs,
                )
            elif q_modality == "image,text":
                query_texts = queries["text"]
                query_embeddings = self.model.get_fused_embeddings(
                    texts=query_texts,
                    images=query_image_dataloader,
                    task_name=task_name,
                    prompt_type=PromptType.query,
                    **self.encode_kwargs,
                )
            else:
                raise ValueError(f"Unsupported modality: {q_modality}")
            
        logger.info("Preparing Corpus...")
        
        corpus_ids = list(corpus["id"])

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")

        result_heaps = {qid: [] for qid in query_ids}

        for chunk_start in range(0, len(corpus), self.corpus_chunk_size):
            chunk = corpus.select(
                range(
                    chunk_start, min(chunk_start + self.corpus_chunk_size, len(corpus))
                )
            )
            chunk_ids = corpus_ids[chunk_start : chunk_start + self.corpus_chunk_size]

            sub_corpus_embeddings = []
            valid_chunk_ids = []
            
            # check the modality one by one
            for i, item in enumerate(chunk):
                modality = item.get("modality")
                try:
                    if modality == "text":
                        embedding = self.model.get_text_embeddings(
                            texts=[item["text"]],
                            task_name=task_name,
                            prompt_type=PromptType.passage,
                            **self.encode_kwargs,
                        )
                    elif modality == "image":
                        dataset = ImageDataset(
                            [item], image_column_name="image", transform=self.transform
                        )
                        dataloader = DataLoader(
                            dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=custom_collate_fn,
                            num_workers=0,
                        )
                        embedding = self.model.get_image_embeddings(
                            images=dataloader,
                            task_name=task_name,
                            prompt_type=PromptType.passage,
                            **self.encode_kwargs,
                        )
                    elif modality == "image,text":
                        dataset = ImageDataset(
                            [item], image_column_name="image", transform=self.transform
                        )
                        dataloader = DataLoader(
                            dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=custom_collate_fn,
                            num_workers=0,
                        )
                        embedding = self.model.get_fused_embeddings(
                            texts=[item["text"]],
                            images=dataloader,
                            task_name=task_name,
                            prompt_type=PromptType.passage,
                            **self.encode_kwargs,
                        )
                    else:
                        logger.warning(f"Unsupported modality: {modality}, skipping.")
                        continue

                    sub_corpus_embeddings.append(embedding)
                    valid_chunk_ids.append(chunk_ids[i])
                except Exception as e:
                    logger.warning(f"Failed to encode item at index {chunk_start + i}: {e}")
                    continue

            if not sub_corpus_embeddings:
                continue

            sub_corpus_embeddings = torch.cat(sub_corpus_embeddings, dim=0)

            cos_scores = score_function(query_embeddings, sub_corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1

            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores,
                min(top_k, cos_scores.size(1)),
                dim=1,
                largest=True,
                sorted=return_sorted,
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]
                ):
                    corpus_id = valid_chunk_ids[sub_corpus_id]
                    if len(result_heaps[query_id]) < top_k:
                        heapq.heappush(result_heaps[query_id], (score, corpus_id))
                    else:
                        heapq.heappushpop(result_heaps[query_id], (score, corpus_id))
                        
        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score

        return self.results
                 
                        
        ### 这里是所有corpus一起编码的版本
        
        result_heaps = {qid: [] for qid in query_ids}
        for chunk_start in range(0, len(corpus), self.corpus_chunk_size):
            chunk = corpus.select(
                range(
                    chunk_start, min(chunk_start + self.corpus_chunk_size, len(corpus))
                )
            )
            chunk_ids = corpus_ids[chunk_start : chunk_start + self.corpus_chunk_size]

            if corpus_modality == "text":
                corpus_texts = chunk["text"]
                sub_corpus_embeddings = self.model.get_text_embeddings(
                    texts=corpus_texts,
                    task_name=task_name,
                    prompt_type=PromptType.passage,
                    **self.encode_kwargs,
                )
            else:
                corpus_dataset = ImageDataset(
                    chunk, image_column_name="image", transform=self.transform
                )
                corpus_image_dataloader = DataLoader(
                    corpus_dataset,
                    batch_size=self.encode_kwargs["batch_size"],
                    shuffle=False,
                    collate_fn=custom_collate_fn,
                    num_workers=min(math.floor(os.cpu_count() / 2), 16),
                )
                if corpus_modality == "image":
                    sub_corpus_embeddings = self.model.get_image_embeddings(
                        images=corpus_image_dataloader,
                        task_name=task_name,
                        prompt_type=PromptType.passage,
                        **self.encode_kwargs,
                    )
                elif corpus_modality == "image,text":
                    corpus_texts = chunk["text"]
                    sub_corpus_embeddings = self.model.get_fused_embeddings(
                        texts=corpus_texts,
                        images=corpus_image_dataloader,
                        task_name=task_name,
                        prompt_type=PromptType.passage,
                        **self.encode_kwargs,
                    )
                else:
                    raise ValueError(f"Unsupported modality: {corpus_modality}")
            
            # 这里是query和一小块一小块的corpus算相似度的，所以没有把所有的都sub-embedding加到一起
            
            cos_scores = score_function(query_embeddings, sub_corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1

            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores,
                min(top_k, cos_scores.size(1)),
                dim=1,
                largest=True,
                sorted=return_sorted,
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(
                    cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]
                ):
                    corpus_id = chunk_ids[sub_corpus_id]
                    if len(result_heaps[query_id]) < top_k:
                        heapq.heappush(result_heaps[query_id], (score, corpus_id))
                    else:
                        heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score

        return self.results

"""

    def load_results_file(self):
        # load the first stage results from file in format {qid: {doc_id: score}}
        if "https://" in self.previous_results:
            # download the file
            if not os.path.exists(self.previous_results):
                url_descriptor = self.previous_results.split("https://")[-1].replace(
                    "/", "--"
                )
                dest_file = os.path.join(
                    "results", f"cached_predictions--{url_descriptor}"
                )
                os.makedirs(os.path.dirname(os.path.abspath(dest_file)), exist_ok=True)
                download(self.previous_results, dest_file)
                logger.info(
                    f"Downloaded the previous results at {self.previous_results} to {dest_file}"
                )
            self.previous_results = dest_file

        with open(self.previous_results) as f:
            previous_results = json.load(f)
        assert isinstance(previous_results, dict)
        assert isinstance(previous_results[list(previous_results.keys())[0]], dict)
        return previous_results


# Adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L9
class Any2AnyRetrievalEvaluator(Evaluator):
    def __init__(
        self,
        retriever=None,
        split_corpus: bool = False,
        task_name: str | None = None,
        k_values: list[int] = [1, 3, 5, 10, 20, 100, 1000],
        score_function: str = "cos_sim",
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.retriever = Any2AnyDenseRetrievalExactSearch(
            retriever, encode_kwargs=encode_kwargs, **kwargs
        )
        self.k_values = k_values
        self.top_k = (
            max(k_values) if "top_k" not in kwargs else kwargs["top_k"]
        )  # can lower it if reranking
        self.score_function = score_function
        self.task_name = task_name
        self.split_corpus = split_corpus

    def __call__(
        self,
        corpus: dict[str, dict[str, str | Image.Image]],
        queries: dict[str, dict[str, str | Image.Image]],
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        print("before embedding split_corpus", self.split_corpus)
        
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")
        
        return self.retriever.search(
            corpus,
            queries,
            self.top_k,
            self.score_function,
            task_name=self.task_name,
            split_corpus=self.split_corpus,
        )

    @staticmethod
    def evaluate(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        split_results: bool = False, # 新增参数：是否进行分类评估
        category_map: dict[str, str] | None = None,
        queries: dict[str, dict[str, str | Image.Image]] | None = None,
        ignore_identical_ids: bool = False,
        skip_first_result: bool = False,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, dict[str, float]],   # <--- 新增：各类别指标
    ]:
        if ignore_identical_ids:
            logger.debug(
                "For evaluation, ``ignore_identical_ids=True`` is set to True, the evaluator will ignore identical query and document ids."
            )
            # Remove identical ids from results dict
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
        else:
            logger.debug(
                "For evaluation, we DO NOT ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=True`` to ignore this."
            )

        all_ndcgs, all_aps, all_recalls, all_precisions, all_cv_recalls = (
            {},
            {},
            {},
            {},
            {},
        )

        for k in k_values:
            all_ndcgs[f"NDCG@{k}"] = []
            all_aps[f"MAP@{k}"] = []
            all_recalls[f"Recall@{k}"] = []
            all_precisions[f"P@{k}"] = []
            all_cv_recalls[f"CV_Recall@{k}"] = []  # (new) CV-style Recall
       
        # 计算出了每个query的所有指标
        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {map_string, ndcg_string, recall_string, precision_string}
        )
        scores = evaluator.evaluate(results)

        sorted_results = {
            qid: sorted(rels.items(), key=lambda item: item[1], reverse=True)
            for qid, rels in results.items()
        }

        if skip_first_result:
            for qid, rels in sorted_results.items():
                sorted_results[qid].pop(0)

        for query_id in scores.keys():
            top_docs = [
                doc_id for doc_id, _ in sorted_results.get(query_id, [])
            ]  # Sorted list of doc IDs
            relevant_docs = set(qrels.get(query_id, {}).keys())

            for k in k_values:
                top_k_docs = top_docs[:k]
                all_ndcgs[f"NDCG@{k}"].append(scores[query_id]["ndcg_cut_" + str(k)])
                all_aps[f"MAP@{k}"].append(scores[query_id]["map_cut_" + str(k)])
                all_recalls[f"Recall@{k}"].append(scores[query_id]["recall_" + str(k)])
                all_precisions[f"P@{k}"].append(scores[query_id]["P_" + str(k)])

                if relevant_docs.intersection(top_k_docs):
                    all_cv_recalls[f"CV_Recall@{k}"].append(1.0)
                else:
                    all_cv_recalls[f"CV_Recall@{k}"].append(0.0)

        ndcg, _map, recall, precision, cv_recall = (
            all_ndcgs.copy(),
            all_aps.copy(),
            all_recalls.copy(),
            all_precisions.copy(),
            all_cv_recalls.copy(),
        )
            
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(sum(ndcg[f"NDCG@{k}"]) / len(scores), 5)
            _map[f"MAP@{k}"] = round(sum(_map[f"MAP@{k}"]) / len(scores), 5)
            recall[f"Recall@{k}"] = round(sum(recall[f"Recall@{k}"]) / len(scores), 5)
            precision[f"P@{k}"] = round(sum(precision[f"P@{k}"]) / len(scores), 5)
            cv_recall[f"CV_Recall@{k}"] = round(
                sum(cv_recall[f"CV_Recall@{k}"]) / len(scores), 5
            )

        naucs = Any2AnyRetrievalEvaluator.evaluate_abstention(
            results,
            {**all_ndcgs, **all_aps, **all_recalls, **all_precisions, **all_cv_recalls},
        )
        
        # === 新增：split_results 分类评估 ===
        split_metrics = {}
        print("split_results: ", split_results, "queries: ", queries)
        queries = {x["id"]: x for x in queries}
        print(type(queries))
        # 调用一定要记得传queries，否则即使split-results是true，也不会进入这个分支！
        if split_results and queries is not None:
            # 先按类别分组
            queries_by_cat = {}
            for qid in scores.keys():
                cat = queries[qid].get("category", "unknown")
                if cat not in queries_by_cat:
                    queries_by_cat[cat] = []
                queries_by_cat[cat].append(qid)

            # 分类别计算指标
            for cat, qids in queries_by_cat.items():
                cat_metrics = {"ndcg": {}, "map": {}, "recall": {}, "precision": {}, "cv_recall": {}}
                for k in k_values:
                    cat_metrics["ndcg"][f"NDCG@{k}"] = round(
                        sum(all_ndcgs[f"NDCG@{k}"][i] for i, qid in enumerate(scores.keys()) if qid in qids) / len(qids), 5
                    )
                    cat_metrics["map"][f"MAP@{k}"] = round(
                        sum(all_aps[f"MAP@{k}"][i] for i, qid in enumerate(scores.keys()) if qid in qids) / len(qids), 5
                    )
                    cat_metrics["recall"][f"Recall@{k}"] = round(
                        sum(all_recalls[f"Recall@{k}"][i] for i, qid in enumerate(scores.keys()) if qid in qids) / len(qids), 5
                    )
                    cat_metrics["precision"][f"P@{k}"] = round(
                        sum(all_precisions[f"P@{k}"][i] for i, qid in enumerate(scores.keys()) if qid in qids) / len(qids), 5
                    )
                    cat_metrics["cv_recall"][f"CV_Recall@{k}"] = round(
                        sum(all_cv_recalls[f"CV_Recall@{k}"][i] for i, qid in enumerate(scores.keys()) if qid in qids) / len(qids), 5
                    )
                split_metrics[cat] = cat_metrics
            
            print(category_map is not None)
            
            # 是否需要重新组合小类成大类
            if category_map is not None:  
                coarse_groups = {}
                for fine_cat, qids in queries_by_cat.items():
                    coarse = category_map.get(fine_cat, "other")
                    coarse_groups.setdefault(coarse, []).extend(qids)

                for gname, qids in coarse_groups.items():
                    g_metrics = {"ndcg": {}, "map": {}, "recall": {}, "precision": {}, "cv_recall": {}}
                    for k in k_values:
                        g_metrics["ndcg"][f"NDCG@{k}"] = round(
                            sum(all_ndcgs[f"NDCG@{k}"][i] for i, qid in enumerate(scores.keys()) if qid in qids) / len(qids), 5
                        )
                        g_metrics["map"][f"MAP@{k}"] = round(
                            sum(all_aps[f"MAP@{k}"][i] for i, qid in enumerate(scores.keys()) if qid in qids) / len(qids), 5
                        )
                        g_metrics["recall"][f"Recall@{k}"] = round(
                            sum(all_recalls[f"Recall@{k}"][i] for i, qid in enumerate(scores.keys()) if qid in qids) / len(qids), 5
                        )
                        g_metrics["precision"][f"P@{k}"] = round(
                            sum(all_precisions[f"P@{k}"][i] for i, qid in enumerate(scores.keys()) if qid in qids) / len(qids), 5
                        )
                        g_metrics["cv_recall"][f"CV_Recall@{k}"] = round(
                            sum(all_cv_recalls[f"CV_Recall@{k}"][i] for i, qid in enumerate(scores.keys()) if qid in qids) / len(qids), 5
                        )
                    # 存到 split_metrics，加个前缀区分，例如 "coarse/groupA"
                    split_metrics[f"coarse/{gname}"] = g_metrics

        return ndcg, _map, recall, precision, cv_recall, naucs, split_metrics
    
    
    @staticmethod
    def evaluate_custom(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        metric: str,
        output_type: str = "all",
    ) -> tuple[dict[str, float]]:
        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            metric_scores = mrr(qrels, results, k_values, output_type)

        elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
            metric_scores = recall_cap(qrels, results, k_values, output_type)

        elif metric.lower() in ["hole", "hole@k"]:
            metric_scores = hole(qrels, results, k_values, output_type)

        elif metric.lower() in [
            "acc",
            "top_k_acc",
            "accuracy",
            "accuracy@k",
            "top_k_accuracy",
        ]:
            metric_scores = top_k_accuracy(qrels, results, k_values, output_type)

        naucs = Any2AnyRetrievalEvaluator.evaluate_abstention(results, metric_scores)
        metric_scores_avg = {k: sum(v) / len(v) for k, v in metric_scores.items()}

        return metric_scores_avg, naucs

    @staticmethod
    def evaluate_abstention(
        results: dict[str, dict[str, float]],
        metric_scores: dict[str, list[float]],
    ) -> dict[str, float]:
        """Computes normalized Area Under the Curve on a set of evaluated instances as presented in the paper https://arxiv.org/abs/2402.12997"""
        # Filter out queries with empty similarity scores (not relevant to any documents)
        qids = list(results.keys())
        filtered_qids = []
        filtered_sim_scores = []
        
        for qid in qids:
            sim_scores = list(results[qid].values())
            if len(sim_scores) > 0:
                filtered_qids.append(qid)
                filtered_sim_scores.append(sim_scores)
            else:
                print(f"INFO: Skipping qid {qid} - no similarity scores (not relevant to any documents)")
        
        print(f"INFO: Filtered {len(qids) - len(filtered_qids)} queries with empty scores out of {len(qids)} total queries")
        
        # Handle case where all queries are filtered out
        if not filtered_sim_scores:
            print("WARNING: All queries have empty similarity scores. Returning empty nAUC results.")
            return {}
        
        all_conf_scores = [
            confidence_scores(sim_scores) for sim_scores in filtered_sim_scores
        ]
        conf_fcts = list(all_conf_scores[0].keys()) if all_conf_scores else []
        all_conf_scores = {
            fct: np.array([x[fct] for x in all_conf_scores]) for fct in conf_fcts
        }
        
        # Filter metric_scores to match filtered queries
        # Assuming metric_scores values are lists aligned with the original qids order
        original_qids = list(results.keys())
        filtered_indices = [i for i, qid in enumerate(original_qids) if qid in filtered_qids]
        
        filtered_metric_scores = {}
        for metric_name, scores in metric_scores.items():
            if isinstance(scores, list) and len(scores) == len(original_qids):
                filtered_metric_scores[metric_name] = [scores[i] for i in filtered_indices]
            else:
                # If scores don't align with qids, keep as is
                filtered_metric_scores[metric_name] = scores
        
        filtered_metric_scores = {k: np.array(v) for k, v in filtered_metric_scores.items()}
        naucs = {}

        for metric_name, scores in filtered_metric_scores.items():
            for fct, conf_scores in all_conf_scores.items():
                if len(conf_scores) == len(scores):
                    naucs[f"nAUC_{metric_name}_{fct}"] = nAUC(conf_scores, scores)
                else:
                    print(f"WARNING: Length mismatch for {metric_name}: conf_scores={len(conf_scores)}, scores={len(scores)}")

        return naucs

    @staticmethod
    def calculate_cv_style_recall(
        qrels: dict[str, dict[str, int]], results: dict[str, dict[str, float]], k: int
    ) -> dict[str, float]:
        """Calculate CV-style recall: Recall is 1 if any relevant document is
        retrieved in the top k, otherwise 0.
        """
        cv_recalls = {}
        for query_id, relevant_docs in qrels.items():
            retrieved_docs = list(results.get(query_id, {}).keys())[
                :k
            ]  # Retrieve top k documents
            if any(doc_id in relevant_docs for doc_id in retrieved_docs):
                cv_recalls[query_id] = (
                    1.0  # If any relevant doc is found in top k, recall is 1
                )
            else:
                cv_recalls[query_id] = 0.0  # Otherwise, recall is 0
        return cv_recalls
