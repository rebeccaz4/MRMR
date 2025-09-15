from __future__ import annotations

import os
import torch
from functools import partial

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.instruct_wrapper import InstructSentenceTransformerWrapper

import logging
from functools import partial
from typing import Any, Literal

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, AutoModel

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.requires_package import (
    requires_image_dependencies,
    requires_package,
    suggest_package,
)

class Qwen3Wrapper:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        self.device = device
        self.pooling = "last"
        self.normalize = True
        self.hidden_size = 4096  # 根据模型调整

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        # 加载模型
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.use_cache = False
        self.mdl = AutoModel.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
        ).to(device)
        self.mdl.eval()

    def encode(self, sentences: list[str], **kwargs) -> torch.Tensor:
        """
        兼容 MTEB 的 encode 方法
        """
        return self.get_text_embeddings(texts=sentences, **kwargs)

    def encode_input(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        编码输入，返回 [batch_size, hidden_dim]
        """
        with torch.no_grad():
            outputs = self.mdl(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state  # 或 outputs.hidden_states[-1]
            
            # 取第一个 token（CLS）作为文本表示
            reps = hidden_states[:, 0, :]
            if self.normalize:
                reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def get_text_embeddings(
        self,
        texts: list[str],
        batch_size: int = 32,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        纯文本版本的 embeddings
        """
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            batch_embeddings = self.encode_input(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            all_embeddings.append(batch_embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    # 如果你不需要图像，也可以直接删掉
    def calculate_probs(self, text_embeddings, image_embeddings):
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        logits = torch.matmul(image_embeddings, text_embeddings.T)
        probs = (logits * 100).softmax(dim=-1)
        return probs


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    if not instruction or prompt_type == PromptType.passage:
        return ""
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = list(instruction.values())[0]  # TODO
        else:
            instruction = instruction[prompt_type]
    return f"Instruct: {instruction}\nQuery:"


multilingual_langs = [
    "afr-Latn",
    "ara-Arab",
    "aze-Latn",
    "bel-Cyrl",
    "bul-Cyrl",
    "ben-Beng",
    "cat-Latn",
    "ceb-Latn",
    "ces-Latn",
    "cym-Latn",
    "dan-Latn",
    "deu-Latn",
    "ell-Grek",
    "eng-Latn",
    "spa-Latn",
    "est-Latn",
    "eus-Latn",
    "fas-Arab",
    "fin-Latn",
    "fra-Latn",
    "glg-Latn",
    "guj-Gujr",
    "heb-Hebr",
    "hin-Deva",
    "hrv-Latn",
    "hat-Latn",
    "hun-Latn",
    "hye-Armn",
    "ind-Latn",
    "isl-Latn",
    "ita-Latn",
    "jpn-Jpan",
    "jav-Latn",
    "kat-Geor",
    "kaz-Cyrl",
    "khm-Khmr",
    "kan-Knda",
    "kor-Hang",
    "kir-Cyrl",
    "lao-Laoo",
    "lit-Latn",
    "lav-Latn",
    "mkd-Cyrl",
    "mal-Mlym",
    "mon-Cyrl",
    "mar-Deva",
    "msa-Latn",
    "mya-Mymr",
    "nep-Deva",
    "nld-Latn",
    "nor-Latn",
    "pan-Guru",
    "pol-Latn",
    "por-Latn",
    "que-Latn",
    "ron-Latn",
    "rus-Cyrl",
    "sin-Sinh",
    "slk-Latn",
    "slv-Latn",
    "swa-Latn",
    "tam-Taml",
    "tel-Telu",
    "tha-Thai",
    "tgl-Latn",
    "tur-Latn",
    "ukr-Cyrl",
    "urd-Arab",
    "vie-Latn",
    "yor-Latn",
    "zho-Hans",
]

training_data = {
    "T2Retrieval": ["train"],
    "DuRetrieval": ["train"],
    "MMarcoReranking": ["train"],
    "CMedQAv2-reranking": ["train"],
    "NQ": ["train"],
    "MSMARCO": ["train"],
    "HotpotQA": ["train"],
    "FEVER": ["train"],
    "MrTidyRetrieval": ["train"],
    "MIRACLRetrieval": ["train"],
    "CodeSearchNet": ["train"],
}


def q3e_instruct_loader(model_name_or_path, **kwargs):
    model = InstructSentenceTransformerWrapper(
        model_name_or_path,
        revision=kwargs.pop("revision", None),
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
        **kwargs,
    )
    encoder = model.model._first_module()
    if encoder.auto_model.config._attn_implementation == "flash_attention_2":
        # The Qwen3 code only use left padding in flash_attention_2 mode.
        encoder.tokenizer.padding_side = "left"
    return model


Qwen3_Embedding_0B6 = ModelMeta(
    loader=partial(  # type: ignore
        q3e_instruct_loader,
        model_name_or_path=os.environ.get("Q3E_0B6_PATH", "Qwen/Qwen3-Embedding-0.6B"),
    ),
    name="Qwen/Qwen3-Embedding-0.6B",
    languages=multilingual_langs,
    open_weights=True,
    revision="b22da495047858cce924d27d76261e96be6febc0",  # Commit of @tomaarsen
    release_date="2025-06-05",
    n_parameters=595776512,
    memory_usage_mb=2272,
    embed_dim=1024,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
)

Qwen3_Embedding_4B = ModelMeta(
    loader=partial(  # type: ignore
        q3e_instruct_loader,
        model_name_or_path=os.environ.get("Q3E_4B_PATH", "Qwen/Qwen3-Embedding-4B"),
    ),
    name="Qwen/Qwen3-Embedding-4B",
    languages=multilingual_langs,
    open_weights=True,
    revision="636cd9bf47d976946cdbb2b0c3ca0cb2f8eea5ff",  # Commit of @tomaarsen
    release_date="2025-06-05",
    n_parameters=4021774336,
    memory_usage_mb=15341,
    embed_dim=2560,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-4B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
)

Qwen3_Embedding_8B = ModelMeta(
    loader=partial(  # type: ignore
        Qwen3Wrapper,
        model_name_or_path=os.environ.get("Q3E_8B_PATH", "Qwen/Qwen3-Embedding-8B"),
    ),
    name="Qwen/Qwen3-Embedding-8B",
    languages=multilingual_langs,
    open_weights=True,
    revision="4e423935c619ae4df87b646a3ce949610c66241c",  # Commit of @tomaarsen
    release_date="2025-06-05",
    n_parameters=7567295488,
    memory_usage_mb=28866,
    embed_dim=4096,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-8B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
)
