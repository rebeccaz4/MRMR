from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper
from mteb.requires_package import requires_package


class CustomWrapper(Wrapper):
    def __init__(self, model_name, revision):
        super().__init__()
        self.model = SentenceTransformer(
            model_name, revision=revision, trust_remote_code=True
        )
        self.output_dim = 1536

    def encode(self, sentences: list[str], **kwargs) -> np.ndarray:
        embeddings = self.model.encode(sentences, **kwargs)
        return embeddings[:, : self.output_dim]


class OpsMMEmbeddingWrapper(Wrapper):
    """Wrapper for OpenSearch-AI/Ops-MM-embedding-v1-7B multimodal model."""
    
    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        max_length: int | None = None,
        **kwargs,
    ):
        requires_package(
            self, "transformers", model_name, "pip install transformers>=4.51.0"
        )
        super().__init__()
        
        # Import here to avoid dependency issues
        import requests
        import base64
        import math
        from io import BytesIO
        from typing import List, Optional, TypeAlias, Union
        import torch
        import torch.nn as nn
        from PIL import Image
        from tqdm import tqdm
        from transformers import AutoModelForImageTextToText, AutoProcessor
        
        # Store imports for later use
        self._torch = torch
        self._Image = Image
        self._requests = requests
        self._base64 = base64
        self._math = math
        self._BytesIO = BytesIO
        self._tqdm = tqdm
        self._nn = nn
        
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(device)
        # device = "cuda:2"
        # print(f"!! MANUAL VLM2VecWrapper: moving model to {device}")
        
        
        self.device = device
        self.max_length = max_length
        
        # Use appropriate dtype based on device
        if device == "cpu":
            model_dtype = torch.float32
        else:
            model_dtype = torch.bfloat16
        
        # Load the model and processor
        self.base_model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            revision=revision,
            min_pixels=256 * 28 * 28, 
            max_pixels=1280 * 28 * 28,
            trust_remote_code=True
        )
        self.processor.tokenizer.padding_side = "left"
        
        self.default_instruction = "You are a helpful assistant."
        self.base_model.eval()
        
        # Image processing constants
        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 256 * 28 * 28
        self.MAX_PIXELS = 1280 * 28 * 28
        self.MAX_RATIO = 200

    def _pooling(self, last_hidden_state):
        """Extract embeddings using last token pooling."""
        batch_size = last_hidden_state.shape[0]
        reps = last_hidden_state[self._torch.arange(batch_size), -1, :]
        reps = self._torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def _encode_input(self, input_dict):
        """Encode input and extract embeddings."""
        with self._torch.inference_mode():
            hidden_states = self.base_model(**input_dict, return_dict=True, output_hidden_states=True)
            hidden_states = hidden_states.hidden_states[-1]
            pooled_output = self._pooling(hidden_states)
        return pooled_output

    def _round_by_factor(self, number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    def _ceil_by_factor(self, number: int | float, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return self._math.ceil(number / factor) * factor

    def _floor_by_factor(self, number: int | float, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return self._math.floor(number / factor) * factor

    def _smart_resize(
        self,
        height: int,
        width: int,
        factor: int = 28,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ) -> tuple[int, int]:
        """
        Rescales the image so that the following conditions are met:
        1. Both dimensions (height and width) are divisible by 'factor'.
        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.
        """
        h_bar = max(factor, self._round_by_factor(height, factor))
        w_bar = max(factor, self._round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = self._math.sqrt((height * width) / max_pixels)
            h_bar = self._floor_by_factor(height / beta, factor)
            w_bar = self._floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = self._math.sqrt(min_pixels / (height * width))
            h_bar = self._ceil_by_factor(height * beta, factor)
            w_bar = self._ceil_by_factor(width * beta, factor)

        if max(h_bar, w_bar) / min(h_bar, w_bar) > self.MAX_RATIO:
            import logging
            logging.warning(f"Absolute aspect ratio must be smaller than {self.MAX_RATIO}, got {max(h_bar, w_bar) / min(h_bar, w_bar)}")
            if h_bar > w_bar:
                h_bar = w_bar * self.MAX_RATIO
            else:
                w_bar = h_bar * self.MAX_RATIO
        return h_bar, w_bar

    def _fetch_image(self, image):
        """Fetch and process image from various sources."""
        image_obj = None
        if isinstance(image, self._Image.Image):
            image_obj = image
        elif isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                image_obj = self._Image.open(self._requests.get(image, stream=True).raw)
            elif image.startswith("file://"):
                image_obj = self._Image.open(image[7:])
            elif image.startswith("data:image"):
                if "base64," in image:
                    _, base64_data = image.split("base64,", 1)
                    data = self._base64.b64decode(base64_data)
                    image_obj = self._Image.open(self._BytesIO(data))
            else:
                image_obj = self._Image.open(image)
        elif isinstance(image, self._torch.Tensor):
            # Convert tensor to PIL Image
            # Assume CHW format, convert to HWC
            if image.dim() == 3 and image.shape[0] in [1, 3, 4]:
                image_np = image.permute(1, 2, 0).cpu().numpy()
                if image.shape[0] == 1:  # grayscale
                    image_np = image_np.squeeze(2)
                    image_obj = self._Image.fromarray(image_np, mode='L')
                else:
                    image_obj = self._Image.fromarray(image_np)
            else:
                raise ValueError(f"Unsupported tensor shape for image: {image.shape}")
        
        if image_obj is None:
            raise ValueError(f"Unrecognized image input, support local path, http url, base64, PIL.Image, and torch.Tensor, got {type(image)}")
            
        image = image_obj.convert("RGB")
        width, height = image.size
        resized_height, resized_width = self._smart_resize(
            height,
            width,
            factor=self.IMAGE_FACTOR,
            min_pixels=self.MIN_PIXELS,
            max_pixels=self.MAX_PIXELS,
        )
        image = image.resize((resized_width, resized_height))

        return image

    def _process_images(self, images):
        """Convert single image or list of images to processed format"""
        if isinstance(images, self._Image.Image) or isinstance(images, str):
            return [self._fetch_image(images)]
        return [self._fetch_image(i) for i in images]

    def get_text_embeddings(
        self,
        texts: list[str],
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        instruction: str | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get embeddings for text inputs."""
        if instruction is None:
            instruction = Wrapper.get_instruction(task_name=task_name, prompt_type=prompt_type)
        is_query = prompt_type == PromptType.query
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            input_texts = []
            for text in batch_texts:
                msg = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>"
                input_texts.append(msg)
            inputs = self.processor(
                text=input_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            batch_embeddings = self._encode_input(inputs)
            all_embeddings.append(batch_embeddings.cpu().float().numpy())
        embeddings = np.vstack(all_embeddings)
        if "convert_to_tensor" in kwargs and kwargs["convert_to_tensor"]:
            return self._torch.from_numpy(embeddings)
        return embeddings

    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        instruction: str | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get embeddings for image inputs."""
        if instruction is None:
            instruction = Wrapper.get_instruction(task_name=task_name, prompt_type=prompt_type)
        if isinstance(images, DataLoader):
            # Convert DataLoader to list of images
            image_list = []
            for batch in images:
                if isinstance(batch, dict) and "image" in batch:
                    image_list.extend(batch["image"])
                else:
                    image_list.extend(batch)
            images = image_list
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            # Process images
            processed_images = []
            input_texts = []
            for image in batch_images:
                if not isinstance(image, self._Image.Image):
                    image = self._fetch_image(image)
                processed_images.append(image)
                # Create input text with image token
                input_str = "<|vision_start|><|image_pad|><|vision_end|>"
                msg = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>"
                input_texts.append(msg)
            inputs = self.processor(
                text=input_texts,
                images=processed_images,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            batch_embeddings = self._encode_input(inputs)
            all_embeddings.append(batch_embeddings.cpu().float().numpy())
        embeddings = np.vstack(all_embeddings)
        if "convert_to_tensor" in kwargs and kwargs["convert_to_tensor"]:
            return self._torch.from_numpy(embeddings)
        return embeddings

    def get_fused_embeddings(
        self,
        texts: list[str] | None = None,
        images: list[Image.Image] | DataLoader | None = None,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        instruction: str | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get embeddings for combined text and image inputs."""
        if instruction is None:
            instruction = Wrapper.get_instruction(task_name=task_name, prompt_type=prompt_type)
        is_query = prompt_type == PromptType.query
        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")
        # Convert DataLoader to list if needed
        if isinstance(images, DataLoader):
            image_list = []
            for batch in images:
                if isinstance(batch, dict) and "image" in batch:
                    image_list.extend(batch["image"])
                else:
                    image_list.extend(batch)
            images = image_list
        if texts is not None and images is not None:
            if len(texts) != len(images):
                raise ValueError("Number of texts and images must match")
        total_items = len(texts) if texts is not None else len(images)
        all_embeddings = []
        for i in range(0, total_items, batch_size):
            batch_texts = texts[i:i + batch_size] if texts is not None else [None] * min(batch_size, total_items - i)
            batch_images = images[i:i + batch_size] if images is not None else [None] * min(batch_size, total_items - i)
            input_texts = []
            processed_images = []
            for j, (text, image) in enumerate(zip(batch_texts, batch_images)):
                input_str = ""
                processed_image = None
                if image is not None:
                    if not isinstance(image, self._Image.Image):
                        image = self._fetch_image(image)
                    processed_image = [image]
                    input_str += "<|vision_start|><|image_pad|><|vision_end|>" * len(processed_image)

                if text is not None:
                    input_str += text
                    msg = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input_str}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>"
                input_texts.append(msg)
                processed_images.append(processed_image)
            images_to_process = None
            if any(img is not None for img in processed_images):
                images_to_process = []
                for img_list in processed_images:
                    if img_list is not None:
                        images_to_process.extend(img_list)
                if len(images_to_process) == 0:
                    images_to_process = None
            inputs = self.processor(
                text=input_texts,
                images=images_to_process,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            batch_embeddings = self._encode_input(inputs)
            all_embeddings.append(batch_embeddings.cpu().float().numpy())
        embeddings = np.vstack(all_embeddings)
        if "convert_to_tensor" in kwargs and kwargs["convert_to_tensor"]:
            return self._torch.from_numpy(embeddings)
        return embeddings

    def encode(self, sentences: list[str], **kwargs) -> np.ndarray:
        """Encode sentences using text embeddings."""
        return self.get_text_embeddings(sentences, **kwargs)


ops_moa_conan_embedding = ModelMeta(
    name="OpenSearch-AI/Ops-MoA-Conan-embedding-v1",
    revision="46dcd58753f3daa920c66f89e47086a534089350",
    release_date="2025-03-26",
    languages=["zho-Hans"],
    loader=partial(
        CustomWrapper,
        "OpenSearch-AI/Ops-MoA-Conan-embedding-v1",
        "46dcd58753f3daa920c66f89e47086a534089350",
    ),
    n_parameters=343 * 1e6,
    memory_usage_mb=2e3,
    max_tokens=512,
    embed_dim=1536,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/OpenSearch-AI/Ops-MoA-Conan-embedding-v1",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={
        "T2Retrieval": ["train"],
        "MMarcoRetrieval": ["train"],
        "DuRetrieval": ["train"],
        "CovidRetrieval": ["train"],
        "CmedqaRetrieval": ["train"],
        "EcomRetrieval": ["train"],
        "MedicalRetrieval": ["train"],
        "VideoRetrieval": ["train"],
    },
    superseded_by=None,
)

ops_moa_yuan_embedding = ModelMeta(
    name="OpenSearch-AI/Ops-MoA-Yuan-embedding-1.0",
    revision="23712d0766417b0eb88a2513c6e212a58b543268",
    release_date="2025-03-26",
    languages=["zho-Hans"],
    loader=partial(
        CustomWrapper,
        "OpenSearch-AI/Ops-MoA-Yuan-embedding-1.0",
        "23712d0766417b0eb88a2513c6e212a58b543268",
    ),
    n_parameters=343 * 1e6,
    memory_usage_mb=2e3,
    max_tokens=512,
    embed_dim=1536,
    license="cc-by-nc-4.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Sentence Transformers"],
    reference="https://huggingface.co/OpenSearch-AI/Ops-MoA-Yuan-embedding-1.0",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={
        "T2Retrieval": ["train"],
        "MMarcoRetrieval": ["train"],
        "DuRetrieval": ["train"],
        "CovidRetrieval": ["train"],
        "CmedqaRetrieval": ["train"],
        "EcomRetrieval": ["train"],
        "MedicalRetrieval": ["train"],
        "VideoRetrieval": ["train"],
    },
    superseded_by=None,
)

ops_mm_embedding_v1_7b = ModelMeta(
    name="OpenSearch-AI/Ops-MM-embedding-v1-7B",
    revision="79fbd6f0ea5c6cd02424acdb97c3cb4ea4cff0f0",
    release_date="2024-10-03",
    languages=["eng-Latn"],
    modalities=["text", "image"],
    loader=partial(
        OpsMMEmbeddingWrapper,
        "OpenSearch-AI/Ops-MM-embedding-v1-7B",
        "79fbd6f0ea5c6cd02424acdb97c3cb4ea4cff0f0",
    ),
    n_parameters=7.1 * 1e9,
    memory_usage_mb=16e3,
    max_tokens=32768,
    embed_dim=3584,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch"],
    reference="https://huggingface.co/OpenSearch-AI/Ops-MM-embedding-v1-7B",
    similarity_fn_name=None,
    use_instructions=True,
    training_datasets={
        "MMEB-train": ["train"],
        "CC-3M": ["train"],
        "ColPali": ["train"],
    },
    superseded_by=None,
)
