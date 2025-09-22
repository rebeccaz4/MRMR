from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from mteb.encoder_interface import ImageEncoder, PromptType
from mteb.models.wrapper import Wrapper
from mteb.requires_package import requires_package

logger = logging.getLogger(__name__)


class MultimodalWrapper(Wrapper, ImageEncoder):
    """Wrapper for multimodal models that support both text and image inputs like MM-Embed."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        instruction_template: str | callable | None = None,
        max_seq_length: int | None = None,
        device: str | None = None,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ):
        """Initialize the multimodal wrapper.
        
        Args:
            model_name: HuggingFace model name
            revision: Model revision
            instruction_template: Template for formatting instructions
            max_seq_length: Maximum sequence length
            device: Device to use for inference
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional arguments for model loading
        """
        requires_package(
            MultimodalWrapper, "transformers", model_name, "pip install transformers"
        )
        requires_package(
            MultimodalWrapper, "torch", model_name, "pip install torch"
        )
        requires_package(
            MultimodalWrapper, "PIL", model_name, "pip install pillow"
        )
        
        from transformers import AutoModel, AutoTokenizer
        
        self.model_name = model_name
        self.revision = revision
        self.instruction_template = instruction_template
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if max_seq_length is not None:
            self.max_seq_length = max_seq_length
        else:
            # Default max length for MM-Embed
            self.max_seq_length = 4096



    def encode(
        self,
        queries: list[dict],
        instruction: str,
        is_query: bool,
        max_length: int | None = 4096,
        batch_size: int = 16,
        show_progress_bar: bool = False,
        convert_to_tensor: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode a list of queries (text, image, or multimodal) into embeddings with batching.

        Args:
            queries: List of dicts, e.g., {'txt': ..., 'img': ...} or just {'txt': ...} / {'img': ...}
            is_query: Whether this is a query (vs. corpus)
            max_length: Maximum sequence length
            batch_size: Number of samples per batch
            show_progress_bar: Whether to show a progress bar
            convert_to_tensor: Whether to return a torch.Tensor (True) or numpy.ndarray (False)
            **kwargs: Additional arguments for model.encode
        Returns:
            torch.Tensor or np.ndarray of embeddings
        """
        all_embeddings = []
        max_length = max_length or self.max_seq_length
        print("batch_size", batch_size)
        # 如果需要显示进度条
        iterator = range(0, len(queries), batch_size)
        if show_progress_bar:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Encoding batches")

        for i in iterator:
            batch = queries[i:i + batch_size]

            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch,
                    is_query=is_query,
                    max_length=max_length,
                    instruction=instruction,
                    **kwargs
                )

                if isinstance(batch_embeddings, dict) and 'hidden_states' in batch_embeddings:
                    batch_embeddings = batch_embeddings['hidden_states']

                # 转成 numpy
                if isinstance(batch_embeddings, torch.Tensor):
                    batch_embeddings = batch_embeddings.cpu().numpy()

                all_embeddings.append(batch_embeddings)

        all_embeddings = np.concatenate(all_embeddings, axis=0)

        if convert_to_tensor:
            return torch.from_numpy(all_embeddings)
        return all_embeddings

    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Get embeddings for images only.
        
        Args:
            images: List of PIL Images or DataLoader
            **kwargs: Additional arguments
            
        Returns:
            Numpy array of image embeddings
        """
        # Convert images to the format expected by MM-Embed
        
        if isinstance(images, list):
            queries = [{'img': img} for img in images]
        else:
            # Handle DataLoader case
            queries = []
            for batch in images:
                print(type(batch))
                if isinstance(batch, dict) and 'img' in batch:
                    queries.extend([{'img': img} for img in batch['img']])
                else:
                    queries.extend([{'img': img} for img in batch])
                    
        instruction = Wrapper.get_instruction(task_name=task_name, prompt_type=prompt_type)
        
        is_query = True if prompt_type==PromptType.query else False
        print("is_query", is_query)
        
        return self.encode(queries, is_query=is_query, max_length=self.max_seq_length, instruction=instruction, **kwargs)



    def get_text_embeddings(
        self,
        texts: list[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Get embeddings for text only.
        
        Args:
            texts: List of text strings
            **kwargs: Additional arguments
            
        Returns:
            Numpy array of text embeddings
        """
        queries = [{'txt': text} for text in texts]
        instruction = Wrapper.get_instruction(task_name=task_name, prompt_type=prompt_type)
        
        is_query = True if prompt_type==PromptType.query else False
        print("is_query", is_query)
        
        return self.encode(queries, is_query=is_query, max_length=self.max_seq_length, instruction=instruction, **kwargs)

    def get_fused_embeddings(
        self,
        task_name: str,
        prompt_type: PromptType | None = None,
        texts: list[str] | None = None,
        images: list[Image.Image] | DataLoader | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Get fused embeddings for text and image pairs.
        
        Args:
            texts: List of text strings
            images: List of PIL Images or DataLoader
            **kwargs: Additional arguments
            
        Returns:
            Numpy array of fused embeddings
        """
        instruction = Wrapper.get_instruction(task_name=task_name, prompt_type=prompt_type)
        
        is_query = True if prompt_type==PromptType.query else False
        print("is_query", is_query)
        
        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")
        
        # Create multimodal queries
        queries = []
        if texts is not None and images is not None:
            if len(texts) != len(images):
                raise ValueError("Number of texts and images must match")
            queries = [{'txt': text.replace('<image>', '<image 1>'), 'img': img} for text, img in zip(texts, images)]
        elif texts is not None:
            queries = [{'txt': text} for text in texts]
        else:  # images is not None
            if isinstance(images, list):
                queries = [{'img': img} for img in images]
            else:
                # Handle DataLoader case
                for batch in images:
                    if isinstance(batch, dict) and 'img' in batch:
                        queries.extend([{'img': img} for img in batch['img']])
                    else:
                        queries.extend([{'img': img} for img in batch])
                      
        return self.encode(queries, is_query=is_query, max_length=self.max_seq_length, instruction=instruction, **kwargs)