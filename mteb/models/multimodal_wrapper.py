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
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode text sentences using the multimodal model.
        
        Args:
            sentences: List of text sentences to encode
            task_name: Task name for instruction formatting
            prompt_type: Type of prompt (query or passage)
            **kwargs: Additional encoding arguments
            
        Returns:
            Numpy array of embeddings
        """
        # Format sentences with instructions if needed
        instruction = self.get_task_instruction(task_name, prompt_type)
        
        # Prepare queries in the format expected by MM-Embed
        queries = [{'txt': sent} for sent in sentences]
        
        logger.info(f"Using instruction: '{instruction}' for task: '{task_name}'")
        
        # Use the model's encode method for text-only encoding
        with torch.no_grad():
            embeddings = self.model.encode(
                queries,
                is_query=(prompt_type == PromptType.query),
                instruction=instruction if instruction else None,
                max_length=self.max_seq_length,
                **kwargs
            )
            
            # Extract the hidden states from the model output
            if isinstance(embeddings, dict) and 'hidden_states' in embeddings:
                embeddings = embeddings['hidden_states']
            
            # Convert to numpy if needed
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
                
        return embeddings

    def encode_multimodal(
        self,
        queries: Sequence[dict[str, Any]],
        passages: Sequence[dict[str, Any]] | None = None,
        instruction: str | None = None,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Encode multimodal queries and passages.
        
        Args:
            queries: List of dictionaries containing 'txt' and optionally 'img' keys
            passages: List of dictionaries containing 'txt' and optionally 'img' keys
            instruction: Instruction for query encoding
            max_length: Maximum sequence length
            **kwargs: Additional encoding arguments
            
        Returns:
            Dictionary with 'query_embeddings' and optionally 'passage_embeddings'
        """
        max_length = max_length or self.max_seq_length
        
        with torch.no_grad():
            # Encode queries
            query_embeddings = self.model.encode(
                queries,
                is_query=True,
                instruction=instruction,
                max_length=max_length,
                **kwargs
            )
            
            result = {'query_embeddings': query_embeddings['hidden_states']}
            
            # Encode passages if provided
            if passages is not None:
                passage_embeddings = self.model.encode(
                    passages,
                    is_query=False,
                    max_length=max_length,
                    **kwargs
                )
                result['passage_embeddings'] = passage_embeddings['hidden_states']
            
            # Convert to numpy arrays
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.cpu().numpy()
                    
        return result

    def get_image_embeddings(
        self,
        images: list[Image.Image] | DataLoader,
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
                if isinstance(batch, dict) and 'img' in batch:
                    queries.extend([{'img': img} for img in batch['img']])
                else:
                    queries.extend([{'img': img} for img in batch])
        
        with torch.no_grad():
            embeddings = self.model.encode(
                queries,
                is_query=False,
                max_length=self.max_seq_length,
                **kwargs
            )
            
            if isinstance(embeddings, dict) and 'hidden_states' in embeddings:
                embeddings = embeddings['hidden_states']
            
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
                
        return embeddings

    def get_text_embeddings(
        self,
        texts: list[str],
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
        
        with torch.no_grad():
            embeddings = self.model.encode(
                queries,
                is_query=False,
                max_length=self.max_seq_length,
                **kwargs
            )
            
            if isinstance(embeddings, dict) and 'hidden_states' in embeddings:
                embeddings = embeddings['hidden_states']
            
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
                
        return embeddings

    def get_fused_embeddings(
        self,
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
        if texts is None and images is None:
            raise ValueError("Either texts or images must be provided")
        
        # Create multimodal queries
        queries = []
        if texts is not None and images is not None:
            if len(texts) != len(images):
                raise ValueError("Number of texts and images must match")
            queries = [{'txt': text, 'img': img} for text, img in zip(texts, images)]
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
        
        with torch.no_grad():
            embeddings = self.model.encode(
                queries,
                is_query=False,
                max_length=self.max_seq_length,
                **kwargs
            )
            
            if isinstance(embeddings, dict) and 'hidden_states' in embeddings:
                embeddings = embeddings['hidden_states']
            
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
                
        return embeddings