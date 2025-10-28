#!/usr/bin/env python3
"""
Debug test to identify the BFloat16 issue
"""

import sys
import os
sys.path.insert(0, '/home/siyue/Projects/mm-mteb')

def debug_bfloat16_issue():
    """Debug the BFloat16 issue step by step"""
    try:
        from mteb.models.ops_moa_models import OpsMMEmbeddingWrapper
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU 2 available: {torch.cuda.device_count() > 2}")
        
        # Initialize wrapper
        print("Initializing wrapper...")
        wrapper = OpsMMEmbeddingWrapper(
            model_name="OpenSearch-AI/Ops-MM-embedding-v1-7B",
            device="cuda:2",
            max_length=512
        )
        print("✓ Wrapper initialized successfully")
        
        # Test simple text processing
        print("Testing simple text processing...")
        test_texts = ["Hello, world!"]
        
        # Format texts with instruction template
        input_texts = []
        for text in test_texts:
            msg = f"<|im_start|>system\n{wrapper.default_instruction}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>"
            input_texts.append(msg)
        
        print("✓ Text formatting complete")
        
        # Process texts
        print("Processing with tokenizer...")
        inputs = wrapper.processor(
            text=input_texts,
            padding=True,
            truncation=True,
            max_length=wrapper.max_length,
            return_tensors="pt"
        )
        print(f"✓ Processor output keys: {inputs.keys()}")
        
        # Check tensor dtypes before moving to device
        for key, value in inputs.items():
            if hasattr(value, 'dtype'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        print("Moving tensors to device...")
        inputs = {k: v.to(wrapper.device) for k, v in inputs.items()}
        print("✓ Tensors moved to device")
        
        # Check tensor dtypes after moving to device
        for key, value in inputs.items():
            if hasattr(value, 'dtype'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
        
        print("Running model inference...")
        batch_embeddings = wrapper._encode_input(inputs)
        print(f"✓ Model inference complete: {batch_embeddings.shape}, dtype={batch_embeddings.dtype}")
        
        print("Converting to numpy...")
        numpy_embeddings = batch_embeddings.cpu().float().numpy()
        print(f"✓ Numpy conversion complete: {numpy_embeddings.shape}, dtype={numpy_embeddings.dtype}")
        
        return True
        
    except Exception as e:
        print(f"✗ Debug test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_bfloat16_issue()