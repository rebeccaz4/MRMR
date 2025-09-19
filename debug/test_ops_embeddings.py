#!/usr/bin/env python3
"""
Comprehensive test script for OpsMMEmbeddingWrapper functionality
"""

import sys
import os
import numpy as np
sys.path.insert(0, '/home/siyue/Projects/mm-mteb')

def test_text_embeddings():
    """Test text embedding functionality"""
    try:
        from mteb.models.ops_moa_models import OpsMMEmbeddingWrapper
        
        # Initialize wrapper
        wrapper = OpsMMEmbeddingWrapper(
            model_name="OpenSearch-AI/Ops-MM-embedding-v1-7B",
            device="cuda:2",
            max_length=512
        )
        
        # Test text embeddings
        test_texts = [
            "Hello, world!",
            "This is a test sentence.",
            "Machine learning is fascinating."
        ]
        
        print("Testing text embeddings...")
        embeddings = wrapper.get_text_embeddings(test_texts, batch_size=2)
        
        print(f"✓ Text embeddings shape: {embeddings.shape}")
        print(f"✓ Expected shape: ({len(test_texts)}, embed_dim)")
        print(f"✓ Embedding dtype: {embeddings.dtype}")
        
        # Check embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"✓ Embedding norms (should be ~1.0): {norms}")
        
        if embeddings.shape[0] == len(test_texts) and embeddings.shape[1] > 0:
            print("✓ Text embeddings test passed!")
            return True
        else:
            print("✗ Text embeddings test failed!")
            return False
            
    except Exception as e:
        print(f"✗ Text embeddings test failed: {e}")
        return False

def test_image_embeddings():
    """Test image embedding functionality"""
    try:
        from mteb.models.ops_moa_models import OpsMMEmbeddingWrapper
        from PIL import Image
        import numpy as np
        
        # Initialize wrapper
        wrapper = OpsMMEmbeddingWrapper(
            model_name="OpenSearch-AI/Ops-MM-embedding-v1-7B",
            device="cuda:2",
            max_length=512
        )
        
        # Create test images (simple colored squares)
        test_images = []
        for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            img = Image.new('RGB', (224, 224), color)
            test_images.append(img)
        
        print("Testing image embeddings...")
        embeddings = wrapper.get_image_embeddings(test_images, batch_size=2)
        
        print(f"✓ Image embeddings shape: {embeddings.shape}")
        print(f"✓ Expected shape: ({len(test_images)}, embed_dim)")
        print(f"✓ Embedding dtype: {embeddings.dtype}")
        
        # Check embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"✓ Embedding norms (should be ~1.0): {norms}")
        
        if embeddings.shape[0] == len(test_images) and embeddings.shape[1] > 0:
            print("✓ Image embeddings test passed!")
            return True
        else:
            print("✗ Image embeddings test failed!")
            return False
            
    except Exception as e:
        print(f"✗ Image embeddings test failed: {e}")
        return False

def test_fused_embeddings():
    """Test combined text+image embedding functionality"""
    try:
        from mteb.models.ops_moa_models import OpsMMEmbeddingWrapper
        from PIL import Image
        import numpy as np
        
        # Initialize wrapper
        wrapper = OpsMMEmbeddingWrapper(
            model_name="OpenSearch-AI/Ops-MM-embedding-v1-7B",
            device="cuda:2",
            max_length=512
        )
        
        # Create test data
        test_texts = [
            "A red square",
            "A green square", 
            "A blue square"
        ]
        
        test_images = []
        for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            img = Image.new('RGB', (224, 224), color)
            test_images.append(img)
        
        print("Testing fused embeddings...")
        embeddings = wrapper.get_fused_embeddings(
            texts=test_texts, 
            images=test_images, 
            batch_size=2
        )
        
        print(f"✓ Fused embeddings shape: {embeddings.shape}")
        print(f"✓ Expected shape: ({len(test_texts)}, embed_dim)")
        print(f"✓ Embedding dtype: {embeddings.dtype}")
        
        # Check embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"✓ Embedding norms (should be ~1.0): {norms}")
        
        if embeddings.shape[0] == len(test_texts) and embeddings.shape[1] > 0:
            print("✓ Fused embeddings test passed!")
            return True
        else:
            print("✗ Fused embeddings test failed!")
            return False
            
    except Exception as e:
        print(f"✗ Fused embeddings test failed: {e}")
        return False

def main():
    """Run all embedding tests"""
    print("Testing OpsMMEmbeddingWrapper embedding functionality...")
    print("=" * 60)
    
    tests = [
        test_text_embeddings,
        test_image_embeddings,
        test_fused_embeddings,
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        print("-" * 40)
        results.append(test())
        print()
        
    print("=" * 60)
    print(f"Results: {sum(results)}/{len(results)} embedding tests passed")
    
    if all(results):
        print("🎉 All embedding tests passed! The wrapper is working correctly.")
    else:
        print("❌ Some embedding tests failed. Please check the implementation.")
        
    return all(results)

if __name__ == "__main__":
    main()