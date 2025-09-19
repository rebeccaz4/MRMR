"""
Simple test case for NVIDIA MM-Embed model using cuda:1
"""

import mteb
import torch
import os
from PIL import Image
import numpy as np

# Use cuda:1 to avoid memory conflicts
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def test_mm_embed_basic():
    """Test basic MM-Embed functionality"""
    print("🧪 Testing MM-Embed Basic Functionality")
    print("=" * 50)
    
    try:
        # Load MM-Embed model
        model = mteb.get_model('nvidia/MM-Embed', device='cuda:0')  # cuda:0 will map to cuda:1 due to CUDA_VISIBLE_DEVICES
        print("✅ MM-Embed loaded successfully!")
        
        # Test text encoding with CUB200I2IRetrieval task
        test_sentences = [
            "A beautiful red cardinal bird",
            "A blue jay sitting on a tree branch", 
            "A small sparrow with brown feathers"
        ]
        
        print("\n📝 Testing text encoding...")
        embeddings = model.encode(
            test_sentences, 
            task_name='CUB200I2IRetrieval',
            prompt_type=mteb.encoder_interface.PromptType.query
        )
        
        print(f"✅ Text encoding successful!")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
        print(f"   Number of sentences: {embeddings.shape[0]}")
        
        # Check embedding properties
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"   Embedding norms: {norms}")
        print(f"   Normalized: {np.allclose(norms, 1.0, atol=1e-3)}")
        
        # Test similarity computation
        similarity_matrix = np.dot(embeddings, embeddings.T)
        print(f"   Similarity matrix shape: {similarity_matrix.shape}")
        print(f"   Self-similarities (diagonal): {np.diag(similarity_matrix)}")
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mm_embed_multimodal_direct():
    """Test MM-Embed multimodal encoding directly"""
    print("\n🖼️ Testing MM-Embed Multimodal Capabilities")
    print("=" * 50)
    
    try:
        model = mteb.get_model('nvidia/MM-Embed', device='cuda:0')
        
        # Test if multimodal methods are available
        if hasattr(model, 'encode_multimodal'):
            print("✅ Multimodal encoding method found")
            
            # Create simple test image
            test_image = Image.new('RGB', (224, 224), color='red')
            
            # Test multimodal queries
            queries = [
                {'txt': 'What color is this bird?', 'img': test_image},
                {'txt': 'A red bird flying'},  # text-only
            ]
            
            passages = [
                {'txt': 'This is a red cardinal bird with bright red feathers.'},
                {'txt': 'This is a blue jay with blue and white plumage.'},
            ]
            
            result = model.encode_multimodal(
                queries=queries,
                passages=passages,
                instruction="Retrieve information about birds based on the query."
            )
            
            print(f"✅ Multimodal encoding successful!")
            print(f"   Query embeddings shape: {result['query_embeddings'].shape}")
            print(f"   Passage embeddings shape: {result['passage_embeddings'].shape}")
            
            # Calculate similarity scores
            query_emb = result['query_embeddings']
            passage_emb = result['passage_embeddings']
            scores = (query_emb @ passage_emb.T) * 100
            
            print(f"   Similarity scores: {scores.tolist()}")
            print("✅ Multimodal test passed!")
            return True
            
        else:
            print("⚠️ Multimodal encoding method not available")
            return False
            
    except Exception as e:
        print(f"❌ Multimodal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run MM-Embed tests"""
    print("🚀 Starting MM-Embed Tests with cuda:1")
    print("=" * 60)
    
    # Test 1: Basic functionality
    basic_success = test_mm_embed_basic()
    
    # Test 2: Multimodal capabilities (if basic test passes)
    multimodal_success = False
    if basic_success:
        multimodal_success = test_mm_embed_multimodal_direct()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    print(f"Basic Text Encoding: {'✅ PASSED' if basic_success else '❌ FAILED'}")
    print(f"Multimodal Encoding: {'✅ PASSED' if multimodal_success else '❌ FAILED'}")
    
    total_passed = sum([basic_success, multimodal_success])
    print(f"\nOverall: {total_passed}/2 tests passed")
    
    if basic_success:
        print("🎉 MM-Embed is working in MTEB!")
        print("✅ Ready for evaluation on vision tasks like CUB200I2IRetrieval")
    else:
        print("❌ MM-Embed needs debugging")

if __name__ == "__main__":
    main()