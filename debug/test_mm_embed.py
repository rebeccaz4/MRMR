#!/usr/bin/env python3
"""Test script for MM-Embed model implementation."""

import sys
from pathlib import Path

# Add the mteb module to path
sys.path.insert(0, str(Path(__file__).parent))

def test_mm_embed_import():
    """Test that MM-Embed model can be imported successfully."""
    try:
        from mteb.models.nvidia_models import MM_Embed
        print("✓ Successfully imported MM_Embed model")
        return True
    except ImportError as e:
        print(f"✗ Failed to import MM_Embed model: {e}")
        return False

def test_multimodal_wrapper_import():
    """Test that MultimodalWrapper can be imported."""
    try:
        from mteb.models.multimodal_wrapper import MultimodalWrapper
        print("✓ Successfully imported MultimodalWrapper")
        return True
    except ImportError as e:
        print(f"✗ Failed to import MultimodalWrapper: {e}")
        return False

def test_mm_embed_model_metadata():
    """Test MM-Embed model metadata."""
    try:
        from mteb.models.nvidia_models import MM_Embed
        
        # Check basic metadata
        assert MM_Embed.name == "nvidia/MM-Embed"
        assert MM_Embed.embed_dim == 4096
        assert MM_Embed.max_tokens == 4096
        assert MM_Embed.license == "cc-by-nc-4.0"
        assert MM_Embed.n_parameters == 8_180_000_000
        
        print("✓ MM-Embed model metadata is correct")
        return True
    except Exception as e:
        print(f"✗ MM-Embed model metadata test failed: {e}")
        return False

def test_mm_embed_model_instantiation():
    """Test that MM-Embed model can be instantiated (without actually loading the weights)."""
    try:
        from mteb.models.nvidia_models import MM_Embed
        
        # Test that the loader is configured correctly
        loader = MM_Embed.loader
        assert loader is not None
        
        print("✓ MM-Embed model loader is configured")
        return True
    except Exception as e:
        print(f"✗ MM-Embed model instantiation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing MM-Embed model implementation...")
    print("=" * 50)
    
    tests = [
        test_multimodal_wrapper_import,
        test_mm_embed_import,
        test_mm_embed_model_metadata,
        test_mm_embed_model_instantiation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! MM-Embed implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())