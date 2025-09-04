#!/usr/bin/env python3
"""
Quick test script to verify haystack imports work in Docker
"""

def test_imports():
    print("Testing critical imports...")
    
    try:
        import haystack
        print(f"✅ haystack imported successfully - version: {haystack.__version__}")
    except ImportError as e:
        print(f"❌ haystack import failed: {e}")
        return False
    
    try:
        from haystack_integrations.components.generators.ollama import OllamaGenerator
        print("✅ ollama-haystack integration imported successfully")
    except ImportError as e:
        print(f"❌ ollama-haystack import failed: {e}")
        return False
    
    try:
        from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
        print("✅ opensearch-haystack integration imported successfully")
    except ImportError as e:
        print(f"❌ opensearch-haystack import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ transformers imported successfully - version: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ transformers import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✅ torch imported successfully - version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        return False
        
    print("\n🎉 All critical imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
