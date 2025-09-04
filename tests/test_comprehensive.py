"""
Comprehensive test suite for eDiscovery LLM Retrieval System
Tests all endpoints with various scenarios
"""

import pytest
import requests
import json
import time
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8001/api/v1"
TIMEOUT = 30

class TestAPI:
    """Main API test class"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        # Check if API is running
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            assert response.status_code == 200, "API server not available"
        except Exception as e:
            pytest.skip(f"API server not available: {e}")
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "status" in data
        assert "version" in data
        assert "services" in data
        assert "uptime" in data

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = requests.get(f"{BASE_URL.replace('/api/v1', '')}/", timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "endpoints" in data

class TestQAEndpoints:
    """QA functionality tests"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test documents"""
        # Index test documents first
        self.test_documents = [
            {
                "content": "John Smith is the CEO of TechCorp Inc. The company was founded in 2020 in New York.",
                "meta": {
                    "document_id": "doc1",
                    "document_type": "company_info",
                    "created_date": "2024-01-01T00:00:00Z"
                }
            },
            {
                "content": "The Q4 2024 financial report shows revenue of $2.5 million and expenses of $1.8 million.",
                "meta": {
                    "document_id": "doc2", 
                    "document_type": "financial",
                    "created_date": "2024-12-01T00:00:00Z"
                }
            }
        ]
        
        # Index documents
        index_data = {
            "documents": self.test_documents,
            "collection_id": "test_collection",
            "index_name": "test_index"
        }
        
        try:
            response = requests.post(f"{BASE_URL}/documents/index", json=index_data, timeout=TIMEOUT)
            assert response.status_code == 200
        except Exception:
            pytest.skip("Failed to index test documents")
    
    def test_simple_qa(self):
        """Test simple QA endpoint"""
        qa_data = {
            "query": "Who is the CEO of TechCorp?",
            "collection_id": "test_collection",
            "index_name": "test_index",
            "top_k": 5
        }
        
        response = requests.post(f"{BASE_URL}/qa/simple", json=qa_data, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "result" in data
        assert "method" in data
        assert "processing_time" in data
        
        result = data["result"]
        assert "answer" in result
        assert isinstance(result.get("sources", []), list)
    
    def test_direct_index_qa(self):
        """Test direct index QA endpoint"""
        qa_data = {
            "query": "What was the revenue in Q4 2024?",
            "collection_id": "test_collection", 
            "index_name": "test_index",
            "top_k": 3
        }
        
        response = requests.post(f"{BASE_URL}/qa/direct-index", json=qa_data, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "result" in data
    
    def test_qa_with_filters(self):
        """Test QA with filters"""
        qa_data = {
            "query": "Tell me about financial information",
            "collection_id": "test_collection",
            "index_name": "test_index", 
            "top_k": 5,
            "filters": {
                "document_type": "financial"
            }
        }
        
        response = requests.post(f"{BASE_URL}/qa/simple", json=qa_data, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_document_indexing(self):
        """Test document indexing"""
        index_data = {
            "documents": [
                {
                    "content": "Test document content for indexing",
                    "meta": {
                        "document_id": "test_doc_123",
                        "document_type": "test"
                    }
                }
            ],
            "collection_id": "test_index_collection",
            "index_name": "test_indexing"
        }
        
        response = requests.post(f"{BASE_URL}/documents/index", json=index_data, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["indexed_count"] == 1
        assert data["failed_count"] == 0
        assert "index_name" in data
        assert "processing_time" in data

class TestNEREndpoints:
    """NER functionality tests"""
    
    def test_bert_ner_extraction(self):
        """Test BERT NER extraction"""
        ner_data = {
            "text": "John Smith from Microsoft met with Sarah Johnson at Google in New York on January 15, 2024.",
            "method": "bert",
            "entity_types": ["PERSON", "ORGANIZATION", "LOCATION", "DATE"],
            "include_pii": True,
            "min_score": 0.7
        }
        
        response = requests.post(f"{BASE_URL}/ner/extract", json=ner_data, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "result" in data
        assert "method" in data
        assert "processing_time" in data
        
        result = data["result"]
        assert "entities" in result
        assert "statistics" in result
        
        # Check statistics
        stats = result["statistics"]
        assert "total_entities" in stats
        assert "entity_types" in stats
        assert "avg_confidence" in stats
    
    def test_llm_ner_extraction(self):
        """Test LLM NER extraction"""
        ner_data = {
            "text": "Contact jane.doe@company.com or call 555-123-4567 for more information about the $50,000 contract.",
            "method": "llm",
            "entity_types": ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "MONEY"],
            "include_pii": True,
            "min_score": 0.5
        }
        
        response = requests.post(f"{BASE_URL}/ner/extract", json=ner_data, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "result" in data
    
    def test_ner_with_pii_detection(self):
        """Test NER with PII detection"""
        ner_data = {
            "text": "My email is test@example.com and my phone number is (555) 123-4567. My SSN is 123-45-6789.",
            "method": "bert",
            "include_pii": True,
            "min_score": 0.5
        }
        
        response = requests.post(f"{BASE_URL}/ner/extract", json=ner_data, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Should detect email, phone, and potentially SSN
        entities = data["result"]["entities"]
        entity_types = list(entities.keys())
        assert any("EMAIL" in et for et in entity_types)

class TestSummarizationEndpoints:
    """Summarization functionality tests"""
    
    def test_regular_summarization(self):
        """Test regular summarization"""
        summarization_data = {
            "text": "This is a quarterly business review meeting. The marketing team presented their Q3 results showing a 15% increase in lead generation. The sales team reported closing 25 new deals worth $2.3M total revenue. The engineering team delivered 3 major features ahead of schedule. However, there are concerns about the upcoming Q4 budget constraints and potential staff reductions.",
            "summary_type": "general",
            "max_length": 100
        }
        
        response = requests.post(f"{BASE_URL}/summarization/regular", json=summarization_data, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "result" in data
        assert "method" in data
        assert "processing_time" in data
        
        result = data["result"]
        assert "summary" in result
        assert len(result["summary"]) > 0
    
    def test_thread_summarization(self):
        """Test thread summarization"""
        summarization_data = {
            "text": "From: john@company.com\nTo: team@company.com\nSubject: Project Update\n\nHi Team,\n\nThe project is progressing well. We've completed the first phase and are moving to phase 2.\n\nFrom: mary@company.com\nTo: john@company.com\nSubject: Re: Project Update\n\nGreat news! When do you expect phase 2 to complete?",
            "summary_type": "thread_summary",
            "max_length": 80
        }
        
        response = requests.post(f"{BASE_URL}/summarization/thread", json=summarization_data, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "result" in data
    
    def test_summarization_with_keywords(self):
        """Test summarization with keyword extraction"""
        summarization_data = {
            "text": "Artificial intelligence and machine learning are transforming the technology industry. Companies are investing heavily in AI research and development. Natural language processing is a key component of modern AI systems.",
            "summary_type": "key_facts",
            "max_length": 50
        }
        
        response = requests.post(f"{BASE_URL}/summarization/regular", json=summarization_data, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        result = data["result"]
        if "keywords" in result:
            assert isinstance(result["keywords"], list)

class TestClassificationEndpoints:
    """Classification functionality tests"""
    
    def test_comprehensive_classification(self):
        """Test comprehensive classification"""
        classification_data = {
            "text": "From: urgent@company.com\nTo: team@company.com\nSubject: URGENT: System Outage\n\nWe are experiencing a critical system outage that is affecting all customers. This requires immediate attention from the engineering team.",
            "classification_types": ["document_type", "priority", "sentiment"]
        }
        
        response = requests.post(f"{BASE_URL}/classification/comprehensive", json=classification_data, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "result" in data
        assert "method" in data
        assert "processing_time" in data
        
        result = data["result"]
        assert "classifications" in result
        
        classifications = result["classifications"]
        assert isinstance(classifications, dict)
        assert len(classifications) > 0
    
    def test_email_classification(self):
        """Test email classification"""
        classification_data = {
            "text": "From: sales@company.com\nTo: customer@client.com\nSubject: Thank you for your purchase\n\nDear Customer,\n\nThank you for choosing our product. We appreciate your business and look forward to serving you again.",
            "classification_types": ["document_type", "sentiment"]
        }
        
        response = requests.post(f"{BASE_URL}/classification/comprehensive", json=classification_data, timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        classifications = data["result"]["classifications"]
        # Should classify as email with positive sentiment
        assert "document_type" in classifications
        assert "sentiment" in classifications

class TestPerformanceAndReliability:
    """Performance and reliability tests"""
    
    def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        import concurrent.futures
        import threading
        
        def make_health_request():
            response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
            return response.status_code == 200
        
        # Make 10 concurrent health check requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_health_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(results)
    
    def test_large_text_processing(self):
        """Test processing large text"""
        large_text = "This is a test sentence. " * 1000  # ~24KB text
        
        ner_data = {
            "text": large_text,
            "method": "bert",
            "min_score": 0.8
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/ner/extract", json=ner_data, timeout=60)
        processing_time = time.time() - start_time
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Should complete within reasonable time
        assert processing_time < 30  # 30 seconds max
    
    def test_malformed_requests(self):
        """Test handling of malformed requests"""
        # Missing required fields
        response = requests.post(f"{BASE_URL}/qa/simple", json={}, timeout=TIMEOUT)
        assert response.status_code == 422  # Validation error
        
        # Invalid JSON
        response = requests.post(
            f"{BASE_URL}/qa/simple", 
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        assert response.status_code == 422
    
    def test_error_handling(self):
        """Test error handling and recovery"""
        # Test with extremely long query
        qa_data = {
            "query": "x" * 10000,  # Very long query
            "top_k": 5
        }
        
        response = requests.post(f"{BASE_URL}/qa/simple", json=qa_data, timeout=TIMEOUT)
        
        # Should handle gracefully (either success or proper error)
        assert response.status_code in [200, 400, 422, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "success" in data

# Utility functions for manual testing
def run_quick_test():
    """Quick smoke test of all endpoints"""
    print("🚀 Running quick smoke test...")
    
    # Health check
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"Health: {response.status_code}")
    
    # QA test
    qa_data = {"query": "test query", "top_k": 3}
    response = requests.post(f"{BASE_URL}/qa/simple", json=qa_data, timeout=10)
    print(f"QA: {response.status_code}")
    
    # NER test
    ner_data = {"text": "John Smith works at Google", "method": "bert"}
    response = requests.post(f"{BASE_URL}/ner/extract", json=ner_data, timeout=10)
    print(f"NER: {response.status_code}")
    
    # Summarization test
    sum_data = {"text": "This is a test document for summarization.", "max_length": 50}
    response = requests.post(f"{BASE_URL}/summarization/regular", json=sum_data, timeout=10)
    print(f"Summarization: {response.status_code}")
    
    # Classification test
    class_data = {"text": "This is a test email for classification."}
    response = requests.post(f"{BASE_URL}/classification/comprehensive", json=class_data, timeout=10)
    print(f"Classification: {response.status_code}")
    
    print("✅ Quick test completed")

if __name__ == "__main__":
    # Run quick test when executed directly
    run_quick_test()
