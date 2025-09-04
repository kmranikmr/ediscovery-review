#!/usr/bin/env python3
"""
Fixed Indexing and QA Implementation
Provides robust document indexing and retrieval for Ollama and HuggingFace servers
"""

import json
import logging
from typing import List, Dict, Any, Optional
from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

class ImprovedDocumentManager:
    """
    Improved document management with proper indexing and retrieval
    """
    
    def __init__(self, use_opensearch=False, opensearch_host="localhost", opensearch_port=9200):
        self.use_opensearch = use_opensearch
        self.document_stores = {}  # Store per collection
        self.retrievers = {}  # Retriever per collection
        self.collection_stats = {}  # Stats per collection
        
        if use_opensearch:
            try:
                from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
                from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever
                self.opensearch_available = True
                self.opensearch_host = opensearch_host
                self.opensearch_port = opensearch_port
                print("✅ OpenSearch integration available")
            except ImportError:
                print("⚠️ OpenSearch not available, falling back to InMemory")
                self.opensearch_available = False
                self.use_opensearch = False
        else:
            self.opensearch_available = False
    
    def _get_document_store(self, collection_id: str, index_name: Optional[str] = None):
        """Get or create document store for collection"""
        store_key = f"{collection_id}:{index_name or 'default'}"
        
        if store_key not in self.document_stores:
            if self.use_opensearch and self.opensearch_available:
                from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
                actual_index = index_name or f"documents_{collection_id}"
                self.document_stores[store_key] = OpenSearchDocumentStore(
                    hosts=[f"http://{self.opensearch_host}:{self.opensearch_port}"],
                    index=actual_index
                )
                print(f"✅ Created OpenSearch store for collection '{collection_id}' with index '{actual_index}'")
            else:
                self.document_stores[store_key] = InMemoryDocumentStore()
                print(f"✅ Created InMemory store for collection '{collection_id}'")
        
        return self.document_stores[store_key]
    
    def _get_retriever(self, collection_id: str, index_name: Optional[str] = None):
        """Get or create retriever for collection"""
        store_key = f"{collection_id}:{index_name or 'default'}"
        
        if store_key not in self.retrievers:
            document_store = self._get_document_store(collection_id, index_name)
            
            if self.use_opensearch and self.opensearch_available:
                from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever
                self.retrievers[store_key] = OpenSearchBM25Retriever(document_store=document_store)
            else:
                self.retrievers[store_key] = InMemoryBM25Retriever(document_store=document_store)
            
            print(f"✅ Created retriever for collection '{collection_id}'")
        
        return self.retrievers[store_key]
    
    def index_documents(self, documents: List[Document], collection_id: str, index_name: Optional[str] = None) -> Dict[str, Any]:
        """Index documents with proper error handling and stats tracking"""
        try:
            # Add collection_id to document metadata
            for doc in documents:
                if not doc.meta:
                    doc.meta = {}
                doc.meta["collection_id"] = collection_id
                if index_name:
                    doc.meta["index_name"] = index_name
            
            # Get document store and index documents
            document_store = self._get_document_store(collection_id, index_name)
            document_store.write_documents(documents)
            
            # Update stats
            store_key = f"{collection_id}:{index_name or 'default'}"
            if store_key not in self.collection_stats:
                self.collection_stats[store_key] = {
                    "collection_id": collection_id,
                    "index_name": index_name,
                    "document_count": 0,
                    "last_updated": None
                }
            
            self.collection_stats[store_key]["document_count"] += len(documents)
            self.collection_stats[store_key]["last_updated"] = str(self._get_current_time())
            
            print(f"✅ Indexed {len(documents)} documents in collection '{collection_id}'")
            
            return {
                "success": True,
                "indexed_count": len(documents),
                "collection_id": collection_id,
                "index_name": index_name,
                "total_documents": self.collection_stats[store_key]["document_count"],
                "message": f"Successfully indexed {len(documents)} documents"
            }
            
        except Exception as e:
            print(f"❌ Indexing error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "collection_id": collection_id
            }
    
    def retrieve_documents(self, query: str, collection_id: str, index_name: Optional[str] = None, top_k: int = 5) -> List[Document]:
        """Retrieve relevant documents for a query"""
        try:
            retriever = self._get_retriever(collection_id, index_name)
            
            # Add collection filter
            filters = {"field": "collection_id", "operator": "==", "value": collection_id}
            
            # Retrieve documents
            result = retriever.run(query=query, top_k=top_k, filters=filters)
            documents = result.get("documents", [])
            
            print(f"🔍 Retrieved {len(documents)} documents for query in collection '{collection_id}'")
            return documents
            
        except Exception as e:
            print(f"❌ Retrieval error: {str(e)}")
            return []
    
    def get_collection_stats(self, collection_id: str, index_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a collection"""
        store_key = f"{collection_id}:{index_name or 'default'}"
        
        if store_key in self.collection_stats:
            return self.collection_stats[store_key]
        
        # Try to get stats from document store
        try:
            document_store = self._get_document_store(collection_id, index_name)
            count = document_store.count_documents()
            
            stats = {
                "collection_id": collection_id,
                "index_name": index_name,
                "document_count": count,
                "last_updated": str(self._get_current_time())
            }
            
            self.collection_stats[store_key] = stats
            return stats
            
        except Exception as e:
            print(f"❌ Stats error: {str(e)}")
            return {
                "collection_id": collection_id,
                "index_name": index_name,
                "document_count": 0,
                "error": str(e)
            }
    
    def list_all_collections(self) -> Dict[str, Dict[str, Any]]:
        """List all collections with their stats"""
        all_collections = {}
        
        for store_key, stats in self.collection_stats.items():
            collection_id = stats["collection_id"]
            all_collections[collection_id] = stats
        
        return all_collections
    
    def _get_current_time(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

# Global document manager instances
document_managers = {}

def get_document_manager(use_opensearch=False) -> ImprovedDocumentManager:
    """Get or create document manager"""
    key = "opensearch" if use_opensearch else "memory"
    
    if key not in document_managers:
        document_managers[key] = ImprovedDocumentManager(use_opensearch=use_opensearch)
    
    return document_managers[key]

def create_improved_qa_pipeline(llm_component, use_opensearch=False):
    """Create improved QA pipeline with proper retrieval"""
    from haystack import Pipeline
    from haystack.components.builders import PromptBuilder
    
    # Create pipeline
    pipeline = Pipeline()
    
    # Add prompt builder
    prompt_template = """
    You are an expert assistant. Answer the following question based on the provided documents.
    If the answer cannot be found in the documents, say "I cannot find the answer in the provided documents."
    
    Documents:
    {% for document in documents %}
    {{ document.content }}
    ---
    {% endfor %}
    
    Question: {{ query }}
    
    Answer:
    """
    
    pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
    pipeline.add_component("llm", llm_component)
    
    return pipeline

def run_improved_qa(pipeline, query: str, collection_id: str, index_name: Optional[str] = None, use_opensearch=False) -> Dict[str, Any]:
    """Run QA with improved document retrieval"""
    try:
        # Get document manager and retrieve documents
        doc_manager = get_document_manager(use_opensearch=use_opensearch)
        documents = doc_manager.retrieve_documents(query, collection_id, index_name)
        
        if not documents:
            return {
                "success": False,
                "error": f"No documents found in collection '{collection_id}'. Please index documents first.",
                "answer": "No relevant documents found."
            }
        
        # Run QA pipeline
        result = pipeline.run({
            "prompt_builder": {
                "query": query,
                "documents": documents
            }
        })
        
        # Extract answer
        if "llm" in result and "replies" in result["llm"]:
            answer = result["llm"]["replies"][0] if result["llm"]["replies"] else "No answer generated"
        else:
            answer = "Error: Could not generate answer"
        
        return {
            "success": True,
            "answer": answer,
            "retrieved_docs": len(documents),
            "collection_id": collection_id,
            "index_name": index_name
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "answer": f"Error during QA: {str(e)}"
        }

if __name__ == "__main__":
    print("Improved Document Indexing and QA System")
    print("=" * 50)
    
    # Test the document manager
    doc_manager = get_document_manager(use_opensearch=False)
    
    # Test documents
    test_docs = [
        Document(
            content="The Q4 budget for the analytics project is $200,000.",
            meta={"document_id": "budget_001", "type": "email"}
        ),
        Document(
            content="The project timeline is 6 months with expected ROI of 25%.",
            meta={"document_id": "timeline_001", "type": "email"}
        )
    ]
    
    # Test indexing
    result = doc_manager.index_documents(test_docs, "test_collection")
    print(f"Indexing result: {result}")
    
    # Test stats
    stats = doc_manager.get_collection_stats("test_collection")
    print(f"Collection stats: {stats}")
    
    # Test retrieval
    docs = doc_manager.retrieve_documents("budget", "test_collection")
    print(f"Retrieved {len(docs)} documents")
