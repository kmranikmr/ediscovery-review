"""
Haystack Service - Core document processing and QA functionality
Production-ready service for document indexing, QA, summarization, and classification
"""

import logging
import time
import os
from typing import Dict, List, Any, Optional
from haystack import Pipeline
from haystack.dataclasses import Document

from app.core.config import settings

logger = logging.getLogger(__name__)

class HaystackService:
    """Main Haystack service for document processing"""
    
    def __init__(self):
        self.pipelines = {}
        self.document_store = None
        self.retriever = None
        self.initialized = False
        self.ollama_available = False
        
    async def initialize(self):
        """Initialize Haystack service"""
        try:
            logger.info("🔄 Initializing Haystack service...")
            
            # Initialize document store
            await self._initialize_document_store()
            
            # Initialize retriever
            await self._initialize_retriever()
            
            # Initialize pipelines
            await self._initialize_pipelines()
            
            # Check Ollama availability
            await self._check_ollama()
            
            self.initialized = True
            logger.info("✅ Haystack service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Haystack service: {e}")
            raise
    
    async def _initialize_document_store(self):
        """Initialize document store (OpenSearch or InMemory)"""
        try:
            # Try OpenSearch first
            if await self._try_opensearch():
                from haystack.document_stores import OpenSearchDocumentStore
                
                self.document_store = OpenSearchDocumentStore(
                    host=settings.OPENSEARCH_HOST,
                    port=settings.OPENSEARCH_PORT,
                    use_ssl=settings.OPENSEARCH_USE_SSL,
                    verify_certs=settings.OPENSEARCH_VERIFY_CERTS,
                    index="documents"
                )
                logger.info("✅ Using OpenSearch document store")
            else:
                # Fallback to InMemory
                from haystack.document_stores import InMemoryDocumentStore
                self.document_store = InMemoryDocumentStore()
                logger.info("✅ Using InMemory document store (OpenSearch not available)")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize document store: {e}")
            # Final fallback
            from haystack.document_stores import InMemoryDocumentStore
            self.document_store = InMemoryDocumentStore()
            logger.info("✅ Using InMemory document store (fallback)")
    
    async def _try_opensearch(self) -> bool:
        """Test OpenSearch connection"""
        try:
            import aiohttp
            url = f"{'https' if settings.OPENSEARCH_USE_SSL else 'http'}://{settings.OPENSEARCH_HOST}:{settings.OPENSEARCH_PORT}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _initialize_retriever(self):
        """Initialize retriever"""
        try:
            from haystack.nodes import BM25Retriever
            
            self.retriever = BM25Retriever(
                document_store=self.document_store,
                top_k=10
            )
            logger.info("✅ BM25 retriever initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize retriever: {e}")
            raise
    
    async def _initialize_pipelines(self):
        """Initialize processing pipelines"""
        try:
            # QA Pipeline
            self.pipelines["qa"] = await self._create_qa_pipeline()
            
            # Indexing Pipeline  
            self.pipelines["indexing"] = await self._create_indexing_pipeline()
            
            logger.info(f"✅ Initialized {len(self.pipelines)} pipelines")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize pipelines: {e}")
            raise
    
    async def _create_qa_pipeline(self) -> Pipeline:
        """Create QA pipeline"""
        try:
            from haystack.nodes import PromptNode, PromptTemplate
            
            # Create QA prompt template
            qa_template = PromptTemplate(
                prompt="""Answer the following question based on the provided context. If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."

Context:
{join(documents)}

Question: {query}
Answer:""",
                output_parser=lambda x: x
            )
            
            # Create prompt node (will use Ollama if available)
            prompt_node = PromptNode(
                model_name_or_path=settings.OLLAMA_MODEL,
                api_key="", # Not needed for Ollama
                default_prompt_template=qa_template,
                model_kwargs={"base_url": settings.OLLAMA_BASE_URL}
            )
            
            # Create pipeline
            pipeline = Pipeline()
            pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])
            pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])
            
            return pipeline
            
        except Exception as e:
            logger.error(f"❌ Failed to create QA pipeline: {e}")
            # Create a minimal pipeline without LLM
            pipeline = Pipeline()
            pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])
            return pipeline
    
    async def _create_indexing_pipeline(self) -> Pipeline:
        """Create document indexing pipeline"""
        try:
            from haystack.nodes import PreProcessor
            
            # Create preprocessor
            preprocessor = PreProcessor(
                clean_empty_lines=True,
                clean_whitespace=True,
                split_by="word",
                split_length=1000,
                split_respect_sentence_boundary=True,
                split_overlap=50
            )
            
            # Create pipeline
            pipeline = Pipeline()
            pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["File"])
            pipeline.add_node(component=self.document_store, name="DocumentStore", inputs=["PreProcessor"])
            
            return pipeline
            
        except Exception as e:
            logger.error(f"❌ Failed to create indexing pipeline: {e}")
            raise
    
    async def _check_ollama(self):
        """Check Ollama availability"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{settings.OLLAMA_BASE_URL}/api/tags") as response:
                    if response.status == 200:
                        self.ollama_available = True
                        logger.info("✅ Ollama service available")
                    else:
                        logger.warning("⚠️ Ollama service not responding")
        except Exception as e:
            logger.warning(f"⚠️ Ollama not available: {e}")
    
    async def process_qa(self, 
                        query: str, 
                        collection_id: Optional[str] = None,
                        index_name: Optional[str] = None,
                        top_k: int = 5,
                        filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process QA request"""
        start_time = time.time()
        
        try:
            if not self.initialized:
                raise RuntimeError("Haystack service not initialized")
            
            # Prepare filters
            if filters is None:
                filters = {}
            if collection_id:
                filters["collection_id"] = collection_id
            
            # Update retriever top_k
            self.retriever.top_k = top_k
            
            # Run QA pipeline
            if "PromptNode" in [node.name for node in self.pipelines["qa"].graph.nodes()]:
                # Full QA with LLM
                result = self.pipelines["qa"].run(
                    query=query,
                    params={
                        "Retriever": {"filters": filters} if filters else {}
                    }
                )
                
                answer = result.get("answers", [{}])[0].get("answer", "No answer generated")
                documents = result.get("documents", [])
                
            else:
                # Retrieval only (no LLM)
                result = self.retriever.retrieve(query=query, filters=filters)
                documents = result
                
                # Create a simple answer from top documents
                if documents:
                    answer = f"Found {len(documents)} relevant documents. Top document: {documents[0].content[:200]}..."
                else:
                    answer = "No relevant documents found"
            
            # Format response
            return {
                "success": True,
                "result": {
                    "answer": answer,
                    "sources": [
                        {
                            "content": doc.content,
                            "meta": doc.meta,
                            "score": getattr(doc, "score", 0.0)
                        }
                        for doc in documents[:top_k]
                    ],
                    "total_documents_searched": len(documents)
                },
                "method": "haystack_ollama" if self.ollama_available else "haystack_retrieval",
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"QA processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def index_documents(self, 
                             documents: List[Dict[str, Any]],
                             collection_id: Optional[str] = None,
                             index_name: Optional[str] = None) -> Dict[str, Any]:
        """Index documents"""
        start_time = time.time()
        
        try:
            if not self.initialized:
                raise RuntimeError("Haystack service not initialized")
            
            # Convert to Haystack documents
            haystack_docs = []
            for doc_data in documents:
                meta = doc_data.get("meta", {})
                if collection_id:
                    meta["collection_id"] = collection_id
                if index_name:
                    meta["index_name"] = index_name
                
                doc = Document(
                    content=doc_data["content"],
                    meta=meta
                )
                haystack_docs.append(doc)
            
            # Index documents
            self.document_store.write_documents(haystack_docs)
            
            return {
                "success": True,
                "indexed_count": len(haystack_docs),
                "failed_count": 0,
                "index_name": index_name or "default",
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "indexed_count": 0,
                "failed_count": len(documents),
                "processing_time": time.time() - start_time
            }
    
    async def process_summarization(self, 
                                   text: str,
                                   summary_type: str = "regular",
                                   max_length: int = 150) -> Dict[str, Any]:
        """Process summarization request"""
        start_time = time.time()
        
        try:
            if self.ollama_available:
                result = await self._summarize_with_ollama(text, summary_type, max_length)
            else:
                # Fallback to simple extraction
                result = await self._simple_summarization(text, max_length)
            
            return {
                "success": True,
                "result": result,
                "method": "ollama_direct" if self.ollama_available else "extractive",
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _summarize_with_ollama(self, text: str, summary_type: str, max_length: int) -> Dict[str, Any]:
        """Summarize using Ollama"""
        try:
            import aiohttp
            
            prompt = f"""Summarize the following text in approximately {max_length} words. Focus on {summary_type} aspects.

Text: {text}

Summary:"""

            payload = {
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        summary = result.get("response", "").strip()
                        
                        return {
                            "summary": summary,
                            "keywords": self._extract_keywords(text),
                            "summary_type": summary_type
                        }
                    else:
                        raise Exception(f"Ollama API error: {response.status}")
                        
        except Exception as e:
            logger.error(f"Ollama summarization failed: {e}")
            return await self._simple_summarization(text, max_length)
    
    async def _simple_summarization(self, text: str, max_length: int) -> Dict[str, Any]:
        """Simple extractive summarization fallback"""
        sentences = text.split(". ")
        if len(sentences) <= 3:
            summary = text
        else:
            # Take first and last sentences, plus middle one
            summary = ". ".join([
                sentences[0],
                sentences[len(sentences)//2],
                sentences[-1]
            ])
        
        return {
            "summary": summary,
            "keywords": self._extract_keywords(text),
            "summary_type": "extractive"
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction"""
        # Remove common stop words and extract important terms
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = text.lower().split()
        keywords = [word.strip(".,!?;:") for word in words if len(word) > 3 and word not in stop_words]
        
        # Count frequency and return top keywords
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:10]
    
    async def process_classification(self, text: str) -> Dict[str, Any]:
        """Process classification request"""
        start_time = time.time()
        
        try:
            if self.ollama_available:
                result = await self._classify_with_ollama(text)
            else:
                result = await self._simple_classification(text)
            
            return {
                "success": True,
                "result": result,
                "method": "ollama_direct" if self.ollama_available else "rule_based",
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _classify_with_ollama(self, text: str) -> Dict[str, Any]:
        """Classify using Ollama"""
        try:
            import aiohttp
            
            prompt = f"""Classify the following text according to these categories:

Document Type: email, report, contract, memo, legal, financial, technical, other
Priority: high, medium, low
Sentiment: positive, negative, neutral

Text: {text}

Return only a JSON object with the classifications:
{{
  "document_type": "category",
  "priority": "level", 
  "sentiment": "sentiment"
}}

JSON:"""

            payload = {
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get("response", "")
                        
                        # Parse JSON response
                        import json
                        try:
                            json_start = response_text.find("{")
                            json_end = response_text.rfind("}") + 1
                            if json_start >= 0 and json_end > json_start:
                                classification_json = response_text[json_start:json_end]
                                classification = json.loads(classification_json)
                                return {"classifications": classification}
                        except json.JSONDecodeError:
                            pass
                            
        except Exception as e:
            logger.error(f"Ollama classification failed: {e}")
        
        return await self._simple_classification(text)
    
    async def _simple_classification(self, text: str) -> Dict[str, Any]:
        """Simple rule-based classification fallback"""
        text_lower = text.lower()
        
        # Document type classification
        if any(word in text_lower for word in ["from:", "to:", "subject:", "@"]):
            doc_type = "email"
        elif any(word in text_lower for word in ["contract", "agreement", "terms"]):
            doc_type = "contract"
        elif any(word in text_lower for word in ["report", "analysis", "findings"]):
            doc_type = "report"
        else:
            doc_type = "other"
        
        # Priority classification
        urgent_words = ["urgent", "asap", "immediately", "critical", "emergency"]
        if any(word in text_lower for word in urgent_words):
            priority = "high"
        elif len(text) > 1000:
            priority = "medium"
        else:
            priority = "low"
        
        # Sentiment classification
        positive_words = ["good", "great", "excellent", "pleased", "happy", "success"]
        negative_words = ["bad", "terrible", "disappointed", "failed", "problem", "issue"]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = "positive"
        elif neg_count > pos_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "classifications": {
                "document_type": doc_type,
                "priority": priority,
                "sentiment": sentiment
            }
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            "status": "healthy" if self.initialized else "unhealthy",
            "initialized": self.initialized,
            "document_store_type": type(self.document_store).__name__ if self.document_store else None,
            "pipelines_loaded": len(self.pipelines),
            "ollama_available": self.ollama_available
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Close document store connections if needed
            if hasattr(self.document_store, 'client'):
                self.document_store.client.close()
            logger.info("✅ Haystack service cleanup complete")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
