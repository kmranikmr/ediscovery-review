import json
import os
import yaml
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Haystack 2.x imports 
#from haystack import Pipeline, Document
from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers import InMemoryBM25Retriever 
# OpenSearch imports for production document indexing
try:
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
    from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False
    print("Warning: OpenSearch integrations not available, using InMemory store")
from haystack.components.generators import OpenAIGenerator, HuggingFaceLocalGenerator

from haystack.components.builders import PromptBuilder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.generators.ollama import OllamaGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_huggingface_generator(model_config):
    """Helper function to create HuggingFace generator with proper task"""
    # Determine task based on model name
    if any(model_type in model_config.model_name.lower() for model_type in ["t5", "flan"]):
        task = "text2text-generation"
    else:
        task = "text-generation"
    
    # Ensure we have adequate generation parameters for classification
    max_tokens = max(model_config.max_tokens, 200)  # Ensure at least 200 tokens for JSON response
    
    return HuggingFaceLocalGenerator(
        model=model_config.model_name,
        task=task,
        generation_kwargs={
            "temperature": model_config.temperature, 
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "top_p": 0.95,
            "pad_token_id": 0  # Ensure proper padding
        }
    )

@dataclass
class ModelConfig:
    """Configuration for different model types"""
    model_name: str
    model_type: str  # 'openai', 'ollama', 'huggingface'
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 1000

class EnhancedEmailSummarizerNode:
    """Enhanced summarizer node with multi-model support"""

    def __init__(self, model_config: ModelConfig, task_type: str = "single_document"):
        self.model_config = model_config
        self.task_type = task_type
        self.prompt_template = self._get_prompt_template()
        self.generator = self._create_generator()

    def _create_generator(self):
        """Create Generator based on model configuration"""
        if self.model_config.model_type == "openai":
            # For Haystack 2.x, use environment variable for serialization compatibility
            from haystack.utils import Secret
            api_key = Secret.from_env_var("OPENAI_API_KEY") if self.model_config.api_key else None
            return OpenAIGenerator(
                model=self.model_config.model_name,
                api_key=api_key,
                generation_kwargs={"temperature": self.model_config.temperature, "max_tokens": self.model_config.max_tokens}
            )
        elif self.model_config.model_type == "ollama":
            return OllamaGenerator(
                model=self.model_config.model_name,
                url= "http://localhost:11434",  # Base URL without /api
                generation_kwargs={"temperature": self.model_config.temperature, "max_tokens": self.model_config.max_tokens}
            )
        elif self.model_config.model_type == "huggingface":
            return create_huggingface_generator(self.model_config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_config.model_type}")

    def _get_prompt_template(self) -> str:
        """Get appropriate prompt template based on task type"""
        if self.task_type == "single_document":
            return """
Analyze this email and extract ALL specific details:

{% for document in documents %}
{{ document.content }}
{% endfor %}

Create a summary with these exact details:
- Budget/Money: List exact dollar amounts
- Timeline: List exact timeframes and deadlines  
- Features: List all technical features mentioned
- People: List names and roles
- Actions: List what needs to be done and when

Be specific and include exact numbers, dates, and names from the email.

Summary:
            """
        elif self.task_type == "family_summarization":
            return """
            You are an expert email analyst. Summarize the following email and its attachments as a cohesive unit.

            Email and Attachments:
            {% for document in documents %}
            {{ document.content }}
            {% endfor %}

            Provide a comprehensive summary that:
            1. Summarizes the main email content
            2. Incorporates key information from attachments
            3. Identifies relationships between email and attachments
            4. Highlights important decisions or action items
            5. Notes any discrepancies or additional context from attachments

            Summary:
            """
        elif self.task_type == "thread_summarization":
            return """
            You are an expert email analyst. Summarize the following email thread chronologically.

            Email Thread:
            {% for document in documents %}
            {{ document.content }}
            {% endfor %}

            Provide a thread summary that:
            1. Shows the conversation flow and progression
            2. Identifies key participants and their roles
            3. Tracks decisions made throughout the thread
            4. Highlights unresolved issues or pending actions
            5. Notes the current status/conclusion

            Thread Summary:
            """
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def to_components(self) -> Dict[str, Any]:
        """Returns the Haystack 2.x components for this node."""
        return {
            "prompt_builder": PromptBuilder(template=self.prompt_template, required_variables=["documents"]),
            "generator": self.generator
        }


class EnhancedEmailQANode:
    """Enhanced QA node with multi-model support"""

    def __init__(self, model_config: ModelConfig, task_type: str = "single_document"):
        self.model_config = model_config
        self.task_type = task_type
        self.prompt_template = self._get_qa_prompt_template()
        self.generator = self._create_generator()

    def _create_generator(self):
        """Create Generator based on model configuration"""
        if self.model_config.model_type == "openai":
            # For Haystack 2.x, use environment variable for serialization compatibility
            from haystack.utils import Secret
            api_key = Secret.from_env_var("OPENAI_API_KEY") if self.model_config.api_key else None
            return OpenAIGenerator(
                model=self.model_config.model_name,
                api_key=api_key,
                generation_kwargs={"temperature": self.model_config.temperature, "max_tokens": self.model_config.max_tokens}
            )
        elif self.model_config.model_type == "ollama":
            return OllamaGenerator(
                model=self.model_config.model_name,
                url=self.model_config.base_url or "http://localhost:11434",  # Base URL without /api
                generation_kwargs={"temperature": self.model_config.temperature, "max_tokens": self.model_config.max_tokens}
            )
        elif self.model_config.model_type == "huggingface":
            return create_huggingface_generator(self.model_config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_config.model_type}")

    def _get_qa_prompt_template(self) -> str:
        """Get QA prompt template"""
        if self.task_type == "single_document":
            return """
            You are an expert document analyst. Answer the following question based on the provided documents.

            Question: {{ query }}

            Retrieved Documents:
            {% for document in documents %}
            ---
            Document {{ loop.index }}:
            {{ document.content }}
            ---
            {% endfor %}

            Instructions:
            1. Carefully review ALL provided documents to find relevant information
            2. Use information from ANY of the documents that help answer the question
            3. If multiple documents contain relevant information, synthesize them in your answer
            4. Cite which document(s) contain the supporting information (e.g., "According to Document 1...")
            5. If the answer requires information from multiple documents, combine them coherently
            6. Only say "Information not found" if NONE of the documents contain relevant information
            7. Be comprehensive but precise - include all relevant details found across documents
            8. If documents contain conflicting information, note the discrepancy

            Answer:
            """
        elif self.task_type == "family_qa":
            return """
            You are an expert document analyst. Answer the following question based on the email and its attachments.

            Question: {{ query }}

            Email and Attachments:
            {% for document in documents %}
            ---
            Document {{ loop.index }}:
            {{ document.content }}
            ---
            {% endfor %}

            Instructions:
            1. Search comprehensively across ALL documents (main email and attachments)
            2. Identify which specific document(s) contain relevant information
            3. Synthesize information from multiple documents if needed
            4. Cite your sources clearly (e.g., "Document 2 shows...", "According to the attachment in Document 3...")
            5. If information spans multiple documents, provide a coherent combined answer
            6. Include specific text snippets that support your answer
            7. Only say "Information not found" if NONE of the documents contain relevant information
            8. Be thorough - don't miss relevant details in any document

            Answer:
            """
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def to_components(self) -> Dict[str, Any]:
        """Returns the Haystack 2.x components for this node."""
        return {
            "prompt_builder": PromptBuilder(template=self.prompt_template, required_variables=["query", "documents"]),
            "generator": self.generator
        }


class EnhancedDocumentClassifierNode:
    """Enhanced classification node with multi-model support"""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.generator = self._create_generator()

    def _create_generator(self):
        """Create Generator based on model configuration"""
        if self.model_config.model_type == "openai":
            # For Haystack 2.x, use environment variable for serialization compatibility
            from haystack.utils import Secret
            api_key = Secret.from_env_var("OPENAI_API_KEY") if self.model_config.api_key else None
            return OpenAIGenerator(
                model=self.model_config.model_name,
                api_key=api_key,
                generation_kwargs={"temperature": self.model_config.temperature, "max_tokens": self.model_config.max_tokens}
            )
        elif self.model_config.model_type == "ollama":
            return OllamaGenerator(
                model=self.model_config.model_name,
                url=self.model_config.base_url or "http://localhost:11434",  # Base URL without /api
                generation_kwargs={"temperature": self.model_config.temperature, "max_tokens": self.model_config.max_tokens}
            )
        elif self.model_config.model_type == "huggingface":
            return create_huggingface_generator(self.model_config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_config.model_type}")

    def create_classification_prompt(self, classifications: List[Dict], metadata: Dict, 
                                   user_inputs: List[str] = None, user_preferences: Dict = None) -> str:
        """Create dynamic classification prompt with simple user input support"""
        metadata_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()]) if metadata else "No metadata provided"

        # Add user preferences context
        user_context = ""
        if user_preferences:
            user_context = "\nUser Review Preferences:\n"
            for key, value in user_preferences.items():
                if value is not None:
                    user_context += f"- {key.replace('_', ' ').title()}: {value}\n"

        # Add user input guidance
        user_input_guidance = ""
        if user_inputs and any(user_inputs):
            user_input_guidance = """

IMPORTANT: User inputs have been provided for some documents. These are free-text notes from reviewers that may include:
- Suggestions about document classification or responsiveness
- Important topics or themes the user identified
- Context about business relevance or legal significance
- Personal observations about the document content
- Keywords or concepts the user found important

Consider these user inputs as helpful context, but always validate against the actual document content.
"""

        tasks_str = ""
        for i, classification in enumerate(classifications, 1):
            tasks_str += f"""
Task {i}: {classification['category']}
Labels: {', '.join(classification['labels'])}
Instructions: {classification['instructions']}
"""

        return f"""
You are a legal document reviewer with AI assistance. Use the following metadata, user input, and document content to perform multiple classification tasks.

Metadata:
{metadata_str}
{user_context}
{user_input_guidance}

{tasks_str}

Document Content:
{{% for document in documents %}}
{{{{ document.content }}}}
{{% endfor %}}

For each document and task, provide:
1. Selected label
2. Confidence score (0.0-1.0)
3. Brief reason for the classification
4. User input consideration (if applicable)
5. Any observations about user input vs AI analysis

Return your response in the following JSON format:
{{
    "results": [
        {{
            "document_id": "Document index or ID",
            "category": "Task name",
            "label": "Selected label",
            "confidence": 0.95,
            "reason": "Brief explanation",
            "user_input_considered": true/false,
            "user_input_alignment": "agrees/disagrees/neutral",
            "requires_human_review": true/false
        }}
    ],
    "summary": {{
        "total_documents": 0,
        "high_confidence_classifications": 0,
        "user_input_conflicts": 0,
        "user_inputs_helpful": 0,
        "processing_notes": "Overall assessment"
    }}
}}

Classification Results:
"""

    def to_components(self) -> Dict[str, Any]:
        """Returns the Haystack 2.x components for this node."""
        return {
            "generator": self.generator
        }


class HaystackRestAPIManager:
    """Enhanced Haystack REST API manager with multi-model support"""

    def __init__(self, model_config: ModelConfig, use_opensearch: bool = True, index_name: str = None):
        """
        Initialize the Haystack REST API Manager
        
        Args:
            model_config: Configuration for the model to use
            use_opensearch: Whether to use OpenSearch (True) or InMemory store (False)
            index_name: Custom index name to use (defaults to "ediscovery_documents")
        """
        self.model_config = model_config
        self.use_opensearch = use_opensearch and OPENSEARCH_AVAILABLE
        self.index_name = index_name or "ediscovery_documents"
        
        # Initialize document store based on availability and preference
        if self.use_opensearch:
            try:
                self.document_store = OpenSearchDocumentStore(
                    hosts=[{
                        "host": os.getenv("OPENSEARCH_HOST", "localhost"), 
                        "port": int(os.getenv("OPENSEARCH_PORT", 9200)), 
                        "scheme": "http"
                    }], 
                    index=self.index_name,
                    embedding_dim=768  # Standard embedding dimension
                )
                logger.info(f"Using OpenSearch document store with index: {self.index_name}")
            except Exception as e:
                logger.warning(f"Failed to connect to OpenSearch: {e}. Falling back to InMemory store")
                self.document_store = InMemoryDocumentStore()
                self.use_opensearch = False
        else:
            self.document_store = InMemoryDocumentStore()
            logger.info("Using InMemory document store for development")

        self.pipelines = {}
        self._initialize_pipelines()
        
        # Initialize standalone generator for direct text generation
        self.generator = self._create_standalone_generator()

    def _create_standalone_generator(self):
        """Create a standalone generator for direct text generation"""
        try:
            if self.model_config.model_type == "openai":
                return OpenAIGenerator(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model=self.model_config.model_name,
                    generation_kwargs={"temperature": self.model_config.temperature, "max_tokens": self.model_config.max_tokens}
                )
            elif self.model_config.model_type == "ollama":
                return OllamaGenerator(
                    model=self.model_config.model_name,
                    url=self.model_config.base_url,
                    generation_kwargs={"temperature": self.model_config.temperature}
                )
            elif self.model_config.model_type == "huggingface":
                return create_huggingface_generator(self.model_config)
            else:
                logger.warning(f"Unknown model type: {self.model_config.model_type}. Using Ollama as fallback.")
                return OllamaGenerator(
                    model="mistral",
                    url="http://localhost:11434",
                    generation_kwargs={"temperature": 0.1}
                )
        except Exception as e:
            logger.error(f"Failed to create standalone generator: {e}")
            return None

    def _initialize_pipelines(self):
        """Initialize all pipelines with model configuration"""
        logger.info(f"Initializing pipelines with {self.model_config.model_type} model: {self.model_config.model_name}")

        # Create pipelines - each pipeline will create its own component instances
        self.pipelines = {
            "summarization": self._create_summarization_pipeline(),
            "family_summarization": self._create_family_summarization_pipeline(),
            "thread_summarization": self._create_thread_summarization_pipeline(),
            "qa": self._create_qa_pipeline(),
            "family_qa": self._create_family_qa_pipeline(),
            "classification": self._create_classification_pipeline()
        }

        logger.info("All pipelines initialized successfully")

    def index_documents(self, documents: List[Document], collection_id: str = "default") -> Dict[str, Any]:
        """Index documents in the document store - only uses collection_id for organization"""
        try:
            # Add collection_id to metadata for document organization - this is the key step!
            for doc in documents:
                if doc.meta is None:
                    doc.meta = {}
                # Ensure collection_id is properly set for each document
                doc.meta["collection_id"] = collection_id
                print(f"DEBUG: Adding document to collection '{collection_id}'")
                
            # Write documents to the document store with overwrite policy to handle existing documents
            try:
                if hasattr(self.document_store, 'write_documents'):
                    # For newer document stores, try with overwrite policy
                    self.document_store.write_documents(documents, policy="OVERWRITE")
                else:
                    # Fallback for older versions
                    self.document_store.write_documents(documents)
            except Exception as write_error:
                if "version conflict" in str(write_error).lower():
                    # Handle version conflicts by trying to delete and re-add
                    print(f"DEBUG: Version conflict detected, attempting to update documents")
                    for doc in documents:
                        try:
                            # Try to delete existing document with same ID
                            if hasattr(doc, 'id') and doc.id:
                                existing_docs = self.document_store.filter_documents(
                                    filters={"field": "id", "operator": "==", "value": doc.id}
                                )
                                if existing_docs:
                                    self.document_store.delete_documents([doc.id])
                        except Exception as del_error:
                            print(f"DEBUG: Could not delete existing document: {del_error}")
                    
                    # Try writing again
                    self.document_store.write_documents(documents)
                else:
                    raise write_error
            
            logger.info(f"Indexed {len(documents)} documents in collection '{collection_id}'")
            return {
                "success": True,
                "documents_indexed": len(documents),
                "collection_id": collection_id,
                "document_store_type": "OpenSearch" if self.use_opensearch else "InMemory",
            }
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_collection_stats(self, collection_id: str = "default") -> Dict[str, Any]:
        """Get statistics about a document collection - uses only collection_id"""
        try:
            # Prepare filter for collection_id
            collection_filter = {"field": "collection_id", "operator": "==", "value": collection_id}
            
            # Get all documents with this collection_id
            all_docs = self.document_store.filter_documents(filters=collection_filter)
            
            # Debug
            print(f"DEBUG: Found {len(all_docs)} documents in collection '{collection_id}'")
            
            return {
                "collection_id": collection_id,
                "document_count": len(all_docs),
                "document_store_type": "OpenSearch" if self.use_opensearch else "InMemory"
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "collection_id": collection_id,
                "document_count": 0,
                "error": str(e)
            }
            
    def process_qa_request(self, retriever_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process a QA request using only collection_id for document filtering"""
        try:
            import time
            start_time = time.time()
            
            # Extract parameters from retriever input
            query = retriever_input.get("query", "")
            collection_id = retriever_input.get("collection_id", "default")
            
            # Custom prompt template (optional)
            custom_prompt = retriever_input.get("prompt_template", None)
            
            # Debug info
            print(f"DEBUG: Processing QA request for collection: {collection_id}")
            print(f"DEBUG: Query: {query}")
            
            # Handle collection filtering intelligently
            if collection_id and collection_id.strip() != "":
                # Create filter for specific collection
                collection_filter = {"field": "collection_id", "operator": "==", "value": collection_id}
                print(f"DEBUG: Using collection filter: {collection_filter}")
            else:
                # For empty collection_id, search existing documents without collection_id filtering
                collection_filter = None
                print("DEBUG: No collection filter - searching all existing documents")
            
            # First, check if documents exist
            try:
                if collection_filter:
                    # Check documents in specific collection
                    all_docs = self.document_store.filter_documents(filters=collection_filter)
                    print(f"DEBUG: Found {len(all_docs)} documents with collection_id '{collection_id}'")
                else:
                    # Check all documents in the document store for existing docs
                    try:
                        # For OpenSearch, try to get a count of all documents
                        if hasattr(self.document_store, 'count_documents'):
                            doc_count = self.document_store.count_documents()
                            print(f"DEBUG: Document store contains {doc_count} total documents")
                            if doc_count > 0:
                                # Get a sample of documents to verify access
                                all_docs = self.document_store.filter_documents(filters={})[:10]
                                print(f"DEBUG: Successfully retrieved {len(all_docs)} sample documents")
                            else:
                                all_docs = []
                        else:
                            # Fallback: try to get documents without filters
                            all_docs = self.document_store.filter_documents(filters={})[:10]
                            print(f"DEBUG: Retrieved {len(all_docs)} documents using filter_documents")
                    except Exception as doc_error:
                        print(f"DEBUG: Error accessing documents: {doc_error}")
                        # Last resort: assume documents exist and try retrieval anyway
                        all_docs = ["dummy"]  # Fake entry to allow pipeline to run
                        print("DEBUG: Using fallback approach - will try pipeline anyway")
                
                # For empty collection_id, always try the pipeline even if document check fails
                if len(all_docs) == 0 and collection_filter is not None:
                    return {
                        "answer": f"No documents found in collection '{collection_id}'",
                        "sources": [],
                        "collection_id": collection_id,
                        "time_taken": time.time() - start_time
                    }
                    
                # Debug: Show sample document metadata
                if all_docs:
                    print(f"DEBUG: Sample document metadata: {all_docs[0].meta}")
                    
            except Exception as e:
                print(f"DEBUG: Error checking documents: {e}")
            
            # Enhanced pipeline input with adaptive top_k based on query complexity
            query_length = len(query.split())
            # Adaptive top_k: more documents for complex queries, minimum 10 for comprehensive coverage
            adaptive_top_k = max(10, min(20, query_length * 2))
            
            # Build pipeline input - only include filters if we have them
            pipeline_input = {
                "retriever": {
                    "query": query,
                    "top_k": adaptive_top_k  # Use adaptive retrieval for better coverage
                }
            }
            
            # Add filters only if we have a specific collection
            if collection_filter:
                pipeline_input["retriever"]["filters"] = collection_filter
            
            # Add debug for pipeline input
            print(f"DEBUG: Pipeline input: {pipeline_input}")
            
            # Run the pipeline
            pipeline = self.pipelines["qa"]
            result = pipeline.run(pipeline_input)
            
            # Debug pipeline result
            print(f"DEBUG: Pipeline result keys: {result.keys()}")
            if "retriever" in result:
                print(f"DEBUG: Retriever result keys: {result['retriever'].keys()}")
                retrieved_docs = result.get("retriever", {}).get("documents", [])
                print(f"DEBUG: Retrieved {len(retrieved_docs)} documents from pipeline")
            
            # Format the response with sources
            documents = result.get("retriever", {}).get("documents", [])
            
            # Debug retrieval result
            print(f"DEBUG: Retrieved {len(documents)} documents from collection '{collection_id}' with adaptive_top_k={adaptive_top_k}")
            
            # Intelligent fallback: if we got very few relevant documents, try broader search
            if len(documents) < 3 and len(all_docs) > 10:
                print("DEBUG: Few documents retrieved, attempting broader search...")
                
                # Try with relaxed filters or broader query
                broader_pipeline_input = {
                    "retriever": {
                        "query": query,
                        "top_k": min(25, len(all_docs))  # Get more documents
                    }
                }
                
                # Only add filters if we have them
                if collection_filter:
                    broader_pipeline_input["retriever"]["filters"] = collection_filter
                
                broader_result = pipeline.run(broader_pipeline_input)
                broader_documents = broader_result.get("retriever", {}).get("documents", [])
                
                if len(broader_documents) > len(documents):
                    print(f"DEBUG: Broader search found {len(broader_documents)} documents, using those")
                    documents = broader_documents
                    result = broader_result
            
            # Filter documents by relevance score if available
            if documents and hasattr(documents[0], 'score'):
                # Sort by score and potentially filter low-scoring documents
                documents = sorted(documents, key=lambda x: getattr(x, 'score', 0), reverse=True)
                
                # Keep documents with reasonable scores (above average - threshold)
                if len(documents) > 5:
                    scores = [getattr(doc, 'score', 0) for doc in documents]
                    avg_score = sum(scores) / len(scores)
                    threshold = avg_score * 0.7  # Keep docs with score > 70% of average
                    
                    filtered_docs = [doc for doc in documents if getattr(doc, 'score', 0) >= threshold]
                    if len(filtered_docs) >= 3:  # Ensure we keep at least some documents
                        documents = filtered_docs[:15]  # Limit to top 15 even after filtering
                        print(f"DEBUG: Filtered to {len(documents)} documents above relevance threshold")
            
            sources = []
            
            # If no documents retrieved by pipeline but documents exist, try fallback approach
            if len(documents) == 0 and len(all_docs) > 0:
                print("DEBUG: Pipeline retrieval failed, using fallback direct document access")
                # Use direct document access as fallback
                documents = all_docs[:5]  # Take first 5 documents
                print(f"DEBUG: Using {len(documents)} documents from direct access")
            
            # Format source documents
            for i, doc in enumerate(documents):
                source = {
                    "id": getattr(doc, 'id', f"doc_{i}"),
                    "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                    "meta": doc.meta
                }
                sources.append(source)
            
            # Generate answer using the documents
            if documents:
                # If we have documents, try to generate an answer
                try:
                    # Create a simple prompt for answer generation
                    context = "\n\n".join([doc.content for doc in documents[:3]])  # Use top 3 docs
                    simple_prompt = f"""Answer the following question based on the provided context:

Question: {query}

Context:
{context}

Answer:"""
                    
                    # Get generator from pipeline
                    generator = self.pipelines["qa"].get_component("generator")
                    if generator:
                        gen_result = generator.run(prompt=simple_prompt)
                        answer = gen_result.get("replies", ["No answer generated"])[0]
                    else:
                        answer = f"Found {len(documents)} relevant documents but could not generate answer"
                        
                except Exception as gen_error:
                    print(f"DEBUG: Error generating answer: {gen_error}")
                    answer = f"Found {len(documents)} relevant documents in collection '{collection_id}'"
            else:
                answer = f"No relevant documents found for query in collection '{collection_id}'"
            
            response = {
                "answer": answer,
                "sources": sources,
                "collection_id": collection_id,
                "documents_found": len(documents),
                "total_in_collection": len(all_docs) if 'all_docs' in locals() else 0,
                "time_taken": time.time() - start_time
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing QA request: {str(e)}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "collection_id": collection_id if 'collection_id' in locals() else "unknown",
                "time_taken": time.time() - start_time if 'start_time' in locals() else 0
            }

    def _create_summarization_pipeline(self) -> Pipeline:
        """Create single document summarization pipeline - OPTIMIZED FOR EMAILS"""
        summarizer_node = EnhancedEmailSummarizerNode(
            model_config=self.model_config,
            task_type="single_document"
        )
        pipeline = Pipeline()
        
        # Smart document cleaner for emails
        cleaner = DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=True,
            remove_repeated_substrings=False  # Keep email headers and patterns
        )
        
        # Smart splitter that handles large emails intelligently
        # Only split if document is very large (>3000 chars) to avoid truncation
        splitter = DocumentSplitter(
            split_by="word", 
            split_length=4000,  # Larger chunks to preserve email structure
            split_overlap=500   # More overlap to maintain context
        )
        
        pipeline.add_component("cleaner", cleaner)
        pipeline.add_component("splitter", splitter)
        pipeline.add_component("prompt_builder", summarizer_node.to_components()["prompt_builder"])
        pipeline.add_component("generator", summarizer_node.to_components()["generator"])

        # Connect with splitting for large emails
        pipeline.connect("cleaner.documents", "splitter.documents")
        pipeline.connect("splitter.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder.prompt", "generator.prompt")
        return pipeline

    def _create_family_summarization_pipeline(self) -> Pipeline:
        """Create family summarization pipeline"""
        family_summarizer_node = EnhancedEmailSummarizerNode(
            model_config=self.model_config,
            task_type="family_summarization"
        )
        pipeline = Pipeline()
        # Create new component instances for this pipeline
        cleaner = DocumentCleaner()
        splitter = DocumentSplitter(split_by="word", split_length=3000, split_overlap=300)
        
        pipeline.add_component("cleaner", cleaner)
        pipeline.add_component("splitter", splitter)
        pipeline.add_component("prompt_builder", family_summarizer_node.to_components()["prompt_builder"])
        pipeline.add_component("generator", family_summarizer_node.to_components()["generator"])

        pipeline.connect("cleaner.documents", "splitter.documents")
        pipeline.connect("splitter.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder.prompt", "generator.prompt")
        return pipeline

    def _create_thread_summarization_pipeline(self) -> Pipeline:
        """Create thread summarization pipeline"""
        thread_summarizer_node = EnhancedEmailSummarizerNode(
            model_config=self.model_config,
            task_type="thread_summarization"
        )
        pipeline = Pipeline()
        # Create new component instances for this pipeline
        cleaner = DocumentCleaner()
        splitter = DocumentSplitter(split_by="word", split_length=1000, split_overlap=50)
        
        pipeline.add_component("cleaner", cleaner)
        pipeline.add_component("splitter", splitter)
        pipeline.add_component("prompt_builder", thread_summarizer_node.to_components()["prompt_builder"])
        pipeline.add_component("generator", thread_summarizer_node.to_components()["generator"])

        pipeline.connect("cleaner.documents", "splitter.documents")
        pipeline.connect("splitter.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder.prompt", "generator.prompt")
        return pipeline

    def _create_qa_pipeline(self) -> Pipeline:
        """Create single document QA pipeline"""
        qa_node = EnhancedEmailQANode(
            model_config=self.model_config,
            task_type="single_document"
        )
        pipeline = Pipeline()
        
        # Create appropriate retriever based on document store type
        if self.use_opensearch:
            # Create OpenSearch retriever with specific settings
            retriever = OpenSearchBM25Retriever(
                document_store=self.document_store,
                top_k=15,  # Retrieve more documents to ensure comprehensive coverage
                filter_policy="merge"  # Ensure filters are properly applied
            )
            print(f"DEBUG: Created OpenSearch retriever for document store with top_k=15")
        else:
            retriever = InMemoryBM25Retriever(
                document_store=self.document_store,
                top_k=15,  # Retrieve more documents for better coverage
                filter_policy="merge"  # Ensure filters are properly applied
            )
            print("DEBUG: Created InMemory retriever with top_k=15")
        
        pipeline.add_component("retriever", retriever)
        pipeline.add_component("prompt_builder", qa_node.to_components()["prompt_builder"])
        pipeline.add_component("generator", qa_node.to_components()["generator"])

        pipeline.connect("retriever.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder.prompt", "generator.prompt")
        return pipeline

    def _create_family_qa_pipeline(self) -> Pipeline:
        """Create family QA pipeline"""
        family_qa_node = EnhancedEmailQANode(
            model_config=self.model_config,
            task_type="family_qa"
        )
        pipeline = Pipeline()
        
        # Create appropriate retriever based on document store type
        if self.use_opensearch:
            retriever = OpenSearchBM25Retriever(
                document_store=self.document_store,
                top_k=20,  # More documents for family QA to handle attachments
                filter_policy="merge"
            )
            print(f"DEBUG: Created OpenSearch family QA retriever with top_k=20")
        else:
            retriever = InMemoryBM25Retriever(
                document_store=self.document_store,
                top_k=20,  # More documents for comprehensive family analysis
                filter_policy="merge"
            )
            print("DEBUG: Created InMemory family QA retriever with top_k=20")
        
        pipeline.add_component("retriever", retriever)
        pipeline.add_component("prompt_builder", family_qa_node.to_components()["prompt_builder"])
        pipeline.add_component("generator", family_qa_node.to_components()["generator"])

        pipeline.connect("retriever.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder.prompt", "generator.prompt")
        return pipeline

    def _create_classification_pipeline(self) -> Pipeline:
        """Create classification pipeline"""
        classifier_node = EnhancedDocumentClassifierNode(
            model_config=self.model_config
        )
        pipeline = Pipeline()
        # Create new component instances for this pipeline
        cleaner = DocumentCleaner()
        splitter = DocumentSplitter(split_by="word", split_length=1000, split_overlap=50)
        
        # Create a simple classification prompt template optimized for HuggingFace models
        classification_template = """
Classify this legal document for privilege and responsiveness.

Document:
{% for document in documents %}
{{ document.content }}
{% endfor %}

Task: Classify the document as:
- Privilege: "privileged" or "non-privileged" 
- Responsiveness: "responsive" or "non-responsive"

Output the classification in JSON format:
{"privilege": "privileged", "responsiveness": "responsive", "confidence": 0.9}

Classification result:"""
        
        pipeline.add_component("cleaner", cleaner)
        pipeline.add_component("splitter", splitter)
        pipeline.add_component("prompt_builder", PromptBuilder(template=classification_template, required_variables=["documents"]))
        pipeline.add_component("generator", classifier_node.to_components()["generator"])

        pipeline.connect("cleaner.documents", "splitter.documents")
        pipeline.connect("splitter.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder.prompt", "generator.prompt")
        return pipeline

    def generate_haystack_yaml_configs(self, output_dir: str = "./rest_api_configs"):
        """Generate Haystack REST API YAML configurations"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Haystack 2.x export to YAML is directly from the pipeline object
        for pipeline_name, pipeline_obj in self.pipelines.items():
            config_path = os.path.join(output_dir, f"{pipeline_name}.yaml")
            pipeline_obj.draw(config_path.replace(".yaml", ".png")) # Optional: visualize pipeline
            yaml_config = pipeline_obj.to_dict() # Serialize pipeline to dictionary
            with open(config_path, "w") as f:
                yaml.dump(yaml_config, f, default_flow_style=False)

            logger.info(f"Generated Haystack REST API config: {config_path}")


# Model configuration factory
def create_model_config(model_type: str, model_name: str, **kwargs) -> ModelConfig:
    """Create model configuration for different providers"""
    configs = {
        "openai": ModelConfig(
            model_name=model_name,
            model_type="openai",
            api_key=kwargs.get("api_key", os.getenv("OPENAI_API_KEY")),
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 1000)
        ),
        "ollama": ModelConfig(
            model_name=model_name,
            model_type="ollama",
            base_url=kwargs.get("base_url", "http://localhost:11434"), # Remove /api to avoid double /api
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 1000)
        ),
        "huggingface": ModelConfig(
            model_name=model_name,
            model_type="huggingface",
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 1000)
        )
    }

    if model_type not in configs:
        raise ValueError(f"Unsupported model type: {model_type}")

    return configs[model_type]

# Example usage and initialization
def initialize_haystack_rest_api(model_type: str = "ollama", model_name: str = "mistral", use_opensearch: bool = True, **kwargs):
    """Initialize Haystack REST API with specified model and document store"""

    # Create model configuration
    model_config = create_model_config(model_type, model_name, **kwargs)

    # Initialize API manager with OpenSearch option
    api_manager = HaystackRestAPIManager(model_config, use_opensearch=use_opensearch)

    # Generate YAML configs for REST API
    api_manager.generate_haystack_yaml_configs()

    return api_manager

# Usage examples:
if __name__ == "__main__":
    # Example 1: Using Ollama with Mistral
    print("Initializing with Ollama/Mistral...")
    api_manager_ollama = initialize_haystack_rest_api(
        model_type="ollama",
        model_name="mistral",
        base_url="http://localhost:11434", # Remove /api to avoid double /api
        temperature=0.1
    )

    # Example 2: Using OpenAI GPT-4 (only if API key is available)
    if os.getenv("OPENAI_API_KEY"):
        print("Initializing with OpenAI GPT-4...")
        try:
            api_manager_openai = initialize_haystack_rest_api(
                model_type="openai",
                model_name="gpt-4",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.1
            )
            print("OpenAI configuration generated successfully!")
        except Exception as e:
            print(f"Failed to initialize OpenAI: {e}")
    else:
        print("Skipping OpenAI initialization - OPENAI_API_KEY not found")

    # Example 3: Using Hugging Face model (optional)
    try:
        print("Initializing with Hugging Face model...")
        api_manager_hf = initialize_haystack_rest_api(
            model_type="huggingface",
            model_name="distilbert/distilbert-base-uncased", # A common small model for demonstration
            temperature=0.1
        )
        print("Hugging Face configuration generated successfully!")
    except Exception as e:
        print(f"Failed to initialize Hugging Face: {e}")

    print("\nOllama configurations generated successfully!")
    print("You can now start the Haystack REST API server with:")
    print("haystack serve rest-api --pipeline-config-path ./rest_api_configs/")
    print("\nOr for a specific pipeline:")
    print("haystack serve rest-api --pipeline-config-path ./rest_api_configs/summarization.yaml")