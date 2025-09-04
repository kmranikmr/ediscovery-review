#!/usr/bin/env python3
"""
Enhanced OpenSearch Query Support
Allows users to pass raw OpenSearch queries directly
"""

from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field, validator
import json
import logging

logger = logging.getLogger(__name__)

class RawOpenSearchRequest(BaseModel):
    """Raw OpenSearch query request"""
    index_name: str = Field(..., description="Target index name")
    query_body: Dict[str, Any] = Field(..., description="Complete OpenSearch query body")
    include_qa: bool = Field(default=False, description="Include QA answer generation")
    qa_query: Optional[str] = Field(default=None, description="Question for QA if include_qa=True")
    validate_query: bool = Field(default=True, description="Validate query syntax before execution")
    
    @validator('query_body')
    def validate_opensearch_query(cls, v):
        """Basic validation of OpenSearch query structure"""
        if not isinstance(v, dict):
            raise ValueError("query_body must be a dictionary")
        
        # Check for required or dangerous operations
        if '_delete_by_query' in v or 'delete' in str(v).lower():
            raise ValueError("Delete operations not allowed")
        
        return v

class AdvancedQueryRequest(BaseModel):
    """Advanced query with both text query and raw OpenSearch components"""
    text_query: str = Field(..., description="Natural language query")
    index_name: str = Field(..., description="Target index name") 
    
    # Raw OpenSearch components
    raw_query: Optional[Dict[str, Any]] = Field(default=None, description="Raw OpenSearch query clause")
    raw_filters: Optional[Dict[str, Any]] = Field(default=None, description="Raw OpenSearch filter clause")
    raw_aggregations: Optional[Dict[str, Any]] = Field(default=None, description="Raw OpenSearch aggregations")
    raw_sort: Optional[List[Dict[str, Any]]] = Field(default=None, description="Raw OpenSearch sort clause")
    raw_highlight: Optional[Dict[str, Any]] = Field(default=None, description="Raw OpenSearch highlight clause")
    
    # Standard options
    size: int = Field(default=10, description="Number of results")
    from_: int = Field(default=0, alias="from", description="Offset for pagination")
    include_qa: bool = Field(default=True, description="Include QA answer generation")
    
class OpenSearchQueryBuilder:
    """Helper class to build and validate OpenSearch queries"""
    
    def __init__(self):
        self.forbidden_operations = [
            '_delete_by_query', 'delete', '_update_by_query', 
            'create', 'index', '_bulk', '_reindex'
        ]
    
    def build_hybrid_query(self, request: AdvancedQueryRequest) -> Dict[str, Any]:
        """Build hybrid query combining text search and raw OpenSearch components"""
        
        query_body = {
            "size": request.size,
            "from": request.from_
        }
        
        # Build query section
        if request.raw_query:
            # Use raw query directly
            query_body["query"] = request.raw_query
        else:
            # Build text-based query with optional raw filters
            text_query = {
                "multi_match": {
                    "query": request.text_query,
                    "fields": ["content^2", "meta.title^1.5", "meta.*"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            }
            
            if request.raw_filters:
                query_body["query"] = {
                    "bool": {
                        "must": [text_query],
                        "filter": [request.raw_filters]
                    }
                }
            else:
                query_body["query"] = text_query
        
        # Add other raw components
        if request.raw_aggregations:
            query_body["aggs"] = request.raw_aggregations
        
        if request.raw_sort:
            query_body["sort"] = request.raw_sort
        
        if request.raw_highlight:
            query_body["highlight"] = request.raw_highlight
        else:
            # Default highlight
            query_body["highlight"] = {
                "fields": {
                    "content": {"fragment_size": 150, "number_of_fragments": 3}
                }
            }
        
        return query_body
    
    def validate_query_safety(self, query_body: Dict[str, Any]) -> bool:
        """Validate that query doesn't contain dangerous operations"""
        query_str = json.dumps(query_body).lower()
        
        for forbidden_op in self.forbidden_operations:
            if forbidden_op in query_str:
                raise ValueError(f"Forbidden operation detected: {forbidden_op}")
        
        return True
    
    def optimize_query(self, query_body: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize query for better performance"""
        optimized = query_body.copy()
        
        # Add source filtering to reduce response size
        if "_source" not in optimized:
            optimized["_source"] = ["content", "meta"]
        
        # Add timeout
        if "timeout" not in optimized:
            optimized["timeout"] = "30s"
        
        # Limit aggregation size if not specified
        if "aggs" in optimized:
            for agg_name, agg_config in optimized["aggs"].items():
                if "terms" in agg_config and "size" not in agg_config["terms"]:
                    agg_config["terms"]["size"] = 50
        
        return optimized

def execute_raw_opensearch_query(
    request: RawOpenSearchRequest,
    opensearch_client
) -> Dict[str, Any]:
    """Execute raw OpenSearch query with safety checks"""
    
    builder = OpenSearchQueryBuilder()
    
    # Validate query safety
    if request.validate_query:
        builder.validate_query_safety(request.query_body)
    
    # Optimize query
    optimized_query = builder.optimize_query(request.query_body)
    
    logger.info(f"Executing raw OpenSearch query on index: {request.index_name}")
    logger.debug(f"Query body: {json.dumps(optimized_query, indent=2)}")
    
    try:
        # Execute query
        response = opensearch_client.search(
            index=request.index_name,
            body=optimized_query
        )
        
        result = {
            "hits": response.get("hits", {}),
            "aggregations": response.get("aggregations", {}),
            "took": response.get("took", 0),
            "timed_out": response.get("timed_out", False),
            "total_hits": response.get("hits", {}).get("total", {}).get("value", 0)
        }
        
        # Generate QA answer if requested
        if request.include_qa and request.qa_query:
            qa_answer = generate_qa_from_hits(
                request.qa_query, 
                response.get("hits", {}).get("hits", [])
            )
            result["qa_answer"] = qa_answer
        
        return result
        
    except Exception as e:
        logger.error(f"OpenSearch query execution failed: {e}")
        raise

def execute_advanced_query(
    request: AdvancedQueryRequest,
    opensearch_client
) -> Dict[str, Any]:
    """Execute advanced hybrid query"""
    
    builder = OpenSearchQueryBuilder()
    
    # Build hybrid query
    query_body = builder.build_hybrid_query(request)
    
    # Validate and optimize
    builder.validate_query_safety(query_body)
    optimized_query = builder.optimize_query(query_body)
    
    logger.info(f"Executing advanced query on index: {request.index_name}")
    logger.debug(f"Query body: {json.dumps(optimized_query, indent=2)}")
    
    try:
        # Execute query
        response = opensearch_client.search(
            index=request.index_name,
            body=optimized_query
        )
        
        result = {
            "hits": response.get("hits", {}),
            "aggregations": response.get("aggregations", {}),
            "took": response.get("took", 0),
            "timed_out": response.get("timed_out", False),
            "total_hits": response.get("hits", {}).get("total", {}).get("value", 0)
        }
        
        # Generate QA answer if requested
        if request.include_qa:
            qa_answer = generate_qa_from_hits(
                request.text_query, 
                response.get("hits", {}).get("hits", [])
            )
            result["qa_answer"] = qa_answer
        
        return result
        
    except Exception as e:
        logger.error(f"Advanced query execution failed: {e}")
        raise

def generate_qa_from_hits(query: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate QA answer from search hits"""
    if not hits:
        return {
            "answer": "No relevant documents found for the query.",
            "confidence": 0.0,
            "sources_used": 0
        }
    
    # Extract content from hits
    sources = []
    context_parts = []
    
    for hit in hits[:5]:  # Limit to top 5 for QA
        source_data = hit.get('_source', {})
        content = source_data.get('content', '')
        
        if content:
            sources.append({
                "content": content[:500],  # Truncate for display
                "metadata": source_data.get('meta', {}),
                "score": hit.get('_score', 0),
                "highlights": hit.get('highlight', {})
            })
            context_parts.append(content[:1000])  # More content for QA
    
    if not context_parts:
        return {
            "answer": "Found documents but no readable content available.",
            "confidence": 0.1,
            "sources_used": len(hits)
        }
    
    # This would integrate with your existing QA generation logic
    # For now, return a placeholder
    return {
        "answer": f"Based on {len(sources)} documents, information related to '{query}' was found. [QA generation would be integrated here]",
        "confidence": 0.8,
        "sources_used": len(sources),
        "sources": sources
    }

# Example usage patterns
QUERY_EXAMPLES = {
    "raw_boolean_query": {
        "query": {
            "bool": {
                "must": [
                    {"match": {"content": "contract"}},
                    {"range": {"meta.date": {"gte": "2024-01-01"}}}
                ],
                "should": [
                    {"match": {"meta.author": "legal@company.com"}},
                    {"match": {"meta.department": "legal"}}
                ],
                "must_not": [
                    {"term": {"meta.status": "draft"}}
                ],
                "minimum_should_match": 1
            }
        },
        "aggs": {
            "by_author": {
                "terms": {"field": "meta.author.keyword", "size": 10}
            }
        }
    },
    
    "semantic_search_with_filters": {
        "query": {
            "bool": {
                "must": [
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                                "params": {"query_vector": [0.1, 0.2, 0.3]}  # Would be actual embeddings
                            }
                        }
                    }
                ],
                "filter": [
                    {"term": {"meta.type": "email"}},
                    {"range": {"meta.date": {"gte": "2024-01-01"}}}
                ]
            }
        }
    },
    
    "complex_aggregation_query": {
        "size": 0,  # Only aggregations
        "query": {"match_all": {}},
        "aggs": {
            "by_department": {
                "terms": {"field": "meta.department.keyword"},
                "aggs": {
                    "by_month": {
                        "date_histogram": {
                            "field": "meta.date",
                            "calendar_interval": "month"
                        },
                        "aggs": {
                            "avg_importance": {
                                "avg": {"field": "meta.importance_score"}
                            }
                        }
                    }
                }
            }
        }
    }
}
