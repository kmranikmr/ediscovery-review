# OpenSearch QA System - Current Analysis & Improvement Plan

## Current Implementation Overview

### Working Features ✅
1. **Direct Index Access**: `/qa-direct-index` endpoint for direct OpenSearch queries
2. **Basic Filtering**: Document ID, type, author, date range filtering
3. **Streamlit UI**: Interactive filter interface with real-time preview
4. **Multiple QA Types**: Regular QA and Family QA support
5. **Index Selection**: Uses existing `deephousedeephouse_ediscovery_docs_chunks` index

### Current Filter Capabilities
```json
{
  "document_id": "doc123",
  "type": "email", 
  "author": "john@company.com",
  "date": {
    "range": {
      "gte": "2001-01-01T00:00:00",
      "lte": "2001-12-31T23:59:59"
    }
  }
}
```

## Identified Limitations & Improvements

### 1. Limited Filter Types 🔄
**Current**: Basic term, range, and list filters
**Improvement Needed**: Advanced OpenSearch query capabilities

### 2. Static Filter Interface 🔄
**Current**: Fixed set of filter fields in Streamlit
**Improvement Needed**: Dynamic filter discovery based on index schema

### 3. No Query Composition 🔄
**Current**: Single query + basic filters
**Improvement Needed**: Complex boolean queries with nested conditions

### 4. No Aggregation Support 🔄
**Current**: Simple document retrieval
**Improvement Needed**: Faceted search, statistical aggregations

### 5. Limited Search Intelligence 🔄
**Current**: Basic text matching
**Improvement Needed**: Semantic search, relevance scoring, query expansion

## Proposed Enhancements

### Phase 1: Enhanced Filtering Engine

#### 1.1 Advanced Filter Types
```python
# Current basic filters
{
  "field": "value",
  "field": ["val1", "val2"],
  "field": {"range": {"gte": "2024-01-01"}}
}

# Enhanced filter capabilities
{
  "filters": {
    "bool": {
      "must": [
        {"term": {"meta.department": "legal"}},
        {"range": {"meta.date": {"gte": "2024-01-01"}}}
      ],
      "should": [
        {"match": {"content": "contract"}},
        {"match": {"content": "agreement"}}
      ],
      "must_not": [
        {"term": {"meta.status": "draft"}}
      ],
      "filter": [
        {"exists": {"field": "meta.importance"}},
        {"wildcard": {"meta.participants": "*@legal.company.com"}}
      ]
    }
  },
  "aggregations": {
    "by_department": {
      "terms": {"field": "meta.department.keyword"}
    },
    "by_date": {
      "date_histogram": {
        "field": "meta.date",
        "calendar_interval": "month"
      }
    }
  }
}
```

#### 1.2 Dynamic Schema Discovery
```python
def discover_index_schema(index_name: str) -> Dict[str, Any]:
    """Discover available fields and their types in an index"""
    client = create_opensearch_client()
    mapping = client.indices.get_mapping(index=index_name)
    
    schema = {
        "text_fields": [],
        "keyword_fields": [],
        "date_fields": [],
        "numeric_fields": [],
        "nested_fields": [],
        "common_values": {}  # Field -> common values for dropdowns
    }
    
    # Parse mapping to extract field types
    # ... implementation details
    
    return schema
```

#### 1.3 Smart Filter Builder
```python
class SmartFilterBuilder:
    def __init__(self, index_schema: Dict[str, Any]):
        self.schema = index_schema
    
    def build_filters(self, user_filters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert user-friendly filters to OpenSearch query"""
        query = {"bool": {"must": [], "should": [], "filter": []}}
        
        for field, value in user_filters.items():
            filter_clause = self._build_field_filter(field, value)
            query["bool"]["filter"].append(filter_clause)
        
        return query
    
    def _build_field_filter(self, field: str, value: Any) -> Dict[str, Any]:
        """Build appropriate filter based on field type and value"""
        field_type = self.schema.get("field_types", {}).get(field, "keyword")
        
        if field_type == "date":
            return self._build_date_filter(field, value)
        elif field_type == "numeric":
            return self._build_numeric_filter(field, value)
        elif field_type == "text":
            return self._build_text_filter(field, value)
        else:
            return self._build_keyword_filter(field, value)
```

### Phase 2: Intelligent Query Processing

#### 2.1 Query Expansion & Semantic Search
```python
class IntelligentQueryProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.query_expander = QueryExpander()
    
    def enhance_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance user query with semantic understanding"""
        
        # 1. Extract entities and intent
        entities = self.extract_entities(query)
        intent = self.classify_intent(query)
        
        # 2. Expand query with synonyms and related terms
        expanded_terms = self.query_expander.expand(query)
        
        # 3. Build semantic search component
        semantic_query = self.build_semantic_query(query)
        
        # 4. Combine lexical and semantic search
        enhanced_query = {
            "bool": {
                "should": [
                    # Lexical search (current approach)
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["content^2", "meta.title^1.5", "meta.*"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    },
                    # Expanded terms
                    {
                        "multi_match": {
                            "query": " ".join(expanded_terms),
                            "fields": ["content", "meta.*"],
                            "type": "cross_fields"
                        }
                    },
                    # Semantic search (if vector field available)
                    semantic_query
                ],
                "minimum_should_match": 1
            }
        }
        
        return enhanced_query
```

#### 2.2 Context-Aware Filtering
```python
def build_contextual_filters(query: str, base_filters: Dict[str, Any]) -> Dict[str, Any]:
    """Build filters based on query context and user intent"""
    
    contextual_filters = base_filters.copy()
    
    # Extract temporal context
    if temporal_entities := extract_temporal_entities(query):
        contextual_filters.update(build_temporal_filters(temporal_entities))
    
    # Extract organizational context
    if org_entities := extract_organizational_entities(query):
        contextual_filters.update(build_org_filters(org_entities))
    
    # Extract document type hints
    if doc_type_hints := extract_document_type_hints(query):
        contextual_filters.update(build_doc_type_filters(doc_type_hints))
    
    return contextual_filters

def extract_temporal_entities(query: str) -> List[Dict[str, Any]]:
    """Extract dates, years, quarters, etc. from query"""
    # "What happened in Q3 2024?" -> [{"type": "quarter", "quarter": 3, "year": 2024}]
    # "Show me emails from last week" -> [{"type": "relative", "period": "week", "offset": -1}]
    pass

def extract_organizational_entities(query: str) -> List[str]:
    """Extract department, team, person names"""
    # "What did the legal team discuss?" -> ["legal"]
    # "Show me emails from John Smith" -> ["John Smith"]
    pass
```

### Phase 3: Advanced Search Interface

#### 3.1 Dynamic Filter UI
```python
def render_dynamic_filters(index_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Render filter interface based on discovered schema"""
    
    st.markdown("#### 🔍 Smart Filters")
    
    filters = {}
    
    # Text search filters
    for field in index_schema["text_fields"]:
        if field in ["content", "title", "subject"]:
            value = st.text_input(f"Search in {field}:", key=f"text_{field}")
            if value:
                filters[field] = {"match": {"query": value}}
    
    # Keyword filters with auto-complete
    for field in index_schema["keyword_fields"]:
        common_values = index_schema["common_values"].get(field, [])
        if common_values:
            value = st.selectbox(f"{field}:", [""] + common_values, key=f"keyword_{field}")
            if value:
                filters[field] = value
    
    # Date filters with smart presets
    for field in index_schema["date_fields"]:
        with st.expander(f"📅 {field} Filter"):
            preset = st.selectbox(
                "Quick Presets:",
                ["Custom", "Last 7 days", "Last 30 days", "This Quarter", "This Year", "2024", "2023"],
                key=f"date_preset_{field}"
            )
            
            if preset == "Custom":
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(f"From:", key=f"date_start_{field}")
                with col2:
                    end_date = st.date_input(f"To:", key=f"date_end_{field}")
                
                if start_date and end_date:
                    filters[field] = {
                        "range": {
                            "gte": start_date.isoformat(),
                            "lte": end_date.isoformat()
                        }
                    }
            else:
                # Apply preset logic
                filters[field] = build_date_preset_filter(preset)
    
    # Numeric filters with range sliders
    for field in index_schema["numeric_fields"]:
        min_val, max_val = index_schema["field_ranges"].get(field, (0, 100))
        value_range = st.slider(
            f"{field} Range:",
            min_value=float(min_val),
            max_value=float(max_val),
            value=(float(min_val), float(max_val)),
            key=f"numeric_{field}"
        )
        
        if value_range != (min_val, max_val):
            filters[field] = {
                "range": {
                    "gte": value_range[0],
                    "lte": value_range[1]
                }
            }
    
    return filters
```

#### 3.2 Query Builder Interface
```python
def render_advanced_query_builder() -> Dict[str, Any]:
    """Advanced query builder for power users"""
    
    st.markdown("#### 🛠️ Advanced Query Builder")
    
    with st.expander("Boolean Query Builder"):
        query_parts = []
        
        # MUST conditions
        st.markdown("**MUST (all conditions required):**")
        must_conditions = st_add_remove_list("must_conditions", "Add MUST condition")
        
        # SHOULD conditions  
        st.markdown("**SHOULD (any condition matches):**")
        should_conditions = st_add_remove_list("should_conditions", "Add SHOULD condition")
        
        # MUST_NOT conditions
        st.markdown("**MUST NOT (exclude documents with these):**")
        must_not_conditions = st_add_remove_list("must_not_conditions", "Add MUST NOT condition")
        
        # Build boolean query
        bool_query = {
            "bool": {
                "must": [build_condition(cond) for cond in must_conditions],
                "should": [build_condition(cond) for cond in should_conditions],
                "must_not": [build_condition(cond) for cond in must_not_conditions]
            }
        }
        
        if should_conditions:
            bool_query["bool"]["minimum_should_match"] = st.slider(
                "Minimum SHOULD matches:", 1, len(should_conditions), 1
            )
        
        # Show generated query
        with st.expander("Generated OpenSearch Query"):
            st.json(bool_query)
        
        return bool_query

def st_add_remove_list(key: str, add_label: str) -> List[str]:
    """Helper to create add/remove list interface"""
    if key not in st.session_state:
        st.session_state[key] = []
    
    # Add new item
    new_item = st.text_input(f"New condition:", key=f"{key}_input")
    if st.button(add_label, key=f"{key}_add") and new_item:
        st.session_state[key].append(new_item)
        st.session_state[f"{key}_input"] = ""
        st.experimental_rerun()
    
    # Show and allow removal of existing items
    for i, item in enumerate(st.session_state[key]):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.text(item)
        with col2:
            if st.button("🗑️", key=f"{key}_remove_{i}"):
                st.session_state[key].pop(i)
                st.experimental_rerun()
    
    return st.session_state[key]
```

### Phase 4: Enhanced Result Processing

#### 4.1 Faceted Search Results
```python
def perform_faceted_search(query: str, filters: Dict[str, Any], index_name: str) -> Dict[str, Any]:
    """Perform search with faceted results and aggregations"""
    
    search_body = {
        "query": build_enhanced_query(query, filters),
        "size": 20,
        "aggs": {
            "by_type": {
                "terms": {"field": "meta.type.keyword", "size": 10}
            },
            "by_author": {
                "terms": {"field": "meta.author.keyword", "size": 10}
            },
            "by_date": {
                "date_histogram": {
                    "field": "meta.date",
                    "calendar_interval": "month"
                }
            },
            "by_department": {
                "terms": {"field": "meta.department.keyword", "size": 10}
            }
        },
        "highlight": {
            "fields": {
                "content": {"fragment_size": 150, "number_of_fragments": 3},
                "meta.title": {"fragment_size": 100, "number_of_fragments": 1}
            }
        }
    }
    
    client = create_opensearch_client()
    response = client.search(index=index_name, body=search_body)
    
    return {
        "documents": process_search_hits(response["hits"]["hits"]),
        "facets": process_aggregations(response["aggregations"]),
        "total": response["hits"]["total"]["value"],
        "query_info": {
            "query": query,
            "filters_applied": filters,
            "execution_time": response["took"]
        }
    }

def render_faceted_results(search_results: Dict[str, Any]):
    """Render search results with faceted navigation"""
    
    col_results, col_facets = st.columns([2, 1])
    
    with col_facets:
        st.markdown("### 📊 Refine Results")
        
        facets = search_results["facets"]
        
        # Document type facets
        if "by_type" in facets:
            st.markdown("**Document Types:**")
            for bucket in facets["by_type"]["buckets"]:
                if st.checkbox(
                    f"{bucket['key']} ({bucket['doc_count']})",
                    key=f"facet_type_{bucket['key']}"
                ):
                    st.session_state.setdefault("active_facets", {})["type"] = bucket["key"]
        
        # Author facets
        if "by_author" in facets:
            st.markdown("**Authors:**")
            for bucket in facets["by_author"]["buckets"][:5]:  # Top 5
                if st.checkbox(
                    f"{bucket['key']} ({bucket['doc_count']})",
                    key=f"facet_author_{bucket['key']}"
                ):
                    st.session_state.setdefault("active_facets", {})["author"] = bucket["key"]
        
        # Date histogram
        if "by_date" in facets:
            st.markdown("**Timeline:**")
            st.bar_chart({
                bucket["key_as_string"]: bucket["doc_count"]
                for bucket in facets["by_date"]["buckets"]
            })
    
    with col_results:
        st.markdown(f"### 📄 Results ({search_results['total']} total)")
        
        for doc in search_results["documents"]:
            with st.expander(f"📄 {doc.get('title', 'Untitled Document')}"):
                # Show highlights
                if highlights := doc.get("highlights"):
                    st.markdown("**Relevant excerpts:**")
                    for highlight in highlights.get("content", []):
                        st.markdown(f"💡 {highlight}")
                
                # Show metadata
                if metadata := doc.get("metadata"):
                    st.markdown("**Metadata:**")
                    for key, value in metadata.items():
                        st.text(f"{key}: {value}")
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📋 Use for QA", key=f"qa_{doc['id']}"):
                        st.session_state["selected_doc_for_qa"] = doc
                with col2:
                    if st.button("🔍 Similar Docs", key=f"similar_{doc['id']}"):
                        find_similar_documents(doc)
                with col3:
                    if st.button("📊 Analyze", key=f"analyze_{doc['id']}"):
                        analyze_document(doc)
```

## Implementation Priority

### Immediate (Week 1)
1. ✅ **Enhanced Filter Builder** - Extend current filter types
2. ✅ **Schema Discovery** - Auto-detect available fields
3. ✅ **Dynamic UI** - Generate filters based on schema

### Short-term (Week 2-3)
1. 🔄 **Smart Query Processing** - Entity extraction and query expansion
2. 🔄 **Faceted Search** - Add aggregations and faceted navigation
3. 🔄 **Advanced UI** - Boolean query builder

### Medium-term (Month 1)
1. ⏳ **Semantic Search** - Vector embeddings and similarity search
2. ⏳ **ML-Enhanced Ranking** - Learning-to-rank models
3. ⏳ **Analytics Dashboard** - Search analytics and insights

### Long-term (Month 2+)
1. ⏳ **Auto-complete & Suggestions** - Real-time query suggestions
2. ⏳ **Saved Searches** - User preference management
3. ⏳ **Export & Reporting** - Search result export capabilities

## Code Examples for Immediate Implementation

The following files need enhancement for Phase 1 implementation:

1. **`enhanced_filter_builder.py`** - Advanced filter construction
2. **`dynamic_schema_discovery.py`** - Index schema analysis
3. **`improved_streamlit_qa.py`** - Enhanced Streamlit interface
4. **`advanced_opensearch_client.py`** - Extended OpenSearch operations

Would you like me to implement any of these specific enhancements first?
