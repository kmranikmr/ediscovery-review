#!/usr/bin/env python3
"""
Working Comprehensive Streamlit App - Extracted from Proven Implementation
Includes QA, Classification, Summarization, and NER capabilities
"""

import streamlit as st
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configure Streamlit page
st.set_page_config(
    page_title="AI Processing Suite - Working Edition",
    page_icon="🚀",
    layout="wide"
)

# API Configuration - Working endpoints
API_BASE_URL = "http://localhost:8001"

def test_api_connection(api_url=None):
    """Test if the API is running"""
    url = api_url or API_BASE_URL
    try:
        response = requests.get(f"{url}/health")
        return response.status_code == 200
    except Exception as e:
        return False

def call_api_endpoint(endpoint: str, data: Dict[str, Any], method: str = "POST", base_url: str = None) -> Dict[str, Any]:
    """Make API call to specified endpoint - from working implementation"""
    url_base = base_url or API_BASE_URL
    try:
        url = f"{url_base}{endpoint}"
        
        # Direct index endpoints - pass data as-is (WORKING APPROACH)
        if endpoint in ["/qa-direct-index", "/search-direct-index", "/index-stats", "/browse-documents"]:
            print(f"🎯 Direct Index API - keeping original format: {data}")
        
        if method == "POST":
            response = requests.post(url, json=data)
        else:
            response = requests.get(url, params=data)
        
        return response.json()
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "endpoint": endpoint,
            "request_data": data
        }

def main():
    st.title("🚀 AI Processing Suite - Working Edition")
    st.markdown("Proven QA, Classification, Summarization, and NER capabilities")
    
    # Check API connection
    if not test_api_connection():
        st.error(f"❌ API server not responding at {API_BASE_URL}")
        st.info("Please start the API server: `python run_rest_api.py`")
        return
    
    st.success(f"✅ Connected to API server at {API_BASE_URL}")
    
    # Sidebar for configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Index Selection (for QA functionality)
    st.sidebar.markdown("### 🗂️ Index Selection")
    index_preset = st.sidebar.selectbox(
        "Choose Index:",
        [
            "Existing Index (deephousedeephouse_ediscovery_docs_chunks)",
            "Default (ediscovery_documents)", 
            "Custom"
        ],
        index=0,  # Default to your existing index
        help="Select the OpenSearch index to query"
    )
    
    if index_preset == "Default (ediscovery_documents)":
        index_name = "ediscovery_documents"
        st.sidebar.info("Using default index")
    elif index_preset == "Existing Index (deephousedeephouse_ediscovery_docs_chunks)":
        index_name = "deephousedeephouse_ediscovery_docs_chunks"
        st.sidebar.success("🎯 Using your existing OpenSearch index")
    else:  # Custom
        index_name = st.sidebar.text_input(
            "Custom Index Name:",
            value="your_custom_index",
            help="Enter your custom index name"
        )
    
    # Store the selected index in session state for use in API calls (WORKING APPROACH)
    if 'selected_index' not in st.session_state or st.session_state.selected_index != index_name:
        st.session_state.selected_index = index_name
    
    st.sidebar.info(f"**Index:** `{index_name}`")
    
    # Main functionality tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 QA & Search", "� Summarization", "� Classification", "🏷️ NER"])
    
    with tab1:
        qa_and_search_tab(index_name)
    
    with tab2:
        summarization_tab()
    
    with tab3:
        classification_tab()
    
    with tab4:
        ner_tab()

def qa_and_search_tab(index_name):
    """QA and Search functionality tab - from working implementation"""
    st.markdown("### 🔍 Question Answering with Advanced Filters")
    st.markdown("Ask questions about your documents with sophisticated filtering options")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 💬 Ask Your Question")
        
        # Question input with sample queries
        sample_queries = [
            "What information do you have about Phillip Allen?",
            "Show me email information",
            "Tell me about documents",
            "What projects are mentioned?",
            "Give me information about meetings",
            "Show me reports"
        ]
        
        query_mode = st.radio("Query Mode:", ["Sample Queries", "Custom Query"])
        
        if query_mode == "Sample Queries":
            query = st.selectbox("Sample Queries:", sample_queries)
        else:
            query = st.text_input(
                "Enter your question:",
                value="What information do you have about Phillip Allen?",
                help="Ask any question about the documents in your index"
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            # QA Type selection (like in original app)
            qa_type = st.selectbox(
                "QA Type:",
                ["Regular QA (/qa-direct-index)", "Family QA (/qa/family)"],
                help="Choose between regular QA or family-based QA"
            )
            
            # Number of results
            top_k = st.slider(
                "Number of documents to retrieve:",
                min_value=1,
                max_value=20,
                value=10,
                help="More documents provide better context but slower responses"
            )
            
            # Raw OpenSearch Query Option
            use_raw_query = st.checkbox(
                "🔬 Use Raw OpenSearch Query", 
                value=False,
                help="Enable advanced users to provide a complete OpenSearch query body for precise document retrieval"
            )
            
            raw_query_body = None
            if use_raw_query:
                st.markdown("**Raw OpenSearch Query Body:**")
                st.info("💡 **Tip**: This overrides text search and basic filters. Use for complex boolean queries, aggregations, custom sorting, etc.")
                
                # Provide some example templates
                example_templates = st.selectbox(
                    "Quick Templates:",
                    [
                        "Custom (write your own)",
                        "Boolean Query with Date Filter", 
                        "Wildcard Search with Sorting",
                        "Multi-field Match with Aggregations",
                        "Complex Filter with Highlighting"
                    ]
                )
                
                if example_templates == "Boolean Query with Date Filter":
                    template = """{
  "query": {
    "bool": {
      "must": [
        {"match": {"content": "your_search_terms"}}
      ],
      "filter": [
        {"range": {"meta.date": {"gte": "2001-01-01", "lte": "2001-12-31"}}}
      ]
    }
  },
  "sort": [{"meta.date": {"order": "desc"}}]
}"""
                elif example_templates == "Wildcard Search with Sorting":
                    template = """{
  "query": {
    "bool": {
      "must": [
        {"wildcard": {"content": "*contract*"}},
        {"wildcard": {"meta.participants": "*@legal.company.com"}}
      ]
    }
  },
  "sort": [{"_score": {"order": "desc"}}, {"meta.date": {"order": "desc"}}]
}"""
                elif example_templates == "Multi-field Match with Aggregations":
                    template = """{
  "query": {
    "multi_match": {
      "query": "merger acquisition",
      "fields": ["content^2", "meta.subject^1.5", "meta.*"]
    }
  },
  "aggs": {
    "by_author": {"terms": {"field": "meta.author.keyword", "size": 10}},
    "by_month": {"date_histogram": {"field": "meta.date", "calendar_interval": "month"}}
  },
  "sort": [{"_score": {"order": "desc"}}]
}"""
                elif example_templates == "Complex Filter with Highlighting":
                    template = """{
  "query": {
    "bool": {
      "must": [
        {"match": {"content": "budget planning"}}
      ],
      "should": [
        {"term": {"meta.department": "finance"}},
        {"term": {"meta.priority": "high"}}
      ],
      "must_not": [
        {"term": {"meta.status": "draft"}}
      ],
      "minimum_should_match": 1
    }
  },
  "highlight": {
    "fields": {
      "content": {"fragment_size": 200, "number_of_fragments": 5},
      "meta.subject": {"fragment_size": 100}
    }
  }
}"""
                else:
                    template = """{
  "query": {
    "bool": {
      "must": [
        {"match": {"content": "your_search_terms"}}
      ]
    }
  },
  "sort": [{"_score": {"order": "desc"}}]
}"""
                
                raw_query_text = st.text_area(
                    "OpenSearch Query JSON:",
                    value=template,
                    height=300,
                    help="Valid OpenSearch query body. The 'size' parameter will be overridden by the 'top_k' setting above."
                )
                
                # Validate JSON
                try:
                    import json
                    raw_query_body = json.loads(raw_query_text)
                    st.success("✅ Valid JSON format")
                except json.JSONDecodeError as e:
                    st.error(f"❌ Invalid JSON: {str(e)}")
                    raw_query_body = None
                
                # Show a note about the relationship with filters
                st.warning("⚠️ **Note**: When using raw OpenSearch query, the basic filters below will be ignored. All filtering must be specified in the raw query above.")
            
            # Enable/disable certain features
            show_sources = st.checkbox("Show source documents", value=True)
            show_metadata = st.checkbox("Show document metadata", value=True)
            show_highlights = st.checkbox("Show highlighted excerpts", value=True)
    
    with col2:
        st.markdown("#### 🔍 Document Filters")
        st.markdown("Apply filters to narrow down the search:")
        
        # Basic filters
        doc_id_filter = st.text_input(
            "Document ID:",
            placeholder="e.g., doc123, email456",
            help="Filter by specific document ID(s)"
        )
        
        # Document type filter
        doc_type = st.selectbox(
            "Document Type:",
            ["", "email", "document", "pdf", "text", "meeting", "report"],
            help="Filter by document type"
        )
        
        # Author filter
        author_filter = st.text_input(
            "Author:",
            placeholder="e.g., john.smith@company.com",
            help="Filter by document author"
        )
        
        # Date range filter
        enable_date_filter = st.checkbox("Enable date filtering")
        if enable_date_filter:
            date_field = st.selectbox(
                "Date Field:",
                ["created_date", "modified_date", "sent_date", "received_date", "date", "timestamp"],
                help="Which date field to filter on"
            )
            
            # Option for year-only filtering
            filter_type = st.selectbox(
                "Filter Type:",
                ["Date Range", "Specific Year"],
                help="Choose between date range or specific year"
            )
            
            if filter_type == "Specific Year":
                year = st.number_input(
                    "Year:",
                    min_value=1900,
                    max_value=2030,
                    value=2001,
                    help="Enter the year to filter (e.g., 2001)"
                )
                from_date = datetime(year, 1, 1).date()
                to_date = datetime(year, 12, 31).date()
                st.info(f"📅 Searching for documents from {year}: {from_date} to {to_date}")
            else:
                col_from, col_to = st.columns(2)
                with col_from:
                    from_date = st.date_input(
                        "From Date:",
                        value=datetime(2001, 1, 1).date(),  # Ensure .date() is called
                        min_value=datetime(1900, 1, 1).date(),
                        max_value=datetime(2030, 12, 31).date(),
                        help="Start date for filtering"
                    )
                with col_to:
                    to_date = st.date_input(
                        "To Date:",
                        value=datetime(2001, 12, 31).date(),  # Ensure .date() is called
                        min_value=datetime(1900, 1, 1).date(),
                        max_value=datetime(2030, 12, 31).date(),
                        help="End date for filtering"
                    )
        
        # Build filters
        filters = {}
        if doc_id_filter:
            if "," in doc_id_filter:
                doc_ids = [id.strip() for id in doc_id_filter.split(",")]
                filters["document_id"] = doc_ids
            else:
                filters["document_id"] = doc_id_filter
        
        if doc_type:
            filters["type"] = doc_type
        
        if author_filter:
            filters["author"] = author_filter
        
        if enable_date_filter and from_date and to_date:
            # Convert dates to proper ISO format with time component
            from_date_str = from_date.strftime("%Y-%m-%dT00:00:00")
            to_date_str = to_date.strftime("%Y-%m-%dT23:59:59")
            
            filters[date_field] = {
                "range": {
                    "gte": from_date_str,
                    "lte": to_date_str
                }
            }
        
        # Show applied filters or raw query status
        if use_raw_query and raw_query_body:
            st.markdown("**🔧 Advanced Query Mode:**")
            st.warning("⚡ Using custom OpenSearch query - basic filters are ignored")
            with st.expander("View Raw Query", expanded=False):
                st.code(raw_query_body, language="json")
        elif filters:
            st.markdown("**Applied Filters:**")
            
            # Show filters in a more user-friendly way
            for filter_name, filter_value in filters.items():
                if isinstance(filter_value, dict) and "range" in filter_value:
                    range_info = filter_value["range"]
                    st.info(f"📅 **{filter_name}**: {range_info.get('gte', 'N/A')} to {range_info.get('lte', 'N/A')}")
                elif isinstance(filter_value, list):
                    st.info(f"📋 **{filter_name}**: {', '.join(map(str, filter_value))}")
                else:
                    st.info(f"🔍 **{filter_name}**: {filter_value}")
                    
            # Add helpful note about date filtering
            if any("date" in str(k).lower() for k in filters.keys()):
                st.warning("💡 **Note**: Date filtering searches multiple field formats (meta.date, date, date.keyword) to ensure matches. If no results appear, try adjusting the date range or check the document date formats.")
        else:
            st.info("No filters applied - searching all documents")
    
    # Search button
    st.markdown("---")
    col_search, col_clear = st.columns([3, 1])
    
    with col_search:
        search_button = st.button("🔍 Search with Filters", type="primary", use_container_width=True)
    
    with col_clear:
        if st.button("🗑️ Clear Filters", use_container_width=True):
            st.experimental_rerun()
    
    # Perform search using WORKING implementation
    if search_button and query.strip():
        with st.spinner("🔍 Searching documents..."):
            # Choose endpoint based on QA type
            if "Family" in qa_type:
                # Use family QA endpoint with proper format - it needs query + documents from index
                data = {
                    "query": query,
                    "documents": [],  # Family QA will fetch documents internally
                    "index_name": st.session_state.get('selected_index', index_name),
                    "top_k": top_k,
                    "filters": filters
                }
                endpoint = "/qa/family"
            else:
                # Use the PROVEN data format that works for regular QA
                data = {
                    "query": query,
                    "index_name": st.session_state.get('selected_index', index_name),
                    "top_k": top_k,
                    "direct_access": True,  # Flag for bypassing collection filtering
                    "filters": filters
                }
                
                # Add raw query body if provided (overrides text search and filters)
                if use_raw_query and raw_query_body:
                    data["raw_query_body"] = raw_query_body
                    st.info("🔬 **Using Raw OpenSearch Query** - Basic filters ignored, using custom query for document retrieval")
                
                endpoint = "/qa-direct-index"
            
            # Call the appropriate endpoint
            st.info(f"🔄 Using endpoint: **{endpoint}** {'(Family QA searches documents first)' if 'family' in endpoint else '(Direct index search)'}")
            
            api_response = call_api_endpoint(endpoint, data)
            
            if api_response and api_response.get("success"):
                # Handle different response formats for Family QA vs Regular QA
                if "Family" in qa_type:
                    # Family QA returns result directly
                    result = api_response.get("result", {})
                    
                    # Extract answer and sources from family QA response
                    if isinstance(result, dict):
                        answer = result.get("answer", str(result))
                        sources = result.get("sources", [])
                        documents_processed = result.get("documents_processed", 0)
                    else:
                        answer = str(result)
                        sources = []
                        documents_processed = 0
                    
                    # Create a unified result structure
                    result = {
                        "answer": answer,
                        "sources": sources,
                        "total_documents_searched": documents_processed
                    }
                else:
                    # Extract the nested result from our API response format (WORKING APPROACH)
                    # API returns: {"success": true, "data": {"success": true, "result": {...}}}
                    # We need to get to the actual result data
                    response_data = api_response.get("data", {})
                    if response_data.get("success") and "result" in response_data:
                        result = response_data["result"]
                    else:
                        result = response_data
                
                # If result is still not properly extracted, try direct access
                if not result or not result.get("answer"):
                    # Try to extract result directly from api_response
                    if "result" in api_response:
                        result = api_response["result"]
                    else:
                        result = api_response
                
                # Display answer
                st.markdown("### 💡 Answer")
                answer = result.get("answer", "No answer generated")
                
                # Color-code the answer based on quality
                if len(answer) > 100 and "No relevant documents" not in answer:
                    st.success(answer)
                elif "Found" in answer and "documents" in answer:
                    st.warning(answer)
                else:
                    st.error(answer)
                
                # Display statistics
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                
                sources = result.get("sources", [])
                
                with col_stats1:
                    st.metric(
                        "Documents Found",
                        result.get("total_documents_searched", len(sources))
                    )
                
                with col_stats2:
                    st.metric(
                        "Sources Used",
                        len(sources)
                    )
                
                with col_stats3:
                    st.metric(
                        "Filters Applied",
                        len(filters)
                    )
                
                # Display sources if enabled (WORKING APPROACH)
                if show_sources and sources:
                    st.markdown("### 📚 Source Documents")
                    
                    for i, source in enumerate(sources[:5]):  # Show top 5
                        with st.expander(f"📄 Document {i+1} (Score: {source.get('score', 'N/A')})"):
                            
                            if show_highlights and source.get('highlights'):
                                st.markdown("**Highlighted Excerpts:**")
                                for highlight in source['highlights'][:3]:  # Show first 3 highlights
                                    st.markdown(f"> {highlight}")
                                st.markdown("---")
                            
                            st.markdown("**Content:**")
                            content = source.get("content", "")
                            if len(content) > 500:
                                st.text(content[:500] + "...")
                                if st.button(f"Show full content {i+1}", key=f"show_full_{i}"):
                                    st.text(content)
                            else:
                                st.text(content)
                            
                            if show_metadata and source.get("metadata"):
                                st.markdown("**Metadata:**")
                                st.json(source["metadata"])
            
            else:
                st.error(f"❌ Query failed: {api_response.get('error', 'Unknown error')}")
    
    elif search_button:
        st.warning("⚠️ Please enter a question to search")

def classification_tab():
    """Enhanced Classification functionality tab with eDiscovery support"""
    st.markdown("### 📊 Enhanced eDiscovery Classification")
    st.markdown("Comprehensive document classification with topic analysis, responsiveness, privilege, and confidentiality assessment")
    
    # Text input
    text_to_classify = st.text_area(
        "Enter document text to classify:",
        value="From: legal@company.com\nTo: ceo@company.com\nSubject: ATTORNEY-CLIENT PRIVILEGED - Contract Review\n\nI've reviewed the merger agreement with ABC Corp. Several clauses need modification to protect confidential information and limit liability exposure. This analysis is protected by attorney-client privilege.",
        height=150,
        help="Enter the document content you want to classify"
    )
    
    # User prompt for context
    col1, col2 = st.columns(2)
    
    with col1:
        user_prompt = st.text_area(
            "Classification Instructions (Optional):",
            placeholder="e.g., 'Focus on employment law relevance', 'This case involves contract disputes', 'Look for responsiveness to data breach discovery requests'",
            height=80,
            help="Provide specific instructions or context to guide the classification"
        )
    
    with col2:
        discovery_context = st.text_area(
            "Discovery Request Context (Optional):",
            placeholder="e.g., 'All communications regarding the XYZ project', 'Documents related to employee termination policies', 'Financial records for Q4 2023'",
            height=80,
            help="Describe the discovery request to help determine responsiveness"
        )
    
    # Response Configuration Options
    st.markdown("#### ⚙️ Response Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        response_format = st.selectbox(
            "Response Detail Level:",
            ["comprehensive", "standard", "minimal"],
            index=0,
            help="Comprehensive: All fields and detailed analysis\nStandard: Core eDiscovery fields with simplified topic analysis\nMinimal: Basic responsiveness, privilege, and confidentiality only"
        )
    
    with col2:
        include_detailed_reasoning = st.checkbox(
            "Include Detailed Reasoning",
            value=True,
            help="Include comprehensive analysis and explanations"
        )
    
    with col3:
        include_topic_analysis = st.checkbox(
            "Include Topic Analysis",
            value=True,
            help="Include detailed topic analysis and key concepts"
        )
    
    # Advanced options in expander
    with st.expander("🔧 Advanced Response Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            include_raw_response = st.checkbox(
                "Include Raw LLM Response",
                value=False,
                help="Include the raw response from the language model for debugging"
            )
        
        with col2:
            custom_fields = st.multiselect(
                "Select Specific Fields (Optional):",
                ["responsiveness", "privilege", "confidentiality", "document_type", "business_relevance", "contains_pii"],
                default=[],
                help="If selected, only these specific fields will be included in the response"
            )
    
    # Classification options
    classification_type = st.selectbox(
        "Classification Type:",
        ["Enhanced eDiscovery Classification", "Standard Classification", "BART-only Classification"],
        help="Choose the type of classification to perform"
    )
    
    # Classification button
    if st.button("📊 Classify Document", type="primary", use_container_width=True):
        if text_to_classify.strip():
            with st.spinner("🔄 Performing comprehensive classification..."):
                
                if classification_type == "BART-only Classification":
                    # Use BART-only endpoint
                    data = {
                        "email_text": text_to_classify,
                        "classification_schemes": ["simple", "family", "thread"],
                        "confidence_threshold": 0.7,
                        "include_advanced_analysis": True
                    }
                    endpoint = "/classify/bart-only"
                elif classification_type == "Enhanced eDiscovery Classification":
                    # Use enhanced eDiscovery classification
                    data = {
                        "documents": [
                            {
                                "content": text_to_classify,
                                "meta": {"source": "user_input"}
                            }
                        ],
                        # Response configuration options
                        "response_format": response_format,
                        "include_detailed_reasoning": include_detailed_reasoning,
                        "include_topic_analysis": include_topic_analysis,
                        "include_raw_response": include_raw_response
                    }
                    
                    # Add optional context fields
                    if user_prompt.strip():
                        data["user_prompt"] = user_prompt.strip()
                    if discovery_context.strip():
                        data["discovery_context"] = discovery_context.strip()
                    if custom_fields:
                        data["fields_to_include"] = custom_fields
                    endpoint = "/classify"
                else:
                    # Use standard classification
                    data = {
                        "documents": [
                            {
                                "content": text_to_classify,
                                "meta": {"source": "user_input"}
                            }
                        ]
                    }
                    endpoint = "/classify"
                
                result = call_api_endpoint(endpoint, data)
                
                if result.get("success"):
                    classification_result = result.get("result", {})
                    
                    st.markdown("### 🎯 Classification Results")
                    
                    if classification_type == "Enhanced eDiscovery Classification":
                        # Display enhanced results with structured layout
                        st.markdown("#### 🎯 Enhanced eDiscovery Results")
                        
                        # Check if we have direct eDiscovery fields (new format)
                        has_direct_fields = any(field in classification_result for field in 
                                              ['responsiveness', 'privilege', 'confidentiality', 'topic_analysis', 'ediscovery_classification'])
                        
                        if has_direct_fields:
                            # NEW FORMAT: Use direct fields from classification_result
                            st.markdown("#### 📊 Primary Classification")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.info(f"**Classification:** {classification_result.get('classification', 'N/A')}")
                            with col2:
                                st.info(f"**Confidence:** {classification_result.get('confidence', 'N/A')}")
                            with col3:
                                st.info(f"**Method:** {classification_result.get('method', 'N/A')}")
                            
                            # eDiscovery Classifications - Direct from API response
                            st.markdown("#### ⚖️ eDiscovery Classifications")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                responsiveness = classification_result.get('responsiveness', 'N/A')
                                if 'responsive' in responsiveness.lower() and 'non-responsive' not in responsiveness.lower():
                                    st.success(f"**Responsiveness:** {responsiveness}")
                                elif 'non-responsive' in responsiveness.lower():
                                    st.error(f"**Responsiveness:** {responsiveness}")
                                else:
                                    st.warning(f"**Responsiveness:** {responsiveness}")
                                
                                privilege = classification_result.get('privilege', 'N/A')
                                if 'privileged' in privilege.lower() and 'not privileged' not in privilege.lower():
                                    st.warning(f"**Privilege:** {privilege}")
                                else:
                                    st.info(f"**Privilege:** {privilege}")
                            
                            with col2:
                                st.info(f"**Document Type:** {classification_result.get('document_type', 'N/A')}")
                                st.info(f"**Business Relevance:** {classification_result.get('business_relevance', 'N/A')}")
                                
                            with col3:
                                confidentiality = classification_result.get('confidentiality', 'N/A')
                                if 'confidential' in confidentiality.lower():
                                    st.warning(f"**Confidentiality:** {confidentiality}")
                                else:
                                    st.info(f"**Confidentiality:** {confidentiality}")
                                
                                pii = classification_result.get('contains_pii', 'N/A')
                                if pii.lower() == 'yes':
                                    st.error(f"**Contains PII:** {pii}")
                                else:
                                    st.info(f"**Contains PII:** {pii}")
                            
                            # Topic Analysis - Direct from API response
                            if "topic_analysis" in classification_result:
                                st.markdown("#### 📋 Topic Analysis")
                                topic_data = classification_result["topic_analysis"]
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.info(f"**Primary Topic:** {topic_data.get('primary_topic', 'N/A')}")
                                    st.info(f"**Subject Matter:** {topic_data.get('subject_matter', 'N/A')}")
                                with col2:
                                    if topic_data.get('secondary_topics'):
                                        st.info(f"**Secondary Topics:** {', '.join(topic_data['secondary_topics'])}")
                                    if topic_data.get('key_concepts'):
                                        st.info(f"**Key Concepts:** {', '.join(topic_data['key_concepts'])}")
                            
                            # Reasoning - Direct from API response
                            if "reasoning" in classification_result:
                                st.markdown("#### 💭 Analysis & Reasoning")
                                reasoning_data = classification_result["reasoning"]
                                
                                if isinstance(reasoning_data, dict):
                                    with st.expander("📝 Detailed Reasoning", expanded=True):
                                        if reasoning_data.get('responsiveness_reasoning'):
                                            st.markdown(f"**Responsiveness:** {reasoning_data['responsiveness_reasoning']}")
                                        if reasoning_data.get('privilege_reasoning'):
                                            st.markdown(f"**Privilege:** {reasoning_data['privilege_reasoning']}")
                                        if reasoning_data.get('context_analysis'):
                                            st.markdown(f"**Context Analysis:** {reasoning_data['context_analysis']}")
                                        if reasoning_data.get('sensitivity_notes'):
                                            st.markdown(f"**Sensitivity Notes:** {reasoning_data['sensitivity_notes']}")
                                        if reasoning_data.get('redaction_recommendations'):
                                            st.markdown(f"**Redaction Recommendations:** {reasoning_data['redaction_recommendations']}")
                                else:
                                    st.info(f"**Reasoning:** {str(reasoning_data)}")
                        
                        else:
                            # FALLBACK: Try to parse from raw_response (old format)
                            st.markdown("#### 📋 Parsing from Raw Response")
                            try:
                                import json
                                import re
                                
                                # Extract JSON from response
                                if isinstance(classification_result, dict) and "raw_response" in classification_result:
                                    raw_text = classification_result["raw_response"]
                                    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                                    if json_match:
                                        parsed_data = json.loads(json_match.group())
                                        st.json(parsed_data)
                                    else:
                                        st.info("Could not parse JSON from raw response")
                                        st.text(raw_text[:500] + "..." if len(raw_text) > 500 else raw_text)
                                else:
                                    st.info("No raw response available for parsing")
                                    st.json(classification_result)
                            except Exception as e:
                                st.warning(f"Could not parse enhanced results: {e}")
                                st.json(classification_result)
                    
                    else:
                        # Handle other classification types (Standard, BART-only)
                        st.markdown("#### 📊 Standard Classification Results")
                        if isinstance(classification_result, dict):
                            # Main classification info
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if "classification" in classification_result:
                                    st.success(f"**Classification:** {classification_result['classification']}")
                                elif "simple_classification" in classification_result:
                                    st.success(f"**Classification:** {classification_result['simple_classification']}")
                                else:
                                    st.info("**Classification:** Not available")
                            
                            with col2:
                                if "confidence" in classification_result:
                                    st.info(f"**Confidence:** {classification_result['confidence']}")
                                if "method" in classification_result:
                                    st.info(f"**Method:** {classification_result['method']}")
                            
                            with col3:
                                # Check for any enhanced fields that might be present
                                enhanced_count = sum(1 for field in ['responsiveness', 'privilege', 'confidentiality'] 
                                                   if field in classification_result)
                                if enhanced_count > 0:
                                    st.info(f"**Enhanced Fields:** {enhanced_count}")
                            
                            # Show basic enhanced fields if present
                            if any(field in classification_result for field in ['responsiveness', 'privilege', 'confidentiality']):
                                st.markdown("#### ⚖️ eDiscovery Fields")
                                if "responsiveness" in classification_result:
                                    st.info(f"**Responsiveness:** {classification_result['responsiveness']}")
                                if "privilege" in classification_result:
                                    st.info(f"**Privilege:** {classification_result['privilege']}")
                                if "confidentiality" in classification_result:
                                    st.info(f"**Confidentiality:** {classification_result['confidentiality']}")
                            
                            # Reasoning
                            if "reasoning" in classification_result:
                                with st.expander("💭 Reasoning"):
                                    st.markdown(f"{classification_result['reasoning']}")
                        
                        elif isinstance(classification_result, str):
                            st.success(f"**Classification:** {classification_result}")
                        else:
                            st.success(f"**Classification:** {str(classification_result)}")
                    
                    # Full results in expandable section
                    with st.expander("🔍 Full Classification Details"):
                        st.json(classification_result)
                
                else:
                    st.error(f"❌ Classification failed: {result.get('error', 'Unknown error')}")
        else:
            st.warning("⚠️ Please enter text to classify")

def summarization_tab():
    """Enhanced summarization functionality tab with clear input explanations"""
    st.markdown("### 📝 Text Summarization")
    st.markdown("Generate summaries using different specialized endpoints")
    
    # Summarization type selection first to show appropriate instructions
    summary_type = st.selectbox(
        "Summarization Type:",
        ["Regular (/summarize)", "Family (/summarize/family)", "Thread (/summarize/thread)", "BART-only Summarization"],
        help="Choose the type of summarization to perform"
    )
    
    # Show endpoint-specific information
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dynamic instructions based on endpoint
        if summary_type == "Regular (/summarize)":
            st.info("📋 **Regular Summarization**: Takes plain text input and generates a standard summary with keywords")
            default_text = """This is a quarterly business review meeting. The marketing team presented their Q3 results showing a 15% increase in lead generation. The sales team reported closing 25 new deals worth $2.3M total revenue. The engineering team delivered 3 major features ahead of schedule. However, there are concerns about the upcoming Q4 budget constraints and potential staff reductions. The CEO emphasized the need for cost optimization while maintaining growth momentum."""
            
        elif summary_type == "Family (/summarize/family)":
            st.info("👨‍👩‍👧‍👦 **Family Summarization**: Designed for email families (email + attachments as one unit). Takes document format input.")
            default_text = """{
  "documents": [
    {
      "content": "From: john.smith@company.com\\nTo: team@company.com\\nSubject: Q4 Project Status\\n\\nHi Team,\\n\\nI wanted to update everyone on our Q4 project status. We've completed 75% of the development work and are on track for the December launch.\\n\\nKey achievements:\\n- Database migration completed\\n- API integration tests passed\\n- User interface redesign approved\\n\\nNext steps:\\n- Final testing phase (2 weeks)\\n- Documentation completion\\n- Training material preparation\\n\\nPlease let me know if you have any questions.\\n\\nBest regards,\\nJohn",
      "meta": {
        "source": "email",
        "document_id": "email1",
        "email_subject": "Q4 Project Status"
      }
    },
    {
      "content": "Project Timeline Summary\\n\\nPhase 1: Requirements (Completed)\\nPhase 2: Development (75% complete)\\nPhase 3: Testing (Starts Nov 15)\\nPhase 4: Deployment (Dec 1-15)\\n\\nKey Milestones:\\n- Nov 1: Feature freeze\\n- Nov 15: Testing begins\\n- Dec 1: Production deployment\\n- Dec 15: Go-live\\n\\nBudget Status: $125,000 spent of $150,000 allocated",
      "meta": {
        "source": "attachment",
        "document_id": "attach1", 
        "attachment_name": "project_timeline.pdf"
      }
    },
    {
      "content": "Q4 Budget Allocation Report\\n\\nDevelopment: $75,000\\nTesting: $25,000\\nDeployment: $20,000\\nContingency: $30,000\\nTotal: $150,000\\n\\nSpent to Date: $125,000\\nRemaining: $25,000\\n\\nProjected completion within budget.",
      "meta": {
        "source": "attachment",
        "document_id": "attach2",
        "attachment_name": "budget_report.xlsx"
      }
    }
  ]
}"""
            
        elif summary_type == "Thread (/summarize/thread)":
            st.info("🧵 **Thread Summarization**: Designed for email thread conversations. Takes document format input.")
            default_text = """Message 1:
From: alice@company.com
To: bob@company.com
Subject: Budget Approval Request
Hi Bob, I need approval for the $50,000 marketing budget increase for Q1.

Message 2:
From: bob@company.com  
To: alice@company.com
Subject: Re: Budget Approval Request
Alice, can you provide more details on how this will be allocated?

Message 3:
From: alice@company.com
To: bob@company.com
Subject: Re: Budget Approval Request
Sure! $30K for digital ads, $15K for events, $5K for content creation.

Message 4:
From: bob@company.com
To: alice@company.com
Subject: Re: Budget Approval Request
Approved! Please proceed with the budget allocation as outlined."""
            
        else:  # BART-only
            st.info("🤖 **BART-only Summarization**: Uses pure BART model for comparison. Takes plain text input.")
            default_text = """Subject: Urgent: System Outage Response Plan

Team,

We experienced a critical system outage from 2:00 AM to 6:00 AM EST affecting our main customer portal. Here's what happened and our response:

Root Cause: Database connection timeout due to increased load from automated sync processes.

Immediate Actions Taken:
1. Switched to backup database server
2. Implemented connection pooling optimization  
3. Notified all affected customers via email and status page
4. Escalated to senior engineering team

Current Status: All systems operational as of 6:15 AM EST

Next Steps:
- Conduct full post-mortem review by Friday
- Implement additional monitoring alerts
- Review capacity planning for Q1 growth

Customer Impact: Approximately 2,500 users unable to access portal during outage window.

Please join the emergency debrief meeting at 10 AM today.

Thanks,
IT Operations Team"""
    
    with col2:
        # Endpoint format information
        st.markdown("#### 📊 Input Format")
        if summary_type == "Regular (/summarize)":
            st.code('''
{
  "text": "content...",
  "length": "medium",
  "focus": "main_points",
  "extract_keywords": true
}
            ''', language="json")
        elif summary_type in ["Family (/summarize/family)", "Thread (/summarize/thread)"]:
            st.code('''
{
  "documents": [
    {
      "content": "This is the main email body text.",
      "meta": {
        "source": "email",
        "document_id": "email1",
        "email_subject": "Project Update"
      }
    },
    {
      "content": "This is the extracted text from the attached PDF.",
      "meta": {
        "source": "attachment", 
        "document_id": "attach1",
        "attachment_name": "project_update.pdf"
      }
    }
  ]
}
            ''', language="json")
        else:  # BART-only
            st.code('''
{
  "email_text": "content...",
  "summary_type": "detailed",
  "max_length": 150
}
            ''', language="json")
    
    # Text input with dynamic default
    text_to_summarize = st.text_area(
        f"Enter text for {summary_type}:",
        value=default_text,
        height=250,
        help=f"Enter the text you want to summarize using {summary_type}"
    )
    
    # Additional options in expander
    with st.expander("🔧 Advanced Options"):
        extra_instruction = st.text_area(
            "Extra Instruction (optional)",
            value="",
            help="Add any custom instruction for the summarization model (e.g., focus on risks, use simple language, etc.)"
        )
        summary_format = st.radio(
            "Summary Format:",
            ["paragraph", "bulleted"],
            index=0,
            help="Choose the format for the summary: paragraph or bulleted list"
        )
        if summary_type == "Regular (/summarize)":
            col_len, col_focus = st.columns(2)
            with col_len:
                length = st.selectbox("Summary Length:", ["short", "medium", "long"], index=1)
            with col_focus:
                focus = st.selectbox("Focus Area:", ["general", "main_points", "action_items", "key_facts"], index=1)
            extract_keywords = st.checkbox("Extract Keywords", value=True)
        elif summary_type == "BART-only Summarization":
            col_type, col_len = st.columns(2)
            with col_type:
                bart_summary_type = st.selectbox("Summary Type:", ["business", "detailed", "concise"], index=1)
            with col_len:
                max_length = st.slider("Max Length:", 50, 300, 150)
        elif summary_type in ["Family (/summarize/family)", "Thread (/summarize/thread)"]:
            length = st.selectbox("Summary Length:", ["short", "medium", "long"], index=1)
        else:
            st.info("Family and Thread summarization use default settings optimized for email content.")
    
    # Summarization button
    if st.button("📝 Generate Summary", type="primary", use_container_width=True):
        if text_to_summarize.strip():
            with st.spinner(f"📝 Generating {summary_type} summary..."):
                
                if summary_type == "BART-only Summarization":
                    # Use BART-only endpoint with working format
                    data = {
                        "email_text": text_to_summarize,
                        "summary_type": bart_summary_type,
                        "max_length": max_length,
                        "min_length": max_length // 3,
                        "extra_instruction": extra_instruction.strip() if extra_instruction else None
                    }
                    endpoint = "/summarize/bart-only"
                elif summary_type == "Regular (/summarize)":
                    # Regular summarization uses TextSummarizationRequest format
                    data = {
                        "text": text_to_summarize,
                        "length": length,
                        "format": summary_format,
                        "focus": focus,
                        "extract_keywords": extract_keywords,
                        "extra_instruction": extra_instruction.strip() if extra_instruction else None
                    }
                    endpoint = "/summarize"
                elif summary_type in ["Family (/summarize/family)", "Thread (/summarize/thread)"]:
                    endpoint_map = {
                        "Family (/summarize/family)": "/summarize/family",
                        "Thread (/summarize/thread)": "/summarize/thread"
                    }
                    endpoint = endpoint_map[summary_type]
                    
                    # Parse the text as JSON to get proper documents format
                    try:
                        import json
                        parsed_data = json.loads(text_to_summarize)
                        if "documents" in parsed_data:
                            # Use the parsed documents directly
                            data = {
                                "documents": parsed_data["documents"],
                                "extra_instruction": extra_instruction.strip() if extra_instruction else None
                            }
                        else:
                            # Fallback: treat as single document
                            data = {
                                "documents": [
                                    {
                                        "content": text_to_summarize,
                                        "meta": {
                                            "source": "user_input",
                                            "type": "family" if "family" in summary_type.lower() else "thread",
                                            "timestamp": datetime.now().isoformat()
                                        }
                                    }
                                ],
                                "extra_instruction": extra_instruction.strip() if extra_instruction else None
                            }
                    except json.JSONDecodeError:
                        # If not valid JSON, treat as plain text
                        data = {
                            "documents": [
                                {
                                    "content": text_to_summarize,
                                    "meta": {
                                        "source": "user_input",
                                        "type": "family" if "family" in summary_type.lower() else "thread",
                                        "timestamp": datetime.now().isoformat()
                                    }
                                }
                            ],
                            "extra_instruction": extra_instruction.strip() if extra_instruction else None
                        }
                else:
                    st.warning("Unknown summarization type selected.")
                
                # Show what we're sending to the API
                with st.expander("🔍 API Request Details"):
                    st.markdown(f"**Endpoint:** `{endpoint}`")
                    st.json(data)
                
                result = call_api_endpoint(endpoint, data)
                
                if result.get("success"):
                    summary_result = result.get("result", {})
                    
                    st.markdown("### 📄 Summary Results")
                    
                    # Main summary - handle various response formats
                    if isinstance(summary_result, dict):
                        # Handle dictionary responses
                        if "summary" in summary_result:
                            st.success(f"**Summary:** {summary_result['summary']}")
                        elif "detailed_summary" in summary_result:
                            st.success(f"**Detailed Summary:** {summary_result['detailed_summary']}")
                        else:
                            st.success(f"**Summary:** {str(summary_result)}")
                        
                        # Additional insights and information
                        col_results1, col_results2 = st.columns(2)
                        
                        with col_results1:
                            if isinstance(summary_result, dict) and "insights" in summary_result:
                                st.markdown("#### 💡 Insights")
                                st.info(summary_result["insights"])
                            
                            if isinstance(summary_result, dict) and "key_points" in summary_result and isinstance(summary_result["key_points"], list):
                                st.markdown("#### 🔑 Key Points")
                                for point in summary_result["key_points"]:
                                    st.write(f"• {point}")
                        
                        with col_results2:
                            if isinstance(summary_result, dict) and "keywords" in summary_result and isinstance(summary_result["keywords"], list):
                                st.markdown("#### 🏷️ Keywords")
                                st.write(", ".join(summary_result["keywords"]))
                            
                            if isinstance(summary_result, dict) and "business_facts" in summary_result:
                                st.markdown("#### 📊 Business Facts")
                                facts = summary_result["business_facts"]
                                if isinstance(facts, dict):
                                    for key, value in facts.items():
                                        st.write(f"**{key}:** {value}")
                        
                        # Technical information
                        if isinstance(summary_result, dict):
                            tech_info = []
                            if "model_used" in summary_result:
                                tech_info.append(f"Model: {summary_result['model_used']}")
                            if "processing_time" in summary_result:
                                tech_info.append(f"Time: {summary_result['processing_time']:.2f}s")
                            if "confidence_score" in summary_result:
                                tech_info.append(f"Confidence: {summary_result['confidence_score']:.2f}")
                            
                            if tech_info:
                                st.caption(" | ".join(tech_info))
                    
                    elif isinstance(summary_result, str):
                        # Handle string responses
                        st.success(f"**Summary:** {summary_result}")
                    
                    else:
                        # Handle any other response format
                        st.success(f"**Summary:** {str(summary_result)}")
                        
                else:
                    st.error(f"❌ Summarization failed: {result.get('error', 'Unknown error')}")
                    
                    # Show error details
                    with st.expander("🐛 Error Details"):
                        st.json(result)
        else:
            st.warning("⚠️ Please enter some text to summarize")

def ner_tab():
    """Named Entity Recognition functionality tab - using enhanced NER API"""
    st.markdown("### 🏷️ Named Entity Recognition (NER)")
    st.markdown("Extract entities using BERT models or LLM-based approaches")
    
    # Method selection
    ner_method = st.radio(
        "🔬 NER Method:",
        ["bert", "llm"],
        index=0,
        horizontal=True,
        help="BERT: Fast, accurate model-based NER | LLM: Flexible, context-aware NER"
    )
    
    if ner_method == "bert":
        st.info("🔬 Using BERT model for fast, accurate entity recognition with GPU acceleration")
    else:
        st.info("🤖 Using LLM (Ollama/Mistral) for flexible, context-aware entity recognition")
    
    # Text input
    text_for_ner = st.text_area(
        "Enter text for entity extraction:",
        value="John Smith from Microsoft met with Sarah Johnson at the New York office on January 15th, 2024. They discussed the Q4 budget of $2.5 million and the upcoming project deadline in March.",
        height=150,
        help="Enter text to extract named entities from"
    )
    
    # NER options
    with st.expander("🔧 NER Options"):
        entity_types = st.multiselect(
            "Entity Types to Extract:",
            ["PERSON", "ORGANIZATION", "LOCATION", "DATE", "MONEY", "PERCENT", "TIME", "EMAIL_ADDRESS", "PHONE_NUMBER"],
            default=["PERSON", "ORGANIZATION", "LOCATION", "DATE", "MONEY"],
            help="Select which types of entities to extract"
        )
        
        include_pii = st.checkbox("Include PII Detection", value=True)
        min_score = st.slider("Minimum Confidence Score:", 0.0, 1.0, 0.7)
    
    # NER button
    if st.button("🏷️ Extract Entities", type="primary", use_container_width=True):
        if text_for_ner.strip():
            with st.spinner("🔍 Extracting entities..."):
                # Use enhanced NER API with method selection
                data = {
                    "text": text_for_ner,
                    "entity_types": entity_types if entity_types else None,
                    "include_pii": include_pii,
                    "min_score": min_score,
                    "method": ner_method
                }
                
                # Call main NER API on port 8001
                result = call_api_endpoint("/ner/extract", data, base_url="http://localhost:8001")
                
                if result.get("success"):
                    # Get result directly from API response
                    ner_result = result.get("result", {})
                    method_used = result.get("method", "unknown")
                    
                    # Display method used
                    if method_used == "bert_model":
                        st.success("🔬 **Method Used**: BERT Model (GPU-accelerated)")
                    elif method_used == "ollama_direct":
                        st.success("🤖 **Method Used**: LLM-based (Ollama/Mistral)")
                    else:
                        st.info(f"🔧 **Method Used**: {method_used}")
                    
                    st.markdown("### 🎯 Extracted Entities")
                    
                    # Display entities
                    entities = ner_result.get("entities", {})
                    statistics = ner_result.get("statistics", {})
                    
                    # Display summary
                    if statistics:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Entities", statistics.get("total_entities", 0))
                        with col2:
                            st.metric("Entity Types", statistics.get("entity_types", 0))
                        with col3:
                            avg_conf = statistics.get("avg_confidence", 0)
                            if avg_conf > 0:
                                st.metric("Avg Confidence", f"{avg_conf:.2f}")
                    
                    if entities:
                        for entity_type, entity_list in entities.items():
                            if entity_list:  # Only show if there are entities
                                st.markdown(f"#### {entity_type}")
                                
                                for entity in entity_list:
                                    if isinstance(entity, dict):
                                        text = entity.get("text", "")
                                        confidence = entity.get("confidence", 0)
                                        position = f"{entity.get('start', 'N/A')}-{entity.get('end', 'N/A')}"
                                        
                                        # Color code by confidence
                                        conf_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                                        
                                        st.markdown(f"• **{text}** {conf_color} (confidence: {confidence:.2%}, pos: {position})")
                                    else:
                                        st.write(f"• {entity}")
                    else:
                        st.warning("No entities found with the current settings")
                    
                    # Show detailed results
                    if ner_result:
                        with st.expander("🔍 Detailed Results"):
                            st.json(ner_result)
                
                else:
                    st.error(f"❌ Entity extraction failed: {result.get('error', 'Unknown error')}")
        else:
            st.warning("⚠️ Please enter text for entity extraction")

if __name__ == "__main__":
    main()
