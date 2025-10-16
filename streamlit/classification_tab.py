import streamlit as st
from api_client import call_api_endpoint

def classification_tab():
    """Enhanced Classification functionality tab with eDiscovery support"""
    # Create a radio button to select the classification type
    classification_mode = st.radio(
        "Classification Mode",
        ["Standard Classification", "Structured Attribute Classification", "Document Attribute Classification"],
        horizontal=True
    )
    
    if classification_mode == "Document Attribute Classification":
        # Import and show attribute classification section
        from attribute_classification import attribute_classification_section
        attribute_classification_section()
    elif classification_mode == "Structured Attribute Classification":
        # Import and show structured classification section
        from structured_classification import structured_classification_section
        structured_classification_section()
    else:
        # Standard classification section
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
                                        st.warning(f"**Contains PII:** {pii}")
                                    else:
                                        st.info(f"**Contains PII:** {pii}")
                                
                                # Topic Analysis
                                if include_topic_analysis and 'topic_analysis' in classification_result:
                                    st.markdown("#### 🔍 Topic Analysis")
                                    topic_analysis = classification_result['topic_analysis']
                                    
                                    if isinstance(topic_analysis, dict):
                                        # If it's a structured dict
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown("**Primary Topics:**")
                                            if 'primary_topics' in topic_analysis:
                                                for topic in topic_analysis['primary_topics']:
                                                    st.markdown(f"- {topic}")
                                            else:
                                                st.markdown("- No primary topics identified")
                                        
                                        with col2:
                                            st.markdown("**Key Concepts:**")
                                            if 'key_concepts' in topic_analysis:
                                                for concept in topic_analysis['key_concepts']:
                                                    st.markdown(f"- {concept}")
                                            else:
                                                st.markdown("- No key concepts identified")
                                    else:
                                        # If it's a string or other format
                                        st.markdown(f"{topic_analysis}")
                                
                                # Detailed Reasoning
                                if include_detailed_reasoning and 'reasoning' in classification_result:
                                    with st.expander("🧠 Detailed Reasoning", expanded=False):
                                        st.markdown(classification_result['reasoning'])
                                
                                # Raw Response
                                if include_raw_response and 'raw_response' in classification_result:
                                    with st.expander("🔧 Raw LLM Response", expanded=False):
                                        st.text(classification_result['raw_response'])
                            else:
                                # LEGACY FORMAT: Try to extract fields from nested structure or text
                                st.info(f"Classification Result: {classification_result}")
                                
                                # Try to handle legacy formats - direct display
                                with st.expander("Full Classification Details", expanded=True):
                                    st.json(classification_result)
                        
                        elif classification_type == "BART-only Classification":
                            # Display BART-specific results
                            st.markdown("#### 🤖 BART Model Classification")
                            
                            if isinstance(classification_result, dict):
                                # Structured response
                                for key, value in classification_result.items():
                                    if key.lower() in ['classification', 'label', 'category']:
                                        st.success(f"**{key.title()}:** {value}")
                                    elif key.lower() in ['confidence', 'score', 'probability']:
                                        st.info(f"**{key.title()}:** {value}")
                                    elif key.lower() in ['explanation', 'reason', 'reasoning']:
                                        with st.expander(f"📝 {key.title()}", expanded=True):
                                            st.markdown(value)
                                    else:
                                        st.info(f"**{key.title()}:** {value}")
                            else:
                                # String response
                                st.markdown(classification_result)
                        
                        else:
                            # Standard classification display
                            if isinstance(classification_result, dict):
                                st.json(classification_result)
                            else:
                                st.markdown(classification_result)
                    else:
                        st.error(f"Classification failed: {result.get('error', 'Unknown error')}")
            else:
                st.warning("⚠️ Please enter text to classify")