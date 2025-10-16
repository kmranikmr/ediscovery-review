import streamlit as st
import json
from typing import Dict, List, Any

def fetch_attribute_definitions():
    """
    Fetch detailed attribute definitions from the API
    """
    try:
        # Import the call_api_endpoint function from parent module
        from app import call_api_endpoint
        
        result = call_api_endpoint("/classification/attribute-definitions", data=None, method="GET")
        
        if "error" in result:
            st.error(f"Error fetching attribute definitions: {result['error']}")
            return {}
        
        if result.get("success") and "attribute_definitions" in result:
            # Convert to a dictionary keyed by attribute name for easier lookup
            definitions_dict = {}
            for attr_def in result["attribute_definitions"]:
                definitions_dict[attr_def["name"]] = attr_def
            return definitions_dict
        
        return {}
    except Exception as e:
        st.error(f"Failed to fetch attribute definitions: {str(e)}")
        return {}

def display_attribute_definitions(attribute_name, definitions_dict, attribute_index):
    """
    Display detailed definitions for an attribute if available, with editing capability
    """
    if attribute_name in definitions_dict:
        attr_def = definitions_dict[attribute_name]
        if attr_def.get("definitions"):
            st.markdown("**📚 Detailed Definitions:**")
            
            # Initialize editable definitions in session state if not exists
            session_key = f"editable_definitions_{attribute_name.lower()}_{attribute_index}"
            if session_key not in st.session_state:
                st.session_state[session_key] = attr_def["definitions"].copy()
            
            # Toggle for editing mode
            edit_mode_key = f"edit_mode_{attribute_name.lower()}_{attribute_index}"
            edit_mode = st.checkbox("✏️ Edit Definitions", key=edit_mode_key)
            
            for value_name, value_def in st.session_state[session_key].items():
                with st.expander(f"ℹ️ {value_name}", expanded=False):
                    if edit_mode:
                        # Editable mode
                        st.markdown("**Edit Definition and Criteria:**")
                        
                        # Edit definition
                        new_definition = st.text_area(
                            "Definition:",
                            value=value_def.get('definition', ''),
                            key=f"def_{attribute_name}_{value_name}_{attribute_index}",
                            height=60
                        )
                        st.session_state[session_key][value_name]['definition'] = new_definition
                        
                        # Edit criteria
                        st.markdown("**Criteria (one per line):**")
                        criteria_text = "\n".join(value_def.get('criteria', []))
                        new_criteria = st.text_area(
                            "Criteria:",
                            value=criteria_text,
                            key=f"criteria_{attribute_name}_{value_name}_{attribute_index}",
                            height=100
                        )
                        st.session_state[session_key][value_name]['criteria'] = [
                            c.strip() for c in new_criteria.split('\n') if c.strip()
                        ]
                        
                        # Edit examples
                        st.markdown("**Examples (one per line):**")
                        examples_text = "\n".join(value_def.get('examples', []))
                        new_examples = st.text_area(
                            "Examples:",
                            value=examples_text,
                            key=f"examples_{attribute_name}_{value_name}_{attribute_index}",
                            height=100
                        )
                        st.session_state[session_key][value_name]['examples'] = [
                            e.strip() for e in new_examples.split('\n') if e.strip()
                        ]
                        
                    else:
                        # Read-only mode
                        st.markdown(f"**Definition:** {value_def.get('definition', 'No definition available')}")
                        
                        if value_def.get('criteria'):
                            st.markdown("**Criteria:**")
                            for criterion in value_def['criteria']:
                                st.markdown(f"• {criterion}")
                        
                        if value_def.get('examples'):
                            st.markdown("**Examples:**")
                            for example in value_def['examples']:
                                st.markdown(f"• {example}")
            
            # Save/Reset buttons in edit mode
            if edit_mode:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("💾 Save Changes", key=f"save_{attribute_name}_{attribute_index}"):
                        try:
                            # Update the main definitions dict
                            definitions_dict[attribute_name]["definitions"] = st.session_state[session_key].copy()
                            
                            # Send updates to backend
                            from app import call_api_endpoint
                            update_payload = {attribute_name: st.session_state[session_key]}
                            result = call_api_endpoint("/classification/attribute-definitions/update", update_payload)
                            
                            if result.get("success"):
                                st.success("✅ Definitions saved to backend!")
                            else:
                                st.warning("⚠️ Definitions saved locally but not synced to backend")
                                st.error(f"Backend error: {result.get('error', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"❌ Error saving definitions: {str(e)}")
                            st.session_state[session_key] = definitions_dict[attribute_name]["definitions"].copy()  # Revert
                
                with col2:
                    if st.button("🔄 Reset to Default", key=f"reset_{attribute_name}_{attribute_index}"):
                        # Reset to original definitions
                        original_defs = fetch_attribute_definitions()
                        if attribute_name in original_defs:
                            st.session_state[session_key] = original_defs[attribute_name]["definitions"].copy()
                            st.success("🔄 Reset to default definitions!")
                            st.rerun()

def display_custom_definitions_editor():
    """
    Display a section for creating completely custom attribute definitions
    """
    with st.expander("🛠️ Custom Definitions Manager", expanded=False):
        st.markdown("Create or import custom attribute definitions beyond the default Privilege and Determination categories.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export Current Definitions"):
                # Create downloadable JSON of current definitions
                current_definitions = st.session_state.get('detailed_definitions', {})
                st.download_button(
                    label="💾 Download JSON",
                    data=json.dumps(current_definitions, indent=2),
                    file_name="attribute_definitions.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded_file = st.file_uploader("📤 Import Definitions", type=['json'])
            if uploaded_file:
                try:
                    imported_defs = json.load(uploaded_file)
                    st.session_state['detailed_definitions'].update(imported_defs)
                    st.success("✅ Definitions imported successfully!")
                except Exception as e:
                    st.error(f"❌ Error importing definitions: {str(e)}")

def attribute_classification_section():
    """
    Streamlit UI section for testing the attribute classification endpoint with exact JSON format
    """
    st.markdown("### 🏷️ Attribute Classification")
    st.markdown("Classify documents with multiple attributes using allowed values and get supporting snippets")
    
    # Case Context
    st.markdown("#### Case Context")
    col1, col2 = st.columns(2)
    with col1:
        case_name = st.text_input(
            "Case Name", 
            value="Doe v. ACME Corp.",
            help="Name of the legal case"
        )
    with col2:
        case_detail = st.text_area(
            "Case Detail", 
            value="Employment discrimination dispute. Plaintiff alleges wrongful termination and retaliation after reporting safety violations.",
            height=80,
            help="Detailed description of the case"
        )
    
    # Document Input
    st.markdown("#### Document")
    col1, col2 = st.columns(2)
    with col1:
        doc_id = st.text_input(
            "Document ID", 
            value="email-001",
            help="Unique identifier for the document"
        )
    with col2:
        doc_type = st.text_input(
            "Document Type", 
            value="email",
            help="Type of document (email, contract, etc.)"
        )
    
    # Document Text
    doc_text = st.text_area(
        "Document Text", 
        value="""From: John Smith <jsmith@acme.com>
To: Sarah Jones <sjones@acme.com>, Legal Counsel <counsel@lawfirm.com>
Date: March 5, 2024
Subject: Termination of Jane Doe

Sarah,

As discussed, we need to finalize Jane Doe's termination effective this Friday.
Please coordinate with HR to complete the paperwork.

Counsel, please confirm that there are no issues with proceeding given Jane's recent complaint to OSHA.

Thanks,
John""",
        height=200,
        help="Text content of the document to classify"
    )
    
    # Document Processing Options (for large documents)
    with st.expander("🔧 Document Processing Options", expanded=False):
        st.markdown("**Optimize processing for large documents:**")
        
        col1, col2 = st.columns(2)
        with col1:
            max_length = st.number_input(
                "Max Content Length", 
                min_value=1000, 
                max_value=50000, 
                value=8000,
                help="Maximum characters to send to LLM"
            )
            smart_truncation = st.checkbox(
                "Smart Truncation", 
                value=True,
                help="Use intelligent content selection instead of simple truncation"
            )
        
        with col2:
            focus_headers = st.checkbox(
                "Focus on Headers", 
                value=True,
                help="Prioritize email headers and metadata"
            )
            focus_signatures = st.checkbox(
                "Focus on Signatures", 
                value=True,
                help="Include signature blocks and legal disclaimers"
            )
        
        # Show document stats
        doc_length = len(doc_text)
        if doc_length > max_length:
            st.warning(f"⚠️ Document is {doc_length:,} characters. Will be processed using smart truncation to {max_length:,} characters.")
            
            # Preview processing button
            if st.button("🔍 Preview Document Processing", key="preview_processing"):
                with st.spinner("Analyzing document processing..."):
                    from app import call_api_endpoint
                    preview_data = {
                        "content": doc_text,
                        "processing_options": {
                            "max_content_length": max_length,
                            "smart_truncation": smart_truncation,
                            "focus_on_headers": focus_headers,
                            "focus_on_signatures": focus_signatures
                        }
                    }
                    
                    result = call_api_endpoint("/classification/analyze-document", preview_data)
                    
                    if result.get("success"):
                        analysis = result["analysis"]
                        st.success("📊 Document Processing Analysis:")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Length", f"{analysis['original_length']:,} chars")
                        with col2:
                            st.metric("Processed Length", f"{analysis['processed_length']:,} chars")
                        with col3:
                            st.metric("Compression Ratio", f"{analysis['compression_ratio']:.1%}")
                        
                        st.info(f"**Processing Strategy:** {analysis['processing_strategy']}")
                        
                        if analysis.get('sections_included'):
                            st.info(f"**Sections Included:** {', '.join(analysis['sections_included'])}")
                        
                        st.markdown("**Processed Content Preview:**")
                        st.code(analysis['processed_content_preview'], language="text")
                    else:
                        st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
        else:
            st.info(f"✅ Document is {doc_length:,} characters (within limit)")
    
    # LLM Instructions
    llm_instruction = st.text_area(
        "LLM Instructions",
        value="Review the provided document in the context of the case. For each attribute, select one allowed value. Provide a confidence score (0–1) and a supporting snippet with character start/end offsets from the document text. Return results as JSON.",
        height=80,
        help="Instructions for the LLM on how to classify the document"
    )
    
    # Attribute Definitions Section
    st.markdown("#### Define Attributes to Classify")
    
    # Fetch detailed attribute definitions from API
    if 'detailed_definitions' not in st.session_state:
        with st.spinner("Loading detailed attribute definitions..."):
            st.session_state.detailed_definitions = fetch_attribute_definitions()
    
    # Initialize session state for attributes if not exists
    if 'attribute_definitions' not in st.session_state:
        # Initialize with default attributes
        st.session_state.attribute_definitions = [
            {
                "attribute_id": 1,
                "name": "Determination",
                "description": "Primary production decision field",
                "attribute_type": "SingleChoice",
                "is_required": True,
                "is_exclusive": True,
                "allowed_values": ["Responsive", "Non-Responsive", "Technical Flaw", "Further Review Required"]
            },
            {
                "attribute_id": 2,
                "name": "Privilege",
                "description": "Indicates document privilege status based on legal doctrines",
                "attribute_type": "SingleChoice",
                "is_required": True,
                "is_exclusive": True,
                "allowed_values": ["Not Privileged", "Attorney-Client", "Work Product", "Attorney-Client & Work Product"]
            }
        ]
    
    # Function to add a new attribute
    def add_attribute():
        max_id = 0
        for attr in st.session_state.attribute_definitions:
            if attr["attribute_id"] > max_id:
                max_id = attr["attribute_id"]
        
        st.session_state.attribute_definitions.append({
            "attribute_id": max_id + 1,
            "name": "",
            "description": "",
            "attribute_type": "SingleChoice",
            "is_required": True,
            "is_exclusive": True,
            "allowed_values": []
        })
    
    # Function to remove an attribute
    def remove_attribute(idx):
        st.session_state.attribute_definitions.pop(idx)
    
    # Display each attribute with edit fields
    for i, attribute in enumerate(st.session_state.attribute_definitions):
        with st.expander(f"Attribute {attribute['attribute_id']}: {attribute['name'] or 'New Attribute'}", expanded=i==0):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                attribute['name'] = st.text_input(
                    "Attribute Name", 
                    value=attribute['name'],
                    key=f"attr_name_{i}"
                )
                
                attribute['description'] = st.text_area(
                    "Description", 
                    value=attribute['description'],
                    key=f"attr_desc_{i}", 
                    height=80
                )
                
                # Create columns for the attribute properties
                props_col1, props_col2 = st.columns(2)
                
                with props_col1:
                    attribute['attribute_type'] = st.selectbox(
                        "Type", 
                        ["SingleChoice", "MultiChoice"], 
                        index=0 if attribute['attribute_type'] == "SingleChoice" else 1,
                        key=f"attr_type_{i}"
                    )
                
                with props_col2:
                    col1, col2 = st.columns(2)
                    with col1:
                        attribute['is_required'] = st.checkbox(
                            "Required", 
                            value=attribute['is_required'],
                            key=f"attr_required_{i}"
                        )
                    with col2:
                        attribute['is_exclusive'] = st.checkbox(
                            "Exclusive", 
                            value=attribute['is_exclusive'],
                            key=f"attr_exclusive_{i}"
                        )
                
                # Convert allowed values list to comma-separated string for editing
                allowed_values_str = ", ".join(attribute['allowed_values'])
                new_allowed_values = st.text_input(
                    "Allowed Values (comma-separated)", 
                    value=allowed_values_str,
                    key=f"attr_values_{i}"
                )
                # Update the allowed values list
                attribute['allowed_values'] = [val.strip() for val in new_allowed_values.split(",") if val.strip()]
                
                # Display detailed definitions if available
                display_attribute_definitions(attribute['name'], st.session_state.detailed_definitions, i)
            
            with col2:
                if st.button("Remove", key=f"remove_{i}"):
                    remove_attribute(i)
                    st.rerun()
    
    # Add attribute button
    if st.button("➕ Add Attribute"):
        add_attribute()
        st.rerun()
    
    # Custom definitions manager
    display_custom_definitions_editor()
    
    # Run classification button
    if st.button("🏷️ Run Attribute Classification", type="primary", use_container_width=True):
        if not doc_text.strip():
            st.error("Please enter document text to classify")
            return
        
        if not st.session_state.attribute_definitions:
            st.error("Please add at least one attribute to classify")
            return
        
        # Validate attributes
        valid_attributes = True
        for attr in st.session_state.attribute_definitions:
            if not attr['name'].strip():
                st.error(f"Attribute {attr['attribute_id']} is missing a name")
                valid_attributes = False
            if not attr['allowed_values']:
                st.error(f"Attribute '{attr['name']}' has no allowed values")
                valid_attributes = False
        
        if not valid_attributes:
            return
        
        # Create the request payload
        data = {
            "case_context": {
                "case_name": case_name,
                "case_detail": case_detail
            },
            "document": {
                "id": doc_id,
                "content": doc_text
            },
            "attributes": [
                {
                    "attribute_id": attr["attribute_id"],
                    "name": attr["name"],
                    "description": attr["description"], 
                    "allowed_values": attr["allowed_values"]
                }
                for attr in st.session_state.attribute_definitions
            ],
            "processing_options": {
                "max_content_length": max_length,
                "smart_truncation": smart_truncation,
                "focus_on_headers": focus_headers,
                "focus_on_signatures": focus_signatures
            }
        }
        
        # Call the API endpoint
        with st.spinner("🔄 Classifying document..."):
            # Import the call_api_endpoint function from parent module
            from app import call_api_endpoint
            
            result = call_api_endpoint("/classification/attributes", data)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
                st.json(result)
            else:
                # Display results
                st.success("✅ Classification complete!")
                
                # Display document ID
                st.markdown(f"**Document ID:** {result.get('doc_id', 'Unknown')}")
                
                # Display classification results
                st.markdown("### 🏷️ Classification Results")
                
                for attr in result.get("attributes", []):
                    # Create a colored box based on confidence
                    confidence = attr.get('confidence', 0)
                    if confidence >= 0.8:
                        confidence_color = "green"
                    elif confidence >= 0.5:
                        confidence_color = "orange"
                    else:
                        confidence_color = "red"
                    
                    # Create expandable section for each attribute
                    with st.expander(f"{attr.get('name', 'Unknown')}: {attr.get('value', 'Unknown')} (Confidence: {confidence:.2f})", expanded=True):
                        # Extract source information
                        source = attr.get('source', {})
                        snippet = source.get('text', '')
                        start = source.get('start', 0)
                        end = source.get('end', 0)
                        
                        # Display the snippet with context
                        if snippet and doc_text:
                            # Get text before and after the snippet for context
                            context_before = doc_text[max(0, start-30):start] if start > 0 else ""
                            context_after = doc_text[end:min(len(doc_text), end+30)] if end < len(doc_text) else ""
                            
                            st.markdown("**Source Snippet:**")
                            st.markdown(f"...{context_before}**{snippet}**{context_after}...")
                            st.markdown(f"*Character positions: {start} to {end}*")
                        
                        # Display confidence with color
                        st.markdown(f"**Confidence:** <span style='color:{confidence_color};font-weight:bold;'>{confidence:.2f}</span>", unsafe_allow_html=True)
                
                # Raw JSON response for reference
                with st.expander("Raw API Response"):
                    st.json(result)