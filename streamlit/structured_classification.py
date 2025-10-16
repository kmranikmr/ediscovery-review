import streamlit as st
import json
from typing import Dict, List, Any

def structured_classification_section():
    """
    Streamlit UI section for testing the structured attribute classification endpoint
    """
    st.markdown("### 🏷️ Structured Attribute Classification")
    st.markdown("Classify documents according to custom attributes with allowed values")
    
    # Document input
    document_text = st.text_area(
        "Document to Classify:",
        value="From: legal@company.com\nTo: ceo@company.com\nSubject: ATTORNEY-CLIENT PRIVILEGED - Contract Review\n\nI've reviewed the merger agreement with ABC Corp. Several clauses need modification to protect confidential information and limit liability exposure. This analysis is protected by attorney-client privilege.",
        height=150,
        help="Enter the document content you want to classify"
    )
    
    # Case context
    case_context = st.text_area(
        "Case Context:",
        value="This is a litigation case involving a commercial contract dispute between Company X and ABC Corp over a failed merger.",
        height=80,
        help="Provide context about the case or matter to guide classification"
    )
    
    # Optional instructions
    instructions = st.text_area(
        "Additional Instructions (Optional):",
        value="",
        height=50,
        help="Provide any additional instructions for the classification"
    )
    
    # Attribute definitions section
    st.markdown("#### Define Attributes to Classify")
    st.markdown("Add attributes with allowed values that the LLM should classify")
    
    # Initialize session state for attributes if not exists
    if 'structured_attributes' not in st.session_state:
        # Initialize with some default attributes
        st.session_state.structured_attributes = [
            {
                "name": "Privilege Status",
                "description": "Determine if the document contains privileged information",
                "allowed_values": ["Privileged", "Not Privileged", "Potentially Privileged"],
                "examples": {
                    "Privileged": "Communications with lawyers marked as attorney-client privileged",
                    "Not Privileged": "Regular business communications with no legal advice",
                    "Potentially Privileged": "Communications that mention legal issues but aren't clearly marked"
                }
            },
            {
                "name": "Document Type",
                "description": "Categorize the type of document",
                "allowed_values": ["Email", "Contract", "Invoice", "Memo", "Report", "Other"],
                "examples": {}
            }
        ]
    
    # Function to add a new attribute
    def add_attribute():
        st.session_state.structured_attributes.append({
            "name": "",
            "description": "",
            "allowed_values": [],
            "examples": {}
        })
    
    # Function to remove an attribute
    def remove_attribute(idx):
        st.session_state.structured_attributes.pop(idx)
    
    # Display each attribute with edit fields
    for i, attribute in enumerate(st.session_state.structured_attributes):
        with st.expander(f"Attribute {i+1}: {attribute['name'] or 'New Attribute'}", expanded=i==0):
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
                
                # Convert allowed values list to comma-separated string for editing
                allowed_values_str = ", ".join(attribute['allowed_values'])
                new_allowed_values = st.text_input(
                    "Allowed Values (comma-separated)", 
                    value=allowed_values_str,
                    key=f"attr_values_{i}"
                )
                # Update the allowed values list
                attribute['allowed_values'] = [val.strip() for val in new_allowed_values.split(",") if val.strip()]
                
                # Examples section (advanced)
                with st.expander("Examples (Optional)"):
                    # Display existing examples
                    for value in attribute['allowed_values']:
                        example = attribute['examples'].get(value, "")
                        new_example = st.text_area(
                            f"Example for '{value}'", 
                            value=example,
                            key=f"example_{i}_{value}",
                            height=50
                        )
                        if new_example:
                            attribute['examples'][value] = new_example
                        elif value in attribute['examples']:
                            del attribute['examples'][value]
            
            with col2:
                if st.button("Remove", key=f"remove_{i}"):
                    remove_attribute(i)
                    st.experimental_rerun()
    
    # Add attribute button
    if st.button("➕ Add Attribute"):
        add_attribute()
        st.experimental_rerun()
    
    # Run classification button
    if st.button("🏷️ Run Structured Classification", type="primary", use_container_width=True):
        if not document_text.strip():
            st.error("Please enter document text to classify")
            return
        
        if not st.session_state.structured_attributes:
            st.error("Please add at least one attribute to classify")
            return
        
        # Validate attributes
        valid_attributes = True
        for i, attr in enumerate(st.session_state.structured_attributes):
            if not attr['name'].strip():
                st.error(f"Attribute {i+1} is missing a name")
                valid_attributes = False
            if not attr['allowed_values']:
                st.error(f"Attribute '{attr['name']}' has no allowed values")
                valid_attributes = False
        
        if not valid_attributes:
            return
        
        # Create the request payload
        data = {
            "case_context": case_context,
            "document": document_text,
            "attributes": st.session_state.structured_attributes,
            "instructions": instructions if instructions.strip() else None
        }
        
        # Call the API endpoint
        with st.spinner("🔄 Classifying document..."):
            # Get the call_api_endpoint function from parent module
            import sys
            from app import call_api_endpoint
            
            result = call_api_endpoint("/classification/structured", data)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
                st.json(result)
            else:
                # Display results
                st.success("✅ Classification complete!")
                
                # Display summary if available
                if "summary" in result and result["summary"]:
                    st.markdown("### 📝 Summary")
                    st.info(result["summary"])
                
                # Display classification results
                st.markdown("### 🏷️ Classification Results")
                
                for classification in result.get("classifications", []):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{classification['name']}**: {classification['value']}")
                        st.markdown(f"*Reasoning*: {classification['reasoning']}")
                    
                    with col2:
                        # Show confidence with color coding
                        confidence = classification['confidence']
                        if confidence >= 0.8:
                            st.success(f"Confidence: {confidence:.2f}")
                        elif confidence >= 0.5:
                            st.warning(f"Confidence: {confidence:.2f}")
                        else:
                            st.error(f"Confidence: {confidence:.2f}")
                
                # Raw response for debugging
                with st.expander("Raw API Response"):
                    st.json(result)