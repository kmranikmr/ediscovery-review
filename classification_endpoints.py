"""
Attribute Classification endpoint implementation for the API.
"""
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
import requests
import json
import re
from datetime import datetime

# Import configuration
from config import OLLAMA_API_BASE, OLLAMA_MODEL

router = APIRouter()

class ProcessingOptions(BaseModel):
    max_content_length: int = 8000  # Max chars to send to LLM
    smart_truncation: bool = True   # Intelligent content selection
    focus_on_headers: bool = True   # Prioritize email headers, subject lines
    focus_on_signatures: bool = True # Prioritize signature blocks
    include_metadata: bool = True   # Include doc type, dates, participants
    chunk_overlap: int = 200        # Character overlap between chunks

determination_definitions = {
    "Responsive": {
        "definition": "Document is relevant to the case and should be produced to opposing party",
        "criteria": [
            "Relates to claims or defenses in the case",
            "Contains information relevant to disputed facts",
            "Falls within the scope of discovery requests"
        ],
        "examples": [
            "Email discussing contract terms in dispute",
            "Report containing facts about the case",
            "Document referenced in pleadings"
        ]
    },
    "Non-Responsive": {
        "definition": "Document is not relevant to the case and need not be produced",
        "criteria": [
            "Does not relate to claims or defenses in the case",
            "Contains no information relevant to disputed facts",
            "Falls outside the scope of discovery requests"
        ],
        "examples": [
            "Personal email unrelated to litigation",
            "Document about unrelated business matters"
        ]
    },
    "Technical Flaw": {
        "definition": "Document cannot be reviewed due to technical issues",
        "criteria": [
            "File is corrupted or cannot be opened",
            "Contains unreadable or garbled text",
            "Missing critical metadata or information",
            "Format conversion errors or technical processing problems"
        ],
        "examples": [
            "Unreadable PDF attachment",
            "Corrupted email file"
        ]
    },
    "Further Review Required": {
        "definition": "Document requires additional review to determine responsiveness",
        "criteria": [
            "Complex legal or factual issues requiring expert review",
            "Potential privilege claims need investigation",
            "Unclear relevance requiring additional context",
            "Quality control flagging for secondary review"
        ],
        "examples": [
            "Documents with unclear privilege status",
            "Complex technical documents requiring expert analysis",
            "Partially redacted documents needing review",
            "Documents with conflicting classification indicators"
        ]
    }
}

# Hardcoded privilege definitions - these will be used for any "Privilege" attribute
privilege_definitions = {
    "Not Privileged": {
        "definition": "Document contains no privileged information and is discoverable",
        "criteria": [
            "No attorney-client relationship involved in the communication",
            "No legal advice being sought or provided", 
            "Document not prepared in anticipation of litigation",
            "Information is of a business or factual nature",
            "Legal counsel is NOT copied or consulted",
            "No request for legal guidance or confirmation"
        ],
        "examples": [
            "Regular business communications between employees",
            "Factual reports and updates without legal implications", 
            "Non-legal business decisions and operational matters",
            "Public information or announcements",
            "HR communications not involving legal advice",
            "Internal business discussions without counsel involvement"
        ]
    },
    "Attorney-Client": {
        "definition": "Communication protected by attorney-client privilege",
        "criteria": [
            "Communication between attorney and client (or authorized representative)",
            "Made for the purpose of obtaining or providing legal advice",
            "Confidential communication (not disclosed to third parties)",
            "Privilege has not been waived",
            "Includes emails where legal counsel is copied or consulted",
            "Contains requests for legal guidance or confirmation"
        ],
        "examples": [
            "Email from lawyer to client providing legal advice",
            "Client request to attorney for legal guidance", 
            "Confidential legal consultation notes",
            "Attorney's legal opinion to client",
            "Email copying legal counsel asking for legal confirmation",
            "Communication seeking legal advice about business decisions",
            "Request to attorney about legal implications of actions"
        ]
    },
    "Work Product": {
        "definition": "Materials prepared by or for attorney in anticipation of litigation",
        "criteria": [
            "Document prepared in anticipation of litigation",
            "Prepared by or at direction of attorney",
            "Contains attorney's mental impressions, conclusions, or strategies",
            "Protected from discovery under work product doctrine"
        ],
        "examples": [
            "Attorney's case strategy memoranda",
            "Litigation preparation materials", 
            "Witness interview notes by counsel",
            "Legal research in anticipation of lawsuit"
        ]
    },
    "Attorney-Client & Work Product": {
        "definition": "Document protected by both attorney-client privilege and work product doctrine",
        "criteria": [
            "Meets criteria for both attorney-client privilege AND work product",
            "Confidential communication between attorney and client",
            "Prepared in anticipation of litigation",
            "Contains legal advice AND attorney mental impressions/strategy"
        ],
        "examples": [
            "Attorney's email to client containing legal advice and litigation strategy",
            "Confidential legal memorandum with case analysis and recommendations",
            "Attorney-client communication about litigation tactics",
            "Joint defense agreement discussions with legal strategy"
        ]
    }
}

# Attribute Classification Models
class AttributeDefinition(BaseModel):
    attribute_id: Optional[Union[str, int]] = None
    name: str
    description: str
    attribute_type: Optional[str] = None
    is_required: Optional[bool] = None
    is_exclusive: Optional[bool] = None
    allowed_values: List[str]

class CaseContext(BaseModel):
    case_name: str
    case_detail: str

class DocumentInfo(BaseModel):
    id: str
    content: str
    content_preview: Optional[str] = None  # First 1000 chars for quick preview
    content_length: Optional[int] = None   # Total character count
    chunk_size: Optional[int] = None       # Size of content chunks if splitting
    focus_sections: Optional[List[str]] = None  # Key sections to prioritize

class TextChunk(BaseModel):
    chunk_id: str
    content: str
    start_position: int
    end_position: int
    priority: Optional[int] = 1  # Higher numbers = higher priority

class ChunkedDocumentInfo(BaseModel):
    id: str
    total_length: int
    chunks: List[TextChunk]
    metadata: Optional[Dict[str, Any]] = None

class DetailedAttributeDefinition(BaseModel):
    name: str
    description: str
    allowed_values: List[str]
    definitions: Dict[str, Dict[str, Any]] = None

class AttributeClassificationRequest(BaseModel):
    case_context: CaseContext
    document: DocumentInfo
    attributes: List[AttributeDefinition]
    processing_options: Optional[Dict[str, Any]] = None  # New: processing controls

class ChunkedAttributeClassificationRequest(BaseModel):
    case_context: CaseContext
    document: ChunkedDocumentInfo
    attributes: List[AttributeDefinition]
    processing_options: Optional[Dict[str, Any]] = None

class AttributeValueResponse(BaseModel):
    value: str
    confidence: float
    source_text: str
    start_pos: int
    end_pos: int

class AttributeClassificationResponse(BaseModel):
    document_id: str
    attributes: Dict[str, AttributeValueResponse]

# Helper functions for smart text processing
def smart_text_processor(content: str, max_length: int = 8000, options: ProcessingOptions = None, attributes: List[AttributeDefinition] = None) -> Dict[str, Any]:
    """
    Intelligently process large documents for classification, considering which attributes need to be classified
    """
    if not options:
        options = ProcessingOptions()
    
    content_length = len(content)
    
    if content_length <= max_length:
        return {
            "processed_content": content,
            "truncated": False,
            "strategy": "full_content",
            "original_length": content_length
        }
    
    # Analyze what attributes we need to classify to optimize content selection
    needs_privilege = any(attr.name.lower() == "privilege" for attr in (attributes or []))
    needs_determination = any(attr.name.lower() == "determination" for attr in (attributes or []))
    
    # Strategy 1: Smart extraction for emails
    if is_email_content(content):
        return extract_email_essentials(content, max_length, needs_privilege, needs_determination)
    
    # Strategy 2: Smart extraction for legal documents
    if is_legal_document(content):
        return extract_legal_essentials(content, max_length, needs_privilege, needs_determination)
    
    # Strategy 3: General intelligent truncation
    return intelligent_truncation(content, max_length, options)

def is_email_content(content: str) -> bool:
    """Check if content appears to be an email"""
    email_indicators = ['From:', 'To:', 'Subject:', 'Date:', 'Sent:', '@']
    return sum(1 for indicator in email_indicators if indicator in content) >= 3

def is_legal_document(content: str) -> bool:
    """Check if content appears to be a legal document"""
    legal_indicators = ['attorney', 'counsel', 'legal', 'privileged', 'confidential', 
                       'litigation', 'contract', 'agreement', 'whereas', 'plaintiff', 'defendant']
    content_lower = content.lower()
    return sum(1 for indicator in legal_indicators if indicator in content_lower) >= 2

def extract_email_essentials(content: str, max_length: int, needs_privilege: bool = True, needs_determination: bool = True) -> Dict[str, Any]:
    """Extract key parts of email for classification, considering both privilege and determination needs"""
    lines = content.split('\n')
    
    # Always include headers (From, To, Subject, Date) - important for both privilege and determination
    header_section = []
    body_start = 0
    
    for i, line in enumerate(lines):
        if any(line.startswith(header) for header in ['From:', 'To:', 'Cc:', 'Bcc:', 'Subject:', 'Date:', 'Sent:']):
            header_section.append(line)
            body_start = max(body_start, i + 1)
        elif line.strip() == '' and header_section:
            body_start = i + 1
            break
    
    # Include signature blocks (important for privilege detection)
    signature_section = []
    signature_start = len(lines)
    if needs_privilege:
        for i in range(len(lines) - 1, max(0, len(lines) - 20), -1):
            line = lines[i].strip()
            if any(indicator in line.lower() for indicator in ['regards', 'sincerely', 'best', 'thanks', 'attorney', 'counsel', 'esq']):
                signature_section = lines[i:]
                signature_start = i
                break
    
    # For determination classification, prioritize substantive content over signatures
    if needs_determination:
        # Look for business-relevant content in the body
        body_lines = lines[body_start:signature_start]
        
        # Find lines that might be relevant to case determination
        important_body_lines = []
        for i, line in enumerate(body_lines):
            line_lower = line.lower()
            # Look for business context, decisions, actions, case-relevant content
            if any(term in line_lower for term in ['termination', 'employment', 'lawsuit', 'complaint', 'investigation', 
                                                   'violation', 'policy', 'hr', 'meeting', 'decision', 'action',
                                                   'project', 'contract', 'agreement', 'business', 'financial',
                                                   'report', 'analysis', 'recommendation', 'proposal']):
                # Include context around important lines
                start_ctx = max(0, i - 1)
                end_ctx = min(len(body_lines), i + 2)
                context_lines = body_lines[start_ctx:end_ctx]
                important_body_lines.extend(context_lines)
    
    # Combine sections with priority based on needs
    essential_content = '\n'.join(header_section)
    if essential_content:
        essential_content += '\n\n'
    
    # Calculate space allocation
    header_size = len(essential_content)
    signature_size = len('\n'.join(signature_section)) if signature_section else 0
    
    if needs_determination and 'important_body_lines' in locals():
        # Prioritize important body content for determination
        important_body = '\n'.join(set(important_body_lines))  # Remove duplicates
        available_space = max_length - header_size - signature_size - 100
        
        if len(important_body) <= available_space:
            essential_content += important_body + '\n'
        else:
            # Include as much important content as possible
            essential_content += important_body[:available_space-10] + '\n...\n'
    else:
        # Original logic for privilege-focused processing
        remaining_space = max_length - header_size - signature_size - 50
        if remaining_space > 0:
            body_content = '\n'.join(lines[body_start:signature_start])
            if len(body_content) > remaining_space:
                essential_content += body_content[:remaining_space-10] + '\n...\n'
            else:
                essential_content += body_content + '\n'
    
    # Add signature (more important for privilege than determination)
    if signature_section and (needs_privilege or len(essential_content) + signature_size < max_length):
        essential_content += '\n'.join(signature_section)
    
    sections_included = ["headers"]
    if needs_determination:
        sections_included.append("case_relevant_content")
    else:
        sections_included.append("body_excerpt")
    if signature_section:
        sections_included.append("signature")
    
    return {
        "processed_content": essential_content[:max_length],
        "truncated": True,
        "strategy": "email_essentials_smart",
        "original_length": len(content),
        "sections_included": sections_included
    }

def extract_legal_essentials(content: str, max_length: int, needs_privilege: bool = True, needs_determination: bool = True) -> Dict[str, Any]:
    """Extract key parts of legal documents for both privilege and determination classification"""
    lines = content.split('\n')
    
    # Find lines with legal significance (for privilege)
    privilege_sections = []
    if needs_privilege:
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(term in line_lower for term in ['privileged', 'confidential', 'attorney-client', 
                                                 'work product', 'legal advice', 'counsel']):
                # Include context around important lines
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                section = '\n'.join(lines[start:end])
                privilege_sections.append(section)
    
    # Find lines with determination significance (for responsiveness)
    determination_sections = []
    if needs_determination:
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # Look for case-relevant terms, business decisions, factual information
            if any(term in line_lower for term in ['termination', 'employment', 'discrimination', 'harassment',
                                                  'contract', 'agreement', 'breach', 'damages', 'liability',
                                                  'investigation', 'complaint', 'violation', 'policy',
                                                  'meeting', 'decision', 'recommendation', 'action',
                                                  'financial', 'payment', 'invoice', 'cost', 'expense']):
                # Include more context for determination-relevant content
                start = max(0, i - 3)
                end = min(len(lines), i + 4)
                section = '\n'.join(lines[start:end])
                determination_sections.append(section)
    
    # Combine sections based on priority
    combined_sections = []
    sections_included = []
    
    if needs_determination and determination_sections:
        # Prioritize determination content
        combined_sections.extend(determination_sections)
        sections_included.append("case_relevant_content")
    
    if needs_privilege and privilege_sections:
        # Add privilege content
        combined_sections.extend(privilege_sections)
        sections_included.append("privilege_indicators")
    
    # Remove duplicates and combine
    unique_sections = []
    seen_content = set()
    for section in combined_sections:
        if section not in seen_content:
            unique_sections.append(section)
            seen_content.add(section)
    
    combined_content = '\n...\n'.join(unique_sections)
    
    if len(combined_content) <= max_length:
        return {
            "processed_content": combined_content,
            "truncated": True,
            "strategy": "legal_essentials_smart", 
            "original_length": len(content),
            "sections_included": sections_included
        }
    
    # If still too long, truncate intelligently
    return intelligent_truncation(combined_content, max_length, ProcessingOptions())

def intelligent_truncation(content: str, max_length: int, options: ProcessingOptions) -> Dict[str, Any]:
    """General intelligent truncation strategy"""
    if len(content) <= max_length:
        return {"processed_content": content, "truncated": False, "strategy": "full_content"}
    
    # Take beginning and end with gap indicator
    begin_size = max_length // 2 - 50
    end_size = max_length // 2 - 50
    
    begin_content = content[:begin_size]
    end_content = content[-end_size:]
    
    processed_content = f"{begin_content}\n\n[... content truncated for length ...]\n\n{end_content}"
    
    return {
        "processed_content": processed_content,
        "truncated": True,
        "strategy": "intelligent_truncation",
        "original_length": len(content),
        "begin_chars": begin_size,
        "end_chars": end_size
    }

def create_fallback_response(result_text: str, document_id: str, attributes: List[AttributeDefinition]) -> Dict[str, Any]:
    """Create a fallback response when JSON parsing fails - analyze the text response"""
    fallback_result = {
        "document_id": document_id,
        "attributes": {}
    }
    
    # Try to extract values using text patterns
    for attr in attributes:
        attr_name = attr.name
        best_match = None
        best_confidence = 0.1
        best_source = "No clear classification found"
        
        # Look for attribute mentions in the text with better analysis
        if attr.name.lower() == "privilege":
            # Look for privilege indicators in the response
            privilege_indicators = {
                "Not Privileged": ["not privileged", "no privilege", "not protected", "not confidential"],
                "Attorney-Client": ["attorney-client", "legal advice", "counsel", "attorney", "privileged"],
                "Work Product": ["work product", "prepared for litigation", "trial preparation", "litigation strategy"],
                "Attorney-Client & Work Product": ["both", "attorney-client and work product", "dual privilege"]
            }
            
            for value in attr.allowed_values:
                if value in privilege_indicators:
                    indicators = privilege_indicators[value]
                    matches = sum(1 for indicator in indicators if indicator in result_text.lower())
                    if matches > 0:
                        confidence = min(0.7, 0.3 + (matches * 0.1))
                        if confidence > best_confidence:
                            best_match = value
                            best_confidence = confidence
                            best_source = f"Found {matches} privilege indicators"
        
        elif attr.name.lower() == "determination":
            # Look for determination indicators in the response
            determination_indicators = {
                "Responsive": ["responsive", "relevant", "related to case", "pertains to"],
                "Non-Responsive": ["non-responsive", "not relevant", "unrelated", "not responsive"],
                "Technical Flaw": ["technical flaw", "corrupted", "unreadable", "processing error"],
                "Further Review Required": ["review required", "unclear", "ambiguous", "needs review"]
            }
            
            for value in attr.allowed_values:
                if value in determination_indicators:
                    indicators = determination_indicators[value]
                    matches = sum(1 for indicator in indicators if indicator in result_text.lower())
                    if matches > 0:
                        confidence = min(0.7, 0.3 + (matches * 0.1))
                        if confidence > best_confidence:
                            best_match = value
                            best_confidence = confidence
                            best_source = f"Found {matches} determination indicators"
        
        else:
            # Generic attribute analysis
            for value in attr.allowed_values:
                if value.lower() in result_text.lower():
                    best_match = value
                    best_confidence = 0.5
                    best_source = f"Found text match for: {value}"
                    break
        
        # If no pattern match found, use first allowed value with very low confidence
        if best_match is None:
            best_match = attr.allowed_values[0] if attr.allowed_values else "Unknown"
            best_confidence = 0.1
            best_source = "Fallback classification - no clear indicators found"
        
        fallback_result["attributes"][attr_name] = {
            "value": best_match,
            "confidence": best_confidence,
            "source_text": best_source,
            "start_pos": 0,
            "end_pos": len(best_source)
        }
    
    return fallback_result

# API Endpoints

@router.post("/classification/attributes")
async def classify_document_attributes(request: AttributeClassificationRequest):
    """
    Classify document attributes based on provided attribute definitions with allowed values
    """
    try:
        # Extract the document text
        document_text = request.document.content
        document_id = request.document.id
        
        # Parse processing options
        processing_opts = ProcessingOptions()
        if request.processing_options:
            for key, value in request.processing_options.items():
                if hasattr(processing_opts, key):
                    setattr(processing_opts, key, value)
        
        # Apply smart text processing for large documents
        text_processing_result = smart_text_processor(
            document_text, 
            processing_opts.max_content_length, 
            processing_opts,
            request.attributes  # Pass attributes so processor knows what to optimize for
        )
        processed_text = text_processing_result["processed_content"]
        
        # Prepare case context
        case_context = f"Case: {request.case_context.case_name}\nContext: {request.case_context.case_detail}"
        
        # Create enhanced attribute descriptions with detailed definitions
        attribute_descriptions = []
        for attr in request.attributes:
            attr_desc = f"\n{attr.name}: {attr.description}\n"
            
            # Use hardcoded definitions for Privilege and Determination attributes
            if attr.name.lower() == "privilege":
                for value_name in attr.allowed_values:
                    if value_name in privilege_definitions:
                        def_info = privilege_definitions[value_name]
                        criteria_text = "\n      ".join(def_info["criteria"])
                        examples_text = "\n      ".join(def_info["examples"])
                        
                        attr_desc += f"""
  • {value_name}:
    Definition: {def_info["definition"]}
    Criteria:
      {criteria_text}
    Examples:
      {examples_text}"""
                    else:
                        # Fallback for any privilege values not in our hardcoded definitions
                        attr_desc += f"\n  • {value_name}: Standard privilege classification"
            elif attr.name.lower() == "determination":
                for value_name in attr.allowed_values:
                    if value_name in determination_definitions:
                        def_info = determination_definitions[value_name]
                        criteria_text = "\n      ".join(def_info["criteria"])
                        examples_text = "\n      ".join(def_info["examples"])
                        
                        attr_desc += f"""
  • {value_name}:
    Definition: {def_info["definition"]}
    Criteria:
      {criteria_text}
    Examples:
      {examples_text}"""
                    else:
                        # Fallback for any determination values not in our hardcoded definitions
                        attr_desc += f"\n  • {value_name}: Standard determination classification"
            else:
                # For other attributes, use simple list format
                allowed_values_text = ", ".join(attr.allowed_values)
                attr_desc += f"  Allowed values: {allowed_values_text}"
                
            attribute_descriptions.append(attr_desc)
        
        attribute_descriptions_text = "\n".join(attribute_descriptions)
        
        prompt = f"""You are an expert legal document analyst specializing in eDiscovery and privilege determination. You have extensive knowledge of attorney-client privilege, work product doctrine, and legal confidentiality rules.

CASE CONTEXT:
{case_context}

DOCUMENT ID: {document_id}

ATTRIBUTES TO CLASSIFY:
{attribute_descriptions_text}

   - IMPORTANT: If any email header (From, To, CC, BCC) includes legal counsel, attorney, or law firm, classify as Attorney-Client privilege
ANALYSIS INSTRUCTIONS:
1. Carefully read the document text and identify key indicators for each attribute type

2. For DETERMINATION classifications, analyze:
   - Case relevance: Does this document relate to the claims, defenses, or disputed facts?
   - Business context: What business decisions, actions, or events does it describe?
   - Factual content: Does it contain evidence supporting or contradicting party positions?
   - Scope of discovery: Does it fall within what was requested in discovery?
   - Technical issues: Are there any processing or quality problems with the document?

3. For PRIVILEGE determinations, look SPECIFICALLY for:
   - ANY involvement of legal counsel, attorneys, or law firms (even if copied/CC'd)
   - Legal advice being sought or provided (including requests for legal confirmation)
   - Questions about legal implications or compliance issues
   - Communications about litigation strategy or legal risks
   - Confidentiality markings or disclaimers
   - Work product prepared for litigation
   - Mental impressions, strategies, or legal analysis

4. CRITICAL CLASSIFICATION PRIORITIES:
   
   DETERMINATION PRIORITY:
   - If directly related to case facts/claims → likely Responsive
   - If unrelated to case issues → likely Non-Responsive
   - If document has technical problems → Technical Flaw
   - If unclear relevance or needs expert review → Further Review Required
   
   PRIVILEGE PRIORITY:
   - If legal counsel is mentioned/involved → likely Attorney-Client privilege
   - If seeking legal advice/confirmation → likely Attorney-Client privilege  
   - If prepared for litigation → likely Work Product
   - If both legal advice AND litigation prep → likely Attorney-Client & Work Product
   - Only if NO legal involvement → Not Privileged

5. Apply the specific criteria and definitions provided for each allowed value
6. Extract exact supporting text that justifies your classification
7. Assign confidence scores based on strength of evidence

IMPORTANT: Even if the document was processed/truncated, make your best determination based on the available content. The processing preserves the most relevant sections for each attribute type.

CRITICAL SOURCE TEXT INSTRUCTION:
When filling the "source_text" field, you must copy actual words/phrases directly from the document content below. Do not write explanations, summaries, or analysis. Simply find the most relevant sentence or phrase from the document and copy it exactly.

NOW READ THE DOCUMENT CONTENT TO CLASSIFY:

DOCUMENT TEXT:
{processed_text}

CLASSIFICATION REQUIREMENTS:
- You MUST classify ALL attributes provided in the request
- For each attribute, determine the best value from the allowed values list using the detailed definitions and criteria provided
- Extract relevant text FROM THE ACTUAL DOCUMENT CONTENT that supports your classification
- The source_text should be actual sentences or phrases from the document being analyzed
- Provide exact character positions (start and end) for the supporting text within the document
- Assign a confidence score (0.0 to 1.0) for each classification

IMPORTANT FOR SOURCE_TEXT: Copy actual words from the document content, not analysis descriptions. For example:
- If the document says "Meeting with John about Q3 budget" → use "Meeting with John about Q3 budget"  
- Do NOT use "This document relates to business decisions" or similar analysis statements
    IMPORTANT FOR SOURCE_TEXT: Copy actual words from the document content, not analysis descriptions. For example:
    If the document says "Meeting with John about Q3 budget" → use "Meeting with John about Q3 budget"  
    Do NOT use "This document relates to business decisions" or similar analysis statements

    FINAL INSTRUCTION: For each attribute, the source_text must be a verbatim copy of a sentence or phrase from the DOCUMENT TEXT above. Do NOT summarize, paraphrase, or invent text. If no relevant sentence exists, return an empty string for source_text.

Format your response as a valid JSON object with this exact structure - DO NOT include any other text:
{{
    "document_id": "{document_id}",
    "attributes": {{
        "Determination": {{"value": "Responsive", "confidence": 0.95, "source_text": "<actual sentence or phrase from document>", "start_pos": 123, "end_pos": 145}},
        "Privilege": {{"value": "Not Privileged", "confidence": 0.95, "source_text": "<actual sentence or phrase from document>", "start_pos": 150, "end_pos": 180}}
    }}
}}

IMPORTANT: Replace <actual sentence or phrase from document> with a real sentence or phrase copied from the DOCUMENT TEXT above for each attribute.

Ensure your response:
1. Is valid JSON with no trailing commas
2. Includes exactly these {len(request.attributes)} attributes: {', '.join([attr.name for attr in request.attributes])}
3. Uses only values from the allowed values list for each attribute
4. Contains no text outside the JSON structure
"""
        
        print("\n========== LLM PROMPT DEBUG ==========")
        print(prompt)
        print("========== END PROMPT DEBUG ==========")
        
        # Call Ollama for classification
        response = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": "You are an expert legal document analyst specializing in document attribute classification.",
                "stream": False,
                "options": {"temperature": 0.1}  # Lower temperature for more deterministic results
            },
            timeout=60
        )
        
        if response.status_code != 200:
            return {"success": False, "error": f"Ollama API error: {response.status_code} - {response.text}"}
        
        # Extract the JSON response from the LLM
        result_text = response.json().get("response", "")
        
        # Enhanced JSON parsing with multiple pattern attempts
        def enhanced_json_parse(response_text: str, doc_id: str, attributes: List[AttributeDefinition]) -> Dict[str, Any]:
            """Try multiple JSON parsing strategies"""
            
            # Enhanced JSON extraction - be more aggressive
            json_candidates = []
            
            # Strategy 1: Look for complete JSON with required fields
            complete_json = re.search(r'\{[^{}]*"document_id"[^{}]*"attributes"[^{}]*\{.*?\}\s*\}', response_text, re.DOTALL)
            if complete_json:
                json_candidates.append(complete_json.group())
            
            # Strategy 2: Look for JSON blocks (between curly braces)
            json_blocks = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            json_candidates.extend(json_blocks)
            
            # Strategy 3: Look for JSON starting with document_id
            doc_id_json = re.search(r'\{\s*"document_id".*?\}\s*$', response_text, re.DOTALL | re.MULTILINE)
            if doc_id_json:
                json_candidates.append(doc_id_json.group())
            
            # Strategy 4: Extract everything between first { and last }
            if '{' in response_text and '}' in response_text:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start < end:
                    json_candidates.append(response_text[start:end])
            
            if not json_candidates:
                return create_fallback_response(response_text, doc_id, attributes)
            
            # Try multiple cleanup patterns for each JSON candidate
            for json_str in json_candidates:
                patterns_to_try = [
                    json_str,  # Original
                    re.sub(r',(\s*[}\]])', r'\1', json_str),  # Remove trailing commas
                    re.sub(r'([^\\])"([^":,}\s]*?)":', r'\1"\2":', json_str),  # Fix unescaped quotes
                    re.sub(r':\s*"([^"]*?)"([^,}\]])', r': "\1"\2', json_str),  # Fix missing quotes
                    # More aggressive cleaning
                    re.sub(r'([{,]\s*)(\w+)(\s*):', r'\1"\2"\3:', json_str),  # Add quotes to unquoted keys
                    re.sub(r':\s*([^",{\[\]}\s]+)([,}])', r': "\1"\2', json_str),  # Add quotes to unquoted values
                ]
                
                for i, pattern in enumerate(patterns_to_try):
                    try:
                        pattern = pattern.strip()
                        result = json.loads(pattern)
                        
                        # Validate that all required attributes are present
                        if "document_id" in result and "attributes" in result:
                            # Ensure all requested attributes are in the response
                            expected_attrs = set(attr.name for attr in attributes)
                            actual_attrs = set(result["attributes"].keys())
                            
                            if expected_attrs == actual_attrs:
                                return result
                            else:
                                # Add missing attributes with fallback values
                                for attr in attributes:
                                    if attr.name not in result["attributes"]:
                                        result["attributes"][attr.name] = {
                                            "value": attr.allowed_values[0] if attr.allowed_values else "Unknown",
                                            "confidence": 0.3,
                                            "source_text": "Generated due to missing classification",
                                            "start_pos": 0,
                                            "end_pos": 0
                                        }
                                return result
                                        
                    except json.JSONDecodeError:
                        continue
            
            # All patterns failed, use fallback
            return create_fallback_response(response_text, doc_id, attributes)

        classification_result = enhanced_json_parse(result_text, document_id, request.attributes)

        # Inject attribute_id for each attribute in the response, matching the request
        # If the attribute in request has an 'attribute_id' field, add it to the response
        # Otherwise, use the attribute name as id
        for attr in request.attributes:
            attr_name = attr.name
            attr_id = attr.attribute_id if attr.attribute_id is not None else attr_name
            if "attributes" in classification_result and attr_name in classification_result["attributes"]:
                classification_result["attributes"][attr_name]["attribute_id"] = attr_id

        # Return the structured classification result
        return {"success": True, "result": classification_result}
    
    except Exception as e:
        import traceback
        return {"success": False, "error": f"Classification error: {str(e)}", "traceback": traceback.format_exc()}

@router.get("/classification/attribute-definitions")
async def get_attribute_definitions():
    """
    Retrieve available attribute definitions with detailed explanations for supported attributes
    """
    try:
        attribute_definitions = [
            DetailedAttributeDefinition(
                name="Determination",
                description="Primary production decision field",
                allowed_values=["Responsive", "Non-Responsive", "Technical Flaw", "Further Review Required"],
                definitions=determination_definitions
            ),
            DetailedAttributeDefinition(
                name="Privilege",
                description="Indicates document privilege status based on legal doctrines",
                allowed_values=["Not Privileged", "Attorney-Client", "Work Product", "Attorney-Client & Work Product"],
                definitions=privilege_definitions
            )
        ]
        
        return {
            "success": True,
            "attribute_definitions": [attr.dict() for attr in attribute_definitions]
        }
    
    except Exception as e:
        import traceback
        return {"success": False, "error": f"Error retrieving attribute definitions: {str(e)}", "traceback": traceback.format_exc()}

@router.post("/classification/attribute-definitions/update")
async def update_attribute_definitions(definitions_update: Dict[str, Any]):
    """
    Update attribute definitions with custom user modifications
    """
    try:
        global privilege_definitions, determination_definitions
        
        # Update privilege definitions if provided
        if "Privilege" in definitions_update:
            privilege_definitions.update(definitions_update["Privilege"])
        
        # Update determination definitions if provided  
        if "Determination" in definitions_update:
            determination_definitions.update(definitions_update["Determination"])
        
        return {
            "success": True,
            "message": "Attribute definitions updated successfully"
        }
    
    except Exception as e:
        import traceback
        return {"success": False, "error": f"Error updating attribute definitions: {str(e)}", "traceback": traceback.format_exc()}

@router.post("/classification/analyze-document")
async def analyze_document_processing(request: Dict[str, Any]):
    """
    Analyze how a document would be processed without running classification
    """
    try:
        content = request.get("content", "")
        processing_options = request.get("processing_options", {})
        
        # Parse processing options
        processing_opts = ProcessingOptions()
        for key, value in processing_options.items():
            if hasattr(processing_opts, key):
                setattr(processing_opts, key, value)
        
        # Apply smart text processing
        result = smart_text_processor(content, processing_opts.max_content_length, processing_opts)
        
        return {
            "success": True,
            "analysis": {
                "original_length": result["original_length"],
                "processed_length": len(result["processed_content"]),
                "truncated": result["truncated"],
                "processing_strategy": result["strategy"],
                "processed_content_preview": result["processed_content"][:500] + "..." if len(result["processed_content"]) > 500 else result["processed_content"],
                "sections_included": result.get("sections_included", []),
                "compression_ratio": round(len(result["processed_content"]) / result["original_length"], 3) if result["original_length"] > 0 else 1.0
            }
        }
    
    except Exception as e:
        import traceback
        return {"success": False, "error": f"Error analyzing document: {str(e)}", "traceback": traceback.format_exc()}