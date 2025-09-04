#!/usr/bin/env python3
"""
Simple NER Processor for FastAPI Integration
Provides basic NER functionality without complex dependencies
"""
import re
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class SimpleNERProcessor:
    """Simple NER processor using regex patterns and basic transformers"""
    
    def __init__(self):
        self.ner_pipeline = None
        self.device = -1  # CPU by default
        self.initialized = False
        
    def initialize(self):
        """Initialize the NER pipeline"""
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
            
            # Check for GPU
            if torch.cuda.is_available():
                self.device = 0
                logger.info("Using GPU for NER")
            else:
                self.device = -1
                logger.info("Using CPU for NER")
            
            # Initialize model
            model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
            logger.info(f"Loading NER model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            if self.device >= 0:
                model = model.to(self.device)
            
            self.ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=self.device
            )
            
            self.initialized = True
            logger.info("NER pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize NER pipeline: {e}")
            self.initialized = False
            return False
    
    def extract_entities_from_text(self, text: str, entity_types: List[str] = None, 
                                 include_pii: bool = True, min_score: float = 0.7) -> Dict[str, Any]:
        """Extract entities from text"""
        if not self.initialized:
            if not self.initialize():
                return {"entities": [], "error": "NER pipeline not available"}
        
        try:
            # Process with transformer model
            entities = []
            
            if self.ner_pipeline:
                # Process in chunks to handle long text
                chunks = self._chunk_text(text, max_length=400)
                offset = 0
                
                for chunk in chunks:
                    ner_results = self.ner_pipeline(chunk)
                    
                    for entity in ner_results:
                        if entity["score"] >= min_score:
                            # Adjust positions for the full text
                            entities.append({
                                "text": entity["word"],
                                "label": entity["entity_group"],
                                "start": entity["start"] + offset,
                                "end": entity["end"] + offset,
                                "confidence": float(round(entity["score"], 4)),  # Convert to Python float
                                "source": "transformers"
                            })
                    
                    offset += len(chunk)
            
            # Add regex-based PII detection if requested
            if include_pii:
                pii_entities = self._extract_pii_regex(text)
                entities.extend(pii_entities)
            
            # Map BERT entity types to standard types
            entities = self._map_entity_types(entities)
            
            # Filter by entity types if specified
            if entity_types:
                entities = [e for e in entities if e["label"] in entity_types]
            
            # Remove duplicates and sort by position
            entities = self._deduplicate_entities(entities)
            entities.sort(key=lambda x: x["start"])
            
            return {
                "entities": entities,
                "total_entities": len(entities),
                "entity_types": list(set(e["label"] for e in entities)),
                "text_length": len(text),
                "processing_info": {
                    "model_used": "BERT-based NER" if self.ner_pipeline else "Regex patterns",
                    "device": "GPU" if self.device >= 0 else "CPU"
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {"entities": [], "error": str(e)}
    
    def _chunk_text(self, text: str, max_length: int = 400) -> List[str]:
        """Split text into manageable chunks"""
        # Simple sentence-based chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sent in sentences:
            if len(current_chunk) + len(sent) < max_length:
                current_chunk += " " + sent
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sent
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _extract_pii_regex(self, text: str) -> List[Dict[str, Any]]:
        """Extract PII using regex patterns"""
        pii_patterns = {
            "EMAIL_ADDRESS": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "PHONE_NUMBER": r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            "US_SSN": r'\b(?!000|666|9\d{2})([0-8]\d{2}|7([0-6]\d))-(?!00)(\d{2})-(?!0000)(\d{4})\b',
            "CREDIT_CARD": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "URL": r'https?://[^\s<>"{}|\\^`[\]]+',
            "IP_ADDRESS": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        }
        
        entities = []
        for label, pattern in pii_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "text": match.group(),
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                    "score": 0.95,  # High confidence for regex matches
                    "source": "regex"
                })
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities, keeping highest confidence"""
        seen = {}
        for entity in entities:
            key = (entity["text"], entity["label"], entity["start"], entity["end"])
            if key not in seen or entity["score"] > seen[key]["score"]:
                seen[key] = entity
        return list(seen.values())
    
    def extract_entities_from_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Extract entities from a file"""
        try:
            import os
            
            if not os.path.exists(file_path):
                return {"entities": [], "error": f"File not found: {file_path}"}
            
            # Extract text from file
            text = self._extract_text_from_file(file_path)
            if not text:
                return {"entities": [], "error": "No text could be extracted from file"}
            
            # Extract text processing parameters (exclude file-specific ones)
            text_kwargs = {k: v for k, v in kwargs.items() if k not in ['include_content']}
            
            # Process the text
            result = self.extract_entities_from_text(text, **text_kwargs)
            
            # Add file information
            result["file_info"] = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_size": os.path.getsize(file_path)
            }
            
            # Optionally include content
            if kwargs.get("include_content", False):
                max_content = 50000  # 50KB limit
                result["content"] = text[:max_content]
                if len(text) > max_content:
                    result["content"] += "... [content truncated]"
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {"entities": [], "error": str(e)}
    
    def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        import os
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.txt', '.jsonl']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
        
        elif file_ext == '.pdf':
            try:
                # Try to use PyMuPDF if available
                import fitz
                text = ""
                with fitz.open(file_path) as doc:
                    for page in doc:
                        text += page.get_text("text") + "\n\n"
                return text
            except ImportError:
                logger.warning("PyMuPDF not available for PDF processing")
                return ""
            except Exception as e:
                logger.error(f"Error extracting PDF text: {e}")
                return ""
        
        else:
            logger.warning(f"Unsupported file type: {file_ext}")
            return ""
    
    def _map_entity_types(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map BERT entity types to standard entity types"""
        entity_type_mapping = {
            "PER": "PERSON",
            "ORG": "ORGANIZATION", 
            "LOC": "LOCATION",
            "MISC": "MISCELLANEOUS"
        }
        
        mapped_entities = []
        for entity in entities:
            original_label = entity.get("label", "")
            mapped_label = entity_type_mapping.get(original_label, original_label)
            
            # Create a new entity with mapped label
            mapped_entity = entity.copy()
            mapped_entity["label"] = mapped_label
            mapped_entities.append(mapped_entity)
        
        return mapped_entities

# Global instance
ner_processor = SimpleNERProcessor()
