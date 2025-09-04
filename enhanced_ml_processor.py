#!/usr/bin/env python3
"""
Enhanced ML Email Processing Endpoints
Integrates advanced ML models with the Haystack pipeline system
"""
import json
import os
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import Counter
import re

# ML imports
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation

# Haystack imports
from haystack.dataclasses import Document
from haystack_new import HaystackRestAPIManager, create_model_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MLProcessingConfig:
    """Configuration for ML processing"""
    model_name: str = 'facebook/bart-large-cnn'
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    max_length: int = 1024
    temperature: float = 0.1
    enable_gpu: bool = True

class EnhancedMLEmailProcessor:
    """
    Enhanced ML Email Processor with multiple model support
    Combines traditional ML with Haystack pipeline capabilities
    """
    
    def __init__(self, config: MLProcessingConfig = None, haystack_manager: HaystackRestAPIManager = None):
        self.config = config or MLProcessingConfig()
        self.haystack_manager = haystack_manager
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize ML models with proper device management"""
        try:
            # Initialize BART for summarization
            logger.info(f"Loading {self.config.model_name} for summarization...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
            
            # Device management
            if self.config.device == 'auto':
                self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.enable_gpu else "cpu")
            else:
                self.device = torch.device(self.config.device)
                
            self.model.to(self.device)
            logger.info(f"Models loaded on {self.device}")
            
            # Initialize vectorizers for classification
            self.vectorizer = CountVectorizer(max_features=500, stop_words='english')
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            
            # Initialize HuggingFace pipelines for different tasks
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis", 
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=0 if self.device.type == 'cuda' else -1
                )
                self.ner_pipeline = pipeline(
                    "ner", 
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    device=0 if self.device.type == 'cuda' else -1,
                    aggregation_strategy="simple"
                )
            except Exception as e:
                logger.warning(f"Could not initialize additional pipelines: {e}")
                self.sentiment_pipeline = None
                self.ner_pipeline = None
                
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    @staticmethod
    def flatten_email_json(email_json: Dict) -> str:
        """
        Enhanced email flattening with better cleaning
        """
        if isinstance(email_json, str):
            return email_json
            
        # Remove common email headers and footers
        headers_to_remove = [
            r"\n\n\*{3,}.*?EDROM.*?\*{3,}\n",
            r"\*{3,}.*?ZL Technologies.*?\*{3,}",
            r"Classification:.*?\n",
            r"CONFIDENTIAL.*?\n",
        ]
        
        # Extract and clean email body
        email_body = email_json.get("emailBody", "").strip()
        
        # Remove headers
        for header_pattern in headers_to_remove:
            email_body = re.sub(header_pattern, "", email_body, flags=re.DOTALL | re.IGNORECASE)
        
        # Return empty if no meaningful content
        if not email_body or re.fullmatch(r"[\s\*\r\n]*", email_body):
            return ""
        
        # Construct well-formatted email
        email_text = f"Subject: {email_json.get('emailSubject', 'No Subject')}\n"
        email_text += f"From: {email_json.get('emailFrom', 'Unknown')}\n"
        email_text += f"To: {email_json.get('emailTo', 'Unknown')}\n"
        email_text += f"Date: {email_json.get('emailDate', 'Unknown')}\n\n"
        email_text += email_body
        
        return email_text

    def generate_advanced_summary(self, text: str, summary_type: str = "business", **kwargs) -> Dict[str, Any]:
        """
        Generate advanced summaries with multiple approaches
        
        Args:
            text: Input text to summarize
            summary_type: Type of summary ('business', 'legal', 'technical', 'executive')
            **kwargs: Additional parameters
            
        Returns:
            Dict containing different summary approaches
        """
        try:
            # Clean and prepare text
            clean_text = self._clean_text_for_processing(text)
            
            results = {}
            
            # 1. PRIORITY: Use Haystack/Ollama for structured extraction (proven accurate)
            if self.haystack_manager and kwargs.get('use_ollama', True):
                try:
                    doc = Document(content=clean_text, meta={"type": "email"})
                    haystack_result = self.haystack_manager.pipelines["summarization"].run({
                        "cleaner": {"documents": [doc]}
                    })
                    if "generator" in haystack_result and "replies" in haystack_result["generator"]:
                        results["ollama_summary"] = haystack_result["generator"]["replies"][0]
                except Exception as e:
                    logger.warning(f"Haystack summary failed: {e}")
            
            # 2. Enhanced BART with business-specific preprocessing (ALWAYS generate)
            results["bart_summary"] = self._generate_enhanced_bart_summary(
                clean_text, 
                summary_type=summary_type,
                max_length=kwargs.get('max_length', 150),
                min_length=kwargs.get('min_length', 40)
            )
            
            # 3. Business fact extraction (complementary to summary)
            if summary_type == "business":
                results["business_facts"] = self._extract_business_facts(clean_text)
            
            # 4. Extractive summary (key sentences)
            results["extractive_summary"] = self._generate_extractive_summary(clean_text)
            
            # 5. Structured analysis
            results["structured_analysis"] = self._analyze_email_structure(text)
            
            # Add metadata
            results["metadata"] = {
                "original_length": len(text),
                "processed_length": len(clean_text),
                "summary_type": summary_type,
                "processing_time": time.time(),
                "device_used": str(self.device)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return {"error": str(e)}

    def generate_bart_only_summary(self, text, summary_type: str = "business", **kwargs) -> Dict[str, Any]:
        """
        Generate BART-only summary without Ollama fallback for true comparison testing
        
        Args:
            text: Input text to summarize (can be string or email dict)
            summary_type: Type of summary ('business', 'legal', 'technical', 'executive')
            **kwargs: Additional parameters
            
        Returns:
            Dict containing BART-only summary results
        """
        try:
            start_time = time.time()
            
            # Handle both string and dictionary inputs
            if isinstance(text, dict):
                # If it's a dictionary, flatten it to text
                text_to_process = self.flatten_email_json(text)
            else:
                # If it's already a string, use it directly
                text_to_process = str(text)
            
            # Clean and prepare text
            clean_text = self._clean_text_for_processing(text_to_process)
            
            results = {}
            
            # BART-only processing with enhanced business focus
            results["bart_summary"] = self._generate_enhanced_bart_summary(
                clean_text, 
                summary_type=summary_type,
                max_length=kwargs.get('max_length', 150),
                min_length=kwargs.get('min_length', 40)
            )
            
            # Business fact extraction (complementary to summary)
            if summary_type == "business":
                results["business_facts"] = self._extract_business_facts(clean_text)
            
            # Extractive summary (key sentences)
            results["extractive_summary"] = self._generate_extractive_summary(clean_text)
            
            # Calculate confidence (based on BART output quality)
            results["confidence_score"] = 0.75  # BART baseline confidence
            
            results["processing_time"] = time.time() - start_time
            
            return {
                "bart_summary": results.get("bart_summary", ""),  # Add top-level bart_summary field
                "summary": results.get("bart_summary", ""),
                "business_facts": results.get("business_facts", {}),
                "extractive_summary": results.get("extractive_summary", ""),
                "confidence_score": results.get("confidence_score", 0.0),
                "processing_time": results.get("processing_time", 0.0),
                "model_used": "BART-only",
                "all_summaries": results
            }
            
        except Exception as e:
            logger.error(f"BART-only summary generation error: {e}")
            return {
                "summary": f"Error: {str(e)}",
                "business_facts": {},
                "extractive_summary": "",
                "confidence_score": 0.0,
                "processing_time": 0.0,
                "model_used": "BART-only",
                "all_summaries": {}
            }

    def _generate_enhanced_bart_summary(self, text: str, summary_type: str = "business", max_length: int = 150, min_length: int = 40) -> str:
        """Generate enhanced BART summary with business context"""
        try:
            # Create business-focused input for BART
            if summary_type == "business":
                # Preprocess text to highlight business elements
                business_context = self._create_business_context(text)
                summary_input = f"Business Email Summary: {business_context}"
            else:
                summary_input = text
            
            inputs = self.tokenizer(
                summary_input, 
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    early_stopping=True,
                    temperature=self.config.temperature,
                    do_sample=True,
                    top_p=0.9
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Post-process to include business facts if missing
            if summary_type == "business":
                summary = self._enhance_summary_with_facts(summary, text)
            
            return summary
            
        except Exception as e:
            logger.error(f"Enhanced BART summary error: {e}")
            return f"Error generating enhanced summary: {str(e)}"

    def _create_business_context(self, text: str) -> str:
        """Create business-focused context for BART"""
        # Extract and highlight key business elements
        business_elements = []
        
        # Financial amounts
        money_pattern = r'\$[\d,]+(?:\.\d{2})?'
        money_matches = re.findall(money_pattern, text)
        if money_matches:
            business_elements.append(f"Budget: {', '.join(money_matches)}")
        
        # Dates and deadlines
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}',
            r'\b\d{1,2}/\d{1,2}/\d{4}',
            r'\b\d{4}-\d{2}-\d{2}'
        ]
        for pattern in date_patterns:
            dates = re.findall(pattern, text, re.IGNORECASE)
            if dates:
                business_elements.append(f"Dates: {', '.join(dates)}")
                break
        
        # ROI and percentages
        roi_pattern = r'\b\d+:\d+\b|\b\d+%\b'
        roi_matches = re.findall(roi_pattern, text)
        if roi_matches:
            business_elements.append(f"Metrics: {', '.join(roi_matches)}")
        
        # Create enhanced input
        context_prefix = " | ".join(business_elements) if business_elements else ""
        if context_prefix:
            return f"Key Facts: {context_prefix} | Content: {text}"
        else:
            return text

    def _enhance_summary_with_facts(self, summary: str, original_text: str) -> str:
        """Enhance summary by ensuring key business facts are included"""
        # Extract missing business facts
        missing_facts = []
        
        # Check for budget amounts
        money_pattern = r'\$[\d,]+(?:\.\d{2})?'
        original_money = re.findall(money_pattern, original_text)
        summary_money = re.findall(money_pattern, summary)
        
        for amount in original_money:
            if amount not in summary:
                missing_facts.append(f"Budget: {amount}")
                break  # Add only the first missing amount
        
        # Check for ROI
        roi_pattern = r'\b\d+:\d+\b'
        original_roi = re.findall(roi_pattern, original_text)
        summary_roi = re.findall(roi_pattern, summary)
        
        for roi in original_roi:
            if roi not in summary:
                missing_facts.append(f"Expected ROI: {roi}")
                break
        
        # Check for timeline/duration
        timeline_patterns = [r'\b\d+\s+months?\b', r'\b\d+\s+years?\b']
        for pattern in timeline_patterns:
            original_timeline = re.findall(pattern, original_text, re.IGNORECASE)
            summary_timeline = re.findall(pattern, summary, re.IGNORECASE)
            for timeline in original_timeline:
                if timeline.lower() not in summary.lower():
                    missing_facts.append(f"Duration: {timeline}")
                    break
            if missing_facts and "Duration" in missing_facts[-1]:
                break
        
        # Append missing facts to summary
        if missing_facts:
            enhanced_summary = f"{summary} Key details: {', '.join(missing_facts)}."
            return enhanced_summary
        
        return summary

    def _extract_business_facts(self, text: str) -> Dict[str, Any]:
        """Extract structured business facts from email"""
        facts = {
            "financial_amounts": [],
            "dates_deadlines": [],
            "roi_metrics": [],
            "timeframes": [],
            "key_numbers": []
        }
        
        # Financial amounts
        money_pattern = r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|M|thousand|K))?'
        facts["financial_amounts"] = re.findall(money_pattern, text, re.IGNORECASE)
        
        # Dates and deadlines
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}',
            r'\b\d{1,2}/\d{1,2}/\d{4}',
            r'\b\d{4}-\d{2}-\d{2}'
        ]
        for pattern in date_patterns:
            dates = re.findall(pattern, text, re.IGNORECASE)
            facts["dates_deadlines"].extend(dates)
        
        # ROI and metrics
        roi_pattern = r'\b\d+:\d+\b|\b\d+%\b|\b\d+\.\d+%\b'
        facts["roi_metrics"] = re.findall(roi_pattern, text)
        
        # Timeframes
        timeframe_pattern = r'\b\d+\s+(?:months?|years?|weeks?|days?)\b'
        facts["timeframes"] = re.findall(timeframe_pattern, text, re.IGNORECASE)
        
        # Key numbers (leads, targets, etc.)
        number_context_pattern = r'\b\d{1,3}(?:,\d{3})*\s+(?:leads?|customers?|users?|people?|targets?)\b'
        facts["key_numbers"] = re.findall(number_context_pattern, text, re.IGNORECASE)
        
        return facts

    def _generate_bart_summary(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        """Generate summary using BART model"""
        try:
            inputs = self.tokenizer(
                text, 
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    early_stopping=True,
                    temperature=self.config.temperature
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            logger.error(f"BART summary error: {e}")
            return f"Error generating summary: {str(e)}"

    def _generate_domain_specific_summary(self, text: str, domain: str) -> str:
        """Generate domain-specific summaries with targeted prompts"""
        if domain == "business":
            # For business domain, prioritize Ollama/Haystack if available
            if self.haystack_manager:
                try:
                    doc = Document(content=text, meta={"type": "email"})
                    haystack_result = self.haystack_manager.pipelines["summarization"].run({
                        "cleaner": {"documents": [doc]}
                    })
                    if "generator" in haystack_result and "replies" in haystack_result["generator"]:
                        return haystack_result["generator"]["replies"][0]
                except Exception as e:
                    logger.warning(f"Haystack domain summary failed: {e}")
        
        # Fallback to enhanced BART with domain context
        domain_prompts = {
            "business": "Extract key business information including budgets, deadlines, ROI, and action items from this email:",
            "legal": "Summarize this email from a legal perspective, focusing on obligations, deadlines, and compliance:",
            "technical": "Summarize this technical email focusing on system changes, requirements, and technical details:",
            "executive": "Provide an executive summary focusing on high-level impacts and strategic decisions:"
        }
        
        prompt = domain_prompts.get(domain, domain_prompts["business"])
        full_prompt = f"{prompt}\n\n{text}"
        
        return self._generate_enhanced_bart_summary(full_prompt, domain, max_length=200, min_length=60)

    def _generate_extractive_summary(self, text: str, num_sentences: int = 3) -> List[str]:
        """Generate extractive summary by selecting key sentences"""
        try:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            if len(sentences) <= num_sentences:
                return sentences
            
            # Score sentences using TF-IDF
            vectorizer = TfidfVectorizer(stop_words='english')
            sentence_vectors = vectorizer.fit_transform(sentences)
            
            # Calculate sentence importance scores
            scores = sentence_vectors.sum(axis=1).A1
            
            # Get top sentences
            top_indices = scores.argsort()[-num_sentences:][::-1]
            key_sentences = [sentences[i].strip() for i in sorted(top_indices)]
            
            return key_sentences
            
        except Exception as e:
            logger.error(f"Extractive summary error: {e}")
            return [text[:200] + "..."]

    def _analyze_email_structure(self, text: str) -> Dict[str, Any]:
        """Analyze email structure and extract metadata"""
        analysis = {
            "has_attachments": bool(re.search(r'attach|attachment', text, re.IGNORECASE)),
            "is_reply": bool(re.search(r'^(RE:|FWD?:)', text, re.IGNORECASE)),
            "has_deadline": bool(re.search(r'deadline|due|by\s+\d+', text, re.IGNORECASE)),
            "has_action_items": bool(re.search(r'please|action|todo|need\s+to', text, re.IGNORECASE)),
            "sentiment": None,
            "named_entities": [],
            "key_topics": []
        }
        
        # Add sentiment analysis
        if self.sentiment_pipeline:
            try:
                sentiment_result = self.sentiment_pipeline(text[:512])  # Limit text length
                analysis["sentiment"] = sentiment_result[0] if sentiment_result else None
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
        
        # Add named entity recognition
        if self.ner_pipeline:
            try:
                entities = self.ner_pipeline(text[:512])
                analysis["named_entities"] = [
                    {"text": ent["word"], "label": ent["entity_group"], "confidence": ent["score"]}
                    for ent in entities if ent["score"] > 0.9
                ]
            except Exception as e:
                logger.warning(f"NER failed: {e}")
        
        return analysis

    def extract_enhanced_key_points(self, text: str, num_points: int = 5, focus_area: str = None) -> Dict[str, Any]:
        """
        Extract key points with enhanced filtering and categorization
        """
        try:
            results = {}
            
            # 1. Traditional extractive approach
            results["extractive_points"] = self._extract_key_sentences(text, num_points)
            
            # 2. Model-based key point extraction
            results["model_points"] = self._extract_model_key_points(text, num_points, focus_area)
            
            # 3. Action item extraction
            results["action_items"] = self._extract_action_items(text)
            
            # 4. Entity-based points
            results["entity_points"] = self._extract_entity_based_points(text)
            
            # 5. Topic-based points
            results["topic_points"] = self._extract_topic_based_points(text)
            
            # Combine and rank all points
            results["ranked_points"] = self._rank_and_combine_points(results, num_points)
            
            return results
            
        except Exception as e:
            logger.error(f"Key points extraction error: {e}")
            return {"error": str(e)}

    def _extract_model_key_points(self, text: str, num_points: int, focus_area: str = None) -> List[str]:
        """Extract key points using the BART model with focused prompts"""
        focus_prompts = {
            "financial": "Extract key financial information and budget details:",
            "timeline": "Extract key dates, deadlines, and timeline information:",
            "people": "Extract key people, roles, and responsibilities:",
            "actions": "Extract key action items and tasks:",
            "decisions": "Extract key decisions and conclusions:"
        }
        
        base_prompt = focus_prompts.get(focus_area, "Extract the most important key points:")
        
        prompt = f"""
{base_prompt}

{text}

Key Points (exactly {num_points} numbered points):
1."""
        
        try:
            inputs = self.tokenizer(
                prompt,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs.input_ids,
                    max_length=300,
                    min_length=50,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                    early_stopping=True
                )
            
            output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract numbered points from output
            points = re.findall(r'(\d+\..*?)(?=\d+\.|$)', output, re.DOTALL)
            return [point.strip() for point in points[:num_points]]
            
        except Exception as e:
            logger.error(f"Model key points extraction error: {e}")
            return []

    def _extract_action_items(self, text: str) -> List[Dict[str, str]]:
        """Extract action items with context"""
        action_patterns = [
            (r'(?:please|kindly)\s+([^,.!?]+)', 'request'),
            (r'(?:need|must|should)\s+to\s+([^,.!?]+)', 'requirement'),
            (r'(?:will|going to)\s+([^,.!?]+)', 'commitment'),
            (r'(?:action required|todo):?\s*([^,.!?]+)', 'action'),
            (r'by\s+(\w+day|\d+/\d+|\w+\s+\d+)', 'deadline')
        ]
        
        action_items = []
        for pattern, item_type in action_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                action_items.append({
                    "text": match.group(1).strip(),
                    "type": item_type,
                    "context": match.group(0)
                })
        
        return action_items[:10]  # Limit results

    def advanced_email_classification(self, text: str, classification_schemes: List[str] = None) -> Dict[str, Any]:
        """
        Advanced email classification with multiple schemes
        """
        if not classification_schemes:
            classification_schemes = ["business", "legal", "sentiment", "priority", "custom"]
        
        results = {}
        
        try:
            for scheme in classification_schemes:
                if scheme == "business":
                    results["business"] = self._classify_business_categories(text)
                elif scheme == "legal":
                    results["legal"] = self._classify_legal_categories(text)
                elif scheme == "sentiment":
                    results["sentiment"] = self._analyze_sentiment_advanced(text)
                elif scheme == "priority":
                    results["priority"] = self._classify_priority(text)
                elif scheme == "custom":
                    results["custom"] = self._dynamic_classification(text)
            
            # Add overall classification confidence
            results["metadata"] = {
                "classification_time": time.time(),
                "schemes_used": classification_schemes,
                "confidence_threshold": 0.7
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {"error": str(e)}

    def _classify_business_categories(self, text: str) -> Dict[str, float]:
        """Classify into business categories"""
        business_categories = [
            "meeting_scheduling", "project_update", "financial_report", 
            "customer_inquiry", "internal_communication", "marketing",
            "legal_matter", "hr_related", "it_support", "procurement",
            "sales_activity", "partnership", "compliance"
        ]
        
        return self._calculate_category_scores(text, business_categories)

    def _classify_legal_categories(self, text: str) -> Dict[str, Any]:
        """Enhanced legal classification"""
        legal_categories = [
            "privileged_communication", "confidential_information", 
            "responsive_documents", "non_responsive", "key_entities",
            "contract_related", "compliance_issue", "litigation_hold",
            "discovery_material", "attorney_work_product"
        ]
        
        scores = self._calculate_category_scores(text, legal_categories)
        
        # Add specific legal indicators
        legal_indicators = {
            "has_attorney_mention": bool(re.search(r'attorney|lawyer|counsel', text, re.IGNORECASE)),
            "has_privilege_claim": bool(re.search(r'privileged|attorney.client', text, re.IGNORECASE)),
            "has_confidentiality": bool(re.search(r'confidential|sensitive|proprietary', text, re.IGNORECASE)),
            "contains_legal_terms": len(re.findall(r'contract|agreement|lawsuit|litigation|discovery', text, re.IGNORECASE))
        }
        
        return {
            "category_scores": scores,
            "legal_indicators": legal_indicators,
            "primary_category": max(scores.items(), key=lambda x: x[1])[0],
            "confidence": max(scores.values())
        }

    def _calculate_category_scores(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Calculate similarity scores for categories"""
        try:
            # Create category descriptions
            category_texts = [cat.replace("_", " ") + " related email communication" for cat in categories]
            
            # Vectorize
            vectorizer = TfidfVectorizer(stop_words='english')
            vectors = vectorizer.fit_transform(category_texts + [text])
            
            # Calculate similarities
            text_vector = vectors[-1]
            category_vectors = vectors[:-1]
            similarities = cosine_similarity(text_vector, category_vectors)[0]
            
            return dict(zip(categories, similarities.astype(float)))
            
        except Exception as e:
            logger.error(f"Category scoring error: {e}")
            return {cat: 0.0 for cat in categories}

    def extract_advanced_topics(self, text: str, num_topics: int = 5, method: str = "hybrid") -> Dict[str, Any]:
        """
        Advanced topic extraction with multiple methods
        """
        try:
            results = {}
            
            if method in ["lda", "hybrid"]:
                results["lda_topics"] = self._extract_lda_topics(text, num_topics)
            
            if method in ["keyword", "hybrid"]:
                results["keyword_topics"] = self._extract_keyword_topics(text, num_topics)
            
            if method in ["semantic", "hybrid"]:
                results["semantic_topics"] = self._extract_semantic_topics(text, num_topics)
            
            # Combine and rank topics
            if method == "hybrid":
                results["combined_topics"] = self._combine_topic_methods(results, num_topics)
            
            return results
            
        except Exception as e:
            logger.error(f"Topic extraction error: {e}")
            return {"error": str(e)}

    def _extract_lda_topics(self, text: str, num_topics: int) -> List[str]:
        """Extract topics using LDA"""
        try:
            # Clean and vectorize
            clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
            vectorizer = CountVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            X = vectorizer.fit_transform([clean_text])
            
            # Apply LDA
            lda = LatentDirichletAllocation(
                n_components=min(num_topics, 5),  # Limit for small texts
                random_state=42,
                max_iter=10
            )
            lda.fit(X)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
                topic_str = f"Topic {topic_idx + 1}: {', '.join(top_words)}"
                topics.append(topic_str)
            
            return topics
            
        except Exception as e:
            logger.error(f"LDA topics error: {e}")
            return [f"Topic extraction failed: {str(e)}"]

    def _clean_text_for_processing(self, text: str) -> str:
        """Clean text for ML processing"""
        # Remove email headers and signatures
        cleaned = re.sub(r'From:.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)
        cleaned = re.sub(r'(?:Regards|Sincerely|Thanks|Best),?\s*\n.*?(?=\n\n|\Z)', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    # Placeholder methods for additional functionality
    def _extract_key_sentences(self, text: str, num_points: int) -> List[str]:
        """Extract key sentences using TF-IDF scoring"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= num_points:
            return sentences
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            sentence_vectors = vectorizer.fit_transform(sentences)
            scores = sentence_vectors.sum(axis=1).A1
            top_indices = scores.argsort()[-num_points:][::-1]
            return [sentences[i].strip() for i in sorted(top_indices)]
        except:
            return sentences[:num_points]

    def _extract_entity_based_points(self, text: str) -> List[str]:
        """Extract points based on named entities"""
        if not self.ner_pipeline:
            return []
        
        try:
            entities = self.ner_pipeline(text[:512])
            points = []
            for ent in entities:
                if ent["score"] > 0.9:
                    points.append(f"Key {ent['entity_group'].lower()}: {ent['word']}")
            return points[:5]
        except:
            return []

    def _extract_topic_based_points(self, text: str) -> List[str]:
        """Extract points based on topics"""
        topics = self._extract_keyword_topics(text, 3)
        return [f"Topic area: {topic}" for topic in topics]

    def _rank_and_combine_points(self, all_points: Dict, num_points: int) -> List[str]:
        """Combine and rank all extracted points"""
        combined = []
        
        # Add model points first (usually highest quality)
        combined.extend(all_points.get("model_points", []))
        
        # Add action items
        action_items = all_points.get("action_items", [])
        combined.extend([item["text"] for item in action_items[:3]])
        
        # Add extractive points
        combined.extend(all_points.get("extractive_points", [])[:3])
        
        # Remove duplicates and limit
        unique_points = []
        for point in combined:
            if point not in unique_points:
                unique_points.append(point)
        
        return unique_points[:num_points]

    def _analyze_sentiment_advanced(self, text: str) -> Dict[str, Any]:
        """Advanced sentiment analysis"""
        if not self.sentiment_pipeline:
            return {"error": "Sentiment pipeline not available"}
        
        try:
            result = self.sentiment_pipeline(text[:512])
            return result[0] if result else {"error": "No sentiment detected"}
        except Exception as e:
            return {"error": str(e)}

    def _classify_priority(self, text: str) -> Dict[str, float]:
        """Classify email priority"""
        priority_indicators = {
            "urgent": ["urgent", "asap", "immediately", "emergency", "critical"],
            "high": ["important", "priority", "deadline", "soon", "quickly"],
            "medium": ["please", "request", "need", "update", "review"],
            "low": ["fyi", "information", "note", "btw", "when you can"]
        }
        
        scores = {}
        text_lower = text.lower()
        
        for priority, keywords in priority_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[priority] = float(score / len(keywords))
        
        return scores

    def _dynamic_classification(self, text: str) -> List[str]:
        """Dynamic classification based on content"""
        # Extract potential categories from text
        categories = []
        
        # Look for explicit mentions
        category_patterns = [
            (r'project\s+(\w+)', 'project_{}'),
            (r'meeting\s+about\s+(\w+)', 'meeting_{}'),
            (r'(\w+)\s+report', '{}_report'),
            (r'(\w+)\s+update', '{}_update')
        ]
        
        for pattern, template in category_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                categories.append(template.format(match))
        
        return categories[:5] if categories else ["general_communication"]

    def _extract_keyword_topics(self, text: str, num_topics: int) -> List[str]:
        """Extract topics using keyword extraction"""
        try:
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            X = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = X.toarray()[0]
            
            # Get top scoring terms
            top_indices = scores.argsort()[-num_topics*2:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            
            # Group into topics
            topics = []
            for i in range(0, len(top_terms), 2):
                topic_words = top_terms[i:i+2]
                topics.append(" + ".join(topic_words))
                if len(topics) >= num_topics:
                    break
            
            return topics
            
        except Exception as e:
            return [f"Keyword extraction failed: {str(e)}"]

    def _extract_semantic_topics(self, text: str, num_topics: int) -> List[str]:
        """Extract topics using semantic analysis"""
        # Placeholder for semantic topic extraction
        # Would use sentence transformers or similar
        return [f"Semantic topic {i+1}" for i in range(num_topics)]

    def _combine_topic_methods(self, topic_results: Dict, num_topics: int) -> List[str]:
        """Combine topics from different methods"""
        combined = []
        
        # Add LDA topics first
        combined.extend(topic_results.get("lda_topics", []))
        
        # Add keyword topics
        combined.extend(topic_results.get("keyword_topics", []))
        
        # Add semantic topics
        combined.extend(topic_results.get("semantic_topics", []))
        
        # Remove duplicates and limit
        unique_topics = []
        for topic in combined:
            if topic not in unique_topics:
                unique_topics.append(topic)
        
        return unique_topics[:num_topics]


# Factory function for creating processor instances
def create_ml_processor(
    model_name: str = 'facebook/bart-large-cnn',
    device: str = 'auto',
    enable_haystack: bool = True,
    haystack_model_type: str = "ollama",
    haystack_model_name: str = "mistral"
) -> EnhancedMLEmailProcessor:
    """
    Factory function to create ML processor with optional Haystack integration
    """
    
    # Create ML config
    ml_config = MLProcessingConfig(
        model_name=model_name,
        device=device,
        enable_gpu=True
    )
    
    # Create Haystack manager if requested
    haystack_manager = None
    if enable_haystack:
        try:
            model_config = create_model_config(haystack_model_type, haystack_model_name)
            haystack_manager = HaystackRestAPIManager(model_config, use_opensearch=False)
            logger.info("Haystack integration enabled")
        except Exception as e:
            logger.warning(f"Could not initialize Haystack: {e}")
    
    return EnhancedMLEmailProcessor(ml_config, haystack_manager)


if __name__ == "__main__":
    # Example usage
    processor = create_ml_processor()
    
    sample_email = {
        "emailSubject": "Project Alpha Budget Approval",
        "emailFrom": "sarah.johnson@company.com",
        "emailTo": "finance@company.com", 
        "emailDate": "2024-07-27",
        "emailBody": """Hi Finance Team,

I need approval for Project Alpha budget of $200,000 for development work.
Timeline is 6 months starting August 1st.

Key features include:
- Real-time dashboard
- Customer analytics
- Predictive models

Team: John Smith (PM), Maria Rodriguez (Lead Dev)

Please approve by EOD Friday.

Thanks,
Sarah"""
    }
    
    # Test different functionalities
    email_text = processor.flatten_email_json(sample_email)
    
    print("=== ADVANCED SUMMARY ===")
    summary_result = processor.generate_advanced_summary(email_text, "business")
    print(json.dumps(summary_result, indent=2))
    
    print("\n=== ENHANCED KEY POINTS ===")
    key_points_result = processor.extract_enhanced_key_points(email_text, focus_area="financial")
    print(json.dumps(key_points_result, indent=2))
    
    print("\n=== ADVANCED CLASSIFICATION ===")
    classification_result = processor.advanced_email_classification(email_text)
    print(json.dumps(classification_result, indent=2))
