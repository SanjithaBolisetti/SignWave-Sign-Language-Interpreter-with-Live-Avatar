"""
NLP Processing Module
Natural Language Processing for sign language grammar and syntax
"""

import re
import nltk
from typing import List, Dict, Optional, Tuple
from loguru import logger
import json

class SignLanguageNLP:
    """Handles NLP processing for sign language grammar and syntax"""
    
    def __init__(self):
        self.sign_language_rules = self._load_sign_language_rules()
        self.word_to_sign_map = self._load_word_to_sign_mapping()
        self.grammar_rules = self._load_grammar_rules()
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('pos_tags', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            logger.warning(f"NLTK download warning: {e}")
        
        logger.info("SignLanguageNLP initialized")
    
    def _load_sign_language_rules(self) -> Dict:
        """Load sign language grammar rules"""
        return {
            "ASL": {
                "syntax": "SOV",  # Subject-Object-Verb
                "question_markers": ["eyebrow_raise", "head_tilt"],
                "negation": ["head_shake", "not_sign"],
                "time_markers": ["time_first", "time_last"],
                "spatial_references": ["left", "right", "center"],
                "facial_expressions": {
                    "question": "eyebrow_raise",
                    "negation": "head_shake",
                    "emphasis": "eyebrow_furrow"
                }
            },
            "BSL": {
                "syntax": "SVO",  # Subject-Verb-Object
                "question_markers": ["eyebrow_raise", "head_tilt"],
                "negation": ["head_shake", "not_sign"],
                "time_markers": ["time_first", "time_last"],
                "spatial_references": ["left", "right", "center"],
                "facial_expressions": {
                    "question": "eyebrow_raise",
                    "negation": "head_shake",
                    "emphasis": "eyebrow_furrow"
                }
            }
        }
    
    def _load_word_to_sign_mapping(self) -> Dict:
        """Load word to sign mapping"""
        return {
            "hello": "hello_sign",
            "thank you": "thank_you_sign",
            "yes": "yes_sign",
            "no": "no_sign",
            "help": "help_sign",
            "water": "water_sign",
            "food": "food_sign",
            "bathroom": "bathroom_sign",
            "good": "good_sign",
            "bad": "bad_sign",
            "please": "please_sign",
            "sorry": "sorry_sign",
            "love": "love_sign",
            "family": "family_sign",
            "friend": "friend_sign",
            "work": "work_sign",
            "home": "home_sign",
            "school": "school_sign",
            "doctor": "doctor_sign",
            "hospital": "hospital_sign"
        }
    
    def _load_grammar_rules(self) -> Dict:
        """Load grammar transformation rules"""
        return {
            "question_words": {
                "what": "what_sign",
                "where": "where_sign",
                "when": "when_sign",
                "why": "why_sign",
                "how": "how_sign",
                "who": "who_sign"
            },
            "pronouns": {
                "i": "i_sign",
                "you": "you_sign",
                "he": "he_sign",
                "she": "she_sign",
                "we": "we_sign",
                "they": "they_sign",
                "me": "me_sign",
                "him": "him_sign",
                "her": "her_sign",
                "us": "us_sign",
                "them": "them_sign"
            },
            "verbs": {
                "is": "is_sign",
                "are": "are_sign",
                "was": "was_sign",
                "were": "were_sign",
                "have": "have_sign",
                "has": "has_sign",
                "had": "had_sign",
                "do": "do_sign",
                "does": "does_sign",
                "did": "did_sign",
                "will": "will_sign",
                "can": "can_sign",
                "could": "could_sign",
                "should": "should_sign",
                "would": "would_sign"
            },
            "adjectives": {
                "good": "good_sign",
                "bad": "bad_sign",
                "big": "big_sign",
                "small": "small_sign",
                "hot": "hot_sign",
                "cold": "cold_sign",
                "fast": "fast_sign",
                "slow": "slow_sign"
            }
        }
    
    def process_text_for_sign_language(self, text: str, target_language: str = "ASL") -> Dict:
        """Process text for sign language translation"""
        try:
            logger.info(f"Processing text for {target_language}: {text}")
            
            # Clean and normalize text
            cleaned_text = self._clean_text(text)
            
            # Tokenize text
            tokens = self._tokenize_text(cleaned_text)
            
            # Analyze sentence structure
            sentence_analysis = self._analyze_sentence_structure(tokens)
            
            # Apply sign language grammar rules
            sign_sequence = self._apply_sign_language_grammar(
                sentence_analysis, target_language
            )
            
            # Add facial expressions and non-manual markers
            enhanced_sequence = self._add_non_manual_markers(sign_sequence, target_language)
            
            return {
                "original_text": text,
                "cleaned_text": cleaned_text,
                "tokens": tokens,
                "sentence_analysis": sentence_analysis,
                "sign_sequence": enhanced_sequence,
                "target_language": target_language
            }
            
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            return {"error": str(e)}
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation except for question marks
        text = re.sub(r'[^\w\s?]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words"""
        try:
            # Use NLTK tokenizer
            tokens = nltk.word_tokenize(text)
            return tokens
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            # Fallback to simple split
            return text.split()
    
    def _analyze_sentence_structure(self, tokens: List[str]) -> Dict:
        """Analyze sentence structure"""
        try:
            # Get POS tags
            pos_tags = nltk.pos_tag(tokens)
            
            # Identify sentence type
            sentence_type = self._identify_sentence_type(tokens)
            
            # Extract key components
            components = self._extract_sentence_components(tokens, pos_tags)
            
            return {
                "tokens": tokens,
                "pos_tags": pos_tags,
                "sentence_type": sentence_type,
                "components": components
            }
            
        except Exception as e:
            logger.error(f"Sentence analysis error: {e}")
            return {"tokens": tokens, "error": str(e)}
    
    def _identify_sentence_type(self, tokens: List[str]) -> str:
        """Identify sentence type (statement, question, command)"""
        if tokens and tokens[-1] == '?':
            return "question"
        elif tokens and tokens[0] in ['please', 'can', 'could', 'would']:
            return "command"
        else:
            return "statement"
    
    def _extract_sentence_components(self, tokens: List[str], pos_tags: List[Tuple]) -> Dict:
        """Extract sentence components"""
        components = {
            "subject": [],
            "verb": [],
            "object": [],
            "question_words": [],
            "pronouns": [],
            "adjectives": [],
            "adverbs": []
        }
        
        for token, pos in pos_tags:
            if pos.startswith('NN'):  # Nouns
                if not components["subject"]:
                    components["subject"].append(token)
                else:
                    components["object"].append(token)
            elif pos.startswith('VB'):  # Verbs
                components["verb"].append(token)
            elif pos.startswith('PRP'):  # Pronouns
                components["pronouns"].append(token)
            elif pos.startswith('JJ'):  # Adjectives
                components["adjectives"].append(token)
            elif pos.startswith('RB'):  # Adverbs
                components["adverbs"].append(token)
            elif token in self.grammar_rules["question_words"]:
                components["question_words"].append(token)
        
        return components
    
    def _apply_sign_language_grammar(self, sentence_analysis: Dict, target_language: str) -> List[Dict]:
        """Apply sign language grammar rules"""
        try:
            sign_sequence = []
            rules = self.sign_language_rules[target_language]
            
            # Handle question markers first
            if sentence_analysis["sentence_type"] == "question":
                question_marker = {
                    "type": "non_manual",
                    "marker": "eyebrow_raise",
                    "duration": 0.5,
                    "timing": "beginning"
                }
                sign_sequence.append(question_marker)
            
            # Apply syntax rules (SOV for ASL, SVO for BSL)
            if target_language == "ASL":
                # Subject-Object-Verb order
                sign_sequence.extend(self._process_subject(sentence_analysis["components"]["subject"]))
                sign_sequence.extend(self._process_object(sentence_analysis["components"]["object"]))
                sign_sequence.extend(self._process_verb(sentence_analysis["components"]["verb"]))
            else:
                # Subject-Verb-Object order
                sign_sequence.extend(self._process_subject(sentence_analysis["components"]["subject"]))
                sign_sequence.extend(self._process_verb(sentence_analysis["components"]["verb"]))
                sign_sequence.extend(self._process_object(sentence_analysis["components"]["object"]))
            
            # Add question words
            for q_word in sentence_analysis["components"]["question_words"]:
                sign_sequence.append(self._word_to_sign(q_word))
            
            # Add pronouns
            for pronoun in sentence_analysis["components"]["pronouns"]:
                sign_sequence.append(self._word_to_sign(pronoun))
            
            # Add adjectives
            for adj in sentence_analysis["components"]["adjectives"]:
                sign_sequence.append(self._word_to_sign(adj))
            
            return sign_sequence
            
        except Exception as e:
            logger.error(f"Grammar application error: {e}")
            return []
    
    def _process_subject(self, subject_tokens: List[str]) -> List[Dict]:
        """Process subject tokens"""
        signs = []
        for token in subject_tokens:
            signs.append(self._word_to_sign(token))
        return signs
    
    def _process_verb(self, verb_tokens: List[str]) -> List[Dict]:
        """Process verb tokens"""
        signs = []
        for token in verb_tokens:
            signs.append(self._word_to_sign(token))
        return signs
    
    def _process_object(self, object_tokens: List[str]) -> List[Dict]:
        """Process object tokens"""
        signs = []
        for token in object_tokens:
            signs.append(self._word_to_sign(token))
        return signs
    
    def _word_to_sign(self, word: str) -> Dict:
        """Convert word to sign representation"""
        # Check direct mapping first
        if word in self.word_to_sign_map:
            return {
                "type": "sign",
                "word": word,
                "sign": self.word_to_sign_map[word],
                "duration": 1.0,
                "handshape": self._get_handshape_for_sign(self.word_to_sign_map[word]),
                "location": self._get_location_for_sign(self.word_to_sign_map[word]),
                "movement": self._get_movement_for_sign(self.word_to_sign_map[word])
            }
        
        # Check grammar rules
        for category, mappings in self.grammar_rules.items():
            if word in mappings:
                return {
                    "type": "sign",
                    "word": word,
                    "sign": mappings[word],
                    "duration": 1.0,
                    "handshape": self._get_handshape_for_sign(mappings[word]),
                    "location": self._get_location_for_sign(mappings[word]),
                    "movement": self._get_movement_for_sign(mappings[word])
                }
        
        # Fingerspell unknown words
        return {
            "type": "fingerspell",
            "word": word,
            "letters": list(word),
            "duration": len(word) * 0.5
        }
    
    def _get_handshape_for_sign(self, sign: str) -> str:
        """Get handshape for a sign"""
        # This would be loaded from a comprehensive sign database
        handshape_map = {
            "hello_sign": "open_hand",
            "yes_sign": "fist",
            "no_sign": "index_middle_fingers",
            "thank_you_sign": "flat_hand",
            "help_sign": "thumbs_up",
            "water_sign": "w_handshape",
            "food_sign": "flat_hand"
        }
        return handshape_map.get(sign, "neutral")
    
    def _get_location_for_sign(self, sign: str) -> str:
        """Get location for a sign"""
        location_map = {
            "hello_sign": "chest_level",
            "yes_sign": "chest_level",
            "no_sign": "chest_level",
            "thank_you_sign": "mouth_level",
            "help_sign": "chest_level",
            "water_sign": "chin_level",
            "food_sign": "mouth_level"
        }
        return location_map.get(sign, "chest_level")
    
    def _get_movement_for_sign(self, sign: str) -> str:
        """Get movement for a sign"""
        movement_map = {
            "hello_sign": "wave",
            "yes_sign": "nod_up_down",
            "no_sign": "tap_together",
            "thank_you_sign": "forward_out",
            "help_sign": "tap_chest",
            "water_sign": "tap_chin",
            "food_sign": "tap_mouth"
        }
        return movement_map.get(sign, "static")
    
    def _add_non_manual_markers(self, sign_sequence: List[Dict], target_language: str) -> List[Dict]:
        """Add non-manual markers (facial expressions, head movements)"""
        try:
            enhanced_sequence = []
            
            for sign in sign_sequence:
                enhanced_sign = sign.copy()
                
                # Add facial expressions based on context
                if sign.get("type") == "sign":
                    enhanced_sign["facial_expression"] = self._get_facial_expression_for_sign(sign)
                
                # Add head movements
                enhanced_sign["head_movement"] = self._get_head_movement_for_sign(sign)
                
                enhanced_sequence.append(enhanced_sign)
            
            return enhanced_sequence
            
        except Exception as e:
            logger.error(f"Non-manual markers error: {e}")
            return sign_sequence
    
    def _get_facial_expression_for_sign(self, sign: Dict) -> str:
        """Get facial expression for a sign"""
        # This would be more sophisticated in a real implementation
        return "neutral"
    
    def _get_head_movement_for_sign(self, sign: Dict) -> str:
        """Get head movement for a sign"""
        # This would be more sophisticated in a real implementation
        return "neutral"
    
    def optimize_for_real_time(self, sign_sequence: List[Dict]) -> List[Dict]:
        """Optimize sign sequence for real-time performance"""
        try:
            optimized_sequence = []
            
            for sign in sign_sequence:
                optimized_sign = sign.copy()
                
                # Reduce duration for faster communication
                if optimized_sign.get("duration", 1.0) > 1.0:
                    optimized_sign["duration"] = 1.0
                
                # Add timing information
                optimized_sign["start_time"] = sum(s.get("duration", 1.0) for s in optimized_sequence)
                optimized_sign["end_time"] = optimized_sign["start_time"] + optimized_sign["duration"]
                
                optimized_sequence.append(optimized_sign)
            
            return optimized_sequence
            
        except Exception as e:
            logger.error(f"Real-time optimization error: {e}")
            return sign_sequence
