"""
Speech-to-Sign Translation Module
Converts spoken language to sign language gestures
"""

import speech_recognition as sr
import pyttsx3
import numpy as np
from loguru import logger
import threading
import time
from typing import List, Dict, Optional
import json

# Try to import optional dependencies (Whisper ASR)
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception as e:
    # On some platforms (e.g., Windows without libc) whisper can raise non-ImportError
    WHISPER_AVAILABLE = False
    logger.warning(f"Whisper not available ({e}). Will use Google Speech Recognition only.")

class SpeechToSignTranslator:
    """Handles speech recognition and conversion to sign language"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.whisper_model = None
        self.pyaudio_available = False
        
        # Try to initialize microphone (requires PyAudio)
        try:
            self.microphone = sr.Microphone()
            self.pyaudio_available = True
            logger.info("Microphone initialized successfully")
        except OSError as e:
            if "PyAudio" in str(e) or "pyaudio" in str(e).lower():
                logger.warning("PyAudio not available. Microphone input disabled. You can still use text input.")
                self.pyaudio_available = False
            else:
                logger.error(f"Microphone initialization error: {e}")
                self.pyaudio_available = False
        except Exception as e:
            logger.warning(f"Microphone initialization failed: {e}")
            self.pyaudio_available = False
        
        try:
            self.tts_engine = pyttsx3.init()
            # Configure TTS
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
        except Exception as e:
            logger.warning(f"TTS engine initialization failed: {e}")
            self.tts_engine = None
        
        self.sign_dictionary = self._load_sign_dictionary()
        
        # Calibrate microphone if available
        if self.pyaudio_available:
            self._calibrate_microphone()
        
        logger.info("SpeechToSignTranslator initialized")
    
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        if not self.pyaudio_available or not self.microphone:
            return
        
        try:
            with self.microphone as source:
                logger.info("Calibrating microphone...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.info("Microphone calibrated")
        except Exception as e:
            logger.error(f"Microphone calibration failed: {e}")
    
    def _load_sign_dictionary(self) -> Dict[str, Dict]:
        """Load sign language dictionary with gestures"""
        # Basic ASL dictionary - in production, this would be loaded from a database
        return {
            "hello": {
                "handshape": "open_hand",
                "movement": "wave",
                "location": "chest_level",
                "orientation": "palm_out",
                "facial_expression": "smile"
            },
            "thank you": {
                "handshape": "flat_hand",
                "movement": "forward_out",
                "location": "mouth_level",
                "orientation": "palm_up",
                "facial_expression": "neutral"
            },
            "yes": {
                "handshape": "fist",
                "movement": "nod_up_down",
                "location": "chest_level",
                "orientation": "palm_down",
                "facial_expression": "neutral"
            },
            "no": {
                "handshape": "index_middle_fingers",
                "movement": "tap_together",
                "location": "chest_level",
                "orientation": "palm_down",
                "facial_expression": "neutral"
            },
            "help": {
                "handshape": "thumbs_up",
                "movement": "tap_chest",
                "location": "chest_level",
                "orientation": "palm_in",
                "facial_expression": "concerned"
            },
            "water": {
                "handshape": "w_handshape",
                "movement": "tap_chin",
                "location": "chin_level",
                "orientation": "palm_in",
                "facial_expression": "neutral"
            },
            "food": {
                "handshape": "flat_hand",
                "movement": "tap_mouth",
                "location": "mouth_level",
                "orientation": "palm_in",
                "facial_expression": "neutral"
            },
            "bathroom": {
                "handshape": "t_handshape",
                "movement": "shake",
                "location": "chest_level",
                "orientation": "palm_down",
                "facial_expression": "neutral"
            }
        }
    
    def record_and_transcribe(self, timeout: int = 5) -> Optional[str]:
        """Record audio and transcribe to text"""
        if not self.pyaudio_available or not self.microphone:
            logger.error("Microphone not available. PyAudio is required for audio recording.")
            raise RuntimeError(
                "PyAudio is not installed. Audio recording requires PyAudio.\n"
                "Install it with: pip install pyaudio\n"
                "Note: PyAudio requires Visual Studio Build Tools on Windows."
            )
        
        try:
            logger.info("Starting audio recording...")
            
            with self.microphone as source:
                # Listen for audio with timeout
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            logger.info("Audio recorded, transcribing...")
            
            # Try multiple recognition methods
            text = None
            
            # Method 1: Google Speech Recognition
            try:
                text = self.recognizer.recognize_google(audio)
                logger.info(f"Google recognition: {text}")
            except sr.UnknownValueError:
                logger.warning("Google could not understand audio")
            except sr.RequestError as e:
                logger.error(f"Google recognition error: {e}")
            
            # Method 2: Whisper (if Google fails and Whisper is available)
            if not text and WHISPER_AVAILABLE:
                try:
                    if not self.whisper_model:
                        self.whisper_model = whisper.load_model("base")
                    
                    # Save audio to temporary file
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        audio.export(tmp_file.name, format="wav")
                        result = self.whisper_model.transcribe(tmp_file.name)
                        text = result["text"].strip()
                        logger.info(f"Whisper recognition: {text}")
                        
                        # Clean up temp file
                        import os
                        os.unlink(tmp_file.name)
                        
                except Exception as e:
                    logger.error(f"Whisper recognition error: {e}")
            elif not text and not WHISPER_AVAILABLE:
                logger.warning("Whisper not available. Only Google Speech Recognition was attempted.")
            
            return text
            
        except sr.WaitTimeoutError:
            logger.warning("No audio detected within timeout")
            return None
        except Exception as e:
            logger.error(f"Recording/transcription error: {e}")
            return None
    
    def text_to_signs(self, text: str) -> List[Dict]:
        """Convert text to sign language gestures"""
        try:
            logger.info(f"Converting text to signs: {text}")
            
            # Clean and normalize text
            text = text.lower().strip()
            
            # Split into words and phrases
            words = text.split()
            sign_sequence = []
            
            # Process each word/phrase
            for i, word in enumerate(words):
                # Check for multi-word phrases first
                phrase = None
                if i < len(words) - 1:
                    phrase = f"{word} {words[i+1]}"
                
                # Look up in dictionary
                if phrase and phrase in self.sign_dictionary:
                    sign_sequence.append({
                        "word": phrase,
                        "gesture": self.sign_dictionary[phrase],
                        "duration": 1.5  # seconds
                    })
                    i += 1  # Skip next word
                elif word in self.sign_dictionary:
                    sign_sequence.append({
                        "word": word,
                        "gesture": self.sign_dictionary[word],
                        "duration": 1.0  # seconds
                    })
                else:
                    # Fingerspell unknown words
                    fingerspelled = self._fingerspell_word(word)
                    sign_sequence.extend(fingerspelled)
            
            logger.info(f"Generated {len(sign_sequence)} signs")
            return sign_sequence
            
        except Exception as e:
            logger.error(f"Text to signs conversion error: {e}")
            return []
    
    def _fingerspell_word(self, word: str) -> List[Dict]:
        """Convert word to fingerspelling sequence"""
        fingerspelling_map = {
            'a': {'handshape': 'fist', 'movement': 'static', 'location': 'chest_level'},
            'b': {'handshape': 'flat_hand', 'movement': 'static', 'location': 'chest_level'},
            'c': {'handshape': 'c_handshape', 'movement': 'static', 'location': 'chest_level'},
            'd': {'handshape': 'd_handshape', 'movement': 'static', 'location': 'chest_level'},
            'e': {'handshape': 'e_handshape', 'movement': 'static', 'location': 'chest_level'},
            'f': {'handshape': 'f_handshape', 'movement': 'static', 'location': 'chest_level'},
            'g': {'handshape': 'g_handshape', 'movement': 'static', 'location': 'chest_level'},
            'h': {'handshape': 'h_handshape', 'movement': 'static', 'location': 'chest_level'},
            'i': {'handshape': 'i_handshape', 'movement': 'static', 'location': 'chest_level'},
            'j': {'handshape': 'j_handshape', 'movement': 'j_movement', 'location': 'chest_level'},
            'k': {'handshape': 'k_handshape', 'movement': 'static', 'location': 'chest_level'},
            'l': {'handshape': 'l_handshape', 'movement': 'static', 'location': 'chest_level'},
            'm': {'handshape': 'm_handshape', 'movement': 'static', 'location': 'chest_level'},
            'n': {'handshape': 'n_handshape', 'movement': 'static', 'location': 'chest_level'},
            'o': {'handshape': 'o_handshape', 'movement': 'static', 'location': 'chest_level'},
            'p': {'handshape': 'p_handshape', 'movement': 'static', 'location': 'chest_level'},
            'q': {'handshape': 'q_handshape', 'movement': 'static', 'location': 'chest_level'},
            'r': {'handshape': 'r_handshape', 'movement': 'static', 'location': 'chest_level'},
            's': {'handshape': 's_handshape', 'movement': 'static', 'location': 'chest_level'},
            't': {'handshape': 't_handshape', 'movement': 'static', 'location': 'chest_level'},
            'u': {'handshape': 'u_handshape', 'movement': 'static', 'location': 'chest_level'},
            'v': {'handshape': 'v_handshape', 'movement': 'static', 'location': 'chest_level'},
            'w': {'handshape': 'w_handshape', 'movement': 'static', 'location': 'chest_level'},
            'x': {'handshape': 'x_handshape', 'movement': 'static', 'location': 'chest_level'},
            'y': {'handshape': 'y_handshape', 'movement': 'static', 'location': 'chest_level'},
            'z': {'handshape': 'z_handshape', 'movement': 'z_movement', 'location': 'chest_level'}
        }
        
        fingerspelled = []
        for letter in word.lower():
            if letter in fingerspelling_map:
                fingerspelled.append({
                    "word": letter,
                    "gesture": fingerspelling_map[letter],
                    "duration": 0.5
                })
        
        return fingerspelled
    
    def speak_text(self, text: str):
        """Convert text to speech (for feedback)"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS error: {e}")
    
    def get_available_languages(self) -> List[str]:
        """Get list of available sign languages"""
        return ["ASL", "BSL", "LSF", "DGS"]  # American, British, French, German Sign Language
    
    def set_language(self, language: str):
        """Set the target sign language"""
        logger.info(f"Setting sign language to: {language}")
        # In production, this would load different dictionaries
        pass
