"""
Text-to-Speech Module
Enhanced TTS system for sign language translation
"""

import pyttsx3
import threading
import queue
import time
from loguru import logger
from typing import Optional, Dict, List, Callable
import json

class TextToSpeechEngine:
    """Enhanced text-to-speech engine for sign language translation"""
    
    def __init__(self):
        self.engine = pyttsx3.init()
        self.is_speaking = False
        self.speech_queue = queue.Queue()
        self.speech_thread = None
        self.current_text = ""
        
        # Configure TTS engine
        self._configure_engine()
        
        # Voice settings
        self.voice_settings = self._load_voice_settings()
        
        # Speech callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None
        self.on_speech_error: Optional[Callable] = None
        
        logger.info("TextToSpeechEngine initialized")
    
    def _configure_engine(self):
        """Configure TTS engine settings"""
        try:
            # Set default properties
            self.engine.setProperty('rate', 150)  # Speed
            self.engine.setProperty('volume', 0.9)  # Volume
            
            # Get available voices
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to find a good voice
                for voice in voices:
                    if 'english' in voice.name.lower() or 'us' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            logger.info("TTS engine configured")
            
        except Exception as e:
            logger.error(f"TTS configuration error: {e}")
    
    def _load_voice_settings(self) -> Dict:
        """Load voice settings and configurations"""
        return {
            "default_rate": 150,
            "default_volume": 0.9,
            "voices": {
                "male": None,
                "female": None,
                "child": None
            },
            "languages": {
                "en": "English",
                "es": "Spanish",
                "fr": "French",
                "de": "German"
            }
        }
    
    def speak(self, text: str, **kwargs) -> bool:
        """Speak the given text"""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for TTS")
                return False
            
            # Add to speech queue
            speech_item = {
                "text": text.strip(),
                "priority": kwargs.get("priority", 0),
                "interrupt": kwargs.get("interrupt", False),
                "callback": kwargs.get("callback", None)
            }
            
            if speech_item["interrupt"]:
                # Clear queue and stop current speech
                self.stop()
                while not self.speech_queue.empty():
                    try:
                        self.speech_queue.get_nowait()
                    except queue.Empty:
                        break
            
            self.speech_queue.put(speech_item)
            
            # Start speech thread if not running
            if not self.speech_thread or not self.speech_thread.is_alive():
                self.speech_thread = threading.Thread(target=self._speech_worker)
                self.speech_thread.daemon = True
                self.speech_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"TTS speak error: {e}")
            if self.on_speech_error:
                self.on_speech_error(e)
            return False
    
    def _speech_worker(self):
        """Worker thread for processing speech queue"""
        try:
            while True:
                try:
                    # Get next speech item
                    speech_item = self.speech_queue.get(timeout=1.0)
                    
                    if speech_item is None:  # Shutdown signal
                        break
                    
                    # Process speech item
                    self._process_speech_item(speech_item)
                    
                    # Mark task as done
                    self.speech_queue.task_done()
                    
                except queue.Empty:
                    # No items in queue, check if we should continue
                    if self.speech_queue.empty() and not self.is_speaking:
                        break
                    continue
                    
        except Exception as e:
            logger.error(f"Speech worker error: {e}")
            if self.on_speech_error:
                self.on_speech_error(e)
    
    def _process_speech_item(self, speech_item: Dict):
        """Process a single speech item"""
        try:
            text = speech_item["text"]
            callback = speech_item.get("callback")
            
            logger.info(f"Speaking: {text}")
            
            # Set current text
            self.current_text = text
            
            # Notify speech start
            if self.on_speech_start:
                self.on_speech_start(text)
            
            # Speak the text
            self.is_speaking = True
            self.engine.say(text)
            self.engine.runAndWait()
            
            # Notify speech end
            if self.on_speech_end:
                self.on_speech_end(text)
            
            # Call custom callback if provided
            if callback:
                callback(text)
            
            self.is_speaking = False
            self.current_text = ""
            
        except Exception as e:
            logger.error(f"Speech processing error: {e}")
            self.is_speaking = False
            if self.on_speech_error:
                self.on_speech_error(e)
    
    def stop(self):
        """Stop current speech and clear queue"""
        try:
            self.is_speaking = False
            self.engine.stop()
            
            # Clear queue
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    break
            
            logger.info("TTS stopped")
            
        except Exception as e:
            logger.error(f"TTS stop error: {e}")
    
    def pause(self):
        """Pause current speech"""
        try:
            self.engine.stop()
            self.is_speaking = False
            logger.info("TTS paused")
        except Exception as e:
            logger.error(f"TTS pause error: {e}")
    
    def resume(self):
        """Resume paused speech"""
        try:
            # Resume from current position
            if self.current_text:
                self.speak(self.current_text)
            logger.info("TTS resumed")
        except Exception as e:
            logger.error(f"TTS resume error: {e}")
    
    def set_rate(self, rate: int):
        """Set speech rate"""
        try:
            self.engine.setProperty('rate', rate)
            logger.info(f"Speech rate set to {rate}")
        except Exception as e:
            logger.error(f"Rate setting error: {e}")
    
    def set_volume(self, volume: float):
        """Set speech volume"""
        try:
            volume = max(0.0, min(1.0, volume))  # Clamp to valid range
            self.engine.setProperty('volume', volume)
            logger.info(f"Speech volume set to {volume}")
        except Exception as e:
            logger.error(f"Volume setting error: {e}")
    
    def set_voice(self, voice_id: str):
        """Set voice by ID"""
        try:
            self.engine.setProperty('voice', voice_id)
            logger.info(f"Voice set to {voice_id}")
        except Exception as e:
            logger.error(f"Voice setting error: {e}")
    
    def get_available_voices(self) -> List[Dict]:
        """Get list of available voices"""
        try:
            voices = self.engine.getProperty('voices')
            voice_list = []
            
            for voice in voices:
                voice_list.append({
                    "id": voice.id,
                    "name": voice.name,
                    "languages": voice.languages,
                    "gender": getattr(voice, 'gender', 'unknown')
                })
            
            return voice_list
            
        except Exception as e:
            logger.error(f"Voice list error: {e}")
            return []
    
    def get_current_voice(self) -> Optional[Dict]:
        """Get current voice information"""
        try:
            current_voice_id = self.engine.getProperty('voice')
            voices = self.get_available_voices()
            
            for voice in voices:
                if voice["id"] == current_voice_id:
                    return voice
            
            return None
            
        except Exception as e:
            logger.error(f"Current voice error: {e}")
            return None
    
    def is_speaking_now(self) -> bool:
        """Check if currently speaking"""
        return self.is_speaking
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.speech_queue.qsize()
    
    def set_speech_callbacks(self, on_start: Optional[Callable] = None,
                           on_end: Optional[Callable] = None,
                           on_error: Optional[Callable] = None):
        """Set speech event callbacks"""
        self.on_speech_start = on_start
        self.on_speech_end = on_end
        self.on_speech_error = on_error
    
    def speak_with_emphasis(self, text: str, emphasis_words: List[str]):
        """Speak text with emphasis on specific words"""
        try:
            # Add emphasis markers to text
            emphasized_text = text
            
            for word in emphasis_words:
                # Simple emphasis by repeating the word
                emphasized_text = emphasized_text.replace(
                    word, f"{word} {word}"
                )
            
            self.speak(emphasized_text)
            
        except Exception as e:
            logger.error(f"Emphasis speech error: {e}")
    
    def speak_slowly(self, text: str, slow_factor: float = 0.7):
        """Speak text slowly"""
        try:
            original_rate = self.engine.getProperty('rate')
            slow_rate = int(original_rate * slow_factor)
            
            self.set_rate(slow_rate)
            self.speak(text)
            self.set_rate(original_rate)  # Restore original rate
            
        except Exception as e:
            logger.error(f"Slow speech error: {e}")
    
    def speak_with_pauses(self, text: str, pause_duration: float = 0.5):
        """Speak text with pauses between sentences"""
        try:
            # Split text into sentences
            sentences = text.split('.')
            
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    self.speak(sentence.strip())
                    
                    # Add pause between sentences (except last one)
                    if i < len(sentences) - 1:
                        time.sleep(pause_duration)
            
        except Exception as e:
            logger.error(f"Paused speech error: {e}")
    
    def get_speech_status(self) -> Dict:
        """Get current speech status"""
        return {
            "is_speaking": self.is_speaking,
            "current_text": self.current_text,
            "queue_size": self.get_queue_size(),
            "current_voice": self.get_current_voice(),
            "rate": self.engine.getProperty('rate'),
            "volume": self.engine.getProperty('volume')
        }
    
    def shutdown(self):
        """Shutdown TTS engine"""
        try:
            self.stop()
            
            # Send shutdown signal to worker thread
            self.speech_queue.put(None)
            
            if self.speech_thread and self.speech_thread.is_alive():
                self.speech_thread.join(timeout=2.0)
            
            logger.info("TTS engine shutdown complete")
            
        except Exception as e:
            logger.error(f"TTS shutdown error: {e}")
