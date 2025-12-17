"""
Sign-to-Speech Translation Module
Recognizes sign language gestures and converts to speech/text
"""

import numpy as np
import pyttsx3
from loguru import logger
import threading
import time
from typing import List, Dict, Optional, Tuple
import json

# Try to import optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV (cv2) not available. Camera features disabled.")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available. Gesture recognition disabled.")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Some ML features disabled.")

class SignToSpeechTranslator:
    """Handles sign language recognition and conversion to speech"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe models
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.face = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialize TTS
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)
        
        # Sign recognition model (placeholder - would be trained model in production)
        self.sign_model = None
        self.sign_dictionary = self._load_reverse_sign_dictionary()
        
        # Camera and processing
        self.camera = None
        self.is_recording = False
        self.current_frame = None
        self.gesture_buffer = []
        self.buffer_size = 10
        
        logger.info("SignToSpeechTranslator initialized")
    
    def _load_reverse_sign_dictionary(self) -> Dict[str, str]:
        """Load reverse dictionary mapping gestures to words"""
        return {
            "open_hand_wave_chest_level": "hello",
            "flat_hand_forward_out_mouth_level": "thank you",
            "fist_nod_up_down_chest_level": "yes",
            "index_middle_fingers_tap_together_chest_level": "no",
            "thumbs_up_tap_chest_chest_level": "help",
            "w_handshape_tap_chin_chin_level": "water",
            "flat_hand_tap_mouth_mouth_level": "food",
            "t_handshape_shake_chest_level": "bathroom"
        }
    
    def start_camera_feed(self, placeholder=None):
        """Start camera feed for sign recognition"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                logger.error("Could not open camera")
                return False
            
            self.is_recording = True
            logger.info("Camera feed started")
            
            # Start processing thread
            processing_thread = threading.Thread(target=self._process_camera_feed)
            processing_thread.daemon = True
            processing_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Camera start error: {e}")
            return False
    
    def stop_camera_feed(self):
        """Stop camera feed"""
        self.is_recording = False
        if self.camera:
            self.camera.release()
        logger.info("Camera feed stopped")
    
    def _process_camera_feed(self):
        """Process camera feed for sign recognition"""
        while self.is_recording and self.camera:
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame for sign recognition
            processed_frame, gesture_data = self._analyze_frame(frame)
            
            # Update current frame
            self.current_frame = processed_frame
            
            # Add gesture to buffer
            if gesture_data:
                self.gesture_buffer.append(gesture_data)
                if len(self.gesture_buffer) > self.buffer_size:
                    self.gesture_buffer.pop(0)
            
            time.sleep(0.033)  # ~30 FPS
    
    def _analyze_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Dict]]:
        """Analyze frame for sign language gestures"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hands
            hands_results = self.hands.process(rgb_frame)
            
            # Process pose
            pose_results = self.pose.process(rgb_frame)
            
            # Process face
            face_results = self.face.process(rgb_frame)
            
            # Draw landmarks
            annotated_frame = frame.copy()
            
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
            
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )
            
            # Extract gesture features
            gesture_data = self._extract_gesture_features(
                hands_results, pose_results, face_results
            )
            
            return annotated_frame, gesture_data
            
        except Exception as e:
            logger.error(f"Frame analysis error: {e}")
            return frame, None
    
    def _extract_gesture_features(self, hands_results, pose_results, face_results) -> Optional[Dict]:
        """Extract features from detected landmarks"""
        try:
            gesture_features = {
                "timestamp": time.time(),
                "hands": [],
                "pose": None,
                "face": None
            }
            
            # Extract hand features
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    hand_data = {
                        "landmarks": [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark],
                        "handedness": "left" if hands_results.multi_handedness[0].classification[0].label == "Left" else "right"
                    }
                    gesture_features["hands"].append(hand_data)
            
            # Extract pose features
            if pose_results.pose_landmarks:
                pose_data = {
                    "landmarks": [(lm.x, lm.y, lm.z) for lm in pose_results.pose_landmarks.landmark],
                    "visibility": [lm.visibility for lm in pose_results.pose_landmarks.landmark]
                }
                gesture_features["pose"] = pose_data
            
            # Extract face features
            if face_results.multi_face_landmarks:
                face_data = {
                    "landmarks": [(lm.x, lm.y, lm.z) for lm in face_results.multi_face_landmarks[0].landmark]
                }
                gesture_features["face"] = face_data
            
            return gesture_features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
    
    def process_signs(self) -> Optional[str]:
        """Process accumulated gestures and return recognized text"""
        try:
            if not self.gesture_buffer:
                return None
            
            # Analyze gesture sequence
            recognized_gesture = self._recognize_gesture_sequence()
            
            if recognized_gesture:
                return self.sign_dictionary.get(recognized_gesture, "Unknown gesture")
            
            return None
            
        except Exception as e:
            logger.error(f"Sign processing error: {e}")
            return None
    
    def _recognize_gesture_sequence(self) -> Optional[str]:
        """Recognize gesture from sequence of frames"""
        try:
            if len(self.gesture_buffer) < 3:
                return None
            
            # Analyze hand shapes and movements
            hand_features = []
            for gesture in self.gesture_buffer:
                if gesture["hands"]:
                    hand_data = gesture["hands"][0]  # Primary hand
                    hand_features.append(self._classify_handshape(hand_data["landmarks"]))
            
            # Determine gesture based on hand shape sequence
            gesture_key = self._determine_gesture_key(hand_features)
            
            return gesture_key
            
        except Exception as e:
            logger.error(f"Gesture recognition error: {e}")
            return None
    
    def _classify_handshape(self, landmarks: List[Tuple[float, float, float]]) -> str:
        """Classify hand shape from landmarks"""
        try:
            # Extract key points
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]
            
            # Calculate distances and angles
            thumb_index_dist = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
            index_middle_dist = np.linalg.norm(np.array(index_tip) - np.array(middle_tip))
            
            # Simple hand shape classification
            if thumb_index_dist < 0.05 and index_middle_dist < 0.05:
                return "fist"
            elif thumb_index_dist > 0.1 and index_middle_dist < 0.05:
                return "open_hand"
            elif thumb_index_dist < 0.05 and index_middle_dist > 0.1:
                return "index_middle_fingers"
            elif thumb_index_dist > 0.1 and index_middle_dist > 0.1:
                return "flat_hand"
            else:
                return "unknown"
                
        except Exception as e:
            logger.error(f"Hand shape classification error: {e}")
            return "unknown"
    
    def _determine_gesture_key(self, hand_features: List[str]) -> Optional[str]:
        """Determine gesture key from hand feature sequence"""
        try:
            # Analyze movement patterns
            if len(hand_features) < 3:
                return None
            
            # Check for specific gesture patterns
            if "open_hand" in hand_features and len(set(hand_features)) == 1:
                return "open_hand_wave_chest_level"
            elif "fist" in hand_features and len(set(hand_features)) == 1:
                return "fist_nod_up_down_chest_level"
            elif "index_middle_fingers" in hand_features and len(set(hand_features)) == 1:
                return "index_middle_fingers_tap_together_chest_level"
            elif "flat_hand" in hand_features and len(set(hand_features)) == 1:
                return "flat_hand_forward_out_mouth_level"
            
            return None
            
        except Exception as e:
            logger.error(f"Gesture key determination error: {e}")
            return None
    
    def text_to_speech(self, text: str):
        """Convert recognized text to speech"""
        try:
            logger.info(f"Converting to speech: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS error: {e}")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current processed frame"""
        return self.current_frame
    
    def is_camera_active(self) -> bool:
        """Check if camera is active"""
        return self.is_recording and self.camera is not None
    
    def calibrate_camera(self):
        """Calibrate camera for optimal sign recognition"""
        logger.info("Calibrating camera for sign recognition...")
        # In production, this would perform camera calibration
        pass
    
    def set_recognition_threshold(self, threshold: float):
        """Set recognition confidence threshold"""
        logger.info(f"Setting recognition threshold to: {threshold}")
        # Update MediaPipe model thresholds
        pass
