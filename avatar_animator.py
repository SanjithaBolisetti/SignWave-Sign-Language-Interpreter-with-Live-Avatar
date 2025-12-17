"""
Avatar Animator Module
Creates and animates 3D avatar for sign language gestures
"""

import numpy as np
from loguru import logger
import threading
import time
from typing import List, Dict, Optional, Tuple
import math

# Try to import optional dependencies
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logger.warning("Pygame not available. Avatar animation will be limited.")

try:
    import moderngl
    MODERNGL_AVAILABLE = True
except ImportError:
    MODERNGL_AVAILABLE = False
    logger.warning("ModernGL not available. 3D rendering will be limited.")

class AvatarAnimator:
    """Handles 3D avatar creation and sign language animation"""
    
    def __init__(self):
        self.avatar_data = None
        self.current_animation = None
        self.is_animating = False
        self.animation_thread = None
        
        # Avatar skeleton structure
        self.skeleton = self._create_skeleton()
        
        # Animation parameters
        self.frame_rate = 30
        self.current_frame = 0
        self.total_frames = 0
        
        logger.info("AvatarAnimator initialized")
    
    def _create_skeleton(self) -> Dict:
        """Create avatar skeleton structure"""
        return {
            "head": {
                "position": [0, 1.7, 0],
                "rotation": [0, 0, 0],
                "children": ["neck"]
            },
            "neck": {
                "position": [0, 1.5, 0],
                "rotation": [0, 0, 0],
                "children": ["torso"]
            },
            "torso": {
                "position": [0, 1.2, 0],
                "rotation": [0, 0, 0],
                "children": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
            },
            "left_shoulder": {
                "position": [-0.3, 1.3, 0],
                "rotation": [0, 0, 0],
                "children": ["left_elbow"]
            },
            "right_shoulder": {
                "position": [0.3, 1.3, 0],
                "rotation": [0, 0, 0],
                "children": ["right_elbow"]
            },
            "left_elbow": {
                "position": [-0.5, 1.1, 0],
                "rotation": [0, 0, 0],
                "children": ["left_wrist"]
            },
            "right_elbow": {
                "position": [0.5, 1.1, 0],
                "rotation": [0, 0, 0],
                "children": ["right_wrist"]
            },
            "left_wrist": {
                "position": [-0.7, 0.9, 0],
                "rotation": [0, 0, 0],
                "children": ["left_hand"]
            },
            "right_wrist": {
                "position": [0.7, 0.9, 0],
                "rotation": [0, 0, 0],
                "children": ["right_hand"]
            },
            "left_hand": {
                "position": [-0.8, 0.8, 0],
                "rotation": [0, 0, 0],
                "children": []
            },
            "right_hand": {
                "position": [0.8, 0.8, 0],
                "rotation": [0, 0, 0],
                "children": []
            },
            "left_hip": {
                "position": [-0.2, 0.8, 0],
                "rotation": [0, 0, 0],
                "children": ["left_knee"]
            },
            "right_hip": {
                "position": [0.2, 0.8, 0],
                "rotation": [0, 0, 0],
                "children": ["right_knee"]
            },
            "left_knee": {
                "position": [-0.2, 0.4, 0],
                "rotation": [0, 0, 0],
                "children": ["left_ankle"]
            },
            "right_knee": {
                "position": [0.2, 0.4, 0],
                "rotation": [0, 0, 0],
                "children": ["right_ankle"]
            },
            "left_ankle": {
                "position": [-0.2, 0.1, 0],
                "rotation": [0, 0, 0],
                "children": []
            },
            "right_ankle": {
                "position": [0.2, 0.1, 0],
                "rotation": [0, 0, 0],
                "children": []
            }
        }
    
    def play_animation(self, animation_data: List[Dict]):
        """Play sign language animation sequence"""
        try:
            if not animation_data:
                logger.warning("No animation data provided")
                return
            
            logger.info(f"Playing animation with {len(animation_data)} gestures")
            
            # Stop any current animation
            self.stop_animation()
            
            # Prepare animation
            self.current_animation = self._prepare_animation(animation_data)
            self.total_frames = len(self.current_animation)
            self.current_frame = 0
            self.is_animating = True
            
            # Start animation thread
            self.animation_thread = threading.Thread(target=self._animation_loop)
            self.animation_thread.daemon = True
            self.animation_thread.start()
            
        except Exception as e:
            logger.error(f"Animation play error: {e}")
    
    def _prepare_animation(self, animation_data: List[Dict]) -> List[Dict]:
        """Prepare animation data for playback"""
        try:
            prepared_frames = []
            
            for gesture_data in animation_data:
                gesture = gesture_data["gesture"]
                duration = gesture_data["duration"]
                word = gesture_data["word"]
                
                # Calculate frames for this gesture
                gesture_frames = int(duration * self.frame_rate)
                
                # Generate keyframes for the gesture
                keyframes = self._generate_gesture_keyframes(gesture, gesture_frames)
                
                prepared_frames.extend(keyframes)
            
            return prepared_frames
            
        except Exception as e:
            logger.error(f"Animation preparation error: {e}")
            return []
    
    def _generate_gesture_keyframes(self, gesture: Dict, frame_count: int) -> List[Dict]:
        """Generate keyframes for a specific gesture"""
        try:
            keyframes = []
            
            handshape = gesture.get("handshape", "neutral")
            movement = gesture.get("movement", "static")
            location = gesture.get("location", "chest_level")
            orientation = gesture.get("orientation", "palm_out")
            facial_expression = gesture.get("facial_expression", "neutral")
            
            # Generate frames for the gesture
            for frame in range(frame_count):
                progress = frame / frame_count
                
                # Calculate bone positions and rotations
                frame_data = self._calculate_frame_data(
                    handshape, movement, location, orientation, facial_expression, progress
                )
                
                keyframes.append(frame_data)
            
            return keyframes
            
        except Exception as e:
            logger.error(f"Keyframe generation error: {e}")
            return []
    
    def _calculate_frame_data(self, handshape: str, movement: str, location: str, 
                            orientation: str, facial_expression: str, progress: float) -> Dict:
        """Calculate bone positions and rotations for a frame"""
        try:
            frame_data = {
                "bones": {},
                "facial_expression": facial_expression,
                "progress": progress
            }
            
            # Calculate hand positions based on location
            hand_position = self._get_location_position(location)
            
            # Calculate hand rotation based on orientation
            hand_rotation = self._get_orientation_rotation(orientation)
            
            # Apply movement
            if movement != "static":
                hand_position = self._apply_movement(hand_position, movement, progress)
            
            # Update skeleton with calculated positions
            frame_data["bones"]["right_hand"] = {
                "position": hand_position,
                "rotation": hand_rotation,
                "handshape": handshape
            }
            
            # Calculate arm chain positions
            arm_positions = self._calculate_arm_chain(hand_position)
            frame_data["bones"]["right_wrist"] = arm_positions["wrist"]
            frame_data["bones"]["right_elbow"] = arm_positions["elbow"]
            frame_data["bones"]["right_shoulder"] = arm_positions["shoulder"]
            
            return frame_data
            
        except Exception as e:
            logger.error(f"Frame data calculation error: {e}")
            return {"bones": {}, "facial_expression": "neutral", "progress": 0}
    
    def _get_location_position(self, location: str) -> List[float]:
        """Get hand position based on location"""
        location_map = {
            "chest_level": [0.7, 1.2, 0],
            "mouth_level": [0.7, 1.5, 0],
            "chin_level": [0.7, 1.4, 0],
            "head_level": [0.7, 1.7, 0],
            "waist_level": [0.7, 0.9, 0]
        }
        return location_map.get(location, [0.7, 1.2, 0])
    
    def _get_orientation_rotation(self, orientation: str) -> List[float]:
        """Get hand rotation based on orientation"""
        orientation_map = {
            "palm_out": [0, 0, 0],
            "palm_in": [0, 180, 0],
            "palm_up": [90, 0, 0],
            "palm_down": [-90, 0, 0],
            "palm_left": [0, 90, 0],
            "palm_right": [0, -90, 0]
        }
        return orientation_map.get(orientation, [0, 0, 0])
    
    def _apply_movement(self, position: List[float], movement: str, progress: float) -> List[float]:
        """Apply movement to position based on progress"""
        try:
            base_position = position.copy()
            
            if movement == "wave":
                # Wave motion
                wave_offset = 0.1 * math.sin(progress * 4 * math.pi)
                base_position[1] += wave_offset
            elif movement == "nod_up_down":
                # Nodding motion
                nod_offset = 0.1 * math.sin(progress * 2 * math.pi)
                base_position[1] += nod_offset
            elif movement == "tap_together":
                # Tapping motion
                tap_offset = 0.05 * math.sin(progress * 8 * math.pi)
                base_position[2] += tap_offset
            elif movement == "forward_out":
                # Forward motion
                forward_offset = 0.1 * progress
                base_position[2] += forward_offset
            elif movement == "tap_chest":
                # Chest tapping
                chest_offset = 0.05 * math.sin(progress * 4 * math.pi)
                base_position[0] += chest_offset
            elif movement == "tap_chin":
                # Chin tapping
                chin_offset = 0.05 * math.sin(progress * 4 * math.pi)
                base_position[1] += chin_offset
            elif movement == "tap_mouth":
                # Mouth tapping
                mouth_offset = 0.05 * math.sin(progress * 4 * math.pi)
                base_position[1] += mouth_offset
            elif movement == "shake":
                # Shaking motion
                shake_offset = 0.1 * math.sin(progress * 6 * math.pi)
                base_position[0] += shake_offset
            
            return base_position
            
        except Exception as e:
            logger.error(f"Movement application error: {e}")
            return position
    
    def _calculate_arm_chain(self, hand_position: List[float]) -> Dict:
        """Calculate arm chain positions using inverse kinematics"""
        try:
            # Simple IK calculation
            shoulder_pos = [0.3, 1.3, 0]
            elbow_pos = [0.5, 1.1, 0]
            wrist_pos = hand_position
            
            # Calculate elbow position
            arm_length = 0.3
            forearm_length = 0.2
            
            # Vector from shoulder to hand
            shoulder_to_hand = np.array(wrist_pos) - np.array(shoulder_pos)
            distance = np.linalg.norm(shoulder_to_hand)
            
            if distance > 0:
                # Calculate elbow position using law of cosines
                cos_angle = (arm_length**2 + forearm_length**2 - distance**2) / (2 * arm_length * forearm_length)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                
                elbow_angle = math.acos(cos_angle)
                
                # Calculate elbow position
                shoulder_to_hand_normalized = shoulder_to_hand / distance
                elbow_offset = arm_length * math.cos(elbow_angle)
                
                elbow_pos = np.array(shoulder_pos) + shoulder_to_hand_normalized * elbow_offset
                elbow_pos = elbow_pos.tolist()
            
            return {
                "shoulder": shoulder_pos,
                "elbow": elbow_pos,
                "wrist": wrist_pos
            }
            
        except Exception as e:
            logger.error(f"Arm chain calculation error: {e}")
            return {
                "shoulder": [0.3, 1.3, 0],
                "elbow": [0.5, 1.1, 0],
                "wrist": hand_position
            }
    
    def _animation_loop(self):
        """Main animation loop"""
        try:
            while self.is_animating and self.current_frame < self.total_frames:
                if self.current_animation and self.current_frame < len(self.current_animation):
                    # Get current frame data
                    frame_data = self.current_animation[self.current_frame]
                    
                    # Update avatar with frame data
                    self._update_avatar(frame_data)
                    
                    # Advance frame
                    self.current_frame += 1
                    
                    # Wait for next frame
                    time.sleep(1.0 / self.frame_rate)
                else:
                    break
            
            # Animation finished
            self.is_animating = False
            logger.info("Animation completed")
            
        except Exception as e:
            logger.error(f"Animation loop error: {e}")
            self.is_animating = False
    
    def _update_avatar(self, frame_data: Dict):
        """Update avatar with frame data"""
        try:
            # In a real implementation, this would update the 3D model
            # For now, we'll just log the frame data
            if self.current_frame % 10 == 0:  # Log every 10th frame
                logger.debug(f"Frame {self.current_frame}: {frame_data}")
            
        except Exception as e:
            logger.error(f"Avatar update error: {e}")
    
    def stop_animation(self):
        """Stop current animation"""
        self.is_animating = False
        if self.animation_thread and self.animation_thread.is_alive():
            self.animation_thread.join(timeout=1.0)
        logger.info("Animation stopped")
    
    def is_animating(self) -> bool:
        """Check if animation is currently playing"""
        return self.is_animating
    
    def get_current_frame(self) -> int:
        """Get current animation frame"""
        return self.current_frame
    
    def get_total_frames(self) -> int:
        """Get total animation frames"""
        return self.total_frames
    
    def set_frame_rate(self, fps: int):
        """Set animation frame rate"""
        self.frame_rate = fps
        logger.info(f"Frame rate set to {fps} FPS")
    
    def get_animation_progress(self) -> float:
        """Get animation progress (0.0 to 1.0)"""
        if self.total_frames > 0:
            return self.current_frame / self.total_frames
        return 0.0
