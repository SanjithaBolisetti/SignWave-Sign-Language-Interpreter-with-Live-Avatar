"""
UI Components Module
Streamlit UI components for the sign language translation system
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
from loguru import logger
from typing import Optional, Dict, Any
import time

# Try to import optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV (cv2) not available. Some UI features will be disabled.")

class TranslationUI:
    """UI components for the translation system"""
    
    def __init__(self):
        """
        Initialize UI helpers.
        
        Note:
            Page configuration (st.set_page_config) is handled in main.py
            to ensure it is called exactly once and as the first Streamlit
            command, as required by Streamlit.
        """
        logger.info("TranslationUI initialized")
    
    def render_header(self):
        """Render main header"""
        st.title("🤟 Bidirectional Sign Language Translation System")
        st.markdown("""
        **Real-time communication between hearing and deaf/hard-of-hearing individuals**
        
        This system provides:
        - 🎤 **Speech-to-Sign**: Convert spoken language to animated sign language
        - 📹 **Sign-to-Speech**: Recognize sign language and convert to speech/text
        - 🔄 **Bidirectional**: Real-time conversation support
        """)
        st.markdown("---")
    
    def render_mode_selector(self) -> str:
        """Render mode selection sidebar"""
        st.sidebar.header("🎛️ Translation Mode")
        
        mode = st.sidebar.selectbox(
            "Select Mode",
            ["Speech to Sign", "Sign to Speech", "Bidirectional"],
            help="Choose the translation direction"
        )
        
        st.sidebar.markdown("---")
        return mode
    
    def render_settings_panel(self):
        """Render settings panel in sidebar"""
        st.sidebar.header("⚙️ Settings")
        
        # Language selection
        sign_language = st.sidebar.selectbox(
            "Sign Language",
            ["ASL (American)", "BSL (British)", "LSF (French)", "DGS (German)"],
            help="Select the target sign language"
        )
        
        # Audio settings
        st.sidebar.subheader("🎵 Audio Settings")
        volume = st.sidebar.slider("Volume", 0.0, 1.0, 0.8, 0.1)
        speech_rate = st.sidebar.slider("Speech Rate", 100, 300, 150, 10)
        
        # Camera settings
        st.sidebar.subheader("📹 Camera Settings")
        camera_resolution = st.sidebar.selectbox(
            "Resolution",
            ["640x480", "1280x720", "1920x1080"],
            index=1
        )
        
        # Recognition settings
        st.sidebar.subheader("🔍 Recognition Settings")
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold", 0.5, 1.0, 0.7, 0.05
        )
        
        return {
            "sign_language": sign_language,
            "volume": volume,
            "speech_rate": speech_rate,
            "camera_resolution": camera_resolution,
            "confidence_threshold": confidence_threshold
        }
    
    def render_speech_to_sign_interface(self, settings: Dict[str, Any]):
        """Render speech-to-sign interface"""
        st.header("🎤 Speech to Sign Translation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Audio Input")
            
            # Recording controls
            col_start, col_stop = st.columns([1, 1])
            
            with col_start:
                if st.button("🎤 Start Recording", key="start_speech_recording"):
                    st.session_state['recording'] = True
                    st.success("Recording started! Speak now...")
            
            with col_stop:
                if st.button("⏹️ Stop Recording", key="stop_speech_recording"):
                    st.session_state['recording'] = False
                    st.info("Recording stopped")
            
            # Audio visualization
            if st.session_state.get('recording', False):
                st.info("🎵 Recording in progress...")
                # Placeholder for audio waveform visualization
                st.progress(0.7)
            
            # Transcription display
            if 'transcribed_text' in st.session_state:
                st.subheader("📝 Transcribed Text")
                st.success(f"**Recognized:** {st.session_state['transcribed_text']}")
        
        with col2:
            st.subheader("Sign Language Animation")
            
            # Avatar display area
            avatar_placeholder = st.empty()
            
            if 'sign_animation' in st.session_state:
                st.info("🎭 Playing sign language animation...")
                
                # Animation progress
                progress = st.session_state.get('animation_progress', 0)
                st.progress(progress)
                
                # Placeholder for 3D avatar
                avatar_placeholder.info("🤖 3D Avatar performing signs")
            
            # Animation controls
            col_play, col_pause, col_reset = st.columns([1, 1, 1])
            
            with col_play:
                if st.button("▶️ Play", key="play_animation"):
                    st.session_state['animation_playing'] = True
            
            with col_pause:
                if st.button("⏸️ Pause", key="pause_animation"):
                    st.session_state['animation_playing'] = False
            
            with col_reset:
                if st.button("🔄 Reset", key="reset_animation"):
                    st.session_state['animation_progress'] = 0
    
    def render_sign_to_speech_interface(self, settings: Dict[str, Any]):
        """Render sign-to-speech interface"""
        st.header("📹 Sign to Speech Translation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Camera Input")
            
            # Camera controls
            col_start, col_stop = st.columns([1, 1])
            
            with col_start:
                if st.button("📹 Start Camera", key="start_camera"):
                    st.session_state['camera_active'] = True
                    st.success("Camera started! Sign now...")
            
            with col_stop:
                if st.button("⏹️ Stop Camera", key="stop_camera"):
                    st.session_state['camera_active'] = False
                    st.info("Camera stopped")
            
            # Camera feed display
            camera_placeholder = st.empty()
            
            if st.session_state.get('camera_active', False):
                # Placeholder for camera feed
                camera_placeholder.info("📹 Camera feed active")
                
                # Hand detection visualization
                st.subheader("🖐️ Hand Detection")
                detection_placeholder = st.empty()
                detection_placeholder.info("🔍 Detecting hands and gestures...")
            
            # Recognition status
            if 'recognized_gesture' in st.session_state:
                st.subheader("🎯 Recognized Gesture")
                st.success(f"**Detected:** {st.session_state['recognized_gesture']}")
        
        with col2:
            st.subheader("Translation Output")
            
            # Text output
            if 'recognized_text' in st.session_state:
                st.subheader("📝 Recognized Text")
                st.success(f"**Translated:** {st.session_state['recognized_text']}")
            
            # Speech output controls
            st.subheader("🔊 Speech Output")
            
            col_speak, col_repeat = st.columns([1, 1])
            
            with col_speak:
                if st.button("🔊 Speak", key="speak_text"):
                    if 'recognized_text' in st.session_state:
                        st.info("🔊 Speaking...")
                        # TTS would be called here
            
            with col_repeat:
                if st.button("🔄 Repeat", key="repeat_speech"):
                    if 'recognized_text' in st.session_state:
                        st.info("🔄 Repeating...")
    
    def render_bidirectional_interface(self, settings: Dict[str, Any]):
        """Render bidirectional interface"""
        st.header("🔄 Bidirectional Translation")
        st.markdown("**Real-time conversation between hearing and deaf individuals**")
        
        # Conversation area
        st.subheader("💬 Conversation")
        
        # Chat history
        if 'conversation_history' not in st.session_state:
            st.session_state['conversation_history'] = []
        
        # Display conversation history
        for i, message in enumerate(st.session_state['conversation_history']):
            if message['type'] == 'hearing':
                st.info(f"👤 **Hearing Person:** {message['text']}")
            else:
                st.success(f"🤟 **Deaf Person:** {message['text']}")
        
        # Input areas
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("👤 Hearing Person Input")
            
            if st.button("🎤 Speak", key="bidirectional_speak"):
                st.session_state['hearing_speaking'] = True
                st.info("🎤 Recording...")
            
            if st.session_state.get('hearing_speaking', False):
                # Process speech and add to conversation
                if 'hearing_text' in st.session_state:
                    st.session_state['conversation_history'].append({
                        'type': 'hearing',
                        'text': st.session_state['hearing_text'],
                        'timestamp': time.time()
                    })
                    st.session_state['hearing_speaking'] = False
        
        with col2:
            st.subheader("🤟 Deaf Person Input")
            
            if st.button("📹 Sign", key="bidirectional_sign"):
                st.session_state['deaf_signing'] = True
                st.info("📹 Recording signs...")
            
            if st.session_state.get('deaf_signing', False):
                # Process signs and add to conversation
                if 'deaf_text' in st.session_state:
                    st.session_state['conversation_history'].append({
                        'type': 'deaf',
                        'text': st.session_state['deaf_text'],
                        'timestamp': time.time()
                    })
                    st.session_state['deaf_signing'] = False
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", key="clear_conversation"):
            st.session_state['conversation_history'] = []
            st.success("Conversation cleared")
    
    def render_status_panel(self):
        """Render status panel"""
        st.sidebar.header("📊 Status")
        
        # System status
        st.sidebar.subheader("System Status")
        
        # Microphone status
        mic_status = "🟢 Active" if st.session_state.get('recording', False) else "🔴 Inactive"
        st.sidebar.write(f"**Microphone:** {mic_status}")
        
        # Camera status
        camera_status = "🟢 Active" if st.session_state.get('camera_active', False) else "🔴 Inactive"
        st.sidebar.write(f"**Camera:** {camera_status}")
        
        # Animation status
        animation_status = "🟢 Playing" if st.session_state.get('animation_playing', False) else "🔴 Stopped"
        st.sidebar.write(f"**Animation:** {animation_status}")
        
        # Recognition status
        recognition_status = "🟢 Active" if st.session_state.get('camera_active', False) else "🔴 Inactive"
        st.sidebar.write(f"**Recognition:** {recognition_status}")
    
    def render_help_panel(self):
        """Render help panel"""
        st.sidebar.header("❓ Help")
        
        with st.sidebar.expander("How to Use"):
            st.markdown("""
            **Speech to Sign:**
            1. Click "Start Recording"
            2. Speak clearly into microphone
            3. Watch avatar perform signs
            
            **Sign to Speech:**
            1. Click "Start Camera"
            2. Sign clearly in front of camera
            3. Listen to speech output
            
            **Bidirectional:**
            1. Both users can input simultaneously
            2. Conversation history is maintained
            3. Real-time translation
            """)
        
        with st.sidebar.expander("Tips"):
            st.markdown("""
            - Speak clearly and at normal pace
            - Sign with good lighting
            - Keep hands visible in camera
            - Use simple, clear gestures
            - Maintain eye contact with camera
            """)
    
    def display_camera_feed(self, frame: np.ndarray, placeholder):
        """Display camera feed with overlays"""
        try:
            if frame is not None:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame for display
                height, width = rgb_frame.shape[:2]
                max_width = 640
                if width > max_width:
                    scale = max_width / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
                
                # Convert to PIL Image
                pil_image = Image.fromarray(rgb_frame)
                
                # Display in Streamlit
                placeholder.image(pil_image, caption="Camera Feed", use_column_width=True)
            
        except Exception as e:
            logger.error(f"Camera feed display error: {e}")
            placeholder.error("Error displaying camera feed")
    
    def show_error_message(self, message: str):
        """Show error message"""
        st.error(f"❌ {message}")
    
    def show_success_message(self, message: str):
        """Show success message"""
        st.success(f"✅ {message}")
    
    def show_info_message(self, message: str):
        """Show info message"""
        st.info(f"ℹ️ {message}")
    
    def show_warning_message(self, message: str):
        """Show warning message"""
        st.warning(f"⚠️ {message}")
    
    def render_footer(self):
        """Render footer"""
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>🤟 Bidirectional Sign Language Translation System | 
            Built with ❤️ for inclusive communication</p>
        </div>
        """, unsafe_allow_html=True)
