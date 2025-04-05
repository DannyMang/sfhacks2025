#!/usr/bin/env python3
"""
Face detection and landmark extraction module using MediaPipe.
"""

import cv2
import numpy as np
import logging
import time
import os

# Try to import mediapipe, but handle gracefully if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

class FaceDetector:
    def __init__(self, static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the FaceDetector with MediaPipe Face Mesh.
        
        Args:
            static_image_mode: Whether to treat input images as static (not video)
            max_num_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
        """
        self.logger = logging.getLogger(__name__)
        self.max_num_faces = max_num_faces
        
        if MEDIAPIPE_AVAILABLE:
            self.logger.info("Initializing MediaPipe Face Mesh")
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=max_num_faces,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        else:
            self.logger.warning("MediaPipe not available, using dummy face detector")
            self.face_mesh = None
            self.mp_face_mesh = None
            self.mp_drawing = None
        
        # Track processing time for performance monitoring
        self.process_time = 0
        
        # Default values for dummy face detection
        self.dummy_landmarks = self._generate_dummy_landmarks()
    
    def _generate_dummy_landmarks(self):
        """Generate dummy landmarks for testing."""
        # Create a basic face oval with 468 landmarks (MediaPipe Face Mesh standard)
        landmarks = []
        
        # Parameters for the oval
        center_x, center_y = 0.5, 0.5  # Center of the face in normalized coordinates
        width, height = 0.3, 0.4       # Width and height of the face
        
        for i in range(468):
            # Distribute landmarks around an oval
            angle = 2 * np.pi * (i % 80) / 80
            radius_x = width * (0.8 + 0.2 * np.sin(i * 0.1))
            radius_y = height * (0.8 + 0.2 * np.sin(i * 0.1))
            
            x = center_x + radius_x * np.cos(angle)
            y = center_y + radius_y * np.sin(angle)
            z = 0.01 * np.sin(angle)  # Small Z variation for depth
            
            landmarks.append([x, y, z])
        
        return [landmarks]  # Return as a list of face landmarks
    
    def detect(self, image):
        """
        Detect facial landmarks in an image.
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            landmarks_list: List of normalized landmark coordinates, or None if no face detected
        """
        if not MEDIAPIPE_AVAILABLE:
            return self.dummy_landmarks
        
        landmarks_list, _, _ = self.detect_face_landmarks(image)
        return landmarks_list if landmarks_list else None
        
    def detect_face_landmarks(self, image):
        """
        Detect facial landmarks in an image.
        
        Args:
            image: BGR image (OpenCV format)
            
        Returns:
            landmarks_list: List of normalized landmark coordinates [x, y, z]
            face_rect: Rectangle containing the face [x, y, w, h]
            processed_image: Image with landmarks drawn (for visualization only)
        """
        start_time = time.time()
        
        # Check if we're running in dummy mode
        if not MEDIAPIPE_AVAILABLE or self.face_mesh is None:
            self.process_time = time.time() - start_time
            
            # Generate a dummy output
            h, w = image.shape[:2] if image is not None else (512, 512)
            face_rect = [int(w * 0.25), int(h * 0.25), int(w * 0.5), int(h * 0.5)]
            processed_image = image.copy() if image is not None else np.zeros((512, 512, 3), dtype=np.uint8)
            
            # Draw dummy landmarks for visualization
            if processed_image is not None:
                for landmark in self.dummy_landmarks[0]:
                    x, y = int(landmark[0] * w), int(landmark[1] * h)
                    cv2.circle(processed_image, (x, y), 1, (0, 255, 0), -1)
            
            return self.dummy_landmarks, face_rect, processed_image
        
        # Ensure we have a valid image
        if image is None or image.size == 0:
            self.logger.error("Invalid input image")
            return [], None, np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        # Process the image
        results = self.face_mesh.process(image_rgb)
        
        # Default return values
        landmarks_list = []
        face_rect = None
        processed_image = image.copy()
        
        # If face detected, extract landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw landmarks for visualization
                self.mp_drawing.draw_landmarks(
                    image=processed_image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)
                
                # Extract landmark coordinates
                face_landmarks_list = []
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    face_landmarks_list.append([landmark.x, landmark.y, landmark.z])
                    
                    # Update bounding box
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                landmarks_list.append(face_landmarks_list)
                
                # Calculate face bounding rectangle
                face_rect = [x_min, y_min, x_max - x_min, y_max - y_min]
                
                # Only process the first face if max_num_faces is 1
                if len(landmarks_list) >= self.max_num_faces:
                    break
        
        self.process_time = time.time() - start_time
        
        return landmarks_list, face_rect, processed_image
    
    def get_head_pose(self, landmarks_list, image=None):
        """
        Estimate 3D head pose from landmarks.
        
        Args:
            landmarks_list: List of landmark coordinates
            image: Shape of the input image or the image itself (optional)
            
        Returns:
            [pitch, yaw, roll]: Euler angles in degrees
        """
        if not landmarks_list:
            # Return default pose
            return [0.0, 0.0, 0.0]
        
        try:
            # Simple head pose estimation based on facial landmarks
            # This is a simplified version for the hackathon
            
            # Get image dimensions
            if image is not None:
                if isinstance(image, tuple):
                    h, w = image[:2]
                else:
                    h, w = image.shape[:2]
            else:
                h, w = 512, 512
            
            # Use the first face
            landmarks = landmarks_list[0]
            
            # Calculate center of the face
            x_sum = y_sum = z_sum = 0
            for point in landmarks:
                x_sum += point[0]
                y_sum += point[1]
                z_sum += point[2]
            
            center_x = x_sum / len(landmarks)
            center_y = y_sum / len(landmarks)
            center_z = z_sum / len(landmarks)
            
            # Calculate head orientation based on relative positions
            # of key facial points (simplified for the hackathon)
            
            # Find nose tip (average of nose landmarks)
            nose_indices = [1, 2, 3, 4, 5, 6]  # Approximate nose indices
            nose_x = nose_y = nose_z = 0
            for idx in range(min(6, len(landmarks))):
                nose_x += landmarks[idx][0]
                nose_y += landmarks[idx][1]
                nose_z += landmarks[idx][2]
            
            nose_x /= len(nose_indices)
            nose_y /= len(nose_indices)
            nose_z /= len(nose_indices)
            
            # Calculate relative position of nose compared to face center
            rel_x = (nose_x - center_x) * 2  # Scale for more pronounced movement
            rel_y = (nose_y - center_y) * 2
            rel_z = (nose_z - center_z) * 2
            
            # Convert to Euler angles (very simplified)
            pitch = -rel_y * 90  # Vertical tilt
            yaw = rel_x * 90     # Horizontal rotation
            roll = rel_z * 45    # Tilt left/right
            
            return [float(pitch), float(yaw), float(roll)]
            
        except Exception as e:
            self.logger.error(f"Error estimating head pose: {e}")
            return [0.0, 0.0, 0.0]  # Default pose
    
    def get_facial_features(self, landmarks_list):
        """
        Extract facial features (eyes, mouth, nose) from landmarks.
        
        Args:
            landmarks_list: List of landmark coordinates
            
        Returns:
            features: Dictionary with facial feature coordinates
        """
        if not landmarks_list:
            return None
        
        # Use the first face
        landmarks = landmarks_list[0]
        
        # Define facial feature indices (based on MediaPipe Face Mesh)
        # Note: These indices are approximate and might need adjustment based on
        # the specific model or implementation you're using
        LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
        RIGHT_EYE = [362, 385, 387, 380, 373, 390, 249, 263]
        MOUTH = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        NOSE = [1, 2, 98, 327]
        
        # Extract coordinates for each feature
        # Make sure to handle out-of-bounds indices
        features = {}
        
        try:
            features['left_eye'] = [landmarks[min(i, len(landmarks)-1)] for i in LEFT_EYE]
        except:
            features['left_eye'] = []
            
        try:
            features['right_eye'] = [landmarks[min(i, len(landmarks)-1)] for i in RIGHT_EYE]
        except:
            features['right_eye'] = []
            
        try:
            features['mouth'] = [landmarks[min(i, len(landmarks)-1)] for i in MOUTH]
        except:
            features['mouth'] = []
            
        try:
            features['nose'] = [landmarks[min(i, len(landmarks)-1)] for i in NOSE]
        except:
            features['nose'] = []
        
        return features
    
    def release(self):
        """Release resources."""
        if MEDIAPIPE_AVAILABLE and self.face_mesh:
            self.face_mesh.close()