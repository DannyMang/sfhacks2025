#!/usr/bin/env python3
"""
Face detection and landmark extraction module using MediaPipe.
"""

import cv2
import numpy as np
import mediapipe as mp
import time

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
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # Track processing time for performance monitoring
        self.process_time = 0
        
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
                if len(landmarks_list) >= self.face_mesh.max_num_faces:
                    break
        
        self.process_time = time.time() - start_time
        
        return landmarks_list, face_rect, processed_image
    
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
        LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
        RIGHT_EYE = [362, 385, 387, 380, 373, 390, 249, 263]
        MOUTH = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        NOSE = [1, 2, 98, 327]
        
        # Extract coordinates for each feature
        features = {
            'left_eye': [landmarks[i] for i in LEFT_EYE],
            'right_eye': [landmarks[i] for i in RIGHT_EYE],
            'mouth': [landmarks[i] for i in MOUTH],
            'nose': [landmarks[i] for i in NOSE]
        }
        
        return features
    
    def get_head_pose(self, landmarks_list, image_shape):
        """
        Estimate 3D head pose from landmarks.
        
        Args:
            landmarks_list: List of landmark coordinates
            image_shape: Shape of the input image (height, width)
            
        Returns:
            rotation_vector: 3D rotation vector
            translation_vector: 3D translation vector
            euler_angles: Euler angles (pitch, yaw, roll) in degrees
        """
        if not landmarks_list:
            return None, None, None
        
        # Use the first face
        landmarks = landmarks_list[0]
        h, w = image_shape[:2]
        
        # 3D model points (simplified)
        model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ])
        
        # 2D image points (corresponding to 3D points)
        # Using key points from MediaPipe (indices may need adjustment)
        image_points = np.array([
            (landmarks[1][0] * w, landmarks[1][1] * h),     # Nose tip (1)
            (landmarks[152][0] * w, landmarks[152][1] * h), # Chin (152)
            (landmarks[33][0] * w, landmarks[33][1] * h),   # Left eye left corner (33)
            (landmarks[263][0] * w, landmarks[263][1] * h), # Right eye right corner (263)
            (landmarks[61][0] * w, landmarks[61][1] * h),   # Left mouth corner (61)
            (landmarks[291][0] * w, landmarks[291][1] * h)  # Right mouth corner (291)
        ], dtype="double")
        
        # Camera matrix (approximate)
        focal_length = w
        center = (w // 2, h // 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        # Distortion coefficients (assume no distortion)
        dist_coeffs = np.zeros((4, 1))
        
        # Solve for pose
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)
        
        if success:
            # Convert rotation vector to Euler angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
                cv2.vconcat((pose_matrix, np.array([[0, 0, 0, 1]]))))
            
            # Convert to degrees
            euler_angles = [float(angle) for angle in euler_angles]
            
            return rotation_vector, translation_vector, euler_angles
        
        return None, None, None
    
    def release(self):
        """Release resources."""
        self.face_mesh.close() 