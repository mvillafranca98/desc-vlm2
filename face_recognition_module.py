#!/usr/bin/env python3
"""
Face Recognition Module
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Try to import face_recognition
try:
    import face_recognition
    import cv2
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    logger.warning("face_recognition library not available")
    FACE_RECOGNITION_AVAILABLE = False


class FaceRecognizer:
    """Face recognition using reference images."""
    
    def __init__(self, reference_faces_dir: str = "reference_faces"):
        self.reference_faces_dir = Path(reference_faces_dir)
        self.known_face_encodings = []
        self.known_face_names = []
        self.enabled = FACE_RECOGNITION_AVAILABLE
        
        if not self.enabled:
            logger.warning("Face recognition disabled (library not available)")
            return
        
        # Create directory if it doesn't exist
        self.reference_faces_dir.mkdir(exist_ok=True)
        
        # Load reference faces
        self._load_reference_faces()
    
    def _load_reference_faces(self):
        """Load reference faces from directory."""
        if not self.enabled:
            return
        
        try:
            logger.info(f"Loading reference faces from {self.reference_faces_dir}")
            
            # Supported image extensions
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            
            face_files = []
            for ext in extensions:
                face_files.extend(self.reference_faces_dir.glob(f"*{ext}"))
                face_files.extend(self.reference_faces_dir.glob(f"*{ext.upper()}"))
            
            if not face_files:
                logger.info("No reference face images found")
                return
            
            for face_file in face_files:
                try:
                    # Load image
                    img = face_recognition.load_image_file(str(face_file))
                    
                    # Get face encodings
                    encodings = face_recognition.face_encodings(img)
                    
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        # Use filename without extension as name
                        name = face_file.stem
                        self.known_face_names.append(name)
                        logger.info(f"Loaded reference face: {name}")
                    else:
                        logger.warning(f"No face found in {face_file}")
                        
                except Exception as e:
                    logger.error(f"Error loading {face_file}: {e}")
            
            logger.info(f"Loaded {len(self.known_face_encodings)} reference faces")
            
        except Exception as e:
            logger.error(f"Error loading reference faces: {e}")
    
    def recognize_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Recognize faces in a frame."""
        if not self.enabled or not self.known_face_encodings:
            return []
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find faces in frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            recognized_faces = []
            
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings,
                    face_encoding,
                    tolerance=0.6
                )
                
                name = "Unknown"
                confidence = 0.0
                
                if True in matches:
                    # Find best match
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings,
                        face_encoding
                    )
                    
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1.0 - face_distances[best_match_index]
                
                # Get bounding box
                top, right, bottom, left = face_location
                
                recognized_faces.append({
                    'name': name,
                    'confidence': float(confidence),
                    'location': {
                        'top': top,
                        'right': right,
                        'bottom': bottom,
                        'left': left
                    }
                })
            
            return recognized_faces
            
        except Exception as e:
            logger.error(f"Error recognizing faces: {e}")
            return []
    
    def draw_faces(self, frame: np.ndarray, faces: List[Dict[str, Any]]) -> np.ndarray:
        """Draw bounding boxes and names on frame."""
        if not faces:
            return frame
        
        for face in faces:
            loc = face['location']
            name = face['name']
            confidence = face['confidence']
            
            # Draw rectangle
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(
                frame,
                (loc['left'], loc['top']),
                (loc['right'], loc['bottom']),
                color,
                2
            )
            
            # Draw label
            label = f"{name}"
            if name != "Unknown":
                label += f" ({confidence:.2f})"
            
            cv2.putText(
                frame,
                label,
                (loc['left'], loc['top'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        return frame

