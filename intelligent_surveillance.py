import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine, euclidean
import time
import json


class intelligent_surveillance:
    def __init__(self):
        print("🚀 Initializing Enhanced Face Attendance System...")

        # Adjusted thresholds for better compatibility
        self.duplicate_threshold = 0.50
        self.match_similarity = 0.40  # CHANGED FROM 0.45 to 0.40
        self.match_distance = 1.0
        self.min_face_size = 40
        self.confidence_threshold = 0.6
        # Performance optimization
        self.frame_skip_interval = 2
        self.frame_count = 0

        # Initialize detectors
        self.detector = None
        self.detector_type = None
        self.facenet = None
        self.face_quality_threshold = 0.6  # Lower quality threshold

        # Initialize with better compatibility
        self.initialize_compatible_detectors()

        # Enhanced FaceNet
        try:
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
            print("✅ Enhanced FaceNet initialized successfully")
        except Exception as e:
            print(f"❌ FaceNet failed: {e}")
            self.facenet = None

        print("🎯 Enhanced face system ready!")

    def initialize_compatible_detectors(self):
        """Initialize detectors with better compatibility"""
        # Try multiple detector types
        detectors = [
            self.initialize_mtcnn_detector,
            self.initialize_haar_detector
        ]

        for detector_init in detectors:
            try:
                if detector_init():
                    return
            except Exception as e:
                print(f"❌ Detector initialization failed: {e}")
                continue

        print("❌ All detector initializations failed")

    def initialize_mtcnn_detector(self):
        """Initialize MTCNN with better compatibility"""
        try:
            self.detector = MTCNN(
                min_face_size=30,  # Smaller minimum size
                thresholds=[0.6, 0.7, 0.7],  # Lower thresholds
                factor=0.709,
                keep_all=False,  # Single face for better performance
                device='cpu'
            )
            self.detector_type = 'mtcnn'
            print("✅ MTCNN initialized successfully")
            return True
        except Exception as e:
            print(f"❌ MTCNN initialization failed: {e}")
            return False

    def initialize_haar_detector(self):
        """Initialize Haar Cascade as fallback"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            if not self.detector.empty():
                self.detector_type = 'haar'
                print("✅ Haar Cascade initialized as fallback")
                return True
        except Exception as e:
            print(f"❌ Haar Cascade initialization failed: {e}")
        return False


    def parse_face_embedding(self, embedding_value):
        """Safely parse a face embedding stored as a list, numpy array, or JSON string."""
        if isinstance(embedding_value, np.ndarray):
            return embedding_value.astype(np.float32)

        if isinstance(embedding_value, list):
            return np.array(embedding_value, dtype=np.float32)

        if isinstance(embedding_value, str):
            embedding_value = embedding_value.strip()
            if not embedding_value:
                raise ValueError("Empty face embedding")
            return np.array(json.loads(embedding_value), dtype=np.float32)

        raise ValueError("Unsupported face embedding format")

    def enhance_image_quality(self, frame):
        """Simple image enhancement"""
        try:
            if len(frame.shape) == 3:
                # Simple contrast enhancement
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:, :, 0])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return frame
        except Exception as e:
            print(f"Image enhancement error: {e}")
            return frame

    def detect_faces_compatible(self, frame):
        """Compatible face detection that works with low resolution"""
        if self.detector is None:
            return []

        try:
            # Resize frame if too small
            if frame.shape[0] < 240 or frame.shape[1] < 320:
                frame = cv2.resize(frame, (320, 240))

            enhanced_frame = self.enhance_image_quality(frame)

            if self.detector_type == 'mtcnn':
                return self.detect_faces_mtcnn_compatible(enhanced_frame)
            else:
                return self.detect_faces_haar_compatible(enhanced_frame)

        except Exception as e:
            print(f"❌ Face detection error: {e}")
            return []

    def detect_faces_mtcnn_compatible(self, frame):
        """MTCNN detection with better compatibility"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            results = self.detector.detect(rgb_frame)

            if results is None:
                return []

            boxes, confidences = results[0], results[1]

            if boxes is None or len(boxes) == 0:
                return []

            detected_faces = []
            for i, (box, confidence) in enumerate(zip(boxes, confidences)):
                if confidence > self.confidence_threshold:
                    x1, y1, x2, y2 = box.astype(int)
                    w, h = x2 - x1, y2 - y1

                    # More lenient size validation
                    if w >= 40 and h >= 40:  # Smaller minimum size
                        detected_faces.append({
                            'box': [x1, y1, w, h],
                            'confidence': confidence,
                            'keypoints': None
                        })

            print(f"🔍 MTCNN detected {len(detected_faces)} faces")
            return detected_faces

        except Exception as e:
            print(f"❌ MTCNN detection failed: {e}")
            return []

    def detect_faces_haar_compatible(self, frame):
        """Haar cascade detection with better compatibility"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,  # Fewer neighbors for more detection
                minSize=(40, 40),  # Smaller minimum size
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            detected_faces = []
            for (x, y, w, h) in faces:
                detected_faces.append({
                    'box': [int(x), int(y), int(w), int(h)],
                    'confidence': 0.8,  # Default confidence
                    'keypoints': {}
                })

            print(f"🔍 Haar detected {len(detected_faces)} faces")
            return detected_faces

        except Exception as e:
            print(f"❌ Haar detection error: {e}")
            return []

    def get_face_embedding(self, frame, is_capture=False):
        """Compatible face embedding extraction"""
        try:
            if self.detector is None or self.facenet is None:
                return None, "System not ready"

            print(f"📷 Processing frame: {frame.shape}")

            # Detect faces
            faces = self.detect_faces_compatible(frame)
            if not faces:
                return None, "No face detected - ensure face is visible"

            # Use the best face
            face = max(faces, key=lambda f: f['confidence'])
            x, y, w, h = face['box']

            print(f"👤 Face found: {w}x{h} at ({x},{y})")

            # Extract face with padding
            padding = int(min(w, h) * 0.1)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                return None, "Invalid face region"

            # Simple preprocessing
            face_resized = cv2.resize(face_roi, (160, 160))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_normalized = face_rgb.astype(np.float32) / 255.0
            face_normalized = (face_normalized - 0.5) / 0.5

            # Convert to tensor
            face_tensor = torch.from_numpy(face_normalized.transpose(2, 0, 1)).unsqueeze(0).float()

            # Generate embedding
            with torch.no_grad():
                embedding = self.facenet(face_tensor).numpy().flatten()

            # Normalize embedding
            embedding_norm = embedding / np.linalg.norm(embedding)

            print(f"✅ Embedding generated: {len(embedding_norm)} dimensions")
            return embedding_norm.tolist(), (x, y, w, h)

        except Exception as e:
            print(f"❌ Embedding extraction error: {e}")
            return None, f"Detection error: {str(e)}"

    def process_frame(self, frame, is_capture=False):
        """Compatible frame processing"""
        try:
            print(f"🎯 Processing frame - capture: {is_capture}")

            embedding, message_or_box = self.get_face_embedding(frame, is_capture)

            if embedding is None:
                return {
                    'success': False,
                    'message': message_or_box
                }

            # Prepare result
            if isinstance(message_or_box, tuple):
                box = [int(x) for x in message_or_box]
            else:
                box = message_or_box

            result = {
                'success': True,
                'box': box,
                'message': 'Face detected successfully'
            }

            if is_capture:
                result['embedding'] = embedding
                result['message'] = 'Face captured for registration'

            return self.ensure_json_serializable(result)

        except Exception as e:
            print(f"❌ Process frame error: {e}")
            return {
                'success': False,
                'message': 'Error processing frame'
            }

    def find_matching_user(self, embedding, users):
        """Find matching user from database with improved similarity calculation"""
        try:
            embedding = np.array(embedding, dtype=np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize

            best_match = None
            best_similarity = 0
            best_user_name = "Unknown"

            for user in users:
                if user.get('face_embedding'):
                    try:
                        db_embedding_str = user['face_embedding'].strip()

                        # Safely parse embedding string/list without unsafe eval()
                        if db_embedding_str.startswith('[') and db_embedding_str.endswith(']'):
                            db_embedding = self.parse_face_embedding(db_embedding_str)
                        else:
                            continue  # Skip invalid embeddings

                        # Normalize database embedding
                        db_embedding = db_embedding / np.linalg.norm(db_embedding)

                        # ✅ IMPROVED SIMILARITY CALCULATION
                        similarity = 1 - cosine(embedding, db_embedding)

                        # ✅ ADD EUCLIDEAN DISTANCE AS FALLBACK
                        euclidean_sim = 1 / (1 + euclidean(embedding, db_embedding))

                        # Use the better similarity score
                        final_similarity = max(similarity, euclidean_sim)

                        user_name = user.get('Name', 'Unknown')
                        print(
                            f"👤 {user_name}: cosine={similarity:.3f}, euclidean={euclidean_sim:.3f}, final={final_similarity:.3f}")

                        if final_similarity > best_similarity:
                            best_similarity = final_similarity
                            best_match = user
                            best_user_name = user_name

                    except Exception as e:
                        print(f"⚠️ Error processing {user.get('Name')}: {e}")
                        continue

            # ✅ DYNAMIC MATCHING THRESHOLD
            dynamic_threshold = max(self.match_similarity, 0.3)  # Minimum 0.3 threshold

            if best_match and best_similarity > dynamic_threshold:
                print(
                    f"✅ MATCH: {best_user_name} (similarity: {best_similarity:.3f}, threshold: {dynamic_threshold:.3f})")
                return best_match
            else:
                print(f"❌ NO MATCH: {best_user_name} (best: {best_similarity:.3f}, needed: {dynamic_threshold:.3f})")
                return None

        except Exception as e:
            print(f"❌ Matching error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def is_duplicate_face(self, new_embedding, existing_faces, threshold=None):
        """Duplicate face checking with improved similarity calculation"""
        try:
            if threshold is None:
                threshold = self.duplicate_threshold

            new_embedding = np.array(new_embedding, dtype=np.float32)
            new_embedding = new_embedding / np.linalg.norm(new_embedding)

            best_similarity = 0
            best_match = None

            for user in existing_faces:
                if user.get('face_embedding'):
                    try:
                        db_embedding_str = user['face_embedding'].strip()
                        if db_embedding_str.startswith('[') and db_embedding_str.endswith(']'):
                            db_embedding = self.parse_face_embedding(db_embedding_str)
                            db_embedding = db_embedding / np.linalg.norm(db_embedding)

                            # ✅ USE BOTH SIMILARITY METRICS
                            cosine_sim = 1 - cosine(new_embedding, db_embedding)
                            euclidean_sim = 1 / (1 + euclidean(new_embedding, db_embedding))
                            similarity = max(cosine_sim, euclidean_sim)

                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = user.get('Name')

                            if similarity > threshold:
                                return True, user.get('Name'), similarity

                    except Exception as e:
                        print(f"⚠️ Error comparing with {user.get('Name')}: {e}")
                        continue

            print(f"📊 Duplicate check - Best similarity: {best_similarity:.3f} (threshold: {threshold:.3f})")
            return False, best_match, best_similarity

        except Exception as e:
            print(f"❌ Duplicate check error: {e}")
            return False, None, 0.0
    def ensure_json_serializable(self, obj):
        """Make objects JSON serializable"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.ensure_json_serializable(x) for x in obj]
        else:
            return obj