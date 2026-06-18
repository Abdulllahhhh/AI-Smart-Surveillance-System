import threading
import time
from flask import Flask, render_template, Response, session, request, redirect, url_for, jsonify, send_file
from werkzeug.utils import secure_filename
# import datetime
import csv
import os
import cv2
import logging
from logging.config import dictConfig
from intelligent_surveillance import intelligent_surveillance
import numpy as np
from datetime import datetime, time as dt_time  # Rename to avoid conflict
import json

from scipy.spatial.distance import cosine, euclidean
from werkzeug.security import generate_password_hash, check_password_hash
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
import io

import hashlib

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s: %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'WARNING',
        'handlers': ['wsgi']
    }
})

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

log = logging.getLogger('werkzeug')
log.setLevel(logging.INFO)

DEFAULT_ADMIN_PASSWORD = os.environ.get('DEFAULT_ADMIN_PASSWORD', 'change-me')
DEFAULT_CHECK_IN_START = "07:30"
DEFAULT_CHECK_IN_END = "09:30"
DEFAULT_CHECK_OUT_START = "16:00"
DEFAULT_CHECK_OUT_END = "18:00"
DEFAULT_WORKING_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]  # Default working days
DEFAULT_DETECTION_CONFIDENCE = 0.5
DEFAULT_MATCH_DISTANCE = 0.7
DEFAULT_MATCH_SIMILARITY = 0.8
DEFAULT_MIN_FACE_SIZE = 50


# Global variables
camera = None
last_frame = None
camera_lock = threading.Lock()
FACE_DATA_FILE = 'face_data.csv'
is_camera_initialized = False
face_system = intelligent_surveillance()

is_recording = False
video_writer = None
recording_thread = None
recording_start_time = None


@app.route('/recordings')
def recordings_page():
    """Serve the recordings history page"""
    is_admin = session.get('user_role') == 'admin'
    return render_template('view_history_recording.html',
                           active_page='reports',
                           is_admin=is_admin)


@app.route('/api/recordings')
def get_recordings_list():
    """API to get list of all recordings"""
    try:
        recordings_dir = 'recordings'
        if not os.path.exists(recordings_dir):
            return jsonify({'success': True, 'recordings': []})

        recordings = []
        for filename in os.listdir(recordings_dir):
            if filename.endswith('.mp4'):
                filepath = os.path.join(recordings_dir, filename)
                file_stats = os.stat(filepath)

                # Calculate duration (you might want to extract this from video metadata)
                file_size_mb = round(file_stats.st_size / (1024 * 1024), 2)

                recordings.append({
                    'filename': filename,
                    'filepath': filepath,
                    'size': file_size_mb,
                    'size_bytes': file_stats.st_size,
                    'created_time': datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                    'duration': 'Unknown'  # You can enhance this with video metadata
                })

        # Sort by creation time (newest first)
        recordings.sort(key=lambda x: x['created_time'], reverse=True)

        return jsonify({'success': True, 'recordings': recordings})

    except Exception as e:
        print(f"❌ Error getting recordings: {str(e)}")
        return jsonify({'success': False, 'message': f'Error getting recordings: {str(e)}'})


@app.route('/api/recordings/<filename>')
def serve_recording(filename):
    """Serve video file for playback with proper headers"""
    try:
        filename = secure_filename(filename)
        recordings_dir = 'recordings'
        filepath = os.path.join(recordings_dir, filename)

        # Security check to prevent directory traversal
        if not os.path.exists(filepath) or '..' in filename or not filename.endswith('.mp4'):
            return jsonify({'success': False, 'message': 'File not found'}), 404

        # Get file size for Content-Length header
        file_size = os.path.getsize(filepath)

        # Set proper headers for video streaming
        response = send_file(
            filepath,
            as_attachment=False,
            mimetype='video/mp4'
        )

        # Enable range requests for seeking
        response.headers.add('Accept-Ranges', 'bytes')
        response.headers.add('Content-Length', str(file_size))

        return response

    except Exception as e:
        print(f"❌ Error serving recording: {str(e)}")
        return jsonify({'success': False, 'message': 'Error serving recording'}), 500


@app.route('/video/<filename>')
def stream_video(filename):
    """Stream video with better compatibility"""
    try:
        filename = secure_filename(filename)
        recordings_dir = 'recordings'
        filepath = os.path.join(recordings_dir, filename)

        # Security check
        if not os.path.exists(filepath) or '..' in filename or not filename.endswith('.mp4'):
            return "File not found", 404

        # Use send_file with conditional response for better compatibility
        response = send_file(
            filepath,
            as_attachment=False,
            mimetype='video/mp4',
            conditional=True  # This enables range requests automatically
        )

        response.headers.update({
            'Accept-Ranges': 'bytes',
            'Cache-Control': 'no-cache',
            'Content-Disposition': f'inline; filename="{filename}"'
        })

        return response

    except Exception as e:
        print(f"❌ Error streaming video: {str(e)}")
        return "Error streaming video", 500

@app.route('/api/recordings/<filename>', methods=['DELETE'])
def delete_recording(filename):
    """Delete a recording file"""
    try:
        filename = secure_filename(filename)
        recordings_dir = 'recordings'
        filepath = os.path.join(recordings_dir, filename)

        # Security check
        if not os.path.exists(filepath) or '..' in filename or not filename.endswith('.mp4'):
            return jsonify({'success': False, 'message': 'File not found'}), 404

        # Delete the file
        os.remove(filepath)
        print(f"🗑️ Recording deleted: {filename}")

        return jsonify({'success': True, 'message': 'Recording deleted successfully'})

    except Exception as e:
        print(f"❌ Error deleting recording: {str(e)}")
        return jsonify({'success': False, 'message': f'Error deleting recording: {str(e)}'}), 500


@app.route('/api/convert_video/<filename>', methods=['POST'])
def convert_video(filename):
    """Convert video to browser-compatible format"""
    try:
        filename = secure_filename(filename)
        recordings_dir = 'recordings'
        filepath = os.path.join(recordings_dir, filename)

        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'File not found'})

        # Create converted filename
        name_without_ext = os.path.splitext(filename)[0]
        converted_filename = f"{name_without_ext}_converted.mp4"
        converted_path = os.path.join(recordings_dir, converted_filename)

        # Use ffmpeg to convert (if available)
        try:
            import subprocess
            # Convert to H.264 codec for browser compatibility
            cmd = [
                'ffmpeg', '-i', filepath,
                '-c:v', 'libx264',  # H.264 video codec
                '-c:a', 'aac',  # AAC audio codec
                '-movflags', '+faststart',  # Enable streaming
                '-y',  # Overwrite output file
                converted_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return jsonify({
                    'success': True,
                    'message': 'Video converted successfully',
                    'converted_filename': converted_filename
                })
            else:
                return jsonify({
                    'success': False,
                    'message': f'Conversion failed: {result.stderr}'
                })

        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'FFmpeg not available: {str(e)}'
            })

    except Exception as e:
        print(f"❌ Error converting video: {str(e)}")
        return jsonify({'success': False, 'message': f'Error converting video: {str(e)}'})

@app.route('/api/recordings/bulk-delete', methods=['POST'])
def bulk_delete_recordings():
    """Delete multiple recordings at once"""
    try:
        data = request.json
        filenames = data.get('filenames', [])

        if not filenames:
            return jsonify({'success': False, 'message': 'No files specified'})

        recordings_dir = 'recordings'
        deleted_count = 0
        errors = []

        for filename in filenames:
            try:
                filename = secure_filename(filename)
                filepath = os.path.join(recordings_dir, filename)

                # Security check
                if os.path.exists(filepath) and not '..' in filename and filename.endswith('.mp4'):
                    os.remove(filepath)
                    deleted_count += 1
                    print(f"🗑️ Recording deleted: {filename}")
                else:
                    errors.append(f"Invalid file: {filename}")

            except Exception as e:
                errors.append(f"Error deleting {filename}: {str(e)}")

        message = f"Successfully deleted {deleted_count} recording(s)"
        if errors:
            message += f". Errors: {', '.join(errors)}"

        return jsonify({
            'success': True,
            'message': message,
            'deleted_count': deleted_count,
            'errors': errors
        })

    except Exception as e:
        print(f"❌ Error in bulk delete: {str(e)}")
        return jsonify({'success': False, 'message': f'Error deleting recordings: {str(e)}'}), 500


@app.route('/play/<filename>')
def play_video(filename):
    """Simple video playback route"""
    try:
        filename = secure_filename(filename)
        recordings_dir = 'recordings'
        filepath = os.path.join(recordings_dir, filename)

        # Security check
        if not os.path.exists(filepath) or '..' in filename or not filename.endswith('.mp4'):
            return "File not found", 404

        return send_file(filepath)

    except Exception as e:
        print(f"❌ Error playing video: {str(e)}")
        return "Error playing video", 500

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Start recording live video feed"""
    global is_recording, video_writer, recording_start_time

    try:
        if is_recording:
            return jsonify({'success': False, 'message': 'Recording already in progress'})

        # Create recordings directory if it doesn't exist
        recordings_dir = 'recordings'
        if not os.path.exists(recordings_dir):
            os.makedirs(recordings_dir)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'recording_{timestamp}.mp4'
        filepath = os.path.join(recordings_dir, filename)

        # Initialize video writer with BROWSER-COMPATIBLE codec
        frame_width = 640
        frame_height = 480
        fps = 15.0

        # Try different codecs in order of compatibility
        codecs = [
            'avc1',  # H.264 - Most compatible
            'mp4v',  # MPEG-4 - Fallback
            'XVID'  # XVID - Another fallback
        ]

        video_writer = None
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video_writer = cv2.VideoWriter(filepath, fourcc, fps, (frame_width, frame_height))
                if video_writer.isOpened():
                    print(f"✅ Using codec: {codec}")
                    break
                else:
                    video_writer = None
            except Exception as e:
                print(f"❌ Codec {codec} failed: {e}")
                continue

        if video_writer is None:
            return jsonify({'success': False, 'message': 'Failed to initialize video writer with any codec'})

        is_recording = True
        recording_start_time = datetime.now()

        # Start recording thread
        recording_thread = threading.Thread(target=record_frames)
        recording_thread.daemon = True
        recording_thread.start()

        print(f"🎥 Recording started: {filename}")
        return jsonify({
            'success': True,
            'message': 'Recording started',
            'filename': filename,
            'start_time': recording_start_time.strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        print(f"❌ Error starting recording: {str(e)}")
        return jsonify({'success': False, 'message': f'Error starting recording: {str(e)}'})


@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """Stop recording and save video"""
    global is_recording, video_writer, recording_start_time

    try:
        if not is_recording:
            return jsonify({'success': False, 'message': 'No recording in progress'})

        is_recording = False

        # Wait for thread to finish
        if recording_thread and recording_thread.is_alive():
            recording_thread.join(timeout=2.0)

        # Release video writer
        if video_writer:
            video_writer.release()
            video_writer = None

        recording_duration = datetime.now() - recording_start_time
        recording_start_time = None

        print(f"🛑 Recording stopped. Duration: {recording_duration}")
        return jsonify({
            'success': True,
            'message': 'Recording stopped and saved',
            'duration': str(recording_duration)
        })

    except Exception as e:
        print(f"❌ Error stopping recording: {str(e)}")
        return jsonify({'success': False, 'message': f'Error stopping recording: {str(e)}'})


@app.route('/recording_status', methods=['GET'])
def recording_status():
    """Get current recording status"""
    global is_recording, recording_start_time

    status = {
        'is_recording': is_recording,
        'start_time': recording_start_time.strftime('%Y-%m-%d %H:%M:%S') if recording_start_time else None,
        'duration': str(datetime.now() - recording_start_time) if recording_start_time else None
    }

    return jsonify(status)


def record_frames():
    """Background thread to record frames"""
    global is_recording, video_writer, last_frame

    print("🎬 Recording thread started...")
    frames_recorded = 0

    while is_recording:
        try:
            if last_frame is not None and video_writer is not None:
                # Resize frame to consistent size
                frame = cv2.resize(last_frame, (640, 480))
                video_writer.write(frame)
                frames_recorded += 1

                # Small delay to maintain frame rate
                time.sleep(0.066)  # ~15 FPS
            else:
                time.sleep(0.1)

        except Exception as e:
            print(f"❌ Error in recording thread: {e}")
            break

    print(f"📹 Recording thread stopped. Frames recorded: {frames_recorded}")


@app.route('/get_recordings', methods=['GET'])
def get_recordings():
    """Get list of saved recordings"""
    try:
        recordings_dir = 'recordings'
        if not os.path.exists(recordings_dir):
            return jsonify({'success': True, 'recordings': []})

        recordings = []
        for filename in os.listdir(recordings_dir):
            if filename.endswith('.mp4'):
                filepath = os.path.join(recordings_dir, filename)
                file_stats = os.stat(filepath)
                recordings.append({
                    'filename': filename,
                    'filepath': filepath,
                    'size': file_stats.st_size,
                    'created_time': datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                })

        # Sort by creation time (newest first)
        recordings.sort(key=lambda x: x['created_time'], reverse=True)

        return jsonify({'success': True, 'recordings': recordings})

    except Exception as e:
        print(f"❌ Error getting recordings: {str(e)}")
        return jsonify({'success': False, 'message': f'Error getting recordings: {str(e)}'})


def ensure_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(x) for x in obj]
    else:
        return obj


def parse_face_embedding(embedding_value):
    """Safely parse a face embedding stored as a list or JSON string."""
    if isinstance(embedding_value, np.ndarray):
        return embedding_value.astype(np.float32)

    if isinstance(embedding_value, list):
        return np.array(embedding_value, dtype=np.float32)

    if isinstance(embedding_value, str):
        return np.array(json.loads(embedding_value), dtype=np.float32)

    raise ValueError('Unsupported face embedding format')

@app.route('/recognize_face', methods=['POST'])
def recognize_face_enhanced():
    """Enhanced face recognition with better accuracy"""
    global camera, last_frame
    print("🎯 Enhanced face recognition started...")

    if camera is None:
        print("Initializing camera for enhanced recognition...")
        initialize_camera()

    if camera is None or not camera.isOpened():
        print("Camera not available for enhanced recognition")
        return jsonify({'success': False, 'message': 'Camera not available'})

    try:
        # Read frame from camera
        success, frame = camera.read()
        if not success:
            print("Failed to capture frame for enhanced recognition")
            return jsonify({'success': False, 'message': 'Failed to capture frame'})

        # Store the frame
        last_frame = frame.copy()

        print("🔄 Enhanced processing frame for recognition...")
        # Use optimized processing for recognition
        result = face_system.process_frame(frame, is_capture=True)

        if not result['success']:
            print(f"Enhanced recognition failed: {result['message']}")
            return jsonify({'success': False, 'message': result['message']})

        # Get registered faces
        registered_faces = read_face_data()

        if not registered_faces:
            print("No faces registered in database")
            return jsonify({
                'success': True,
                'box': result['box'],
                'message': 'Face detected but no faces registered in database'
            })

        print(f"🔍 Enhanced checking against {len(registered_faces)} registered faces...")

        # Find matching face using enhanced matching
        matched_face = face_system.find_matching_user(result['embedding'], registered_faces)

        # ✅ ADD DEBUG INFO
        if matched_face:
            # Calculate detailed similarity for debugging
            embedding_array = np.array(result['embedding'], dtype=np.float32)
            db_embedding_str = matched_face['face_embedding'].strip()
            db_embedding = parse_face_embedding(db_embedding_str)

            embedding_array = embedding_array / np.linalg.norm(embedding_array)
            db_embedding = db_embedding / np.linalg.norm(db_embedding)

            cosine_sim = 1 - cosine(embedding_array, db_embedding)
            euclidean_dist = euclidean(embedding_array, db_embedding)
            euclidean_sim = 1 / (1 + euclidean_dist)

            final_similarity = max(cosine_sim, euclidean_sim)
            confidence = min(final_similarity * 1.3, 0.99)  # More generous scaling

            print(f"✅ ENHANCED RECOGNITION: {matched_face['Name']}")
            print(f"   Cosine Similarity: {cosine_sim:.3f}")
            print(f"   Euclidean Similarity: {euclidean_sim:.3f}")
            print(f"   Final Similarity: {final_similarity:.3f}")
            print(f"   Confidence: {confidence:.3f}")

            return jsonify({
                'success': True,
                'box': result['box'],
                'recognized_user': {
                    'name': matched_face['Name'],
                    'id': matched_face['ID']
                },
                'confidence': confidence,
                'similarity': final_similarity,
                'cosine_similarity': cosine_sim,
                'euclidean_similarity': euclidean_sim,
                'message': f"Recognized: {matched_face['Name']} ({(confidence * 100):.1f}% confidence)"
            })
        else:
            print("❌ Face detected but not recognized")
            print(f"   Try lowering match_similarity threshold (current: {face_system.match_similarity})")
            return jsonify({
                'success': True,
                'box': result['box'],
                'message': 'Face detected but not recognized in database',
                'debug': 'Try registering the face or adjusting similarity thresholds'
            })

    except Exception as e:
        print(f"❌ Error in enhanced face recognition: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Error in enhanced face recognition'})


@app.route('/register_face', methods=['POST'])
def register_face():
    """Register a new face in the face database with enhanced duplicate checking"""
    try:
        face_data = request.json
        print(f"🔐 Registering face for: {face_data.get('name')}")

        # Validate required fields
        if not face_data.get('name') or not face_data.get('id') or not face_data.get('face_embedding'):
            return jsonify({'success': False, 'message': 'Name, ID, and face embedding are required'})

        # Validate embedding format
        try:
            embedding_array = np.array(face_data['face_embedding'], dtype=np.float32)
            if embedding_array.size < 100 or embedding_array.size > 1000:
                return jsonify({'success': False, 'message': 'Invalid face embedding format'})
        except (ValueError, TypeError) as e:
            return jsonify({'success': False, 'message': 'Invalid face embedding format'})

        # Check if face ID already exists
        existing_faces = read_face_data()
        if any(face['ID'] == face_data['id'] for face in existing_faces):
            return jsonify({'success': False, 'message': 'Face ID already exists'})

        # ENHANCED DUPLICATE FACE EMBEDDING CHECK
        try:
            is_duplicate, existing_name, similarity_score = face_system.is_duplicate_face(
                face_data['face_embedding'],
                existing_faces,
                threshold=0.65  # Slightly lower threshold for stricter checking
            )

            if is_duplicate:
                similarity_percent = similarity_score * 100
                return jsonify({
                    'success': False,
                    'message': f'🚫 This face is already registered as: {existing_name} (similarity: {similarity_percent:.1f}%)',
                    'duplicate': True,
                    'existing_user': existing_name,
                    'similarity_score': similarity_score
                })

            print(f"✅ Duplicate check passed - similarity with closest match: {similarity_score:.4f}")

        except Exception as e:
            print(f"⚠️ Duplicate check failed, proceeding with registration: {e}")
            # In case of error, we can choose to be safe and reject registration
            return jsonify({
                'success': False,
                'message': 'Cannot verify face uniqueness. Please try again.'
            })

        # Save face data
        save_face_data({
            'Name': face_data['name'],
            'ID': face_data['id'],
            'face_embedding': face_data['face_embedding'],
            'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        print(f"✅ Face registered successfully: {face_data['name']}")
        return jsonify({'success': True, 'message': 'Face registered successfully'})

    except Exception as e:
        print(f"❌ Error registering face: {str(e)}")
        return jsonify({'success': False, 'message': 'Error registering face'})

def validate_and_clean_face_data():
    """Validate and clean face data to remove corrupted entries"""
    faces = []
    corrupted_count = 0

    if not os.path.exists(FACE_DATA_FILE):
        print("📝 No face database found - will create new one")
        return faces

    try:
        with open(FACE_DATA_FILE, 'r', newline='', encoding='utf-8') as file:
            # Read all lines first to handle potential CSV issues
            lines = file.readlines()

        # Reset file pointer and read as CSV
        with open(FACE_DATA_FILE, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            # Get expected fieldnames
            expected_fields = ['Name', 'ID', 'face_embedding', 'registration_date']

            for row_num, row in enumerate(reader, 1):
                try:
                    # Skip empty rows
                    if not row or all(value == '' for value in row.values()):
                        corrupted_count += 1
                        continue

                    # Check required fields
                    if not row.get('Name') or not row.get('ID') or not row.get('face_embedding'):
                        print(f"⚠️ Row {row_num}: Missing required fields - removing")
                        corrupted_count += 1
                        continue

                    # Validate embedding format
                    try:
                        embedding_str = row['face_embedding'].strip()
                        if not embedding_str or embedding_str == 'null' or embedding_str == 'None':
                            print(f"⚠️ Row {row_num}: Empty embedding for {row['Name']} - removing")
                            corrupted_count += 1
                            continue

                        embedding_data = json.loads(embedding_str)
                        embedding_array = np.array(embedding_data, dtype=np.float32)

                        # Check if embedding has reasonable shape (should be 512-dimensional)
                        if embedding_array.size < 100 or embedding_array.size > 1000:
                            print(
                                f"⚠️ Row {row_num}: Invalid embedding size {embedding_array.size} for {row['Name']} - removing")
                            corrupted_count += 1
                            continue

                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        print(f"⚠️ Row {row_num}: Corrupted embedding for {row['Name']}: {e} - removing")
                        corrupted_count += 1
                        continue

                    # Create clean row with only expected fields
                    clean_row = {
                        'Name': row['Name'].strip(),
                        'ID': row['ID'].strip(),
                        'face_embedding': row['face_embedding'],
                        'registration_date': row.get('registration_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    }

                    faces.append(clean_row)
                    print(f"✅ Row {row_num}: Validated {row['Name']}")

                except Exception as e:
                    print(f"❌ Row {row_num}: Unexpected error: {e} - removing")
                    corrupted_count += 1
                    continue

    except Exception as e:
        print(f"❌ Error reading face data file: {str(e)}")
        # If file is completely corrupted, return empty list
        return []

    if corrupted_count > 0:
        print(f"🔄 Removed {corrupted_count} corrupted face entries")
        try:
            # Rewrite the file with clean data
            with open(FACE_DATA_FILE, 'w', newline='', encoding='utf-8') as file:
                fieldnames = ['Name', 'ID', 'face_embedding', 'registration_date']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for face in faces:
                    writer.writerow(face)
            print(f"💾 Saved {len(faces)} clean face entries")
        except Exception as e:
            print(f"❌ Error saving cleaned data: {e}")
            # If we can't save, at least return the cleaned data
            return faces
    else:
        print(f"✅ Database validated - {len(faces)} faces are clean")

    return faces

# ✅ ADD THIS - Run data cleanup when app starts with error handling
try:
    print("🔄 Validating face database...")
    validate_and_clean_face_data()
    print("✅ Database validation complete")
except Exception as e:
    print(f"⚠️ Database validation failed: {e}")
    print("⚠️ Continuing with potentially corrupted data...")
def read_face_data():
    """Read registered face data with validation"""
    return validate_and_clean_face_data()


def save_face_data(face_data):
    """Save face data to CSV with duplicate prevention"""
    file_exists = os.path.exists(FACE_DATA_FILE)

    # Read existing data to prevent duplicates
    existing_faces = []
    if file_exists:
        existing_faces = read_face_data()

    # Check for ID duplicates (additional safety)
    if any(face['ID'] == face_data['ID'] for face in existing_faces):
        raise ValueError(f"Face ID {face_data['ID']} already exists")

    with open(FACE_DATA_FILE, 'a', newline='', encoding='utf-8') as file:
        fieldnames = ['Name', 'ID', 'face_embedding', 'registration_date']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(face_data)
    print(f"💾 Saved face data for: {face_data['Name']}")


def initialize_camera(camera_index=0, max_retries=3):
    """Stable camera initialization with better resolution"""
    global camera, is_camera_initialized

    with camera_lock:
        if is_camera_initialized and camera is not None and camera.isOpened():
            return camera

        for attempt in range(max_retries):
            try:
                # Release existing camera
                if camera is not None:
                    try:
                        camera.release()
                    except:
                        pass
                    camera = None

                time.sleep(1)

                print(f"🔄 Camera initialization attempt {attempt + 1}/{max_retries}")

                # Try different backends
                backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]

                for backend in backends:
                    try:
                        camera = cv2.VideoCapture(camera_index, backend)
                        if camera.isOpened():
                            # SET HIGHER RESOLUTION
                            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            camera.set(cv2.CAP_PROP_FPS, 15)

                            success, frame = camera.read()
                            if success and frame is not None:
                                print(f"✅ Camera initialized: {frame.shape[1]}x{frame.shape[0]}")
                                is_camera_initialized = True
                                return camera
                            else:
                                camera.release()
                                camera = None
                    except Exception as e:
                        print(f"❌ Backend {backend} failed: {e}")
                        if camera is not None:
                            camera.release()
                            camera = None

            except Exception as e:
                print(f"❌ Camera initialization attempt {attempt + 1} failed: {e}")
                if camera is not None:
                    try:
                        camera.release()
                    except:
                        pass
                    camera = None

        print("❌ All camera initialization attempts failed")
        is_camera_initialized = False
        return None

def release_camera():
    """Safely release camera resources"""
    global camera, last_frame, is_camera_initialized

    with camera_lock:
        try:
            if camera is not None:
                camera.release()
                print("✅ Camera released successfully")
            camera = None
            last_frame = None
            is_camera_initialized = False
        except Exception as e:
            print(f"⚠️ Warning releasing camera: {str(e)}")
            camera = None
            last_frame = None
            is_camera_initialized = False


def get_stable_frame(max_attempts=5):
    """Get a stable frame from camera with retry logic"""
    global camera, last_frame

    if camera is None or not camera.isOpened():
        if not initialize_camera():
            return None

    for attempt in range(max_attempts):
        try:
            with camera_lock:
                if camera is None or not camera.isOpened():
                    continue

                success, frame = camera.read()
                if success and frame is not None:
                    # Resize for consistency
                    if frame.shape[0] != 480 or frame.shape[1] != 640:
                        frame = cv2.resize(frame, (640, 480))
                    last_frame = frame.copy()
                    return frame
                else:
                    print(f"⚠️ Failed to read frame (attempt {attempt + 1})")
                    time.sleep(0.1)

        except Exception as e:
            print(f"❌ Error reading frame (attempt {attempt + 1}): {e}")
            time.sleep(0.1)

    print("❌ Could not get stable frame")
    return None


def generate_frames():
    """Generate video frames with simple error handling"""
    global camera, last_frame

    # Initialize camera if needed
    if camera is None:
        if not initialize_camera():
            # Return error frame if camera fails
            error_frame = get_error_frame()
            while True:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
            return

    try:
        while True:
            with camera_lock:
                if camera is None or not camera.isOpened():
                    break

                success, frame = camera.read()
                if not success:
                    print("⚠️ Failed to read frame")
                    break

                # Store last frame for other operations
                last_frame = frame.copy()

                # Resize frame for consistent output
                if frame.shape[0] != 480 or frame.shape[1] != 640:
                    frame = cv2.resize(frame, (640, 480))

                # Encode frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    print("⚠️ Failed to encode frame")
                    break

                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    except Exception as e:
        print(f"❌ Error in video feed: {str(e)}")

    # If we get here, yield error frame
    error_frame = get_error_frame()
    while True:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')


def get_error_frame():
    """Generate error frame when camera fails"""
    try:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera Not Available", (150, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Please check connection", (140, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes() if ret else b''
    except:
        return b''


@app.route('/generate_face_embedding', methods=['POST'])
def generate_face_embedding():
    """Generate face embedding with enhanced duplicate detection and error handling"""
    global last_frame

    print("🎯 Generating face embedding...")

    try:
        # Get a stable frame
        frame = get_stable_frame()
        if frame is None:
            return jsonify({
                'success': False,
                'message': '❌ Camera not ready - please ensure face is detected first'
            })

        print("🔄 Processing frame for embedding...")
        result = face_system.process_frame(frame, is_capture=True)

        if not result['success']:
            return jsonify({
                'success': False,
                'message': result['message']
            })

        # Enhanced duplicate checking with proper error handling
        existing_faces = read_face_data()
        if existing_faces:
            try:
                # Ensure the function returns 3 values
                is_duplicate, existing_name, similarity_score = face_system.is_duplicate_face(
                    result['embedding'],
                    existing_faces,
                    threshold=0.7
                )

                if is_duplicate:
                    print(f"❌ Duplicate detected: {existing_name} (similarity: {similarity_score:.4f})")
                    return jsonify({
                        'success': False,
                        'message': f'❌ This face is already registered as: {existing_name} (similarity: {similarity_score:.1%})',
                        'duplicate': True,
                        'existing_user': existing_name,
                        'similarity_score': similarity_score
                    })

            except ValueError as e:
                if "not enough values to unpack" in str(e):
                    print("❌ Error: is_duplicate_face function is not returning 3 values")
                    # Fallback: check if it's returning 2 values
                    try:
                        duplicate_result = face_system.is_duplicate_face(
                            result['embedding'],
                            existing_faces,
                            threshold=0.7
                        )
                        if len(duplicate_result) == 2:
                            is_duplicate, existing_name = duplicate_result
                            similarity_score = 0.8 if is_duplicate else 0.0
                            if is_duplicate:
                                return jsonify({
                                    'success': False,
                                    'message': f'❌ This face is already registered as: {existing_name}',
                                    'duplicate': True,
                                    'existing_user': existing_name,
                                    'similarity_score': similarity_score
                                })
                        else:
                            raise e
                    except Exception as fallback_error:
                        print(f"❌ Fallback duplicate check failed: {fallback_error}")
                else:
                    raise e
            except Exception as e:
                print(f"❌ Error during duplicate check: {e}")
                # Continue without duplicate check if there's an error

        result = ensure_json_serializable(result)
        print("✅ Embedding generated successfully - no duplicates found")
        return jsonify(result)

    except Exception as e:
        print(f"❌ Error generating embedding: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'❌ Error generating face embedding: {str(e)}'
        })


@app.route('/video_feed')
def video_feed():
    """Video feed with stable camera management"""

    def generate():
        while True:
            frame = get_stable_frame()
            if frame is not None:
                try:
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    else:
                        time.sleep(0.1)
                except Exception as e:
                    print(f"❌ Error in video feed: {e}")
                    time.sleep(0.1)
            else:
                # Return error frame
                error_frame = get_error_frame()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
                time.sleep(1)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/release_camera', methods=['POST'])
def release_camera_route():
    """Release camera route"""
    print("Releasing camera...")
    try:
        release_camera()
        return jsonify({'success': True, 'message': 'Camera released'})
    except Exception as e:
        print(f"Error in release_camera route: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/check_face_position', methods=['POST'])
def check_face_position():
    """Check face position for registration with stable frame capture"""
    global last_frame

    print("🔍 Checking face position...")

    try:
        # Get a stable frame
        frame = get_stable_frame()
        if frame is None:
            return jsonify({
                'success': False,
                'message': '❌ Camera not available - please check camera connection'
            })

        print("🔄 Processing frame for face detection...")
        result = face_system.process_frame(frame, is_capture=False)

        # Ensure the result is JSON serializable
        result = ensure_json_serializable(result)

        print(f"📊 Result: {result['success']} - {result['message']}")
        return jsonify(result)

    except Exception as e:
        print(f"❌ Error in check_face_position: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'❌ Error detecting face: {str(e)}'
        })

@app.route('/')
def index():
    # Debug info
    print(f"Index route - User logged in: {session.get('user_logged_in')}")
    print(f"Index route - User role: {session.get('user_role')}")

    # Use the new session system
    if not session.get('user_logged_in'):
        return render_template('base.html')  # This shows login form

    # Keep your existing admin logic
    is_admin = session.get('is_admin', False)
    return render_template('index.html', active_page='check-in', is_admin=is_admin)


@app.route('/camera_status')
def camera_status():
    """Check camera status"""
    global camera

    status = {
        'available': False,
        'message': 'Camera not initialized',
        'camera_index': 0
    }

    if camera is not None and camera.isOpened():
        try:
            success, frame = camera.read()
            if success and frame is not None:
                status.update({
                    'available': True,
                    'message': 'Camera working',
                    'frame_size': f"{frame.shape[1]}x{frame.shape[0]}"
                })
            else:
                status.update({
                    'available': False,
                    'message': 'Camera opened but cannot read frames'
                })
        except Exception as e:
            status.update({
                'available': False,
                'message': f'Camera error: {str(e)}'
            })

    return jsonify(status)


@app.route('/reset_camera', methods=['POST'])
def reset_camera():
    """Force reset camera"""
    global camera
    try:
        release_camera()
        time.sleep(1)  # Use time.sleep instead of datetime.time.sleep
        success = initialize_camera() is not None

        return jsonify({
            'success': success,
            'message': 'Camera reset successfully' if success else 'Camera reset failed'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error resetting camera: {str(e)}'
        })


def read_users():
    users = []
    if os.path.exists('users.csv'):
        with open('users.csv', 'r', newline='') as file:
            csv_reader = csv.DictReader(file)
            users = list(csv_reader)
    return users


def write_user(user):
    users = read_users()
    users.append({
        'Name': user['name'],
        'ID': user['id'],
        'Email': user['email'],
        'Role': user['role'],
        'Phone': user['phone'],
        'Password': user['password']  # Store hashed password
    })

    with open('users.csv', 'w', newline='') as file:
        fieldnames = ['Name', 'ID', 'Email', 'Role', 'Phone', 'Password']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(users)


def authenticate_user(email, password):
    """Authenticate user against users.csv database using existing functions"""
    users = read_users()  # Your existing function

    for user in users:
        if user['Email'].lower() == email.lower():  # Case-insensitive email check
            # Verify password using your existing function
            if verify_password(user.get('Password', ''), password):
                return user
            else:
                continue
    return None


def get_current_user():
    """Get current logged-in user data"""
    if session.get('user_logged_in'):
        return {
            'id': session.get('user_id'),
            'name': session.get('user_name'),
            'email': session.get('user_email'),
            'role': session.get('user_role'),
            'is_admin': session.get('is_admin')
        }
    return None


@app.route('/user-login', methods=['POST'])
def user_login():
    try:
        login_data = request.json
        email = login_data.get('email')
        password = login_data.get('password')

        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password are required'})

        # Authenticate user using your existing functions
        user = authenticate_user(email, password)

        if user:
            # Set ALL session variables needed for role-based navigation
            session['user_logged_in'] = True
            session['user_id'] = user['ID']
            session['user_name'] = user['Name']
            session['user_email'] = user['Email']
            session['user_role'] = user.get('Role', 'employee')  # Default to employee if not set
            session['is_admin'] = (user.get('Role') == 'admin')  # For backward compatibility

            print(f"User logged in: {user['Name']} with role: {user.get('Role', 'employee')}")  # Debug

            return jsonify({
                'success': True,
                'message': 'Login successful',
                'user': {
                    'name': user['Name'],
                    'role': user.get('Role', 'employee')
                }
            })
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password'})

    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({'success': False, 'message': 'Login failed'})


@app.route('/user-logout')
def user_logout():
    # Clear ALL user session data to ensure clean logout
    session.pop('user_logged_in', None)
    session.pop('user_id', None)
    session.pop('user_name', None)
    session.pop('user_email', None)
    session.pop('user_role', None)
    session.pop('is_admin', None)  # Clear admin session too

    print("User logged out successfully")  # Debug

    return redirect(url_for('index'))

@app.before_request
def require_login():
    # List of routes that don't require login
    public_routes = ['user_login', 'admin_login', 'static', 'index']

    if request.endpoint and request.endpoint not in public_routes:
        if not session.get('user_logged_in'):
            print(f"Redirecting to login - no session for: {request.endpoint}")  # Debug
            return redirect(url_for('index'))

@app.route('/debug-session')
def debug_session():
    """Debug route to check session data.

    Disabled by default for public/portfolio deployments.
    Enable only in local development by setting FLASK_DEBUG=1.
    """
    if os.environ.get('FLASK_DEBUG') != '1':
        return jsonify({'success': False, 'message': 'Debug route is disabled'}), 404

    return jsonify({
        'user_logged_in': session.get('user_logged_in'),
        'user_id': session.get('user_id'),
        'user_name': session.get('user_name'),
        'user_email': session.get('user_email'),
        'user_role': session.get('user_role'),
        'is_admin': session.get('is_admin')
    })


@app.route('/users')
def users():
    if not session.get('is_admin', False):
        return redirect(url_for('index'))

    users_list = read_users()
    formatted_users = []

    for user in users_list:
        formatted_users.append({
            'Name': user['Name'],
            'ID': user['ID'],
            'Email': user['Email'],
            'Role': user.get('Role', 'employee'),  # Default to 'employee' if not set
            'Phone': user.get('Phone', '')  # Empty string if not set
            # Removed 'face_embedding' field
        })

    return render_template('users.html', active_page='users', is_admin=True, users=formatted_users)

@app.route('/add_user', methods=['POST'])
def add_user():
    if not session.get('is_admin', False):
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        user_data = request.json

        # Check if user ID already exists
        users = read_users()
        if any(user['ID'] == user_data['id'] for user in users):
            return jsonify({
                'success': False,
                'message': 'User ID already exists'
            })

        # Check if email already exists
        if any(user['Email'] == user_data['email'] for user in users):
            return jsonify({
                'success': False,
                'message': 'Email already exists'
            })

        # Hash the password
        hashed_password = hash_password(user_data['password'])

        # Save user data to CSV
        write_user({
            'name': user_data['name'],
            'id': user_data['id'],
            'email': user_data['email'],
            'role': user_data['role'],
            'phone': user_data['phone'],
            'password': hashed_password
        })

        return jsonify({
            'success': True,
            'message': 'User added successfully'
        })

    except Exception as e:
        print(f"Error adding user: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error adding user'
        })

def hash_password(password):
    """Hash a password for storing using Werkzeug's salted password hashing."""
    return generate_password_hash(password)

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user.

    Supports modern Werkzeug hashes and legacy SHA-256 hashes for older local CSV entries.
    """
    if not stored_password:
        return False

    try:
        if stored_password.startswith(('pbkdf2:', 'scrypt:')):
            return check_password_hash(stored_password, provided_password)
    except Exception:
        return False

    # Legacy fallback for old local users.csv records.
    return stored_password == hashlib.sha256(provided_password.encode()).hexdigest()

@app.route('/edit_user', methods=['POST'])
def edit_user():
    if not session.get('is_admin', False):
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        user_data = request.json
        users = read_users()

        # Find and update the user
        updated = False
        for i, user in enumerate(users):
            if user['ID'] == user_data['id']:
                # Keep existing password if no new password provided
                password = user['Password']
                if user_data.get('password'):
                    password = hash_password(user_data['password'])

                users[i] = {
                    'Name': user_data['name'],
                    'ID': user_data['id'],
                    'Email': user_data['email'],
                    'Role': user_data['role'],
                    'Phone': user_data['phone'],
                    'Password': password
                }
                updated = True
                break

        if not updated:
            return jsonify({
                'success': False,
                'message': 'User not found'
            })

        # Write updated users back to CSV
        with open('users.csv', 'w', newline='') as file:
            fieldnames = ['Name', 'ID', 'Email', 'Role', 'Phone', 'Password']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(users)

        return jsonify({
            'success': True,
            'message': 'User updated successfully'
        })

    except Exception as e:
        print(f"Error updating user: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error updating user'
        })


@app.route('/delete_user', methods=['POST'])
def delete_user():
    if not session.get('is_admin', False):
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        user_id = request.json.get('id')
        users = read_users()

        # Filter out the user to delete
        updated_users = [user for user in users if user['ID'] != user_id]

        if len(updated_users) == len(users):
            return jsonify({
                'success': False,
                'message': 'User not found'
            })

        # Write updated users back to CSV
        with open('users.csv', 'w', newline='') as file:
            fieldnames = ['Name', 'ID', 'Email', 'Role', 'Phone', 'Password']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_users)

        return jsonify({
            'success': True,
            'message': 'User deleted successfully'
        })

    except Exception as e:
        print(f"Error deleting user: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Error deleting user'
        })


def debug_check_users_file():
    """Debug function to check users.csv format"""
    try:
        with open('users.csv', 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                print(f"\nUser: {row.get('Name')}")
                print(f"ID: {row.get('ID')}")
                print(f"Email: {row.get('Email')}")
                print(f"Role: {row.get('Role')}")
                print(f"Phone: {row.get('Phone')}")
                print(f"Has Password: {'Yes' if row.get('Password') else 'No'}")
    except Exception as e:
        print(f"Error reading users.csv: {str(e)}")

# Application entry point
if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG') == '1'

    print("\nStarting AI Smart Surveillance System...")
    print("Server running at: http://127.0.0.1:5000")
    app.run(debug=debug_mode)
