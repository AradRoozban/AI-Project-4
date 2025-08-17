from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS
import cv2
import numpy as np
import os
import time
import json
from datetime import datetime
import threading
import base64
from ultralytics import YOLO
from deepface import DeepFace
import face_recognition
import mediapipe as mp
import pyautogui
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Global variables
analysis_running = False
current_session = None
session_data = {
    'start_time': None,
    'metrics': {
        'productivity_score': 0.0,
        'focus_score': 0.0,
        'phone_detections': 0,
        'emotion_data': {},
        'activity_states': {
            'focused': 0,
            'phone_distraction': 0,
            'away_from_screen': 0,
            'multiple_faces': 0
        }
    },
    'frames_processed': 0
}

# Initialize models
try:
    yolo_model = YOLO('yolov8n.pt')
    print("✅ YOLO model loaded successfully")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    yolo_model = None

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

# Face cascade for OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Camera setup
camera = None
camera_lock = threading.Lock()

def initialize_camera():
    """Initialize camera with proper error handling"""
    global camera
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera = cv2.VideoCapture(1)  # Try alternative camera index
        if not camera.isOpened():
            raise RuntimeError("No camera available")
        
        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Camera initialization failed: {e}")
        return False

def release_camera():
    """Release camera resources"""
    global camera
    if camera:
        camera.release()
        camera = None

def analyze_video(video_path):
    """Analyze uploaded video file"""
    global session_data
    
    try:
        print(f"🎬 Starting video analysis: {video_path}")
        
        # Open video file
        video_cap = cv2.VideoCapture(video_path)
        if not video_cap.isOpened():
            print(f"❌ Error opening video file: {video_path}")
            return
        
        frame_count = 0
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📊 Video info: {total_frames} frames, {fps:.2f} FPS")
        
        while True:
            ret, frame = video_cap.read()
            if not ret:
                break
            
            # Process every 3rd frame for performance (10 FPS analysis)
            if frame_count % 3 == 0:
                # Process frame
                processed_frame, results = process_frame(frame)
                
                # Update progress
                progress = (frame_count / total_frames) * 100
                print(f"📈 Progress: {progress:.1f}% - Frame {frame_count}/{total_frames}")
                
                # Update session data
                if 'phone_count' in results:
                    session_data['metrics']['phone_detections'] = max(
                        session_data['metrics']['phone_detections'], 
                        results['phone_count']
                    )
                
                if 'emotion' in results and results['emotion'] != 'unknown':
                    session_data['metrics']['emotion_data'] = results.get('emotion_scores', {})
                
                # Update frame count
                session_data['frames_processed'] += 1
                
                # Simulate real-time processing delay
                time.sleep(0.1)
            
            frame_count += 1
        
        video_cap.release()
        print(f"✅ Video analysis completed. Processed {session_data['frames_processed']} frames")
        
        # Calculate final metrics
        if session_data['frames_processed'] > 0:
            # Adjust productivity score based on analysis results
            phone_ratio = session_data['metrics']['phone_detections'] / session_data['frames_processed']
            if phone_ratio < 0.1:  # Less than 10% phone usage
                session_data['metrics']['productivity_score'] = min(10.0, 
                    session_data['metrics']['productivity_score'] + 2.0)
            elif phone_ratio > 0.3:  # More than 30% phone usage
                session_data['metrics']['productivity_score'] = max(0.0, 
                    session_data['metrics']['productivity_score'] - 2.0)
            
            # Calculate focus score
            focus_emotions = ['happy', 'neutral', 'surprise']
            if session_data['metrics']['emotion_data']:
                dominant_emotion = max(session_data['metrics']['emotion_data'].items(), 
                                    key=lambda x: x[1])[0]
                if dominant_emotion in focus_emotions:
                    session_data['metrics']['focus_score'] = 0.8
                else:
                    session_data['metrics']['focus_score'] = 0.3
        
    except Exception as e:
        print(f"❌ Video analysis error: {e}")
        import traceback
        traceback.print_exc()

def process_frame(frame):
    """Process a single frame for analysis"""
    if frame is None:
        return frame, {}
    
    results = {}
    
    try:
        # Resize frame for processing
        frame_small = cv2.resize(frame, (640, 480))
        
        # YOLO object detection
        if yolo_model:
            yolo_results = yolo_model(frame_small, verbose=False)[0]
            phone_count = 0
            for box in yolo_results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = yolo_model.names[cls]
                if class_name == 'cell phone':
                    phone_count += 1
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Phone {conf:.2f}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            results['phone_count'] = phone_count
            session_data['metrics']['phone_detections'] = phone_count
        
        # Face detection and emotion analysis
        faces = face_cascade.detectMultiScale(frame_small, 1.1, 5)
        if len(faces) > 0:
            results['face_count'] = len(faces)
            
            # Analyze first face for emotion
            if len(faces) > 0:
                (fx, fy, fw, fh) = faces[0]
                face_crop = frame_small[fy:fy+fh, fx:fx+fw]
                
                try:
                    emotion_analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                    dominant_emotion = emotion_analysis[0].get("dominant_emotion", "neutral")
                    emotion_scores = emotion_analysis[0].get("emotion", {})
                    
                    results['emotion'] = dominant_emotion
                    results['emotion_scores'] = emotion_scores
                    
                    # Update session data
                    session_data['metrics']['emotion_data'] = emotion_scores
                    
                    # Calculate focus score based on emotion
                    focus_emotions = ['happy', 'neutral', 'surprise']
                    if dominant_emotion in focus_emotions:
                        session_data['metrics']['focus_score'] = min(1.0, session_data['metrics']['focus_score'] + 0.01)
                    else:
                        session_data['metrics']['focus_score'] = max(0.0, session_data['metrics']['focus_score'] - 0.01)
                    
                except Exception as e:
                    print(f"Emotion analysis error: {e}")
                    results['emotion'] = 'unknown'
            
            # Draw face rectangles
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # MediaPipe pose detection
        rgb_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)
        
        if pose_results.pose_landmarks:
            # Draw pose landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Analyze pose for productivity
            landmarks = pose_results.pose_landmarks.landmark
            if landmarks:
                # Check if person is sitting properly (shoulders level)
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                
                if abs(left_shoulder.y - right_shoulder.y) < 0.1:  # Shoulders level
                    session_data['metrics']['productivity_score'] = min(10.0, 
                        session_data['metrics']['productivity_score'] + 0.1)
                else:
                    session_data['metrics']['productivity_score'] = max(0.0, 
                        session_data['metrics']['productivity_score'] - 0.05)
        
        # MediaPipe hand detection
        hands_results = hands.process(rgb_frame)
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Update activity states
        if results.get('phone_count', 0) > 0:
            session_data['metrics']['activity_states']['phone_distraction'] += 1
        else:
            session_data['metrics']['activity_states']['focused'] += 1
        
        if results.get('face_count', 0) == 0:
            session_data['metrics']['activity_states']['away_from_screen'] += 1
        elif results.get('face_count', 0) > 1:
            session_data['metrics']['activity_states']['multiple_faces'] += 1
        
        # Update frame count
        session_data['frames_processed'] += 1
        
        # Add metrics overlay to frame
        y_offset = 30
        cv2.putText(frame, f"Productivity: {session_data['metrics']['productivity_score']:.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Focus: {session_data['metrics']['focus_score']:.2f}", 
                   (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Phones: {results.get('phone_count', 0)}", 
                   (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if 'emotion' in results:
            cv2.putText(frame, f"Emotion: {results['emotion']}", 
                       (10, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
    except Exception as e:
        print(f"Frame processing error: {e}")
    
    return frame, results

def generate_frames():
    """Generate video frames for streaming"""
    global analysis_running, camera
    
    while analysis_running:
        with camera_lock:
            if camera is None or not camera.isOpened():
                break
            
            ret, frame = camera.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, _ = process_frame(frame)
            
            if processed_frame is not None:
                # Encode frame for streaming
                ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': {
            'yolo': yolo_model is not None,
            'deepface': True,
            'mediapipe': True
        }
    })

@app.route('/api/start_analysis', methods=['POST'])
def start_analysis():
    """Start the analysis session"""
    global analysis_running, current_session, session_data
    
    try:
        data = request.get_json()
        analysis_type = data.get('type', 'camera')
        
        if analysis_type == 'camera':
            if not initialize_camera():
                return jsonify({'success': False, 'error': 'Camera initialization failed'})
            
            analysis_running = True
            current_session = f"session_{int(time.time())}"
            session_data['start_time'] = time.time()
            session_data['frames_processed'] = 0
            
            # Reset metrics
            session_data['metrics'] = {
                'productivity_score': 5.0,  # Start with neutral score
                'focus_score': 0.5,
                'phone_detections': 0,
                'emotion_data': {},
                'activity_states': {
                    'focused': 0,
                    'phone_distraction': 0,
                    'away_from_screen': 0,
                    'multiple_faces': 0
                }
            }
            
            return jsonify({
                'success': True,
                'session_id': current_session,
                'message': 'Camera analysis started'
            })
        else:
            # Video analysis
            video_path = data.get('video_path')
            if not video_path or not os.path.exists(video_path):
                return jsonify({'success': False, 'error': 'Video file not found'})
            
            analysis_running = True
            current_session = f"session_{int(time.time())}"
            session_data['start_time'] = time.time()
            session_data['frames_processed'] = 0
            
            # Reset metrics
            session_data['metrics'] = {
                'productivity_score': 5.0,  # Start with neutral score
                'focus_score': 0.5,
                'phone_detections': 0,
                'emotion_data': {},
                'activity_states': {
                    'focused': 0,
                    'phone_distraction': 0,
                    'away_from_screen': 0,
                    'multiple_faces': 0
                }
            }
            
            # Start video analysis in background thread
            import threading
            video_thread = threading.Thread(target=analyze_video, args=(video_path,))
            video_thread.daemon = True
            video_thread.start()
            
            return jsonify({
                'success': True,
                'session_id': current_session,
                'message': 'Video analysis started'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop_analysis', methods=['POST'])
def stop_analysis():
    """Stop the analysis session"""
    global analysis_running, current_session
    
    try:
        analysis_running = False
        release_camera()
        
        # Save session data
        if current_session:
            session_file = f"sessions/{current_session}.json"
            os.makedirs("sessions", exist_ok=True)
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
        
        current_session = None
        return jsonify({'success': True, 'message': 'Analysis stopped'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/video_feed')
def video_feed():
    """Video streaming endpoint"""
    if not analysis_running:
        return "Analysis not running", 400
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/current_metrics')
def current_metrics():
    """Get current analysis metrics"""
    if not analysis_running:
        return jsonify({'success': False, 'error': 'Analysis not running'})
    
    # Calculate session duration
    duration = 0
    if session_data['start_time']:
        duration = time.time() - session_data['start_time']
    
    metrics = {
        'current_productivity': session_data['metrics']['productivity_score'],
        'current_focus': session_data['metrics']['focus_score'],
        'phone_detections': session_data['metrics']['phone_detections'],
        'session_duration': duration,
        'frames_processed': session_data['frames_processed'],
        'activity_states': session_data['metrics']['activity_states'],
        'emotion_data': session_data['metrics']['emotion_data']
    }
    
    return jsonify({'success': True, 'metrics': metrics})

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """Upload video file for analysis"""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'})
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save video file
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        filename = f"video_{int(time.time())}.mp4"
        filepath = os.path.join(uploads_dir, filename)
        video_file.save(filepath)
        
        return jsonify({
            'success': True,
            'filepath': filepath,
            'message': 'Video uploaded successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/download_report/<session_id>')
def download_report(session_id):
    """Download analysis report"""
    try:
        session_file = f"sessions/{session_id}.json"
        if not os.path.exists(session_file):
            return jsonify({'success': False, 'error': 'Session not found'})
        
        # Generate report
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        # Create report text
        report_lines = [
            f"Smart Productivity Analysis Report",
            f"Session ID: {session_id}",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"Session Duration: {session_data.get('session_duration', 0):.1f} seconds",
            f"Frames Processed: {session_data.get('frames_processed', 0)}",
            f"",
            f"Final Metrics:",
            f"- Productivity Score: {session_data['metrics']['productivity_score']:.1f}/10",
            f"- Focus Score: {session_data['metrics']['focus_score']:.2f}",
            f"- Phone Detections: {session_data['metrics']['phone_detections']}",
            f"",
            f"Activity Distribution:",
        ]
        
        for activity, count in session_data['metrics']['activity_states'].items():
            report_lines.append(f"- {activity.replace('_', ' ').title()}: {count}")
        
        if session_data['metrics']['emotion_data']:
            report_lines.append(f"")
            report_lines.append(f"Emotion Analysis:")
            for emotion, score in session_data['metrics']['emotion_data'].items():
                report_lines.append(f"- {emotion}: {score:.1f}%")
        
        report_text = "\n".join(report_lines)
        
        # Create file response
        report_bytes = report_text.encode('utf-8')
        report_io = io.BytesIO(report_bytes)
        report_io.seek(0)
        
        return send_file(
            report_io,
            mimetype='text/plain',
            as_attachment=True,
            download_name=f"productivity_report_{session_id}.txt"
        )
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analysis_history')
def analysis_history():
    """Get analysis history"""
    try:
        sessions_dir = "sessions"
        if not os.path.exists(sessions_dir):
            return jsonify({'success': True, 'reports': []})
        
        reports = []
        for filename in os.listdir(sessions_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(sessions_dir, filename)
                with open(filepath, 'r') as f:
                    session_data = json.load(f)
                
                reports.append({
                    'session_id': filename.replace('.json', ''),
                    'start_time': session_data.get('start_time'),
                    'duration': session_data.get('session_duration', 0),
                    'productivity_score': session_data['metrics']['productivity_score']
                })
        
        # Sort by start time (newest first)
        reports.sort(key=lambda x: x['start_time'] or 0, reverse=True)
        
        return jsonify({'success': True, 'reports': reports})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/video_progress')
def video_progress():
    """Get video analysis progress"""
    try:
        if not analysis_running:
            return jsonify({'success': False, 'error': 'No analysis running'})
        
        # Calculate progress based on frames processed
        total_frames = session_data.get('total_video_frames', 1000)  # Default
        frames_processed = session_data.get('frames_processed', 0)
        progress = min(100, (frames_processed / total_frames) * 100) if total_frames > 0 else 0
        
        return jsonify({
            'success': True,
            'progress': progress,
            'frames_processed': frames_processed,
            'total_frames': total_frames,
            'status': 'analyzing'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs("sessions", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    print("🚀 Starting Smart Productivity Analyzer...")
    print("📁 Created directories: sessions, uploads, templates")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
