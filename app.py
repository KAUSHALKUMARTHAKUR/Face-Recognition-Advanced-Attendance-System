import os
import io
import base64
from datetime import datetime, timezone

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from pymongo import MongoClient
from dotenv import load_dotenv

# Lightweight downloader for Google Drive files (handles large files)
# pip install gdown
import gdown

# Optional: Pillow for simple overlay passthrough/annotation if needed
from PIL import Image, ImageDraw, ImageFont

# Load environment variables
load_dotenv()

# Flask app
app = Flask(__name__, template_folder=os.path.join("app", "templates"), static_folder=os.path.join("app", "static"))

# Enable CORS for API routes (adjust origins in production as needed)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# MongoDB setup
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB = os.getenv("MONGODB_DB", "attendance_db")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "attendance")

mongo_client = None
mongo_db = None
attendance_col = None

def init_mongo():
    global mongo_client, mongo_db, attendance_col
    if mongo_client is None and MONGODB_URI:
        mongo_client = MongoClient(MONGODB_URI)  # SRV uses TLS by default
        mongo_db = mongo_client[MONGODB_DB]
        attendance_col = mongo_db[MONGODB_COLLECTION]

# ---------------------------
# Model download management
# ---------------------------

MODELS_DIR = "models"
ANTI_SPOOF_DIR = os.path.join(MODELS_DIR, "anti-spoofing")

# Map expected local file paths -> Google Drive File IDs
# The file IDs must correspond to the exact files in Drive with the same names/structure.
GDRIVE_FILE_IDS = {
    # shape predictor 68 face landmarks
    "models/shape_predictor_68_face_landmarks.dat": "1Y2e589TCdrwd1y_4AqlEf9hjh8k12PJN",

    # dlib face recognition resnet model
    "models/dlib_face_recognition_resnet_model_v1.dat": "1sybYq9GGriXN6sY8YV1-RXMeVqYzhDrV",

    # YOLOv5 face detector (ONNX)
    "models/yolov5s-face.onnx": "1JCaGS82kfopFu9a9OHgSG9Ed0HbbfZy1",

    # Anti-spoofing model (ONNX)
    "models/anti-spoofing/AntiSpoofing_bin_1.5_128.onnx": "1nH5G7dAHFE2KlW_H65txc8GDKSB7Zpy4",
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def download_from_gdrive(file_id: str, dst_path: str) -> bool:
    """
    Downloads a file from Google Drive to dst_path using its File ID.
    Returns True on success, False otherwise.
    """
    try:
        ensure_dir(os.path.dirname(dst_path))
        # gdown supports id=... format directly
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url=url, output=dst_path, quiet=False)
        return os.path.exists(dst_path) and os.path.getsize(dst_path) > 0
    except Exception as e:
        app.logger.error(f"Failed to download {dst_path} from Google Drive: {e}")
        return False

def ensure_models_available() -> dict:
    """
    Ensures that all required model files are present locally.
    Downloads any missing files from Google Drive using the GDRIVE_FILE_IDS map.
    Returns a dict with success flag and details.
    """
    ensure_dir(MODELS_DIR)
    ensure_dir(ANTI_SPOOF_DIR)

    missing = []
    failed = []
    for rel_path, file_id in GDRIVE_FILE_IDS.items():
        local_path = os.path.normpath(rel_path)
        if not os.path.exists(local_path):
            missing.append(local_path)

    if not missing:
        return {"success": True, "downloaded": [], "missing": []}

    downloaded = []
    for rel_path in missing:
        file_id = GDRIVE_FILE_IDS.get(rel_path)
        if not file_id or file_id.startswith("YOUR_FILE_ID_"):
            failed.append({"path": rel_path, "reason": "Missing Google Drive File ID in app.py GDRIVE_FILE_IDS"})
            continue
        ok = download_from_gdrive(file_id, rel_path)
        if ok:
            downloaded.append(rel_path)
        else:
            failed.append({"path": rel_path, "reason": "Download failed"})

    return {
        "success": len(failed) == 0,
        "downloaded": downloaded,
        "missing": failed,
    }

# ---------------------------
# Utilities
# ---------------------------

def decode_data_url_to_image_bytes(data_url: str) -> bytes:
    """
    Accepts a 'data:image/jpeg;base64,...' or 'data:image/png;base64,...'
    Returns raw bytes.
    """
    if not data_url:
        return b""
    if "," in data_url:
        _, b64data = data_url.split(",", 1)
    else:
        b64data = data_url
    return base64.b64decode(b64data)

def pil_bytes_to_data_url(img_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def draw_simple_overlay(img: Image.Image, text: str = "LIVE", color=(0, 200, 0)) -> Image.Image:
    """
    Draws a simple text overlay on the image for demo purposes.
    Replace with real bbox drawing from your detector/anti-spoofing pipeline.
    """
    draw = ImageDraw.Draw(img)
    try:
        # Default font
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.rectangle([(5, 5), (140, 40)], fill=(0, 0, 0, 180))
    draw.text((10, 12), text, fill=color, font=font)
    return img

# ---------------------------
# Placeholder ML functions
# Replace with your actual detector, anti-spoofing and recognition logic.
# ---------------------------

def check_liveness(image_bytes: bytes) -> tuple[bool, float]:
    """
    Anti-spoofing check.
    Return (is_live: bool, live_probability: float)
    TODO: Load and run your ONNX anti-spoofing model here.
    """
    # For demo purposes, always treat as live with high probability
    return True, 0.99

def recognize_face(image_bytes: bytes) -> dict:
    """
    Run your face recognition and return a dict with recognition details.
    Example return:
    {
        "recognized": True/False,
        "person_id": "student_123",
        "score": 0.87
    }
    TODO: Integrate your YOLOv5-face detection + descriptor + matching.
    """
    # Demo implementation: no recognition
    return {"recognized": False, "person_id": None, "score": 0.0}

# ---------------------------
# Frontend routes (unchanged)
# ---------------------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register.html")

# Add other pages if you have them:
# @app.route("/attendance")
# def attendance_page():
#     return render_template("attendance.html")

# ---------------------------
# API routes
# ---------------------------

@app.route("/api/liveness_preview", methods=["POST"])
def api_liveness_preview():
    """
    Live preview endpoint (low-res frames). Returns an overlay image and quick status.
    """
    # Download models if missing (first API call)
    model_status = ensure_models_available()
    if not model_status.get("success"):
        return jsonify({
            "success": False,
            "message": "Model download failed or Google Drive IDs missing.",
            "details": model_status,
        }), 500

    data = request.get_json(silent=True) or {}
    face_image_data_url = data.get("face_image", "")

    try:
        img_bytes = decode_data_url_to_image_bytes(face_image_data_url)
        if not img_bytes:
            return jsonify({"success": False, "message": "Invalid or empty image data."}), 400

        # Liveness check (placeholder)
        is_live, prob = check_liveness(img_bytes)

        # Create a simple overlay (placeholder)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        text = "LIVE" if is_live else "SPOOF"
        color = (0, 200, 0) if is_live else (220, 0, 0)
        img_overlay = draw_simple_overlay(img, text=text, color=color)

        buf = io.BytesIO()
        img_overlay.save(buf, format="JPEG", quality=85)
        overlay_data_url = pil_bytes_to_data_url(buf.getvalue(), mime="image/jpeg")

        return jsonify({
            "success": True,
            "live": is_live,
            "live_prob": float(prob),
            "overlay": overlay_data_url,
            "message": f"{text} p={prob:.2f}",
            "models": model_status,  # helpful on first call to observe downloads
        })
    except Exception as e:
        app.logger.exception("Error in /api/liveness_preview")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/mark_attendance", methods=["POST"])
def api_mark_attendance():
    """
    Accepts a captured still image and metadata; runs liveness + recognition (placeholders),
    and stores the attendance record in MongoDB.
    """
    # Ensure models are available (first call will trigger download)
    model_status = ensure_models_available()
    if not model_status.get("success"):
        return jsonify({
            "success": False,
            "message": "Model download failed or Google Drive IDs missing.",
            "details": model_status,
        }), 500

    data = request.get_json(silent=True) or {}

    student_id = data.get("student_id")  # May be None; your recognition may infer it
    program = data.get("program")
    semester = data.get("semester")
    course = data.get("course")
    face_image_data_url = data.get("face_image", "")

    if not program or not semester or not course:
        return jsonify({"success": False, "message": "program, semester, and course are required."}), 400

    try:
        img_bytes = decode_data_url_to_image_bytes(face_image_data_url)
        if not img_bytes:
            return jsonify({"success": False, "message": "Invalid or empty image data."}), 400

        # Liveness check (placeholder)
        is_live, prob = check_liveness(img_bytes)
        if not is_live:
            return jsonify({"success": False, "message": f"Spoof detected (p={prob:.2f})."}), 400

        # Recognition (placeholder)
        recog = recognize_face(img_bytes)
        recognized = bool(recog.get("recognized"))
        matched_id = recog.get("person_id") or student_id
        score = float(recog.get("score") or 0.0)

        # Create final overlay
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        text = "LIVE" if is_live else "SPOOF"
        color = (0, 200, 0) if is_live else (220, 0, 0)
        img_overlay = draw_simple_overlay(img, text=text, color=color)
        buf = io.BytesIO()
        img_overlay.save(buf, format="JPEG", quality=85)
        overlay_data_url = pil_bytes_to_data_url(buf.getvalue(), mime="image/jpeg")

        # Store attendance if MongoDB configured
        init_mongo()
        saved_id = None
        if attendance_col is not None:
            doc = {
                "program": program,
                "semester": semester,
                "course": course,
                "student_id": matched_id,
                "recognized": recognized,
                "recognition_score": score,
                "live": is_live,
                "live_prob": float(prob),
                "timestamp": datetime.now(timezone.utc),
            }
            result = attendance_col.insert_one(doc)
            saved_id = str(result.inserted_id)

        message = "Attendance marked."
        if recognized and matched_id:
            message = f"Attendance marked for {matched_id}."
        elif matched_id:
            message = f"Attendance marked (unverified) for {matched_id}."

        return jsonify({
            "success": True,
            "message": message,
            "overlay": overlay_data_url,
            "recognized": recognized,
            "student_id": matched_id,
            "score": score,
            "live_prob": float(prob),
            "saved_id": saved_id,
            "models": model_status,  # helpful on first call to observe downloads
        })
    except Exception as e:
        app.logger.exception("Error in /api/mark_attendance")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/get_attendance", methods=["GET"])
def api_get_attendance():
    """
    Query attendance records. Optional filters via query params:
      student_id, program, semester, course, from (ISO), to (ISO)
    """
    init_mongo()
    if attendance_col is None:
        return jsonify({"success": False, "message": "MongoDB not configured."}), 500

    q = {}
    student_id = request.args.get("student_id")
    program = request.args.get("program")
    semester = request.args.get("semester")
    course = request.args.get("course")
    dt_from = request.args.get("from")
    dt_to = request.args.get("to")

    if student_id:
        q["student_id"] = student_id
    if program:
        q["program"] = program
    if semester:
        q["semester"] = semester
    if course:
        q["course"] = course

    # Date range filter
    if dt_from or dt_to:
        q["timestamp"] = {}
        if dt_from:
            try:
                q["timestamp"]["$gte"] = datetime.fromisoformat(dt_from)
            except Exception:
                pass
        if dt_to:
            try:
                q["timestamp"]["$lte"] = datetime.fromisoformat(dt_to)
            except Exception:
                pass
        if not q["timestamp"]:
            q.pop("timestamp", None)

    # Fetch latest 200 records by default
    cursor = attendance_col.find(q).sort("timestamp", -1).limit(200)

    records = []
    for doc in cursor:
        records.append({
            "id": str(doc.get("_id")),
            "student_id": doc.get("student_id"),
            "program": doc.get("program"),
            "semester": doc.get("semester"),
            "course": doc.get("course"),
            "recognized": bool(doc.get("recognized", False)),
            "recognition_score": float(doc.get("recognition_score", 0.0)),
            "live": bool(doc.get("live", False)),
            "live_prob": float(doc.get("live_prob", 0.0)),
            "timestamp": doc.get("timestamp").isoformat() if doc.get("timestamp") else None,
        })

    return jsonify({"success": True, "count": len(records), "records": records})

# ---------------------------
# Backwards-compatible aliases (minimal template changes)
# ---------------------------

@app.route("/liveness-preview", methods=["POST"])
def alias_liveness_preview():
    return api_liveness_preview()

@app.route("/mark-attendance", methods=["POST"])
def alias_mark_attendance():
    return api_mark_attendance()

# ---------------------------
# Gunicorn entrypoint
# ---------------------------

if __name__ == "__main__":
    # Local dev server
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
