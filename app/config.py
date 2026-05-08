"""
Configuración de la aplicación
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Directorios base
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
MODELS_DIR = DATA_DIR / "models"

# Crear directorios si no existen
FACES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Configuración del modelo
FACE_DETECTION_MODEL = "haarcascade_frontalface_default.xml"
EMBEDDING_MODEL_NAME = "Facenet"  # Usaremos FaceNet
CLASSIFIER_PATH = MODELS_DIR / "face_classifier.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
EMBEDDINGS_PATH = MODELS_DIR / "embeddings.pkl"

# Configuración de detección
MIN_FACE_SIZE = (50, 50)
CONFIDENCE_THRESHOLD = 0.6

# Configuración de imágenes
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# ── Backend de reconocimiento: "local" o "supabase" ──────────────────────────
FACE_STORE = os.getenv("FACE_STORE", "local")  # switch principal

# Supabase (solo se usa cuando FACE_STORE=supabase)
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")  # service_role key

# ── Configuración de ESPs
ESP32_BRIDGE_URL = os.getenv("ESP32_BRIDGE_URL", "http://192.168.1.100")
ESP32_CAM_URL = os.getenv("ESP32_CAM_URL", "http://192.168.1.101")

# Intervalos de polling (ms)
SENSOR_POLL_INTERVAL = 1000  # 1s para sensor
CAM_POLL_INTERVAL = 2000     # 2s para cámara
