"""
Configuración de la aplicación
"""
import os
from pathlib import Path

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
