"""
Servicio de clasificación y reconocimiento facial
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2

from app.config import (
    CLASSIFIER_PATH,
    LABEL_ENCODER_PATH,
    EMBEDDINGS_PATH,
    FACES_DIR,
    CONFIDENCE_THRESHOLD
)
from app.services.face_detector import FaceDetector
from app.services.face_embedder import FaceEmbedder


class FaceRecognizer:
    """Sistema de reconocimiento facial"""
    
    def __init__(self):
        """Inicializa el sistema de reconocimiento"""
        self.face_detector = FaceDetector()
        self.face_embedder = FaceEmbedder()
        self.classifier: Optional[SVC] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.embeddings_db: Dict[str, List[np.ndarray]] = {}
        
        # Cargar modelos si existen
        self._load_models()
    
    def _load_models(self):
        """Carga los modelos entrenados si existen"""
        try:
            if CLASSIFIER_PATH.exists():
                with open(CLASSIFIER_PATH, 'rb') as f:
                    self.classifier = pickle.load(f)
                print("✓ Clasificador cargado")
            
            if LABEL_ENCODER_PATH.exists():
                with open(LABEL_ENCODER_PATH, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print("✓ Label encoder cargado")
            
            if EMBEDDINGS_PATH.exists():
                with open(EMBEDDINGS_PATH, 'rb') as f:
                    self.embeddings_db = pickle.load(f)
                print("✓ Base de datos de embeddings cargada")
        
        except Exception as e:
            print(f"Error cargando modelos: {e}")
    
    def _save_models(self):
        """Guarda los modelos entrenados"""
        try:
            if self.classifier:
                with open(CLASSIFIER_PATH, 'wb') as f:
                    pickle.dump(self.classifier, f)
            
            if self.label_encoder:
                with open(LABEL_ENCODER_PATH, 'wb') as f:
                    pickle.dump(self.label_encoder, f)
            
            if self.embeddings_db:
                with open(EMBEDDINGS_PATH, 'wb') as f:
                    pickle.dump(self.embeddings_db, f)
            
            print("✓ Modelos guardados correctamente")
            return True
        
        except Exception as e:
            print(f"Error guardando modelos: {e}")
            return False
    
    def add_person(self, name: str, image: np.ndarray) -> Tuple[bool, str, int]:
        """
        Añade una nueva persona al sistema
        
        Args:
            name: Nombre de la persona
            image: Imagen con el rostro de la persona
        
        Returns:
            (éxito, mensaje, número_de_imágenes_guardadas)
        """
        # Detectar rostro
        face = self.face_detector.detect_and_extract_largest_face(image)
        
        if face is None:
            return False, "No se detectó ningún rostro en la imagen", 0
        
        # Generar embedding
        embedding = self.face_embedder.generate_embedding(face)
        
        if embedding is None:
            return False, "Error generando embedding del rostro", 0
        
        # Crear directorio de la persona si no existe
        person_dir = FACES_DIR / name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar la imagen del rostro
        existing_images = list(person_dir.glob("*.jpg"))
        image_number = len(existing_images) + 1
        image_path = person_dir / f"{name}_{image_number}.jpg"
        
        cv2.imwrite(str(image_path), face)
        
        # Agregar embedding a la base de datos
        if name not in self.embeddings_db:
            self.embeddings_db[name] = []
        
        self.embeddings_db[name].append(embedding)
        
        # Guardar embeddings actualizados
        self._save_models()
        
        return True, f"Persona '{name}' añadida exitosamente", 1
    
    def train_classifier(self) -> Tuple[bool, str, Dict]:
        """
        Entrena el clasificador con todos los embeddings guardados
        
        Returns:
            (éxito, mensaje, estadísticas)
        """
        if not self.embeddings_db:
            return False, "No hay datos para entrenar. Añade personas primero.", {}
        
        # Preparar datos de entrenamiento
        X = []  # Embeddings
        y = []  # Etiquetas
        
        for name, embeddings in self.embeddings_db.items():
            for embedding in embeddings:
                X.append(embedding)
                y.append(name)
        
        X = np.array(X)
        y = np.array(y)
        
        # Verificar que hay suficientes muestras
        if len(X) < 2:
            return False, "Se necesitan al menos 2 muestras para entrenar", {}
        
        # Codificar etiquetas
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Entrenar clasificador SVM
        self.classifier = SVC(
            kernel='linear',
            probability=True,
            C=1.0,
            random_state=42
        )
        
        self.classifier.fit(X, y_encoded)
        
        # Guardar modelos
        self._save_models()
        
        # Calcular estadísticas
        stats = {
            "total_people": len(self.embeddings_db),
            "total_samples": len(X),
            "people": {name: len(embs) for name, embs in self.embeddings_db.items()}
        }
        
        return True, "Clasificador entrenado exitosamente", stats
    
    def recognize_face(self, image: np.ndarray) -> Tuple[bool, Optional[str], Optional[float], str]:
        """
        Reconoce un rostro en una imagen
        
        Args:
            image: Imagen con el rostro a reconocer
        
        Returns:
            (éxito, nombre, confianza, mensaje)
        """
        # Verificar que el modelo está entrenado
        if self.classifier is None or self.label_encoder is None:
            return False, None, None, "El modelo no está entrenado. Entrena primero el clasificador."
        
        # Detectar rostro
        face = self.face_detector.detect_and_extract_largest_face(image)
        
        if face is None:
            return False, None, None, "No se detectó ningún rostro en la imagen"
        
        # Generar embedding
        embedding = self.face_embedder.generate_embedding(face)
        
        if embedding is None:
            return False, None, None, "Error generando embedding del rostro"
        
        # Predecir con el clasificador
        embedding_reshaped = embedding.reshape(1, -1)
        
        # Obtener probabilidades
        probabilities = self.classifier.predict_proba(embedding_reshaped)[0]
        predicted_label = self.classifier.predict(embedding_reshaped)[0]
        confidence = float(np.max(probabilities))
        
        # Decodificar etiqueta
        person_name = self.label_encoder.inverse_transform([predicted_label])[0]
        
        # Verificar umbral de confianza
        if confidence < CONFIDENCE_THRESHOLD:
            return True, "Desconocido", confidence, f"Confianza ({confidence:.2f}) menor al umbral"
        
        return True, person_name, confidence, f"Rostro reconocido como '{person_name}'"
    
    def get_status(self) -> Dict:
        """Obtiene el estado del sistema"""
        total_people = len(self.embeddings_db)
        total_samples = sum(len(embs) for embs in self.embeddings_db.values())
        
        return {
            "model_trained": self.classifier is not None,
            "total_people": total_people,
            "total_samples": total_samples,
            "people": list(self.embeddings_db.keys())
        }
    
    def reload_embeddings_from_disk(self) -> Tuple[bool, str, int]:
        """
        Recarga embeddings desde las imágenes guardadas en disco
        Útil si se añadieron imágenes manualmente
        
        Returns:
            (éxito, mensaje, total_de_imágenes_procesadas)
        """
        self.embeddings_db = {}
        total_processed = 0
        
        # Iterar sobre cada carpeta de persona
        for person_dir in FACES_DIR.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                
                # Procesar cada imagen
                for image_path in person_dir.glob("*.jpg"):
                    try:
                        image = cv2.imread(str(image_path))
                        if image is not None:
                            embedding = self.face_embedder.generate_embedding(image)
                            
                            if embedding is not None:
                                if person_name not in self.embeddings_db:
                                    self.embeddings_db[person_name] = []
                                
                                self.embeddings_db[person_name].append(embedding)
                                total_processed += 1
                    
                    except Exception as e:
                        print(f"Error procesando {image_path}: {e}")
        
        # Guardar embeddings
        self._save_models()
        
        return True, f"Embeddings recargados: {total_processed} imágenes procesadas", total_processed
