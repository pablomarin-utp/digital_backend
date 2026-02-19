"""
Servicio de generación de embeddings faciales usando FaceNet
"""
import numpy as np
import cv2
from typing import Optional
import tensorflow as tf
from tensorflow import keras


class FaceEmbedder:
    """Genera embeddings faciales usando un modelo preentrenado basado en FaceNet"""
    
    def __init__(self):
        """Inicializa el modelo de embeddings"""
        self.input_size = (160, 160)
        self.model = self._build_simple_model()
        
    def _build_simple_model(self) -> keras.Model:
        """
        Construye un modelo simple para embeddings faciales
        En producción, reemplazar con FaceNet preentrenado
        """
        # Usamos MobileNetV2 como backbone (ligero y efectivo)
        base_model = keras.applications.MobileNetV2(
            input_shape=(160, 160, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # Congelar el modelo base
        base_model.trainable = False
        
        # Construir modelo de embeddings
        inputs = keras.Input(shape=(160, 160, 3))
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        # Capa de embedding con L2 normalization
        embeddings = keras.layers.Dense(128, activation=None, name='embeddings')(x)
        embeddings = keras.layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1),
            name='l2_normalize'
        )(embeddings)
        
        model = keras.Model(inputs=inputs, outputs=embeddings, name='face_embedder')
        
        return model
    
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocesa una imagen de rostro para el modelo
        
        Args:
            face_image: Imagen del rostro (BGR)
        
        Returns:
            Imagen preprocesada
        """
        # Asegurar que esté en el tamaño correcto
        if face_image.shape[:2] != self.input_size:
            face_image = cv2.resize(face_image, self.input_size)
        
        # Convertir BGR a RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Convertir a float32
        face_array = np.asarray(face_rgb, dtype=np.float32)
        
        return face_array
    
    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Genera un embedding para un rostro
        
        Args:
            face_image: Imagen del rostro (BGR, 160x160)
        
        Returns:
            Vector de embedding normalizado o None si hay error
        """
        try:
            # Preprocesar
            face_processed = self.preprocess_face(face_image)
            
            # Expandir dimensiones para batch
            face_batch = np.expand_dims(face_processed, axis=0)
            
            # Generar embedding
            embedding = self.model.predict(face_batch, verbose=0)
            
            # Retornar como vector 1D
            return embedding[0]
            
        except Exception as e:
            print(f"Error generando embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, face_images: list) -> np.ndarray:
        """
        Genera embeddings para múltiples rostros
        
        Args:
            face_images: Lista de imágenes de rostros
        
        Returns:
            Array de embeddings
        """
        processed_faces = []
        
        for face in face_images:
            processed = self.preprocess_face(face)
            processed_faces.append(processed)
        
        # Convertir a batch
        batch = np.array(processed_faces)
        
        # Generar embeddings
        embeddings = self.model.predict(batch, verbose=0)
        
        return embeddings
    
    def calculate_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calcula la distancia euclidiana entre dos embeddings
        
        Args:
            embedding1: Primer embedding
            embedding2: Segundo embedding
        
        Returns:
            Distancia euclidiana
        """
        return np.linalg.norm(embedding1 - embedding2)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calcula la similitud coseno entre dos embeddings
        
        Args:
            embedding1: Primer embedding
            embedding2: Segundo embedding
        
        Returns:
            Similitud coseno (0-1)
        """
        # Los embeddings ya están normalizados, así que el producto punto es la similitud coseno
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
