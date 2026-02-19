"""
Servicio de detección facial usando OpenCV
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path


class FaceDetector:
    """Detector de rostros usando Haar Cascade de OpenCV"""
    
    def __init__(self):
        """Inicializa el detector de rostros"""
        # Cargar el clasificador Haar Cascade preentrenado
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Error al cargar el clasificador Haar Cascade")
    
    def detect_faces(
        self, 
        image: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (50, 50)
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detecta rostros en una imagen
        
        Args:
            image: Imagen en formato numpy array (BGR)
            scale_factor: Factor de escala para detección
            min_neighbors: Mínimo de vecinos para considerar detección válida
            min_size: Tamaño mínimo del rostro (ancho, alto)
        
        Returns:
            Lista de tuplas (x, y, w, h) con las coordenadas de los rostros
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Ecualizar histograma para mejorar contraste
        gray = cv2.equalizeHist(gray)
        
        # Detectar rostros
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces.tolist() if len(faces) > 0 else []
    
    def extract_face(
        self, 
        image: np.ndarray, 
        face_coords: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (160, 160),
        padding: float = 0.2
    ) -> np.ndarray:
        """
        Extrae y normaliza un rostro de la imagen
        
        Args:
            image: Imagen original
            face_coords: Coordenadas del rostro (x, y, w, h)
            target_size: Tamaño objetivo del rostro extraído
            padding: Porcentaje de padding alrededor del rostro
        
        Returns:
            Imagen del rostro extraído y redimensionado
        """
        x, y, w, h = face_coords
        
        # Aplicar padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Calcular coordenadas con padding
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        # Extraer rostro
        face = image[y1:y2, x1:x2]
        
        # Redimensionar al tamaño objetivo
        face_resized = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)
        
        return face_resized
    
    def detect_and_extract_largest_face(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (160, 160)
    ) -> Optional[np.ndarray]:
        """
        Detecta y extrae el rostro más grande de la imagen
        
        Args:
            image: Imagen en formato numpy array
            target_size: Tamaño objetivo del rostro
        
        Returns:
            Rostro extraído o None si no se detectó ninguno
        """
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            return None
        
        # Encontrar el rostro más grande (por área)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        
        # Extraer el rostro
        return self.extract_face(image, largest_face, target_size)
    
    def draw_faces(
        self,
        image: np.ndarray,
        faces: List[Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Dibuja rectángulos alrededor de los rostros detectados
        
        Args:
            image: Imagen original
            faces: Lista de coordenadas de rostros
            color: Color del rectángulo (BGR)
            thickness: Grosor del rectángulo
        
        Returns:
            Imagen con rostros marcados
        """
        image_copy = image.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, thickness)
        
        return image_copy
