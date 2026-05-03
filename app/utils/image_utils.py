"""
Utilidades para manejo de imágenes
"""
import io
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple


def bytes_to_numpy(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Convierte bytes de imagen a numpy array en formato BGR (OpenCV)
    
    Args:
        image_bytes: Bytes de la imagen
    
    Returns:
        Array numpy en formato BGR o None si hay error
    """
    try:
        # Abrir imagen con PIL
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convertir a RGB si es necesario
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convertir a numpy array
        image_array = np.array(pil_image)
        
        # PIL usa RGB, OpenCV usa BGR
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_bgr
    
    except Exception as e:
        print(f"Error convirtiendo bytes a numpy: {e}")
        return None


def numpy_to_bytes(image: np.ndarray, format: str = 'JPEG', quality: int = 90) -> Optional[bytes]:
    """
    Convierte numpy array a bytes de imagen
    
    Args:
        image: Array numpy en formato BGR
        format: Formato de salida ('JPEG', 'PNG')
        quality: Calidad de compresión (1-100)
    
    Returns:
        Bytes de la imagen o None si hay error
    """
    try:
        # Convertir BGR a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convertir a PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Guardar en bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format, quality=quality)
        
        return buffer.getvalue()
    
    except Exception as e:
        print(f"Error convirtiendo numpy a bytes: {e}")
        return None


def validate_image(image_bytes: bytes, max_size: int = 10 * 1024 * 1024) -> Tuple[bool, str]:
    """
    Valida que los bytes corresponden a una imagen válida
    
    Args:
        image_bytes: Bytes de la imagen
        max_size: Tamaño máximo permitido en bytes
    
    Returns:
        (válido, mensaje)
    """
    # Validar tamaño
    if len(image_bytes) > max_size:
        return False, f"Imagen demasiado grande. Máximo: {max_size / 1024 / 1024:.1f}MB"
    
    # Intentar abrir con PIL
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.verify()
        return True, "Imagen válida"
    
    except Exception as e:
        return False, f"Imagen inválida: {str(e)}"


def resize_image(image: np.ndarray, max_width: int = 1024, max_height: int = 1024) -> np.ndarray:
    """
    Redimensiona una imagen manteniendo el aspect ratio
    
    Args:
        image: Imagen en formato numpy array
        max_width: Ancho máximo
        max_height: Alto máximo
    
    Returns:
        Imagen redimensionada
    """
    height, width = image.shape[:2]
    
    # Calcular factor de escala
    scale = min(max_width / width, max_height / height, 1.0)
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image


