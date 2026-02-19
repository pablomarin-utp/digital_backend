"""
Endpoints de la API de reconocimiento facial
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional

from app.models.schemas import (
    RecognitionResponse,
    AddPersonRequest,
    AddPersonResponse,
    RetrainResponse,
    StatusResponse
)
from app.services.face_recognizer import FaceRecognizer
from app.utils.image_utils import bytes_to_numpy, validate_image

# Crear router
router = APIRouter()

# Instancia global del reconocedor
face_recognizer = FaceRecognizer()


@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_face(image: UploadFile = File(...)):
    """
    Reconoce un rostro en una imagen
    
    Args:
        image: Archivo de imagen con el rostro
    
    Returns:
        Información del reconocimiento (nombre y confianza)
    """
    try:
        # Leer bytes de la imagen
        image_bytes = await image.read()
        
        # Validar imagen
        is_valid, message = validate_image(image_bytes)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Convertir a numpy array
        image_array = bytes_to_numpy(image_bytes)
        if image_array is None:
            raise HTTPException(status_code=400, detail="Error procesando la imagen")
        
        # Reconocer rostro
        success, person_name, confidence, msg = face_recognizer.recognize_face(image_array)
        
        if not success:
            return RecognitionResponse(
                success=False,
                person_name=None,
                confidence=None,
                message=msg
            )
        
        return RecognitionResponse(
            success=True,
            person_name=person_name,
            confidence=confidence,
            message=msg
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.post("/add_person", response_model=AddPersonResponse)
async def add_person(
    name: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Añade una nueva persona al sistema
    
    Args:
        name: Nombre de la persona
        image: Imagen con el rostro de la persona
    
    Returns:
        Confirmación de la operación
    """
    try:
        # Validar nombre
        if not name or len(name.strip()) == 0:
            raise HTTPException(status_code=400, detail="El nombre no puede estar vacío")
        
        name = name.strip()
        
        # Leer bytes de la imagen
        image_bytes = await image.read()
        
        # Validar imagen
        is_valid, message = validate_image(image_bytes)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Convertir a numpy array
        image_array = bytes_to_numpy(image_bytes)
        if image_array is None:
            raise HTTPException(status_code=400, detail="Error procesando la imagen")
        
        # Añadir persona
        success, msg, images_saved = face_recognizer.add_person(name, image_array)
        
        if not success:
            raise HTTPException(status_code=400, detail=msg)
        
        return AddPersonResponse(
            success=True,
            message=msg,
            images_saved=images_saved,
            person_name=name
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.post("/train", response_model=RetrainResponse)
async def train_classifier():
    """
    Entrena o reentrena el clasificador con todas las personas registradas
    
    Returns:
        Resultado del entrenamiento
    """
    try:
        success, message, stats = face_recognizer.train_classifier()
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        return RetrainResponse(
            success=True,
            message=message,
            total_people=stats.get("total_people", 0),
            total_samples=stats.get("total_samples", 0)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.post("/reload_embeddings")
async def reload_embeddings():
    """
    Recarga embeddings desde las imágenes guardadas en disco
    Útil si se añadieron imágenes manualmente
    """
    try:
        success, message, total = face_recognizer.reload_embeddings_from_disk()
        
        return {
            "success": success,
            "message": message,
            "total_processed": total
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Obtiene el estado del sistema de reconocimiento facial
    
    Returns:
        Estado del sistema
    """
    try:
        status = face_recognizer.get_status()
        
        return StatusResponse(
            status="ok",
            model_trained=status["model_trained"],
            total_people=status["total_people"],
            total_samples=status["total_samples"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.get("/people")
async def list_people():
    """
    Lista todas las personas registradas en el sistema
    
    Returns:
        Lista de personas con estadísticas
    """
    try:
        status = face_recognizer.get_status()
        
        return {
            "total_people": status["total_people"],
            "people": status["people"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
