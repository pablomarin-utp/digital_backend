"""
Modelos de datos Pydantic para la API
"""
from pydantic import BaseModel, Field
from typing import Optional


class RecognitionResponse(BaseModel):
    """Respuesta del reconocimiento facial"""
    success: bool
    person_name: Optional[str] = None
    confidence: Optional[float] = None
    message: str = ""


class AddPersonRequest(BaseModel):
    """Request para añadir una nueva persona"""
    name: str = Field(..., min_length=1, max_length=100, description="Nombre de la persona")


class AddPersonResponse(BaseModel):
    """Respuesta al añadir una persona"""
    success: bool
    message: str
    images_saved: int
    person_name: str


class RetrainResponse(BaseModel):
    """Respuesta del reentrenamiento"""
    success: bool
    message: str
    total_people: int
    total_samples: int


class StatusResponse(BaseModel):
    """Estado del sistema"""
    status: str
    model_trained: bool
    total_people: int
    total_samples: int
