"""
FastAPI Backend para Reconocimiento Facial con ESP32-CAM
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

#from app.routes import recognition
from app.routes import esp

# Crear aplicación FastAPI
app = FastAPI(
    title="Face Recognition API",
    description="API de reconocimiento facial para ESP32-CAM con OpenCV y modelos preentrenados",
    version="1.0.0"
)

# Configurar CORS para permitir peticiones desde ESP32-CAM
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar IPs del ESP32
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
#app.include_router(
#    recognition.router,
#    prefix="/api/v1",
#    tags=["Face Recognition"]
#)

app.include_router(
    esp.router,
    prefix="/api/v1/esp",
    tags=["ESP32 Devices"]
)


@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "Face Recognition API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


def main():
    """Inicia el servidor"""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()
