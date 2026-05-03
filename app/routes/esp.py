"""
Endpoints para gestionar los dispositivos ESP32
"""
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from app.config import ESP32_BRIDGE_URL

router = APIRouter()

class RegisterRequest(BaseModel):
    device_type: str  # "bridge" o "cam"
    ip: str

class MotionResponse(BaseModel):
    motion: bool
    distance: float | None

class CamStatusResponse(BaseModel):
    status: str
    ip: str | None

class LedResponse(BaseModel):
    led: bool

class RegisterResponse(BaseModel):
    success: bool
    message: str

# Registro en memoria (en producción usar Redis o BD)
registered_devices: dict[str, str] = {}


@router.post("/register", response_model=RegisterResponse)
async def register_device(request: RegisterRequest):
    """
    ESP32 se registra al iniciar
    """
    device_type = request.device_type.lower()
    ip = request.ip

    if device_type not in ["bridge", "cam"]:
        return RegisterResponse(success=False, message=f"Tipo inválido: {device_type}")

    registered_devices[device_type] = ip
    print(f"✅ [{device_type.upper()}] Registrado: {ip}")
    return RegisterResponse(
        success=True,
        message=f"Dispositivo {device_type} registrado con IP {ip}"
    )


@router.get("/devices")
async def get_devices():
    """
    Lista dispositivos registrados
    """
    return {"bridge": registered_devices.get("bridge"), "cam": registered_devices.get("cam")}


def get_bridge_url() -> str | None:
    """Obtiene URL del Bridge"""
    ip = registered_devices.get("bridge")
    return f"http://{ip}" if ip else None


def get_cam_url() -> str | None:
    """Obtiene URL de la CAM"""
    ip = registered_devices.get("cam")
    return f"http://{ip}" if ip else None


@router.get("/motion", response_model=MotionResponse)
async def get_motion():
    """
    Obtiene datos del sensor de movimiento desde el ESP32 Bridge
    """
    bridge_url = get_bridge_url()
    if not bridge_url:
        raise HTTPException(status_code=503, detail="ESP32 Bridge no registrado")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{bridge_url}/api/motion")
            response.raise_for_status()
            data = response.json()
            return MotionResponse(
                motion=data.get("motion", False),
                distance=data.get("distance")
            )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ESP32 Bridge no disponible: {str(e)}")


@router.get("/cam_ip")
async def get_cam_ip():
    """
    Obtiene la IP de la ESP32-CAM
    """
    cam_ip = registered_devices.get("cam")
    if not cam_ip:
        raise HTTPException(status_code=503, detail="ESP32-CAM no registrada")
    return {"ip": cam_ip}


@router.get("/cam/status", response_model=CamStatusResponse)
async def get_cam_status():
    """
    Obtiene el estado de la ESP32-CAM
    """
    cam_url = get_cam_url()
    if not cam_url:
        raise HTTPException(status_code=503, detail="ESP32-CAM no registrada")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{cam_url}/status")
            response.raise_for_status()
            data = response.json()
            return CamStatusResponse(
                status=data.get("status", "unknown"),
                ip=data.get("ip")
            )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ESP32-CAM no disponible: {str(e)}")


@router.post("/led", response_model=LedResponse)
async def toggle_led():
    """
    Toggle del LED via Bridge
    """
    bridge_url = get_bridge_url()
    if not bridge_url:
        raise HTTPException(status_code=503, detail="ESP32 Bridge no registrado")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(f"{bridge_url}/api/led")
            response.raise_for_status()
            data = response.json()
            return LedResponse(led=data.get("led", False))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error controlando LED: {str(e)}")


@router.get("/led", response_model=LedResponse)
async def get_led_status():
    """
    Obtiene el estado del LED
    """
    bridge_url = get_bridge_url()
    if not bridge_url:
        raise HTTPException(status_code=503, detail="ESP32 Bridge no registrado")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{bridge_url}/api/led")
            response.raise_for_status()
            data = response.json()
            return LedResponse(led=data.get("led", False))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error obteniendo estado LED: {str(e)}")


@router.get("/cam/capture")
async def capture_frame():
    """
    Captura un frame de la ESP32-CAM
    """
    cam_url = get_cam_url()
    if not cam_url:
        raise HTTPException(status_code=503, detail="ESP32-CAM no registrada")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{cam_url}/capture")
            response.raise_for_status()
            return Response(
                content=response.content,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-cache"},
            )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error capturando frame: {str(e)}")


@router.get("/cam/stream")
async def stream_cam():
    """
    Proxy del stream MJPEG de la ESP32-CAM.
    Mantiene una sola conexión persistente hacia la CAM y la reenvía al browser.
    """
    cam_url = get_cam_url()
    if not cam_url:
        raise HTTPException(status_code=503, detail="ESP32-CAM no registrada")

    # El cliente no puede cerrarse mientras se hace streaming, se cierra en el generator
    client = httpx.AsyncClient(timeout=httpx.Timeout(None))

    async def generate():
        try:
            async with client.stream("GET", f"{cam_url}/stream") as response:
                async for chunk in response.aiter_bytes(4096):
                    yield chunk
        finally:
            await client.aclose()

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
    )