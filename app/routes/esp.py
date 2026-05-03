"""
Endpoints para gestionar los dispositivos ESP32 y el loop de captura/reconocimiento.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from app.config import ESP32_BRIDGE_URL
from app.utils.image_utils import bytes_to_numpy

router = APIRouter()

# ── Modelos ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    device_type: str
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

# ── Estado en memoria ─────────────────────────────────────────────────────────

registered_devices: dict[str, str] = {}
recent_recognitions: list[dict] = []

# LED tracking — seguimos el estado local para no togglear innecesariamente
_led_on: bool = False
_last_recognition_time: float = 0.0

# Umbral de similitud para abrir la puerta (ligeramente más estricto que el de reconocimiento)
DOOR_OPEN_THRESHOLD = 0.75
# Segundos sin movimiento tras el último reconocimiento antes de cerrar
LED_AUTO_OFF_SECS = 10

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_bridge_url() -> str | None:
    ip = registered_devices.get("bridge")
    return f"http://{ip}" if ip else None

def get_cam_url() -> str | None:
    ip = registered_devices.get("cam")
    return f"http://{ip}" if ip else None

async def _set_bulb(bridge_url: str, desired: bool) -> None:
    global _led_on
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.post(f"{bridge_url}/api/led", json={"state": desired})
            _led_on = r.json().get("led", desired)
        print(f"💡 Bombillo {'ENCENDIDO' if _led_on else 'APAGADO'}")
    except Exception as e:
        print(f"⚠️  Error bombillo: {e}")

# ── Endpoints de registro y dispositivos ──────────────────────────────────────

@router.post("/register", response_model=RegisterResponse)
async def register_device(request: RegisterRequest):
    device_type = request.device_type.lower()
    if device_type not in ["bridge", "cam"]:
        return RegisterResponse(success=False, message=f"Tipo inválido: {device_type}")
    registered_devices[device_type] = request.ip
    print(f"✅ [{device_type.upper()}] Registrado: {request.ip}")
    return RegisterResponse(success=True, message=f"{device_type} registrado con IP {request.ip}")

@router.get("/devices")
async def get_devices():
    return {"bridge": registered_devices.get("bridge"), "cam": registered_devices.get("cam")}

# ── Sensor y cámara ───────────────────────────────────────────────────────────

@router.get("/motion", response_model=MotionResponse)
async def get_motion():
    bridge_url = get_bridge_url()
    if not bridge_url:
        raise HTTPException(status_code=503, detail="ESP32 Bridge no registrado")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{bridge_url}/api/motion")
            r.raise_for_status()
            data = r.json()
            return MotionResponse(motion=data.get("motion", False), distance=data.get("distance"))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Bridge no disponible: {e}")

@router.get("/cam_ip")
async def get_cam_ip():
    ip = registered_devices.get("cam")
    if not ip:
        raise HTTPException(status_code=503, detail="ESP32-CAM no registrada")
    return {"ip": ip}

@router.get("/cam/status", response_model=CamStatusResponse)
async def get_cam_status():
    cam_url = get_cam_url()
    if not cam_url:
        raise HTTPException(status_code=503, detail="ESP32-CAM no registrada")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{cam_url}/status")
            r.raise_for_status()
            data = r.json()
            return CamStatusResponse(status=data.get("status", "unknown"), ip=data.get("ip"))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"CAM no disponible: {e}")

@router.get("/cam/capture")
async def capture_frame():
    cam_url = get_cam_url()
    if not cam_url:
        raise HTTPException(status_code=503, detail="ESP32-CAM no registrada")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{cam_url}/capture")
            r.raise_for_status()
            return Response(content=r.content, media_type="image/jpeg",
                            headers={"Cache-Control": "no-cache"})
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error capturando frame: {e}")

@router.get("/cam/stream")
async def stream_cam():
    cam_url = get_cam_url()
    if not cam_url:
        raise HTTPException(status_code=503, detail="ESP32-CAM no registrada")

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

# ── LED ───────────────────────────────────────────────────────────────────────

@router.post("/led", response_model=LedResponse)
async def toggle_led():
    global _led_on
    bridge_url = get_bridge_url()
    if not bridge_url:
        raise HTTPException(status_code=503, detail="Bridge no registrado")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(f"{bridge_url}/api/led")
            r.raise_for_status()
            _led_on = r.json().get("led", False)
            return LedResponse(led=_led_on)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error controlando LED: {e}")

@router.get("/led", response_model=LedResponse)
async def get_led_status():
    bridge_url = get_bridge_url()
    if not bridge_url:
        raise HTTPException(status_code=503, detail="Bridge no registrado")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{bridge_url}/api/led")
            r.raise_for_status()
            return LedResponse(led=r.json().get("led", False))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error obteniendo LED: {e}")

# ── Ventilador ────────────────────────────────────────────────────────────────

class FanResponse(BaseModel):
    fan: bool

@router.post("/fan", response_model=FanResponse)
async def toggle_fan():
    bridge_url = get_bridge_url()
    if not bridge_url:
        raise HTTPException(status_code=503, detail="Bridge no registrado")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(f"{bridge_url}/api/fan")
            r.raise_for_status()
            return FanResponse(fan=r.json().get("fan", False))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error controlando ventilador: {e}")

@router.get("/fan", response_model=FanResponse)
async def get_fan_status():
    bridge_url = get_bridge_url()
    if not bridge_url:
        raise HTTPException(status_code=503, detail="Bridge no registrado")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{bridge_url}/api/fan")
            r.raise_for_status()
            return FanResponse(fan=r.json().get("fan", False))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error obteniendo ventilador: {e}")

# ── Capturas y reconocimientos ────────────────────────────────────────────────

@router.get("/recognitions")
async def get_recognitions():
    """Últimos eventos de reconocimiento generados por el loop de captura."""
    return recent_recognitions

# ── Loop de captura + reconocimiento ─────────────────────────────────────────

async def motion_capture_loop() -> None:
    global _led_on, _last_recognition_time
    from app.routes.recognition import face_recognizer

    motion_start: float | None = None
    loop = asyncio.get_event_loop()

    while True:
        bridge_url = get_bridge_url()
        cam_url = get_cam_url()

        if not bridge_url or not cam_url:
            motion_start = None
            await asyncio.sleep(2)
            continue

        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{bridge_url}/api/motion")
                motion: bool = res.json().get("motion", False)
        except Exception:
            motion_start = None
            await asyncio.sleep(2)
            continue

        now = loop.time()

        if not motion:
            motion_start = None
            # Auto-apagado si han pasado LED_AUTO_OFF_SECS desde el último reconocimiento
            if _led_on and (now - _last_recognition_time) > LED_AUTO_OFF_SECS:
                await _set_bulb(bridge_url, False)
            await asyncio.sleep(1)
            continue

        if motion_start is None:
            motion_start = now
            await asyncio.sleep(0.3)
            continue

        if (now - motion_start) < 0.5:
            await asyncio.sleep(0.3)
            continue

        # 0.5 s de movimiento continuo → capturar y reconocer
        motion_start = None

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                frame_res = await client.get(f"{cam_url}/capture")

            if frame_res.status_code != 200:
                await asyncio.sleep(3)
                continue

            img_array = bytes_to_numpy(frame_res.content)
            name, confidence, message = "Desconocido", 0.0, "Sin cara detectada"
            if img_array is not None:
                _, rec_name, rec_conf, rec_msg = face_recognizer.recognize_face(img_array)
                name = rec_name or "Desconocido"
                confidence = round(rec_conf, 3) if rec_conf is not None else 0.0
                message = rec_msg

            # Controlar LED: abre la puerta solo si conocido Y sobre el umbral
            door_open = (name != "Desconocido") and (confidence >= DOOR_OPEN_THRESHOLD)
            await _set_bulb(bridge_url, door_open)

            now_dt = datetime.now()
            _last_recognition_time = now

            event = {
                "name": name,
                "confidence": confidence,
                "message": message,
                "timestamp": now_dt.isoformat(),
                "door": door_open,
            }
            recent_recognitions.insert(0, event)
            if len(recent_recognitions) > 10:
                recent_recognitions.pop()

            print(f"👤 {name} | sim={confidence:.2f} | puerta={'ABIERTA' if door_open else 'CERRADA'}")

        except Exception as e:
            print(f"⚠️  Error: {e}")

        await asyncio.sleep(3)  # cooldown
