"""
Endpoints para gestionar los dispositivos ESP32 y el loop de captura/reconocimiento.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from app.config import ESP32_BRIDGE_URL
from app.utils.image_utils import bytes_to_numpy

# Carpeta donde se guardan las capturas disparadas por el sensor de movimiento.
# Sirve para verificar manualmente qué está viendo la ESP32-CAM.
CAPTURES_DIR = Path(__file__).resolve().parent.parent.parent / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

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

# Estado del sensor — el ESP32 hace push aquí cada 500ms
_sensor_state: dict = {"motion": False, "distance": -1.0}

# Último frame recibido de la CAM (push desde la ESP32-CAM)
_latest_frame: bytes | None = None

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

async def _set_external_bulb(bridge_url: str, desired: bool) -> None:
    """Controla el BOMBILLO EXTERNO (LED_EXT, pin 26). El LED interno responde al sensor en el propio Bridge."""
    global _led_on
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.post(f"{bridge_url}/api/led", json={"state": desired})
            _led_on = r.json().get("led", desired)
        print(f"💡 Bombillo EXTERNO {'ENCENDIDO' if _led_on else 'APAGADO'}")
    except Exception as e:
        print(f"⚠️  Error bombillo externo: {e}")

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

@router.post("/motion")
async def push_motion(request: Request):
    """El ESP32 Bridge hace POST aquí cada 500ms con su estado."""
    global _sensor_state
    try:
        data = await request.json()
        _sensor_state = {
            "motion":   bool(data.get("motion", False)),
            "distance": float(data.get("distance", -1.0)),
        }
    except Exception:
        pass
    return {"ok": True}

@router.post("/cam/frame")
async def push_cam_frame(request: Request):
    """La ESP32-CAM hace POST aquí con un JPEG cada ~1 s."""
    global _latest_frame
    _latest_frame = await request.body()
    return {"ok": True}

@router.get("/motion", response_model=MotionResponse)
async def get_motion():
    """El frontend lee el último estado cacheado — sin llamar al ESP."""
    return MotionResponse(
        motion=_sensor_state["motion"],
        distance=_sensor_state["distance"],
    )

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


@router.get("/captures/{filename}")
async def get_capture(filename: str):
    """Devuelve una imagen guardada por el loop de reconocimiento."""
    # Evita path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Nombre inválido")
    path = CAPTURES_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Captura no encontrada")
    return Response(content=path.read_bytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"})

# ── Loop de captura + reconocimiento ─────────────────────────────────────────

async def motion_capture_loop() -> None:
    """
    Loop de control del BOMBILLO EXTERNO. El LED interno del Bridge ya se
    enciende localmente en el ESP cuando el sensor detecta movimiento — no
    pasa por aquí. Aquí solo decidimos si encender el externo en base al
    reconocimiento facial.

    Flujo:
      sensor detecta movimiento  →  Bridge prende LED interno (local)
                                 →  Bridge POSTea /motion al backend (este loop lo ve)
      este loop espera 100 ms de movimiento sostenido
                                 →  toma el último frame cacheado de la CAM
                                 →  embedder.get_embedding → similitud coseno
                                 →  si conocido (sim >= 0.75) → prende BOMBILLO EXTERNO
      sin movimiento por LED_AUTO_OFF_SECS → apaga BOMBILLO EXTERNO
    """
    global _led_on, _last_recognition_time
    from app.routes.recognition import face_recognizer

    print("🚀 motion_capture_loop iniciado")

    motion_start: float | None = None
    prev_motion = False
    loop = asyncio.get_event_loop()

    while True:
        try:
            # Leer desde caché (ESP32 hace push cada 500 ms) — sin HTTP
            motion: bool = _sensor_state["motion"]
            now = loop.time()

            # Log de transición — solo cuando cambia el estado
            if motion != prev_motion:
                if motion:
                    print(f"📡 MOVIMIENTO detectado — dist={_sensor_state['distance']:.1f} cm — LED interno del Bridge ya prendido (local)")
                else:
                    print("🌫️  Sin movimiento")
                prev_motion = motion

            if not motion:
                motion_start = None
                # Auto-apagado del BOMBILLO EXTERNO si pasan LED_AUTO_OFF_SECS sin actividad
                bridge_url = get_bridge_url()
                if bridge_url and _led_on and (now - _last_recognition_time) > LED_AUTO_OFF_SECS:
                    await _set_external_bulb(bridge_url, False)
                await asyncio.sleep(0.1)
                continue

            if motion_start is None:
                motion_start = now
                await asyncio.sleep(0.05)
                continue

            # 100 ms de movimiento sostenido → intentar reconocer
            if (now - motion_start) < 0.1:
                await asyncio.sleep(0.05)
                continue

            motion_start = None
            bridge_url = get_bridge_url()

            if _latest_frame is None:
                print("⚠️  Sin frame de CAM — esperando push (CAM envía cada 1 s)...")
                await asyncio.sleep(0.5)
                continue

            # Guardar la captura a disco para inspección manual.
            # Sirve para confirmar qué está enviando la ESP32-CAM realmente.
            capture_filename: str | None = None
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                capture_filename = f"motion_{ts}.jpg"
                capture_path = CAPTURES_DIR / capture_filename
                capture_path.write_bytes(_latest_frame)
                print(f"📸 Captura guardada: {capture_path}")
            except Exception as e:
                print(f"⚠️  No se pudo guardar captura: {e}")
                capture_filename = None

            img_array = bytes_to_numpy(_latest_frame)
            name, confidence, message = "Desconocido", 0.0, "Sin cara detectada"
            if img_array is not None:
                print(f"🔍 Reconociendo (frame {len(_latest_frame)} bytes)...")
                t0 = loop.time()
                _, rec_name, rec_conf, rec_msg = face_recognizer.recognize_face(img_array)
                elapsed = loop.time() - t0
                name = rec_name or "Desconocido"
                confidence = round(rec_conf, 3) if rec_conf is not None else 0.0
                message = rec_msg
                print(f"   → {name} | similitud_coseno={confidence:.2f} | {elapsed*1000:.0f} ms")
            else:
                print("⚠️  No se pudo decodificar la imagen de la CAM")

            # Bombillo EXTERNO: solo prende si la cara está registrada con confianza
            should_open = (name != "Desconocido") and (confidence >= DOOR_OPEN_THRESHOLD)
            if bridge_url:
                await _set_external_bulb(bridge_url, should_open)

            _last_recognition_time = now

            event = {
                "name": name,
                "confidence": confidence,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "door": should_open,
                "file": capture_filename,
            }
            recent_recognitions.insert(0, event)
            if len(recent_recognitions) > 10:
                recent_recognitions.pop()

            print(f"{'🔓' if should_open else '🔒'} {name} | sim={confidence:.2f} | bombillo externo={'ON' if should_open else 'OFF'}")

        except asyncio.CancelledError:
            print("🛑 motion_capture_loop cancelado")
            return
        except Exception as e:
            print(f"💥 Error inesperado en loop: {type(e).__name__}: {e}")
            await asyncio.sleep(1)

        await asyncio.sleep(1)  # cooldown entre reconocimientos
