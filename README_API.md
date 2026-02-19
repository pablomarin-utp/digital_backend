# 🎯 Face Recognition API - Backend

Backend de reconocimiento facial en Python con FastAPI, diseñado para trabajar con ESP32-CAM.

## 🚀 Características

- ✅ Detección de rostros con OpenCV (Haar Cascade)
- ✅ Generación de embeddings faciales con modelo preentrenado (MobileNetV2)
- ✅ Clasificación con SVM (Support Vector Machine)
- ✅ API REST con FastAPI
- ✅ Gestión automática de datasets por persona
- ✅ Soporte para múltiples personas
- ✅ Reentrenamiento dinámico del modelo

## 📁 Estructura del Proyecto

```
backend/
├── app/
│   ├── __init__.py
│   ├── config.py              # Configuración general
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py         # Modelos Pydantic
│   ├── services/
│   │   ├── __init__.py
│   │   ├── face_detector.py   # Detección con OpenCV
│   │   ├── face_embedder.py   # Embeddings faciales
│   │   └── face_recognizer.py # Sistema de reconocimiento
│   ├── routes/
│   │   ├── __init__.py
│   │   └── recognition.py     # Endpoints de la API
│   └── utils/
│       ├── __init__.py
│       └── image_utils.py     # Utilidades con Pillow
├── data/
│   ├── faces/                 # Dataset de rostros por persona
│   │   ├── persona1/
│   │   ├── persona2/
│   │   └── ...
│   └── models/                # Modelos entrenados
│       ├── face_classifier.pkl
│       ├── label_encoder.pkl
│       └── embeddings.pkl
├── main.py                    # Punto de entrada
├── pyproject.toml            # Dependencias
└── README.md
```

## 🛠️ Instalación

### Prerrequisitos

- Python 3.12+
- uv (gestor de paquetes)

### Pasos

1. **Clonar o navegar al directorio:**
```bash
cd backend
```

2. **Activar el entorno virtual:**
```bash
.venv\Scripts\activate
```

3. **Las dependencias ya están instaladas en pyproject.toml:**
   - FastAPI
   - Uvicorn
   - OpenCV-Python
   - TensorFlow
   - Scikit-learn
   - Pillow
   - NumPy

## 🎮 Uso

### Iniciar el servidor

```bash
python main.py
```

O con uvicorn directamente:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

El servidor estará disponible en: `http://localhost:8000`

Documentación interactiva: `http://localhost:8000/docs`

## 📡 Endpoints de la API

### 1. **Reconocer rostro**
```http
POST /api/v1/recognize
Content-Type: multipart/form-data

image: [archivo de imagen]
```

**Respuesta:**
```json
{
  "success": true,
  "person_name": "Juan",
  "confidence": 0.95,
  "message": "Rostro reconocido como 'Juan'"
}
```

### 2. **Añadir nueva persona**
```http
POST /api/v1/add_person
Content-Type: multipart/form-data

name: "Juan"
image: [archivo de imagen]
```

**Respuesta:**
```json
{
  "success": true,
  "message": "Persona 'Juan' añadida exitosamente",
  "images_saved": 1,
  "person_name": "Juan"
}
```

### 3. **Entrenar/reentrenar clasificador**
```http
POST /api/v1/train
```

**Respuesta:**
```json
{
  "success": true,
  "message": "Clasificador entrenado exitosamente",
  "total_people": 3,
  "total_samples": 15
}
```

### 4. **Obtener estado del sistema**
```http
GET /api/v1/status
```

**Respuesta:**
```json
{
  "status": "ok",
  "model_trained": true,
  "total_people": 3,
  "total_samples": 15
}
```

### 5. **Listar personas registradas**
```http
GET /api/v1/people
```

**Respuesta:**
```json
{
  "total_people": 3,
  "people": ["Juan", "María", "Pedro"]
}
```

### 6. **Recargar embeddings desde disco**
```http
POST /api/v1/reload_embeddings
```

## 🔧 Configuración

Edita `app/config.py` para ajustar:

```python
# Umbral de confianza para reconocimiento
CONFIDENCE_THRESHOLD = 0.6  # 60%

# Tamaño mínimo de rostro a detectar
MIN_FACE_SIZE = (50, 50)

# Tamaño máximo de imagen
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
```

## 📱 Integración con ESP32-CAM

### Ejemplo de código Arduino para ESP32-CAM:

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"

const char* serverUrl = "http://192.168.1.100:8000/api/v1/recognize";

void sendImage() {
  camera_fb_t * fb = esp_camera_fb_get();
  
  if(!fb) {
    Serial.println("Error capturando imagen");
    return;
  }
  
  HTTPClient http;
  http.begin(serverUrl);
  http.addHeader("Content-Type", "multipart/form-data; boundary=boundary");
  
  String body = "--boundary\r\n";
  body += "Content-Disposition: form-data; name=\"image\"; filename=\"photo.jpg\"\r\n";
  body += "Content-Type: image/jpeg\r\n\r\n";
  
  // Enviar imagen
  http.POST(body + String((char*)fb->buf, fb->len) + "\r\n--boundary--\r\n");
  
  String response = http.getString();
  Serial.println(response);
  
  esp_camera_fb_return(fb);
  http.end();
}
```

## 🧪 Flujo de Trabajo Recomendado

1. **Añadir personas al sistema:**
   ```bash
   # Enviar varias imágenes de cada persona
   POST /api/v1/add_person (name="Juan", image=foto1.jpg)
   POST /api/v1/add_person (name="Juan", image=foto2.jpg)
   POST /api/v1/add_person (name="María", image=foto1.jpg)
   ```

2. **Entrenar el modelo:**
   ```bash
   POST /api/v1/train
   ```

3. **Reconocer rostros:**
   ```bash
   POST /api/v1/recognize (image=foto_desconocida.jpg)
   ```

4. **Reentrenar cuando se añadan más personas:**
   ```bash
   POST /api/v1/train
   ```

## 🎯 Características Técnicas

### Detección de Rostros
- **Algoritmo:** Haar Cascade (OpenCV)
- **Preprocesamiento:** Ecualización de histograma
- **Padding:** 20% alrededor del rostro detectado

### Embeddings Faciales
- **Modelo Base:** MobileNetV2 (preentrenado en ImageNet)
- **Dimensión:** 128D
- **Normalización:** L2

### Clasificación
- **Algoritmo:** SVM con kernel lineal
- **Probabilidades:** Habilitadas
- **Umbral de confianza:** 0.6 (configurable)

## 📊 Rendimiento

- **Detección:** ~50-100ms por imagen
- **Embedding:** ~100-200ms por rostro
- **Clasificación:** <10ms
- **Total:** ~200-300ms por imagen

## 🐛 Solución de Problemas

### Error: "No se detectó ningún rostro"
- Asegúrate de que la imagen tenga buena iluminación
- El rostro debe estar frontal y visible
- Tamaño mínimo: 50x50 píxeles

### Error: "El modelo no está entrenado"
- Ejecuta `POST /api/v1/train` después de añadir personas

### Baja confianza en reconocimiento
- Añade más imágenes de la persona (mínimo 3-5)
- Reentrena el modelo
- Ajusta `CONFIDENCE_THRESHOLD` en config.py

## 📝 TODO / Mejoras Futuras

- [ ] Implementar FaceNet real preentrenado
- [ ] Soporte para múltiples rostros en una imagen
- [ ] API para eliminar personas
- [ ] Logs y métricas de uso
- [ ] Autenticación JWT
- [ ] Docker container
- [ ] Tests unitarios

## 📄 Licencia

MIT License

## 👨‍💻 Autor

Proyecto de reconocimiento facial para ESP32-CAM
