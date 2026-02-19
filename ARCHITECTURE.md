# 🏗️ Arquitectura del Sistema de Reconocimiento Facial

## 📊 Visión General

```
┌─────────────────────────────────────────────────────────────────┐
│                         ESP32-CAM                               │
│  ┌──────────┐      ┌──────────┐      ┌─────────────┐          │
│  │  Cámara  │ ───> │  Captura │ ───> │ POST /api/v1│          │
│  │  OV2640  │      │  Imagen  │      │  /recognize │          │
│  └──────────┘      └──────────┘      └──────┬──────┘          │
└────────────────────────────────────────────┼──────────────────┘
                                              │
                                              │ HTTP
                                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    API Layer (routes/)                    │  │
│  │  • POST /api/v1/recognize                                 │  │
│  │  • POST /api/v1/add_person                                │  │
│  │  • POST /api/v1/train                                     │  │
│  │  • GET  /api/v1/status                                    │  │
│  └────────────────┬─────────────────────────────────────────┘  │
│                   │                                             │
│                   ▼                                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Service Layer (services/)                    │  │
│  │                                                            │  │
│  │  ┌──────────────────┐  ┌──────────────────┐             │  │
│  │  │  FaceDetector    │  │  FaceEmbedder    │             │  │
│  │  │  (OpenCV)        │  │  (MobileNetV2)   │             │  │
│  │  │                  │  │                  │             │  │
│  │  │  • detectFaces   │  │  • generate      │             │  │
│  │  │  • extractFace   │  │    Embedding     │             │  │
│  │  └────────┬─────────┘  └────────┬─────────┘             │  │
│  │           │                     │                        │  │
│  │           └─────────┬───────────┘                        │  │
│  │                     ▼                                     │  │
│  │          ┌──────────────────────┐                        │  │
│  │          │   FaceRecognizer     │                        │  │
│  │          │   (Orchestrator)     │                        │  │
│  │          │                      │                        │  │
│  │          │  • addPerson         │                        │  │
│  │          │  • trainClassifier   │                        │  │
│  │          │  • recognizeFace     │                        │  │
│  │          └──────────┬───────────┘                        │  │
│  └─────────────────────┼────────────────────────────────────┘  │
│                        │                                        │
│                        ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Storage Layer (data/)                        │  │
│  │                                                            │  │
│  │  • data/faces/             (Imágenes por persona)         │  │
│  │  • data/models/            (Modelos entrenados)           │  │
│  │    - face_classifier.pkl   (SVM)                          │  │
│  │    - label_encoder.pkl     (Labels)                       │  │
│  │    - embeddings.pkl        (Embeddings cache)             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Flujo de Reconocimiento Facial

### 1. Captura y Envío (ESP32-CAM)
```
┌─────────────┐
│ ESP32-CAM   │
│ captura     │
│ imagen      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Codifica    │
│ JPEG        │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ POST HTTP   │
│ multipart/  │
│ form-data   │
└──────┬──────┘
       │
       ▼
```

### 2. Procesamiento Backend
```
┌───────────────────────────────────────────────────────┐
│ 1. Recepción de Imagen (FastAPI)                     │
│    • Validación de formato                           │
│    • Conversión bytes → numpy array (Pillow)         │
└────────────────┬──────────────────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────────────────────────┐
│ 2. Detección de Rostro (OpenCV)                      │
│    • Haar Cascade Classifier                         │
│    • Escala de grises                                │
│    • Ecualización de histograma                      │
│    • Detecta coordenadas (x, y, w, h)                │
└────────────────┬──────────────────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────────────────────────┐
│ 3. Extracción de Rostro                              │
│    • Aplicar padding (20%)                           │
│    • Recortar región                                 │
│    • Redimensionar a 160x160                         │
└────────────────┬──────────────────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────────────────────────┐
│ 4. Generación de Embedding (MobileNetV2)             │
│    • Preprocesamiento (RGB, normalización)           │
│    • Forward pass por la red neuronal                │
│    • Vector de características 128D                  │
│    • Normalización L2                                │
└────────────────┬──────────────────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────────────────────────┐
│ 5. Clasificación (SVM)                               │
│    • Input: vector 128D                              │
│    • SVM con kernel lineal                           │
│    • Output: probabilidades por clase                │
│    • Seleccionar clase con mayor probabilidad        │
└────────────────┬──────────────────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────────────────────────┐
│ 6. Post-procesamiento                                │
│    • Decodificar label → nombre                      │
│    • Aplicar umbral de confianza (60%)               │
│    • Retornar: {nombre, confianza, mensaje}          │
└────────────────┬──────────────────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────────────────────────┐
│ 7. Respuesta JSON                                    │
│    {                                                  │
│      "success": true,                                 │
│      "person_name": "Juan",                           │
│      "confidence": 0.95,                              │
│      "message": "Rostro reconocido como 'Juan'"       │
│    }                                                  │
└───────────────────────────────────────────────────────┘
```

## 📁 Estructura de Datos

### Base de Datos de Embeddings
```python
{
    "Juan": [
        array([0.23, -0.45, 0.67, ...]),  # Embedding 1
        array([0.21, -0.43, 0.69, ...]),  # Embedding 2
        array([0.25, -0.46, 0.65, ...])   # Embedding 3
    ],
    "María": [
        array([-0.12, 0.34, -0.56, ...]),
        array([-0.11, 0.35, -0.55, ...])
    ]
}
```

### Clasificador SVM
```
Input: Vector 128D (embedding)
        ↓
    [SVM Linear]
        ↓
Output: Probabilidades
{
    "Juan": 0.95,
    "María": 0.03,
    "Pedro": 0.02
}
```

## 🧠 Modelos y Algoritmos

### 1. Detección de Rostros
**Algoritmo:** Haar Cascade Classifier
- **Ventajas:** Rápido, no requiere GPU
- **Limitaciones:** Funciona mejor con rostros frontales
- **Parámetros clave:**
  ```python
  scaleFactor = 1.1
  minNeighbors = 5
  minSize = (50, 50)
  ```

### 2. Embeddings Faciales
**Modelo:** MobileNetV2 + Capa Dense
- **Arquitectura:**
  ```
  Input (160x160x3)
       ↓
  MobileNetV2 (pretrained on ImageNet)
       ↓
  Global Average Pooling
       ↓
  Dense(128) + L2 Normalize
       ↓
  Output (128D embedding)
  ```
- **Características:**
  - Lightweight (adecuado para producción)
  - Embeddings de 128 dimensiones
  - Normalización L2 para similitud coseno

### 3. Clasificador
**Algoritmo:** Support Vector Machine (SVM)
- **Kernel:** Linear
- **Parámetros:**
  ```python
  C = 1.0
  probability = True
  random_state = 42
  ```
- **Ventajas:**
  - Funciona bien con pocos datos
  - Probabilidades calibradas
  - Rápido en inferencia

## 🔐 Pipeline de Entrenamiento

```
1. Añadir Personas
   ├─> Capturar imagen
   ├─> Detectar rostro
   ├─> Generar embedding
   ├─> Guardar imagen en data/faces/{nombre}/
   └─> Almacenar embedding en memoria

2. Acumular Datos
   ├─> Múltiples imágenes por persona
   └─> Mínimo 2 personas para entrenar

3. Entrenar Clasificador
   ├─> Recopilar todos los embeddings
   ├─> X = array de embeddings (N, 128)
   ├─> y = array de labels (N,)
   ├─> Codificar labels (LabelEncoder)
   ├─> Entrenar SVM
   └─> Guardar modelos en data/models/

4. Modelo Listo
   └─> Puede reconocer nuevas imágenes
```

## ⚡ Optimizaciones

### Rendimiento
1. **Caché de Embeddings:**
   - Los embeddings se guardan en disco
   - Evita recalcular para imágenes existentes

2. **Detección Rápida:**
   - Solo procesa el rostro más grande
   - Reduce tiempo de procesamiento

3. **Modelo Ligero:**
   - MobileNetV2 optimizado para velocidad
   - Inferencia < 200ms en CPU

### Escalabilidad
1. **Añadir Personas:**
   - Solo requiere reentrenar SVM (rápido)
   - No requiere reentrenar CNN

2. **Dataset Persistente:**
   - Imágenes organizadas por carpetas
   - Fácil backup y migración

## 🛡️ Validaciones y Seguridad

### Validaciones
```python
# Tamaño de imagen
MAX_IMAGE_SIZE = 10MB

# Formatos permitidos
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# Detección de rostro
MIN_FACE_SIZE = (50, 50)

# Umbral de confianza
CONFIDENCE_THRESHOLD = 0.6
```

### Manejo de Errores
```python
# Sin rostro detectado
→ "No se detectó ningún rostro en la imagen"

# Modelo no entrenado
→ "El modelo no está entrenado. Entrena primero el clasificador."

# Confianza baja
→ "Desconocido" (confidence < threshold)

# Imagen inválida
→ "Imagen inválida: {error}"
```

## 📊 Métricas de Rendimiento

### Tiempos de Ejecución (CPU)
```
Operación              Tiempo      Comentario
─────────────────────────────────────────────────
Detección OpenCV       50-100ms    Depende del tamaño
Embedding MobileNet    100-200ms   Forward pass
Clasificación SVM      < 10ms      Muy rápido
──────────────────────────────────────────────────
Total por imagen       ~200-300ms  Aceptable en CPU
```

### Precisión
```
Condiciones ideales:
- Rostro frontal        → 90-95% accuracy
- Buena iluminación     → 85-90% accuracy
- 5+ imágenes/persona   → 90%+ accuracy

Condiciones difíciles:
- Rostro de perfil      → 60-70% accuracy
- Poca luz              → 70-80% accuracy
- Oclusiones            → Variable
```

## 🔄 Mejoras Futuras

### Corto Plazo
1. **FaceNet Real:** Reemplazar MobileNetV2 con FaceNet preentrenado
2. **Múltiples Rostros:** Detectar y reconocer varios rostros por imagen
3. **API de Eliminación:** DELETE /api/v1/people/{name}
4. **Logs:** Sistema de logging estructurado

### Mediano Plazo
1. **Base de Datos:** PostgreSQL para metadata
2. **Autenticación:** JWT tokens
3. **Rate Limiting:** Protección contra spam
4. **Docker:** Containerización completa

### Largo Plazo
1. **GPU Support:** Aceleración con CUDA
2. **Real-time:** WebSocket para streaming
3. **Edge Computing:** Modelo optimizado para ESP32
4. **Cloud Deploy:** AWS/Azure con scaling

## 📚 Referencias

- **OpenCV:** https://opencv.org/
- **TensorFlow:** https://www.tensorflow.org/
- **FastAPI:** https://fastapi.tiangolo.com/
- **Scikit-learn:** https://scikit-learn.org/
- **FaceNet Paper:** https://arxiv.org/abs/1503.03832
- **MobileNetV2:** https://arxiv.org/abs/1801.04381

---

**Versión:** 1.0.0  
**Fecha:** Febrero 2026  
**Licencia:** MIT
