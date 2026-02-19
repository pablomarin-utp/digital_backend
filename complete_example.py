"""
Ejemplo completo de uso de la API de reconocimiento facial
Demuestra todos los endpoints disponibles
"""
import requests
import json
from pathlib import Path
import time


class FaceRecognitionClient:
    """Cliente para interactuar con la API de reconocimiento facial"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
    
    def check_health(self) -> dict:
        """Verifica que el servidor esté corriendo"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def get_status(self) -> dict:
        """Obtiene el estado del sistema"""
        response = requests.get(f"{self.api_url}/status")
        return response.json()
    
    def list_people(self) -> dict:
        """Lista todas las personas registradas"""
        response = requests.get(f"{self.api_url}/people")
        return response.json()
    
    def add_person(self, name: str, image_path: str) -> dict:
        """
        Añade una persona al sistema
        
        Args:
            name: Nombre de la persona
            image_path: Ruta a la imagen
        
        Returns:
            Respuesta del servidor
        """
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'name': name}
            response = requests.post(
                f"{self.api_url}/add_person",
                files=files,
                data=data
            )
        return response.json()
    
    def train_model(self) -> dict:
        """Entrena el clasificador"""
        response = requests.post(f"{self.api_url}/train")
        return response.json()
    
    def recognize_face(self, image_path: str) -> dict:
        """
        Reconoce un rostro en una imagen
        
        Args:
            image_path: Ruta a la imagen
        
        Returns:
            Respuesta con nombre y confianza
        """
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(
                f"{self.api_url}/recognize",
                files=files
            )
        return response.json()
    
    def reload_embeddings(self) -> dict:
        """Recarga embeddings desde disco"""
        response = requests.post(f"{self.api_url}/reload_embeddings")
        return response.json()


def print_section(title: str):
    """Imprime un título de sección"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(data: dict):
    """Imprime un resultado de forma bonita"""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def example_basic_workflow():
    """
    Ejemplo del flujo básico de uso:
    1. Verificar conexión
    2. Añadir personas
    3. Entrenar
    4. Reconocer
    """
    print_section("🚀 EJEMPLO DE FLUJO BÁSICO")
    
    # Crear cliente
    client = FaceRecognitionClient()
    
    # 1. Verificar conexión
    print("\n1️⃣  Verificando conexión con el servidor...")
    try:
        health = client.check_health()
        print(f"✅ Servidor: {health['status']}")
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        print("   Asegúrate de que el servidor esté corriendo:")
        print("   python main.py")
        return
    
    # 2. Ver estado inicial
    print("\n2️⃣  Estado inicial del sistema:")
    status = client.get_status()
    print_result(status)
    
    # 3. Listar personas registradas
    print("\n3️⃣  Personas registradas:")
    people = client.list_people()
    print_result(people)
    
    # Nota: Para continuar, necesitas tener imágenes de prueba
    print("\n" + "⚠️  " * 20)
    print("Para continuar con el flujo completo:")
    print("1. Prepara imágenes de rostros (JPG/PNG)")
    print("2. Actualiza las rutas en la función example_with_images()")
    print("3. Ejecuta: python complete_example.py")
    print("⚠️  " * 20)


def example_with_images():
    """
    Ejemplo completo usando imágenes reales
    NOTA: Debes ajustar las rutas de las imágenes
    """
    print_section("📸 EJEMPLO COMPLETO CON IMÁGENES")
    
    client = FaceRecognitionClient()
    
    # ==========================================
    # ⚠️  AJUSTA ESTAS RUTAS CON TUS IMÁGENES
    # ==========================================
    images = {
        "Juan": [
            "path/to/juan_1.jpg",
            "path/to/juan_2.jpg",
            "path/to/juan_3.jpg"
        ],
        "María": [
            "path/to/maria_1.jpg",
            "path/to/maria_2.jpg"
        ],
        "Pedro": [
            "path/to/pedro_1.jpg",
            "path/to/pedro_2.jpg"
        ]
    }
    
    test_images = [
        "path/to/test_juan.jpg",
        "path/to/test_maria.jpg",
        "path/to/test_unknown.jpg"
    ]
    # ==========================================
    
    # Validar que existan las imágenes
    all_images = []
    for person_images in images.values():
        all_images.extend(person_images)
    all_images.extend(test_images)
    
    missing = [img for img in all_images if not Path(img).exists()]
    
    if missing:
        print("\n⚠️  Las siguientes imágenes no se encontraron:")
        for img in missing:
            print(f"   - {img}")
        print("\n💡 Actualiza las rutas en la función example_with_images()")
        return
    
    # 1. Añadir personas
    print("\n1️⃣  Añadiendo personas al sistema...")
    for person_name, person_images in images.items():
        print(f"\n   👤 {person_name}:")
        for i, image_path in enumerate(person_images, 1):
            result = client.add_person(person_name, image_path)
            if result.get('success'):
                print(f"      ✅ Imagen {i}/{len(person_images)} añadida")
            else:
                print(f"      ❌ Error: {result.get('message')}")
        time.sleep(0.5)
    
    # 2. Ver estado después de añadir personas
    print("\n2️⃣  Estado después de añadir personas:")
    status = client.get_status()
    print_result(status)
    
    # 3. Entrenar el modelo
    print("\n3️⃣  Entrenando el clasificador...")
    train_result = client.train_model()
    print_result(train_result)
    
    if not train_result.get('success'):
        print("❌ Error en el entrenamiento. No podemos continuar.")
        return
    
    print("✅ Modelo entrenado correctamente")
    
    # 4. Reconocer rostros de prueba
    print("\n4️⃣  Reconociendo rostros de prueba...")
    for i, test_image in enumerate(test_images, 1):
        print(f"\n   📷 Prueba {i}/{len(test_images)}: {Path(test_image).name}")
        
        result = client.recognize_face(test_image)
        
        if result.get('success'):
            person = result.get('person_name', 'Desconocido')
            confidence = result.get('confidence', 0)
            
            if person != "Desconocido":
                emoji = "✅" if confidence > 0.8 else "⚠️ "
                print(f"      {emoji} Persona: {person}")
                print(f"      📊 Confianza: {confidence:.2%}")
            else:
                print(f"      ❓ Persona no reconocida")
                print(f"      📊 Confianza: {confidence:.2%}")
        else:
            print(f"      ❌ Error: {result.get('message')}")
        
        time.sleep(0.5)
    
    # 5. Estado final
    print("\n5️⃣  Estado final del sistema:")
    final_status = client.get_status()
    print_result(final_status)
    
    print("\n✅ Flujo completo terminado exitosamente")
    print("\n💡 Puedes usar la API interactiva en: http://localhost:8000/docs")


def example_continuous_recognition():
    """
    Ejemplo de reconocimiento continuo (como ESP32-CAM)
    """
    print_section("🎥 EJEMPLO DE RECONOCIMIENTO CONTINUO")
    
    client = FaceRecognitionClient()
    
    # Imagen de prueba
    test_image = "path/to/test_image.jpg"
    
    if not Path(test_image).exists():
        print(f"\n⚠️  Imagen no encontrada: {test_image}")
        print("   Actualiza la variable 'test_image' con una ruta válida")
        return
    
    print("\n🎥 Iniciando monitoreo continuo...")
    print("   Presiona Ctrl+C para detener\n")
    
    try:
        count = 0
        while True:
            count += 1
            print(f"[{count}] 📸 Capturando y reconociendo...", end=" ")
            
            result = client.recognize_face(test_image)
            
            if result.get('success'):
                person = result.get('person_name', 'Desconocido')
                confidence = result.get('confidence', 0)
                
                if person != "Desconocido":
                    print(f"👤 {person} ({confidence:.1%})")
                else:
                    print(f"❓ Desconocido ({confidence:.1%})")
            else:
                print(f"❌ Error: {result.get('message')}")
            
            time.sleep(3)  # Esperar 3 segundos
    
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoreo detenido")
        print(f"Total de capturas: {count}")


def main():
    """Menú principal"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🎯 API de Reconocimiento Facial - Ejemplos          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("\n📋 Ejemplos disponibles:\n")
    print("1. 🚀 Flujo básico (sin imágenes)")
    print("2. 📸 Flujo completo con imágenes")
    print("3. 🎥 Reconocimiento continuo")
    print("0. ❌ Salir")
    
    choice = input("\n👉 Selecciona una opción: ").strip()
    
    if choice == "1":
        example_basic_workflow()
    elif choice == "2":
        example_with_images()
    elif choice == "3":
        example_continuous_recognition()
    elif choice == "0":
        print("\n👋 ¡Hasta luego!")
    else:
        print("\n⚠️  Opción inválida")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
