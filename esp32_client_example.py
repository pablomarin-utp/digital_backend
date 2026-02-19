"""
Ejemplo de cliente ESP32-CAM para enviar imágenes al backend

Este código es un ejemplo conceptual en Python que simula
el comportamiento del ESP32-CAM enviando imágenes.

Para el código real de Arduino/ESP32, ver README_API.md
"""
import requests
import time
from pathlib import Path


class ESP32CAMSimulator:
    """Simula el comportamiento de un ESP32-CAM"""
    
    def __init__(self, server_url: str = "http://192.168.1.100:8000"):
        self.server_url = server_url
        self.recognize_endpoint = f"{server_url}/api/v1/recognize"
    
    def capture_and_send(self, image_path: str) -> dict:
        """
        Simula captura y envío de imagen
        
        Args:
            image_path: Ruta a la imagen a enviar
        
        Returns:
            Respuesta del servidor
        """
        try:
            print(f"📸 Capturando imagen: {image_path}")
            
            # Leer imagen
            with open(image_path, 'rb') as f:
                files = {'image': f}
                
                # Enviar al servidor
                print(f"📤 Enviando al servidor: {self.recognize_endpoint}")
                response = requests.post(
                    self.recognize_endpoint,
                    files=files,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Respuesta recibida: {result}")
                    return result
                else:
                    print(f"❌ Error HTTP {response.status_code}: {response.text}")
                    return {"success": False, "error": response.text}
        
        except requests.exceptions.ConnectionError:
            print("❌ Error: No se pudo conectar al servidor")
            return {"success": False, "error": "Connection error"}
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return {"success": False, "error": str(e)}
    
    def continuous_monitoring(self, image_path: str, interval: int = 5):
        """
        Simula monitoreo continuo (como haría el ESP32-CAM)
        
        Args:
            image_path: Imagen de prueba
            interval: Intervalo entre capturas (segundos)
        """
        print(f"🎥 Iniciando monitoreo continuo (intervalo: {interval}s)")
        print("Presiona Ctrl+C para detener\n")
        
        try:
            while True:
                result = self.capture_and_send(image_path)
                
                if result.get("success"):
                    person = result.get("person_name", "Desconocido")
                    confidence = result.get("confidence", 0)
                    
                    if person != "Desconocido":
                        print(f"👤 Persona detectada: {person} ({confidence:.2%})")
                    else:
                        print("👤 Persona desconocida")
                
                print(f"⏳ Esperando {interval} segundos...\n")
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n🛑 Monitoreo detenido")


def main():
    """Ejemplo de uso"""
    print("=" * 60)
    print("🎥 SIMULADOR DE ESP32-CAM")
    print("=" * 60)
    
    # Configurar servidor (ajusta la IP según tu red)
    server_url = "http://localhost:8000"  # Cambiar a IP del servidor
    
    # Crear simulador
    esp32 = ESP32CAMSimulator(server_url)
    
    # Ejemplo 1: Enviar una imagen
    print("\n📷 Ejemplo 1: Enviar imagen única")
    print("-" * 60)
    
    # Ajusta la ruta a una imagen de prueba
    test_image = "path/to/test_image.jpg"
    
    if Path(test_image).exists():
        result = esp32.capture_and_send(test_image)
    else:
        print(f"⚠️  Imagen de prueba no encontrada: {test_image}")
        print("   Ajusta la variable 'test_image' con una ruta válida")
    
    # Ejemplo 2: Monitoreo continuo
    print("\n\n📷 Ejemplo 2: Monitoreo continuo")
    print("-" * 60)
    print("⚠️  Comenta/descomenta esta sección según necesites")
    print()
    
    # Descomenta para activar monitoreo continuo:
    # if Path(test_image).exists():
    #     esp32.continuous_monitoring(test_image, interval=5)


if __name__ == "__main__":
    main()
