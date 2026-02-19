"""
Script de prueba para la API de reconocimiento facial
"""
import requests
import os
from pathlib import Path

# Configuración
BASE_URL = "http://localhost:8000/api/v1"


def test_health():
    """Verifica que el servidor esté corriendo"""
    response = requests.get("http://localhost:8000/health")
    print(f"✓ Health Check: {response.json()}")


def test_status():
    """Obtiene el estado del sistema"""
    response = requests.get(f"{BASE_URL}/status")
    print(f"✓ Status: {response.json()}")


def test_add_person(name: str, image_path: str):
    """Añade una persona al sistema"""
    if not os.path.exists(image_path):
        print(f"✗ Imagen no encontrada: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {'name': name}
        response = requests.post(f"{BASE_URL}/add_person", files=files, data=data)
        
    print(f"✓ Add Person: {response.json()}")


def test_train():
    """Entrena el clasificador"""
    response = requests.post(f"{BASE_URL}/train")
    print(f"✓ Train: {response.json()}")


def test_recognize(image_path: str):
    """Reconoce un rostro"""
    if not os.path.exists(image_path):
        print(f"✗ Imagen no encontrada: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{BASE_URL}/recognize", files=files)
        
    print(f"✓ Recognize: {response.json()}")


def test_list_people():
    """Lista las personas registradas"""
    response = requests.get(f"{BASE_URL}/people")
    print(f"✓ People: {response.json()}")


def main():
    """Ejecuta pruebas de ejemplo"""
    print("=" * 60)
    print("🧪 PRUEBAS DE LA API DE RECONOCIMIENTO FACIAL")
    print("=" * 60)
    
    try:
        # 1. Health check
        print("\n1. Verificando servidor...")
        test_health()
        
        # 2. Estado inicial
        print("\n2. Estado del sistema:")
        test_status()
        
        # 3. Listar personas (si hay)
        print("\n3. Personas registradas:")
        test_list_people()
        
        # Ejemplo de uso (descomentar y ajustar rutas):
        """
        # 4. Añadir personas
        print("\n4. Añadiendo personas...")
        test_add_person("Juan", "path/to/juan1.jpg")
        test_add_person("Juan", "path/to/juan2.jpg")
        test_add_person("María", "path/to/maria1.jpg")
        
        # 5. Entrenar modelo
        print("\n5. Entrenando modelo...")
        test_train()
        
        # 6. Reconocer rostro
        print("\n6. Reconociendo rostro...")
        test_recognize("path/to/test_image.jpg")
        """
        
        print("\n" + "=" * 60)
        print("✅ Pruebas completadas")
        print("=" * 60)
        
        print("\n📝 Para probar con tus propias imágenes:")
        print("   1. Descomenta las líneas en main()")
        print("   2. Ajusta las rutas de las imágenes")
        print("   3. Ejecuta: python test_api.py")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: No se pudo conectar al servidor")
        print("   Asegúrate de que el servidor esté corriendo:")
        print("   python main.py")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
