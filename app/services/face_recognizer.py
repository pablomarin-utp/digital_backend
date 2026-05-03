"""
Reconocimiento facial por similitud coseno sobre embeddings FaceNet.
Sin SVM: funciona desde la primera foto por persona.
"""
from __future__ import annotations

import cv2
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

from app.config import EMBEDDINGS_PATH, FACES_DIR
from app.services.face_embedder import FaceEmbedder

SIMILARITY_THRESHOLD = 0.70  # ajustar entre 0.65-0.80 según la ESP32-CAM


class FaceRecognizer:
    def __init__(self):
        self.embedder = FaceEmbedder()
        # {nombre: embedding_medio_normalizado (512D)}
        self.db: dict[str, np.ndarray] = {}
        self._load()

    # ── persistencia ─────────────────────────────────────────────────────────

    def _load(self):
        if EMBEDDINGS_PATH.exists():
            try:
                with open(EMBEDDINGS_PATH, "rb") as f:
                    self.db = pickle.load(f)
                print(f"✓ Face DB cargada: {list(self.db.keys())}")
            except Exception as e:
                print(f"Error cargando Face DB: {e}")
                self.db = {}

    def _save(self):
        EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(self.db, f)

    # ── operaciones principales ───────────────────────────────────────────────

    def add_person(self, name: str, image: np.ndarray) -> tuple[bool, str, int]:
        """
        Guarda la imagen en disco y actualiza el embedding medio de la persona.
        Retorna (éxito, mensaje, n_imágenes_guardadas).
        """
        emb = self.embedder.get_embedding(image)
        if emb is None:
            return False, "No se detectó ningún rostro en la imagen", 0

        # Guardar imagen original en disco
        person_dir = FACES_DIR / name
        person_dir.mkdir(parents=True, exist_ok=True)
        n = len(list(person_dir.glob("*.jpg"))) + 1
        cv2.imwrite(str(person_dir / f"{name}_{n}.jpg"), image)

        # Recalcular embedding medio con todas las fotos guardadas
        self.db[name] = self._mean_embedding(person_dir)
        self._save()

        total = len(list(person_dir.glob("*.jpg")))
        return True, f"'{name}' actualizado ({total} fotos)", 1

    def train_classifier(self) -> tuple[bool, str, dict]:
        """
        Recalcula todos los embeddings medios leyendo las imágenes de disco.
        Equivale al 'reentrenamiento' — tarda segundos, no GPU necesaria.
        """
        if not FACES_DIR.exists():
            return False, "No hay imágenes guardadas", {}

        new_db: dict[str, np.ndarray] = {}
        stats: dict[str, int] = {}

        for person_dir in sorted(FACES_DIR.iterdir()):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            mean_emb = self._mean_embedding(person_dir)
            if mean_emb is not None:
                new_db[name] = mean_emb
                stats[name] = len(list(person_dir.glob("*.jpg")))

        if not new_db:
            return False, "Ninguna imagen con cara válida encontrada", {}

        self.db = new_db
        self._save()

        total_people = len(new_db)
        total_samples = sum(stats.values())
        return (
            True,
            f"Actualizado: {total_people} personas, {total_samples} fotos",
            {"total_people": total_people, "total_samples": total_samples, "people": stats},
        )

    def recognize_face(
        self, image: np.ndarray
    ) -> tuple[bool, Optional[str], Optional[float], str]:
        """
        Reconoce el rostro en image usando similitud coseno.
        Retorna (éxito, nombre, similitud, mensaje).
        """
        if not self.db:
            return False, None, None, "No hay personas registradas. Añade personas primero."

        emb = self.embedder.get_embedding(image)
        if emb is None:
            return False, None, None, "No se detectó ningún rostro"

        best_name, best_sim = "Desconocido", -1.0
        for name, ref_emb in self.db.items():
            sim = float(np.dot(emb, ref_emb))
            if sim > best_sim:
                best_sim, best_name = sim, name

        if best_sim < SIMILARITY_THRESHOLD:
            return (
                True,
                "Desconocido",
                best_sim,
                f"Similitud {best_sim:.2f} < umbral {SIMILARITY_THRESHOLD}",
            )

        return True, best_name, best_sim, f"Reconocido como '{best_name}' ({best_sim:.2f})"

    def get_status(self) -> dict:
        total_samples = 0
        for name in self.db:
            d = FACES_DIR / name
            if d.exists():
                total_samples += len(list(d.glob("*.jpg")))
        return {
            "model_trained": len(self.db) > 0,
            "total_people": len(self.db),
            "total_samples": total_samples,
            "people": list(self.db.keys()),
        }

    def reload_embeddings_from_disk(self) -> tuple[bool, str, int]:
        success, msg, stats = self.train_classifier()
        total = stats.get("total_samples", 0) if isinstance(stats, dict) else 0
        return success, msg, total

    # ── helpers ───────────────────────────────────────────────────────────────

    def _mean_embedding(self, person_dir: Path) -> np.ndarray | None:
        embeddings: list[np.ndarray] = []
        for img_path in person_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is not None:
                e = self.embedder.get_embedding(img)
                if e is not None:
                    embeddings.append(e)
        if not embeddings:
            return None
        mean = np.mean(embeddings, axis=0)
        return mean / (np.linalg.norm(mean) + 1e-10)
