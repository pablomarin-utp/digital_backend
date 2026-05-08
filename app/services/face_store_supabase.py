"""
Backend de reconocimiento facial usando Supabase (pgvector + Storage).

- Los embeddings se guardan en la tabla `people` (vector 512D).
- Las fotos originales se suben al bucket `faces` de Supabase Storage.
- El reconocimiento usa la RPC `match_face` para la búsqueda vectorial.
- Se mantiene un caché en memoria para evitar llamadas a la DB en cada frame.
"""
from __future__ import annotations

import io
import cv2
import numpy as np
from typing import Optional
from datetime import datetime

from supabase import create_client, Client

from app.config import SUPABASE_URL, SUPABASE_KEY
from app.services.face_embedder import FaceEmbedder

SIMILARITY_THRESHOLD = 0.70
STORAGE_BUCKET = "faces"


class SupabaseFaceRecognizer:
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError(
                "Faltan SUPABASE_URL y/o SUPABASE_KEY en las variables de entorno"
            )
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.embedder = FaceEmbedder()

        # Caché local: {name: embedding 512D} — se recarga desde la DB al arrancar
        # y se actualiza en cada add_person, evitando llamadas RPC en cada frame.
        self._cache: dict[str, np.ndarray] = {}
        self._refresh_cache()

    # ── Caché ─────────────────────────────────────────────────────────────────

    def _refresh_cache(self) -> None:
        """Carga todos los embeddings de Supabase en memoria."""
        try:
            rows = self.client.table("people").select("name, embedding").execute().data
            self._cache = {
                r["name"]: np.array(r["embedding"], dtype=np.float32)
                for r in rows
            }
            print(f"✓ Supabase cache: {list(self._cache.keys())}")
        except Exception as e:
            print(f"Error cargando cache desde Supabase: {e}")

    # ── Operaciones principales ───────────────────────────────────────────────

    def add_person(self, name: str, image: np.ndarray) -> tuple[bool, str, int]:
        """
        1. Extrae el embedding de la imagen.
        2. Upsert del embedding en la tabla `people`.
        3. Sube la foto al bucket `faces`.
        4. Registra la ruta en `face_photos`.
        5. Actualiza el caché local.
        """
        emb = self.embedder.get_embedding(image)
        if emb is None:
            return False, "No se detectó ningún rostro en la imagen", 0

        # ── Upsert embedding ──────────────────────────────────────────────────
        try:
            result = (
                self.client.table("people")
                .upsert(
                    {"name": name, "embedding": emb.tolist()},
                    on_conflict="name",
                )
                .execute()
            )
            person_id = result.data[0]["id"]
        except Exception as e:
            return False, f"Error guardando en Supabase: {e}", 0

        # ── Subir foto a Storage ──────────────────────────────────────────────
        try:
            _, img_bytes = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            storage_path = f"{name}/{timestamp}.jpg"

            self.client.storage.from_(STORAGE_BUCKET).upload(
                path=storage_path,
                file=img_bytes.tobytes(),
                file_options={"content-type": "image/jpeg"},
            )

            # Registrar en face_photos
            self.client.table("face_photos").insert(
                {"person_id": person_id, "storage_path": storage_path}
            ).execute()
        except Exception as e:
            # La foto falla silenciosamente — el embedding ya está guardado
            print(f"Advertencia: no se pudo subir la foto: {e}")

        # ── Actualizar caché ──────────────────────────────────────────────────
        self._cache[name] = emb
        return True, f"'{name}' guardado en Supabase", 1

    def recognize_face(
        self, image: np.ndarray
    ) -> tuple[bool, Optional[str], Optional[float], str]:
        """
        1. Extrae el embedding del frame.
        2. Llama a la RPC `match_face` en Supabase (búsqueda vectorial coseno).
        3. Si la RPC falla, hace fallback al caché local.
        """
        if not self._cache:
            return False, None, None, "No hay personas registradas"

        emb = self.embedder.get_embedding(image)
        if emb is None:
            return False, None, None, "No se detectó ningún rostro"

        # ── RPC vectorial ─────────────────────────────────────────────────────
        try:
            rows = self.client.rpc(
                "match_face",
                {
                    "query_embedding": emb.tolist(),
                    "similarity_threshold": SIMILARITY_THRESHOLD,
                    "match_count": 1,
                },
            ).execute().data

            if rows:
                best = rows[0]
                name, sim = best["name"], float(best["similarity"])
                return True, name, sim, f"Reconocido como '{name}' ({sim:.2f}) [supabase]"
            else:
                # Ningún match sobre el umbral
                return True, "Desconocido", 0.0, f"Similitud < {SIMILARITY_THRESHOLD} [supabase]"

        except Exception as e:
            print(f"RPC falló, usando caché local: {e}")
            return self._recognize_from_cache(emb)

    def _recognize_from_cache(
        self, emb: np.ndarray
    ) -> tuple[bool, Optional[str], Optional[float], str]:
        """Fallback: similitud coseno sobre el caché en memoria."""
        best_name, best_sim = "Desconocido", -1.0
        for name, ref_emb in self._cache.items():
            sim = float(np.dot(emb, ref_emb))
            if sim > best_sim:
                best_sim, best_name = sim, name

        if best_sim < SIMILARITY_THRESHOLD:
            return True, "Desconocido", best_sim, f"Similitud {best_sim:.2f} < umbral [cache]"
        return True, best_name, best_sim, f"Reconocido como '{best_name}' ({best_sim:.2f}) [cache]"

    # ── Gestión ───────────────────────────────────────────────────────────────

    def train_classifier(self) -> tuple[bool, str, dict]:
        """Recarga el caché desde Supabase (equivale al 'reentrenar' local)."""
        self._refresh_cache()
        total = len(self._cache)
        return True, f"Caché actualizado: {total} personas", {
            "total_people": total,
            "total_samples": total,
        }

    def get_status(self) -> dict:
        try:
            rows = self.client.table("people").select("name").execute().data
            names = [r["name"] for r in rows]
            photo_count = (
                self.client.table("face_photos")
                .select("id", count="exact")
                .execute()
                .count or 0
            )
            return {
                "model_trained": len(names) > 0,
                "total_people": len(names),
                "total_samples": photo_count,
                "people": names,
                "store": "supabase",
            }
        except Exception as e:
            return {"model_trained": False, "total_people": 0, "total_samples": 0,
                    "people": [], "store": "supabase", "error": str(e)}

    def reload_embeddings_from_disk(self) -> tuple[bool, str, int]:
        """En modo Supabase no hay disco; recarga desde la DB."""
        self._refresh_cache()
        return True, "Caché recargado desde Supabase", len(self._cache)
