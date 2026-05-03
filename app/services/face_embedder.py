from __future__ import annotations

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


class FaceEmbedder:
    def __init__(self):
        self.device = torch.device("cpu")
        self.detector = MTCNN(
            image_size=160,
            margin=20,
            keep_all=False,
            post_process=True,
            device=self.device,
        )
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        print("✓ FaceNet (VGGFace2) cargado")

    def get_embedding(self, img_bgr: np.ndarray) -> np.ndarray | None:
        try:
            # Downscale to 320px wide before MTCNN — 4× fewer pixels, same crop output
            h, w = img_bgr.shape[:2]
            if w > 320:
                scale = 320 / w
                img_bgr = cv2.resize(img_bgr, (320, int(h * scale)), interpolation=cv2.INTER_AREA)

            img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            face_tensor = self.detector(img_pil)
            if face_tensor is None:
                return None

            with torch.inference_mode():
                emb = self.model(face_tensor.unsqueeze(0))[0].numpy()

            norm = np.linalg.norm(emb)
            return emb / (norm + 1e-10)

        except Exception as e:
            print(f"Error generando embedding: {e}")
            return None

    def generate_embedding(self, face_image: np.ndarray) -> np.ndarray | None:
        return self.get_embedding(face_image)

    def calculate_similarity(self, e1: np.ndarray, e2: np.ndarray) -> float:
        return float(np.dot(e1, e2))
