# backend/embedding_model.py
from abc import ABC, abstractmethod
import numpy as np

class EmbeddingModel(ABC):
    @abstractmethod
    def compute_image_embeddings(self, image_paths):
        pass

    @abstractmethod
    def compute_text_embeddings(self, texts):
        pass

    def compute_embedding_from_pixels(self, img):
        pass

    def quantize_embeddings(self, embeddings):
        # Generic quantization, each model can override this to optimize use of the quantized space
        quantized = np.clip((embeddings + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return quantized
