# backend/clip_model.py
import clip
import torch
from PIL import Image
from .embedding_model import EmbeddingModel
import numpy as np
import cv2


class CLIPModel(EmbeddingModel):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", self.device)

    def compute_image_embeddings(self, image_paths):
        images = [self.preprocess(Image.open(image_path)).unsqueeze(0) for image_path in image_paths]
        images = torch.cat(images).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_image(images).cpu().numpy()
        return embeddings

    def compute_embedding_from_pixels(self, img):
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)

        image = [self.preprocess(im_pil).unsqueeze(0)]
        image = torch.cat(image).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_image(image).cpu().numpy()
        return embeddings

    def compute_text_embeddings(self, texts):
        tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_embeddings = self.model.encode_text(tokens).cpu().numpy()
        return text_embeddings

    def quantize_embeddings(self, embeddings):
        quantized = np.clip((embeddings + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return quantized