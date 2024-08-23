# backend/resnet_model.py
import torchvision.models as models
import torch
from torchvision import transforms
from PIL import Image
from .embedding_model import EmbeddingModel


class ResNetModel(EmbeddingModel):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.resnet152(pretrained=True).to(self.device)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def compute_image_embeddings(self, image_paths):
        images = [self.preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0) for image_path in image_paths]
        images = torch.cat(images).to(self.device)
        with torch.no_grad():
            embeddings = self.model(images).cpu().numpy()
        return embeddings

    def compute_text_embeddings(self, texts):
        raise NotImplementedError("ResNet does not support text embeddings.")
