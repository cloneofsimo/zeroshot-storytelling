'''
Visual Guidance with CLIP.
'''

from typing import List

import torch
from PIL import Image
import clip


class CLIPGuide:
    def __init__(self, device: str = "cuda:1"):
        model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
        self.model_clip = model_clip

        self.preprocess_clip = preprocess_clip

        self.device = device

    @torch.no_grad()
    def encode_image(self, img: Image.Image):
        img_tensor = self.preprocess_clip(img)
        img_tensor = img_tensor.to(self.device).unsqueeze(0)
        img_tensor = self.model_clip.encode_image(img_tensor)
        return img_tensor

    @torch.no_grad()
    def set_img(self, img_path):
        img = Image.open(img_path)
        self.img_tensor = self.encode_image(img)

    @torch.no_grad()
    def score(self, scentences: List[str], max_context: int = 256):
        img_features = self.img_tensor
        assert img_features.shape[0] == 1
        text_inputs = torch.cat([
            clip.tokenize(s[-max_context:]) for s in scentences
        ]).to(self.device)
        text_features = self.model_clip.encode_text(text_inputs)
        text_features /= torch.norm(text_features, dim=-1, keepdim=True)

        score = img_features @ text_features.T

        return score.cpu().float()

    @torch.no_grad()
    def batch_score(self,
                    scentences: List[str],
                    batch_size: int = 16,
                    max_context: int = 100) -> List[float]:
        scores = []
        jdx = 0
        while jdx < len(scentences):
            this_score = (self.score(
                scentences[jdx:jdx + batch_size],
                max_context=max_context).squeeze().tolist())
            scores.extend(this_score)
            jdx += batch_size

        return scores
