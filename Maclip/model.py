import torch
import torch.nn as nn
import clip
from .clip_model import load
import torch.nn.functional as F

from .utils import preprocess

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class Maclip(nn.Module):

    def __init__(self,
                 model_type='clipiqa',
                #  backbone='ViT-B/32',
                 backbone='RN50',
                 pretrained=True,
                 pos_embedding=False) -> None:
        super().__init__()

        self.clip_model = load(backbone, 'cpu')  # avoid saving clip weights
        # Different from original paper, we assemble multiple prompts to improve performance
        self.prompt_pairs = clip.tokenize([
            'Good image', 'bad image',
            'Sharp image', 'blurry image',
            'sharp edges', 'blurry edges',
            'High resolution image', 'low resolution image',
            'Noise-free image', 'noisy image',
        ])

        self.default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)

        self.model_type = model_type
        self.pos_embedding = pos_embedding
   
        for p in self.clip_model.parameters():
            p.requires_grad = False
 
    def box_cox(self, x, lam=0.5, epsilon=1e-6):
        x = (x) / (x.std(dim=1, keepdim=True) + epsilon)  # [B, D]
        if lam == 0:
            transformed = torch.log(x+1)
        else:
            transformed = ((x + 1) ** lam - 1) / lam

        return transformed

    def fusion(self, cos, norm, base_cos=1.0, base_norm=0.6, alpha=1.0):
        d = cos - norm 
        cos_param = base_cos + alpha * d
        norm_param = base_norm - alpha * d
        weights = F.softmax(torch.stack([cos_param, norm_param], dim=-1), dim=-1)  
        w_cos, w_norm = weights.unbind(dim=-1) 
        weighted_metric = w_cos * cos + w_norm * norm
        return weighted_metric, w_cos, w_norm

    def forward(self, x, dataset, box_lam=0.5, base_cos=1.0, base_norm=0.6, alpha=1.0):
        x = preprocess(x, dataset)
        clip_model = self.clip_model.to(x.device)
        prompts = self.prompt_pairs.to(x.device)
        logits_per_image, logits_per_text, image_features_org = clip_model(x, prompts, pos_embedding=self.pos_embedding)
        probs = logits_per_image.reshape(logits_per_image.shape[0], -1, 2).softmax(dim=-1)
        clipiqa = probs[..., 0].mean(dim=1, keepdim=True)

        image_features_org_abs = torch.abs(image_features_org)
        image_features_org_abs_box = self.box_cox(image_features_org_abs, lam=box_lam)
        nrm_score2 = image_features_org_abs_box.mean(dim=-1)
        comb, w1, w2 = self.fusion(clipiqa.squeeze(1), nrm_score2, base_cos, base_norm, alpha)
        comb = torch.mean(comb)
        
        return comb
    