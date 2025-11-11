import torch.nn as nn
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import torchvision
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
import torch
_VALID_DATASETS = {'livec', 'AGIQA-3k', 'AGIQA-1k', 'SPAQ', "CSIQ", "TID2013", "kadid", "koniq", "SPAQ"}

def preprocess(img_path, dataset='livec'):
    """Dataset preprocess for IQA databases
    Args:
        img_path (str): image path
        dataset (str): datasets name
    """
    if dataset not in _VALID_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. "
            f"Currently available options are: {sorted(_VALID_DATASETS)}"
        )

    img = Image.open(img_path).convert("RGB")

    if dataset == 'AGIQA-1k':
        transforms = torchvision.transforms.Compose([
            Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    elif dataset == 'SPAQ':
        transforms = torchvision.transforms.Compose([
            Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    else:
        transforms = torchvision.transforms.Compose([
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    raw_image = transforms(img).unsqueeze(0)
    unfold = nn.Unfold(kernel_size=(224, 224), stride=128)
    img = unfold(raw_image).view(1, 3, 224, 224, -1)[0]
    img = img.permute(3,0,1,2).cuda()

    img_s = F.interpolate(raw_image, size=(224, 224), mode='bilinear', align_corners=False).to('cuda')
    img = torch.cat([img, img_s], dim=0)              

    return img