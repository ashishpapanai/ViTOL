import os
import pdb
import random
from PIL import Image
import munch
import matplotlib
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms 
from wsol_model.vitol import generate_cam

from explainability.ViT_explanation_generator import LRP, Baselines


matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wsol_model.vitol import vitol
from config import get_configs
from data_loaders import get_data_loader
from inference import CAMComputer

def _compute_accuracy(self, loader):
    num_correct = 0
    num_images = 0

    for i, (images, targets, image_ids) in enumerate(tqdm(loader)):
        images = images.cuda()
        targets = targets.cuda()
        output_dict = self.model(images)

        if self.args.architecture_type =='vitol':
            pred = output_dict.argmax(dim=1)
        else:
            pred = output_dict['logits'].argmax(dim=1)

        num_correct += (pred == targets).sum().item()
        num_images += images.size(0)

    classification_acc = num_correct / float(num_images) * 100
    return classification_acc


model = vitol(
            dataset_name='ILSVRC',
            architecture_type='vitol',
            pretrained=False,
            num_classes=1000,
            large_feature_map=False,
            pretrained_path=None,
            adl_drop_rate=0.75,
            adl_drop_threshold=0.9,
            adl_layer = True,
            vit_type='vit_deit',
        )

model = model.cuda()
checkpoint = torch.load('./pretrained_weights/ViTOL-DeiT-B_IMAGENET_last.pth.tar')
model.load_state_dict(checkpoint['state_dict'], strict=True)
model.eval()

dataset = {'train': './dataset/ILSVRC', 'val': './dataset/ILSVRC', 'test': './dataset/ILSVRC'}

loaders = get_data_loader(
            data_roots=dataset,
            metadata_root='./metadata/ILSVRC',
            batch_size=64,
            workers=16,
            resize_size=224,
            crop_size=224,
            proxy_training_set=False,
            num_val_sample_per_class=0,
        )


test=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

img_path = './dataset/ILSVRC/val/ILSVRC2012_val_00000001.JPEG'

image = Image.open(img_path).convert('RGB')
image = test(image)

image = image.unsqueeze(0)
image = image.cuda()

# output_dict = model(image)
# pred = output_dict.argmax(dim=1)
# pred = pred.cpu().numpy()

target = [65]
# print(pred[0], target)

log_folder = './outputs/'

attribution_generator = Baselines(model)

cam = generate_cam(attribution_generator ,image, target, 'grad_rollout') # 16x16 size patch, total 14x14 patches
cam = cam.squeeze(0).cpu().numpy()
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
org_img = Image.open(img_path).convert('RGB')
org_img = org_img.resize((224, 224))
ax[0].imshow(org_img)
ax[0].set_title('Original Image')
ax[1].imshow(cam, cmap='jet', alpha=0.5)
ax[1].set_title('Grad-CAM')
ax[2].imshow(cam, cmap='jet', alpha=0.5)
ax[2].set_title('Grad-CAM Values')
for i in range(14):
    for j in range(14):
        #ax[2].text(j*16, i*16, str(round(cam[i*16, j*16], 2)), fontsize=5, color='white')
        ax[2].text(j, i, str(round(cam[i][j], 2)), ha="center", va="center", color="black", fontsize=5)
for ax in ax:
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(log_folder, 'cam.png'))

print('Saved CAM to {}'.format(os.path.join(log_folder, 'cam.png')))