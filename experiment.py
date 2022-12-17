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
import cv2 
from explainability.ViT_explanation_generator import LRP, Baselines


#matplotlib.use("Agg")
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

cam, headwise, layerwise = generate_cam(attribution_generator ,image, target, 'grad_rollout') # 16x16 size patch, total 14x14 patches
print(cam.shape, headwise[0].shape, layerwise[0].shape)
cam = cam.squeeze(0).detach().cpu().numpy()

layers = []
heads = []
for layerNo in range(12):
    layer_cam = layerwise[layerNo]
    layer_cam = layer_cam[:,0, 1:]
    layer_cam = layer_cam.reshape(1, 14, 14)
    l = []
    l.append(layer_cam)
    l = torch.cat(l, 0)
    layer_cam_final = l.squeeze(0).detach().cpu().numpy()
    layers.append(layer_cam_final)
    head_cam = headwise[layerNo]

    for headNo in range(12):
        heads_single = head_cam[headNo, 0, 1:]
        heads_single = heads_single.reshape(1, 14, 14)
        h = []
        h.append(heads_single)
        h = torch.cat(h, 0)
        heads_single_final = heads_single.squeeze(0).detach().cpu().numpy()
        heads.append(heads_single_final)

print(len(layers))
print(len(heads))

org_img = Image.open(img_path).convert('RGB')
org_img = org_img.resize((224, 224))

# create a figure 12*14 with axes as hidden and all subplots close to each other with no space in between using cv2
fig = plt.figure(dpi=300)
fig.subplots_adjust(hspace=0.1, wspace=0.1)
fig.patch.set_visible(False)

# plot layerwise cam for all 12 layers
fig, axs = plt.subplots(12, 2, figsize=(14, 12), dpi=300, constrained_layout=True)

axs[0, 0].set_title('Original Image', fontsize=5)
axs[0, 1].set_title('Layerwise CAM', fontsize=5)
#fig.subplots_adjust(hspace=0, wspace=0)

for i in range(12):
    axs[i, 0].imshow(org_img)
    axs[i, 0].axis('off')
    axs[i, 1].imshow(org_img)
    # copy each value in layer 16*16 times, to make the array 224*224 along both axis of the array
    layer_cpy = np.repeat(layers[i], 16, axis=0)
    layer_cpy = np.repeat(layer_cpy, 16, axis=1)
    print(layer_cpy.shape)
    axs[i, 1].imshow(layer_cpy, cmap='jet', alpha=0.5)
    axs[i, 1].axis('off')

fig.savefig(os.path.join(log_folder, 'layerwise_cam.png'), dpi=300, bbox_inches="tight")
print('layerwise cam saved at {}'.format(os.path.join(log_folder, 'layerwise_cam.png')))
plt.close()

fig, axs = plt.subplots(12, 14)
fig.subplots_adjust(hspace=0, wspace=0)
for ax in fig.get_axes():
    ax.label_outer()
    ax.tick_params(labelsize=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

axs[0, 0].set_title('Image', fontsize=5)
axs[0, 1].set_title('Layer', fontsize=5)
for i in range(12):
    axs[0, i+2].set_title('Head {}'.format(i+1), fontsize=5)
for i in range(12):
    axs[i, 0].imshow(org_img)
    axs[i, 0].axis('off')
    axs[i, 1].imshow(org_img)
    layer_cpy = np.repeat(layers[i], 16, axis=0)
    layer_cpy = np.repeat(layer_cpy, 16, axis=1)
    axs[i, 1].imshow(layer_cpy, cmap='jet', alpha=0.5)
    axs[i, 1].axis('off')
    for j in range(12):
        axs[i, j+2].imshow(org_img)
        head_cpy = np.repeat(heads[i*12+j], 16, axis=0)
        head_cpy = np.repeat(head_cpy, 16, axis=1)
        axs[i, j+2].imshow(head_cpy, cmap='jet', alpha=0.5)
        axs[i, j+2].axis('off')

fig.savefig(os.path.join(log_folder, 'headwise_cam.png'), dpi=300, bbox_inches="tight")
print('headwise cam saved at {}'.format(os.path.join(log_folder, 'headwise_cam.png')))
plt.close()

# exit()
# fig, ax = plt.subplots(1, 3, figsize=(10, 5))
# org_img = Image.open(img_path).convert('RGB')
# org_img = org_img.resize((224, 224))
# ax[0].imshow(org_img)
# ax[0].set_title('Original Image')
# ax[1].imshow(cam, cmap='jet', alpha=0.5)
# ax[1].set_title('Grad-CAM')
# ax[2].imshow(cam, cmap='jet', alpha=0.5)
# ax[2].set_title('Grad-CAM Values')
# for i in range(14):
#     for j in range(14):
#         #ax[2].text(j*16, i*16, str(round(cam[i*16, j*16], 2)), fontsize=5, color='white')
#         ax[2].text(j, i, str(round(cam[i][j], 2)), ha="center", va="center", color="black", fontsize=5)
# for ax in ax:
#     ax.axis('off')

# plt.tight_layout()
# plt.savefig(os.path.join(log_folder, 'cam_gr.png'))

# # check if plot exists at saved path
# assert os.path.exists(os.path.join(log_folder, 'cam_gr.png'))

# print('Saved CAM to {}'.format(os.path.join(log_folder, 'cam_gr.png')))