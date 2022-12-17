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
import time
from explainability.ViT_explanation_generator import LRP, Baselines


#matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wsol_model.vitol import vitol
from config import get_configs
from data_loaders import get_data_loader
from inference import CAMComputer
start_time = time.time()

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

cam, headwise, headwise_graded, layerwise, prop_lw = generate_cam(attribution_generator ,image, target, 'grad_rollout') # 16x16 size patch, total 14x14 patches
# all_head_attentions, all_head_grad_attentions, all_layer_attentions, prop_lw_attn
prop_lw.insert(0, torch.zeros(1, 197, 197))
print(cam.shape, headwise[0].shape, headwise_graded[0].shape, layerwise[0].shape, prop_lw[0].shape, len(prop_lw))
cam = cam.squeeze(0).detach().cpu().numpy()
#exit()
layers = []
layers_prop = []
heads = []
heads_raw = []
for layerNo in range(12):
    layer_cam = layerwise[layerNo]
    prop_layer = prop_lw[layerNo]
    layer_cam = layer_cam[:,0, 1:]
    prop_layer = prop_layer[:,0, 1:]
    layer_cam = layer_cam.reshape(1, 14, 14)
    prop_layer = prop_layer.reshape(1, 14, 14)
    l = []
    lw = []
    l.append(layer_cam)
    lw.append(prop_layer)
    l = torch.cat(l, 0)
    lw = torch.cat(lw, 0)
    layer_cam_final = l.squeeze(0).detach().cpu().numpy()
    prop_lw_final = lw.squeeze(0).detach().cpu().numpy()
    layers.append(layer_cam_final)
    layers_prop.append(prop_lw_final)
    head_cam = headwise_graded[layerNo]
    head_raw = headwise[layerNo]

    for headNo in range(12):
        heads_single = head_cam[headNo, 0, 1:]
        heads_raw_single = head_raw[headNo, 0, 1:]
        heads_single = heads_single.reshape(1, 14, 14)
        heads_raw_single = heads_raw_single.reshape(1, 14, 14)
        h = []
        hr = []
        h.append(heads_single)
        hr.append(heads_raw_single)
        h = torch.cat(h, 0)
        hr = torch.cat(hr, 0)
        heads_single_final = heads_single.squeeze(0).detach().cpu().numpy()
        heads_raw_single_final = heads_raw_single.squeeze(0).detach().cpu().numpy()
        heads.append(heads_single_final)
        heads_raw.append(heads_raw_single_final)


org_img = Image.open(img_path).convert('RGB')
org_img = org_img.resize((224, 224))

fig, axs = plt.subplots(12, 27)
fig.subplots_adjust(hspace=0, wspace=0)
for ax in fig.get_axes():
    ax.label_outer()
    ax.tick_params(labelsize=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

axs[0, 0].set_title('Img', fontsize=2)
axs[0, 1].set_title('Layer', fontsize=2)
axs[0, 2].set_title('Lyr Prp', fontsize=2)


for i in range(12):
    # write head and head number alternatively
    axs[0, i+3].set_title('Head {}'.format(i), fontsize=2)
    axs[0, i+15].set_title('Head_Raw {}'.format(i), fontsize=1.5)
    

for i in range(12):
    axs[i, 0].imshow(org_img)
    axs[i, 0].axis('off')
    axs[i, 1].imshow(org_img)
    layer_cpy = np.repeat(layers[i], 16, axis=0)
    layer_cpy = np.repeat(layer_cpy, 16, axis=1)
    axs[i, 1].imshow(layer_cpy, cmap='jet', alpha=0.5)
    axs[i, 1].axis('off')
    axs[i, 2].imshow(org_img)
    layer_prop_cpy = np.repeat(layers_prop[i], 16, axis=0)
    layer_prop_cpy = np.repeat(layer_prop_cpy, 16, axis=1)
    axs[i, 2].imshow(layer_prop_cpy, cmap='jet', alpha=0.5)
    axs[i, 2].axis('off')
    for j in range(12):
        axs[i, j+3].imshow(org_img)
        head_cpy = np.repeat(heads[i*12+j], 16, axis=0)
        head_cpy = np.repeat(head_cpy, 16, axis=1)
        axs[i, j+3].imshow(head_cpy, cmap='jet', alpha=0.5)
        axs[i, j+3].axis('off')

    for j in range(12):
        axs[i, j+15].imshow(org_img)
        head_raw_cpy = np.repeat(heads_raw[i*12+j], 16, axis=0)
        head_raw_cpy = np.repeat(head_raw_cpy, 16, axis=1)
        axs[i, j+15].imshow(head_raw_cpy, cmap='jet', alpha=0.5)
        axs[i, j+15].axis('off')


fig.savefig(os.path.join(log_folder, 'headwise_cam_mod.png'), dpi=1200, bbox_inches="tight")
print('Time taken: ', time.time() - start_time)
print('headwise cam saved at {}'.format(os.path.join(log_folder, 'headwise_cam_mod.png')))
plt.close()



# fig, axs = plt.subplots(12, 27)
# fig.subplots_adjust(hspace=0, wspace=0)
# for ax in fig.get_axes():
#     ax.label_outer()
#     ax.tick_params(labelsize=1)
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_aspect('equal')

# axs[0, 0].set_title('Img', fontsize=2)
# axs[0, 1].set_title('Layer', fontsize=2)
# axs[0, 2].set_title('Lyr Prp', fontsize=2)
# for i in range(12):
#     axs[0, i+3].set_title('Head {}'.format(i), fontsize=2)
#     axs[0, i+15].set_title('Head {}'.format(i), fontsize=2)

# for i in range(12):
#     axs[i, 0].imshow(org_img)
#     axs[i, 0].axis('off')
#     layer_cpy = np.repeat(layers[i], 16, axis=0)
#     layer_cpy = np.repeat(layer_cpy, 16, axis=1)
#     axs[i, 1].imshow(layer_cpy, cmap='jet', alpha=0.5)
#     for j in range(14):
#         for k in range(14):
#             axs[i, 1].text(k*16, j*16, '{:.2e}'.format(layers[i][j][k]), horizontalalignment='center', verticalalignment='center', fontsize=1)
#     axs[i, 1].axis('off')
#     layer_prop_cpy = np.repeat(layers_prop[i], 16, axis=0)
#     layer_prop_cpy = np.repeat(layer_prop_cpy, 16, axis=1)
#     axs[i, 2].imshow(layer_prop_cpy, cmap='jet', alpha=0.5)
#     for j in range(14):
#         for k in range(14):
#             axs[i, 2].text(k*16, j*16, '{:.2e}'.format(layers_prop[i][j][k]), horizontalalignment='center', verticalalignment='center', fontsize=1)
#     axs[i, 2].axis('off')
#     for j in range(12):
#         head_cpy = np.repeat(heads[i*12+j], 16, axis=0)
#         head_cpy = np.repeat(head_cpy, 16, axis=1)
#         axs[i, j+3].imshow(head_cpy, cmap='jet', alpha=0.5)
#         for k in range(14):
#             for l in range(14):
#                 axs[i, j+3].text(k*16, l*16, '{:.2e}'.format(heads[i*12+j][k][l]), horizontalalignment='center', verticalalignment='center', fontsize=1)
#         axs[i, j+3].axis('off')

#     for j in range(12):
#         head_raw_cpy = np.repeat(heads_raw[i*12+j], 16, axis=0)
#         head_raw_cpy = np.repeat(head_raw_cpy, 16, axis=1)
#         axs[i, j+15].imshow(head_raw_cpy, cmap='jet', alpha=0.5)
#         for k in range(14):
#             for l in range(14):
#                 axs[i, j+15].text(k*16, l*16, '{:.2e}'.format(heads_raw[i*12+j][k][l]), horizontalalignment='center', verticalalignment='center', fontsize=1)
#         axs[i, j+15].axis('off')

# fig.savefig(os.path.join(log_folder, 'headwise_cam_numeric.png'), dpi=1200, bbox_inches="tight")
# print('headwise cam saved at {}'.format(os.path.join(log_folder, 'headwise_cam_numeric.png')))
# plt.close()









