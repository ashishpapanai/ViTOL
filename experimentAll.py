import os
import pdb
import pandas as pd
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


# matplotlib.use("Agg")
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
    adl_layer=True,
    vit_type='vit_deit',
)

model = model.cuda()
checkpoint = torch.load(
    './pretrained_weights/ViTOL-DeiT-B_IMAGENET_last.pth.tar')
model.load_state_dict(checkpoint['state_dict'], strict=True)
model.eval()

dataset = {'train': './dataset/ILSVRC',
           'val': './dataset/ILSVRC', 'test': './dataset/ILSVRC'}

loaders = get_data_loader(
    data_roots=dataset,
    metadata_root='./metadata/ILSVRC',
    batch_size=1,
    workers=16,
    resize_size=224,
    crop_size=224,
    proxy_training_set=False,
    num_val_sample_per_class=0,
)


test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_loader = loaders['test']
log_folder = './comprehensive/FinalRun_withGT'
GT_Path = './metadata/ILSVRC/test/localization.txt'
# read GT boxes to a pandas dataframe
GT_df = pd.read_csv(GT_Path, sep=',', header=None)
GT_df.columns = ['ImageID', 'XMin', 'YMin', 'XMax', 'YMax']

for count, (images, labels, _) in enumerate(test_loader):
    index = 0
    image = images.cuda()
    image = image.unsqueeze(0)
    target = labels.cuda()
    target_val = target.item()
    attribution_generator = Baselines(model)
    cam, headwise, headwise_graded, layerwise, prop_lw = generate_cam(
        attribution_generator, image, target, 'grad_rollout')
    prop_lw.insert(0, torch.zeros(1, 197, 197))
    cam = cam.squeeze(0)
    cam = cam.reshape(14, 14)
    cam = cam.cpu().detach().numpy()
    cam_org = cam
    layers = []
    layers_prop = []
    heads = []
    heads_raw = []

    cam = cam_org
    cam = cv2.resize(cam, (224, 224))
    print(np.unique(cam))
    # scale to 0-255
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam = cam * 255
    threshold = 0.208
    _, the_gray_heatmap = cv2.threshold(
        src=cam,
        thresh=int(threshold * np.max(cam)),
        maxval=255,
        type=cv2.THRESH_BINARY)

    the_gray_heatmap = the_gray_heatmap.astype(np.uint8)
    contours = cv2.findContours(
        image=the_gray_heatmap,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)[1]

    estimated_boxes = []
    height, width = cam.shape
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        print(x, y, w, h)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    boxes = estimated_boxes
    org_img = images.squeeze(0).detach().cpu().numpy()
    org_img = np.transpose(org_img, (1, 2, 0))
    org_img = org_img * np.array([0.229, 0.224, 0.225]) + \
        np.array([0.485, 0.456, 0.406])
    org_img = np.clip(org_img, 0, 1)
    org_img = org_img * 255
    org_img = org_img.astype(np.uint8)
    org_img = cv2.resize(org_img, (224, 224))
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    org_img2 = org_img.copy()
    # invert image colors
    img = cv2.resize(org_img2, (224, 224))
    for box in boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    image_Val = GT_df.iloc[count]
    #XMin, YMin, XMax, XMin = GT_df.iloc[count]
    XMin = int(image_Val['XMin'])
    YMin = int(image_Val['YMin'])
    XMax = int(image_Val['XMax'])
    YMax = int(image_Val['YMax'])
    img_org = cv2.imread(os.path.join(
        './dataset/ILSVRC/', GT_df['ImageID'][count]))
    # rescale the GT boxes to 224x224 from img_org resolution
    XMin = int(XMin * 224 / img_org.shape[1])
    YMin = int(YMin * 224 / img_org.shape[0])
    XMax = int(XMax * 224 / img_org.shape[1])
    YMax = int(YMax * 224 / img_org.shape[0])
    cv2.rectangle(img, (XMin, YMin), (XMax, YMax), (0, 0, 255), 2)

    # make bbox dir in log folder
    bbox_dir = os.path.join(log_folder, 'bbox')
    if not os.path.exists(bbox_dir):
        os.mkdir(bbox_dir)

    # save bbox image
    target_val = target.item()
    cv2.imwrite(os.path.join(bbox_dir, 'bbox_' +
                str(count)+'_'+str(target_val)+'.jpg'), img)
    print('bbox saved at {}'.format(os.path.join(
        bbox_dir, 'bbox_'+str(count)+'_'+str(target_val)+'.jpg')))

    for layerNo in range(12):
        layer_cam = layerwise[layerNo]
        prop_layer = prop_lw[layerNo]
        layer_cam = layer_cam[:, 0, 1:]
        prop_layer = prop_layer[:, 0, 1:]
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
            heads_raw_single_final = heads_raw_single.squeeze(
                0).detach().cpu().numpy()
            heads.append(heads_single_final)
            heads_raw.append(heads_raw_single_final)

    fig, axs = plt.subplots(12, 27, figsize=(27, 12))
    for ax in fig.get_axes():
        ax.label_outer()
        ax.tick_params(labelsize=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    axs[0, 0].set_title('Img', fontsize=5)
    axs[0, 1].set_title('Layer', fontsize=5)
    axs[0, 2].set_title('Lyr Prp', fontsize=5)

    for i in range(0, 24, 2):
        axs[0, i+3].set_title('Head {}'.format(i//2), fontsize=5)
        axs[0, i+4].set_title('Head_Raw {}'.format(i//2), fontsize=5)

    for i in range(12):
        axs[i, 0].imshow(org_img2)
        axs[i, 0].axis('off')
        axs[i, 1].imshow(org_img2)
        layer_cpy = np.repeat(layers[i], 16, axis=0)
        layer_cpy = np.repeat(layer_cpy, 16, axis=1)
        axs[i, 1].imshow(layer_cpy, cmap='jet', alpha=0.5)
        axs[i, 1].axis('off')
        axs[i, 2].imshow(org_img2)
        layer_prop_cpy = np.repeat(layers_prop[i], 16, axis=0)
        layer_prop_cpy = np.repeat(layer_prop_cpy, 16, axis=1)
        axs[i, 2].imshow(layer_prop_cpy, cmap='jet', alpha=0.5)
        axs[i, 2].axis('off')
        for j in range(0, 24, 2):
            axs[i, j+3].imshow(org_img2)
            head_cpy = np.repeat(heads[index], 16, axis=0)
            head_cpy = np.repeat(head_cpy, 16, axis=1)
            axs[i, j+3].imshow(head_cpy, cmap='jet', alpha=0.5)
            axs[i, j+3].axis('off')
            axs[i, j+4].imshow(org_img2)
            head_raw_cpy = np.repeat(heads_raw[index], 16, axis=0)
            head_raw_cpy = np.repeat(head_raw_cpy, 16, axis=1)
            axs[i, j+4].imshow(head_raw_cpy, cmap='jet', alpha=0.5)
            axs[i, j+4].axis('off')
            index += 1

    fig.savefig(os.path.join(os.path.join(log_folder, 'comp_vis'), 'vis_' +
                str(count)+'_'+str(target_val)+'.png'), dpi=300, bbox_inches="tight")
    print('Time taken: ', time.time() - start_time)
    print('headwise cam saved at {}'.format(os.path.join(os.path.join(
        log_folder, 'comp_vis'), 'vis_'+str(count)+'_'+str(target_val)+'.png')))
    plt.close()
