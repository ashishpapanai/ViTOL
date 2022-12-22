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
            batch_size=1,
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

test_loader = loaders['test']
log_folder = './comprehensive/FinalVisuals_WSOD'
GT_Path = './metadata/ILSVRC/test/localization.txt'
# read GT boxes to a pandas dataframe
GT_df = pd.read_csv(GT_Path, sep=',', header=None)
GT_df.columns = ['ImageID', 'XMin', 'YMin', 'XMax', 'YMax']
target_df = pd.read_csv('./metadata/ILSVRC/test/class_labels.txt', sep=',', header=None)
count = 0
for imgageID in GT_df['ImageID']:
    print(imgageID)
    img = Image.open(os.path.join('./dataset/ILSVRC', imgageID)).convert('RGB')
    img = test(img)
    image = img.cuda()
    img = img.unsqueeze(0)
    image = img
    labels = target_df.loc[target_df[0] == imgageID, 1].values[0]
    labels = [labels]
    target = torch.tensor(labels).cuda()
    print(target)
    target_val = target.item()
    attribution_generator = Baselines(model)
    cam, headwise, headwise_graded, layerwise, prop_lw = generate_cam(attribution_generator ,image, target, 'grad_rollout') 
    prop_lw.insert(0, torch.zeros(1, 197, 197))
    cam = cam.squeeze(0)
    cam = cam.reshape(14, 14)
    cam = cam.cpu().detach().numpy()
    cam_org = cam
    layers = []
    layers_prop = []
    heads = []
    heads_raw = []

    cam  = cam_org
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
    org_img = img.squeeze(0).detach().cpu().numpy()
    org_img = np.transpose(org_img, (1, 2, 0))
    org_img = org_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
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
    img_org = cv2.imread(os.path.join('./dataset/ILSVRC/', GT_df['ImageID'][count]))
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
    # get image ID
    image_id = GT_df['ImageID'][count]
    # save image at ImageID from path= ./dataset/ILSVRC/val/ID
    img_path = os.path.join('./dataset/ILSVRC/', image_id)
    img_check = cv2.imread(img_path)
    # cv2.imwrite(os.path.join(bbox_dir, image_id), img_check)
    # print('Saved at {}'.format(os.path.join(bbox_dir, image_id)))


    cv2.imwrite(os.path.join(bbox_dir, 'bbox_'+str(count)+'_'+str(target_val)+'.jpg'), img)
    print('bbox saved at {}'.format(os.path.join(bbox_dir, 'bbox_'+str(count)+'_'+str(target_val)+'.jpg')))
    count = count + 1