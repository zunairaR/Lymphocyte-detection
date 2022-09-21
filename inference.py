import cv2
import csv
import torch
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F

# import classification.clf_models as mdl
# import pretrainedmodels as ptm
import classification.helpers as utils
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from hybrid.configs import Configurations
from detectron2.engine import DefaultPredictor
import classification.lymp_net2_3class as c_model
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from skimage.measure import label, regionprops, regionprops_table
# 0. Preliminaries and Constants
imgs_root = r'/home/lymphocyte_Dataset'

fnames = os.listdir(imgs_root)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# empty cuda cache
torch.cuda.empty_cache()




# 1. Load Classification Model
model_name = 'Lymph-DilNet'
model = c_model.lymp_net(num_classes=3)


# 2. Load Detection Model
detection_model = 'LymphNet'
cf = Configurations('lyon_LymphNet', detection_model)
cfg = cf.get_configurations()
predictor = DefaultPredictor(cfg)

# 3. Construct Hybrid Pipeline
#   Original RGB Input
#       |-> Classification Model
#           |-> A tensor 1x3
#               |-> Label Based on Argmax and Scores using Softmax
#                   |-> If not Artifact and Stroma then pass
#                       |-> Detection Model

result_analysis = []


for f in tqdm(fnames, position=0, leave=True):
 # ---------------------- Going towards stage 2 ----------------------------
    TWO_STAGE = True
    if TWO_STAGE:
        input_beta_fp = os.path.join(imgs_root, f)
        input_beta = cv2.imread(input_beta_fp)
        output_beta = predictor(input_beta)
        mask = output_beta['instances'].get('pred_masks')
        mask = mask.detach().to('cpu')
        num, h, w = mask.shape
        bin_mask = np.zeros((h, w))
        for m in mask:
            m = m.detach().numpy()
            bin_mask += m
        label_mask = label(bin_mask)
        regions = regionprops(label_mask)
        for props in regions:
            y0, x0 = props.centroid
            area = props.area
            properties = [f, y0, x0, area]
            if area < 200:
                pred_2ignore.append(properties)
            if (y0 < 31 | y0 > 225) | (x0 < 31 | x0 > 225):
                pred_in_overlap.append(properties)
            else:
                result_analysis.append(properties)
        
    else:
        input_p = os.path.join(imgs_root, f)
        input_img = cv2.imread(input_p)
        output= predictor(input_img)
        predicted_boxes = output['instances'].pred_boxes.to('cpu').tensor.tolist()
        predicted_count = len(predicted_boxes)
        for rect in filtered_boxes:
            result = [f, , centerCoord[0], centerCoord[1], predicted_class] + classification_scores
            result_analysis.append(result)

