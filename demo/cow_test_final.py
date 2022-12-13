from cow_rcnn import KeypointDataset, get_frame, get_video, collate_fn, pred_keypoints
from stgcn import match_format, MakeNumpy
import torch

#!/usr/bin/env python
import argparse
import sys
sys.path.append('/home/bum/workspace/ASAP/demo/torchlight')

# torchlight
import torchlight
from torchlight import import_class

import json
import csv
import os
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    ### Keypoint Detection using RCNN
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
        print(DEVICE)

    model_path = '/home/bum/workspace/ASAP/KeypointRCNN/models/cows/models10_0.4921022457862968.pt'
    frame_path = '/home/bum/workspace/ASAP/Downloads/milkcow/image/'


    pred_key = pred_keypoints(frame_path, model_path, DEVICE)
    print(pred_key[:10])

    with open('cow_test_result.csv', 'w', newline = '') as output_file:
        f = csv.writer(output_file)
        f.writerow(['x', 'y', 'invisible'])

        for i in range(len(pred_key)):
            with  open('cow_test_result.csv', 'a', newline = '') as output_file:
                f = csv.writer(output_file)
                # f.writerow(new_file[i])
                for j in range(len(pred_key[i])):
                    f.writerow(pred_key[i][j])

       

