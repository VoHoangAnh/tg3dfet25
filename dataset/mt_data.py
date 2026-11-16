import numpy as np
from torch.utils.data import Dataset
import torch

import glob
import json
import pickle
import os 

from skimage.transform import estimate_transform, warp, resize, rescale
from skimage import io, img_as_float32
from imageio import mimread
import random 
from PIL import Image


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


def get_deca_tform(kpt, scale=1.25, resolution_inp=224):
    # (68, 2) keypoint coordinates in original image scale 

    left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
    top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
    
    old_size = (right - left + bottom - top) / 2 * 1.1
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
    
    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
    DST_PTS = np.array([[0,0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)  # linear transformation (3, 3), [0, 0, 1]   from original image to DECA cropped image
    return tform

class MTDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.png' or '.jpg'
      - folder with all frames
    """
    def __init__(self, root_dir, meta_dir, jsonfile, dataset_name, frame_shape=(256, 256, 3), id_sampling=False, is_train=True):
        ### meta data ###
        self.data = []
        f = open(jsonfile)
        motion_data = json.load(f)
        for item in motion_data:
            self.data.append(item)

        self.length = len(self.data) 
        self.map = {'AN':'anger','CONT':'contempt','FE':'fear',
                    'HA':'happy','SA':'sadness','NE':'neutral',
                    'SU':'surprise','DI':'disgust'}
        
        self.dataset_name = dataset_name
        ###################################
        #>>>>>>>>>>>>>>>>>>>>>>>> train / test split >>>>>>>>>>>>>>>>>>>>>>>>>>>>   
        self.videos = self.data # test_videos
            
        self.root_dir = root_dir
        self.meta_dir = meta_dir

        self.is_train = is_train
        self.deca_size = 224
        

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        caption = self.data[idx]['caption'][0]
        if self.dataset_name=='celebv':
            im_target1_path = f"{self.root_dir}/{self.data[idx]['subj']}/{self.map[self.data[idx]['lb_target1']]}/{self.data[idx]['img_target1'].replace('jpg', 'png')}"
            im_target2_path = f"{self.root_dir}/{self.data[idx]['subj']}/{self.map[self.data[idx]['lb_target2']]}/{self.data[idx]['img_target2'].replace('jpg', 'png')}"
            lm_target1_path = f"{self.meta_dir}/{self.data[idx]['subj']}/{self.map[self.data[idx]['lb_target1']]}/{self.data[idx]['img_target1'].replace('jpg', 'pkl')}"
            lm_target2_path = f"{self.meta_dir}/{self.data[idx]['subj']}/{self.map[self.data[idx]['lb_target2']]}/{self.data[idx]['img_target2'].replace('jpg', 'pkl')}"
        
        elif self.dataset_name=='ck':
            im_target1_path = f"{self.root_dir}/{self.data[idx]['lb_target1']}/{self.data[idx]['img_target1']}"
            im_target2_path = f"{self.root_dir}/{self.data[idx]['lb_target2']}/{self.data[idx]['img_target2']}"
            lm_target1_path = f"{self.meta_dir}/{self.data[idx]['lb_target1']}/{self.data[idx]['img_target1'].replace('png', 'pkl')}"
            lm_target2_path = f"{self.meta_dir}/{self.data[idx]['lb_target2']}/{self.data[idx]['img_target2'].replace('png', 'pkl')}"

        bs = len(caption)
        ## imgs ## 
        im1 = Image.open(im_target1_path)
        im2 = Image.open(im_target2_path)
        im1 = im1.resize((256, 256))
        im2 = im2.resize((256, 256))
        if not im1.mode == "RGB":
            im1 = im1.convert("RGB")
        
        if not im2.mode == "RGB":
            im2 = im2.convert("RGB")

        im1 = np.array(im1)
        im2 = np.array(im2)
        im1 = im1/ 255.
        im2 = im2/ 255.
        
        ## >>>>>>>>>>>>>>>>>>>>>>>>> load image file >>>>>>>>>>>>>>>>>>>>>>>>>>>>> ##
        video_array = []
        video_array.append(im1)
        video_array.append(im2)
        video_array = np.array(video_array)
        frame_idx = [0,1]

        video_array = video_array[frame_idx]        # (2 or T, H, W, 3), val: 0-255
        video_array = video_array / 255.            # val: 0-1

        ## >>>>>>>>>>>>>>>>>>>>>>>>> load meta data (2D landmarks) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        f1 = open(lm_target1_path, 'rb')
        im1_meta = pickle.load(f1)
        f1.close()
        
        f2 = open(lm_target2_path, 'rb')
        im2_meta = pickle.load(f2)
        f2.close()

        video_meta = []
        video_meta.append({'ldmk': im1_meta})
        video_meta.append({'ldmk': im2_meta})
        
        landmarks, tforms, crop_images, crop_lmks = [], [], [], []
        for idx, vid in enumerate(frame_idx):
            landmark = video_meta[vid]['ldmk']      # (68, 2)

            tform = get_deca_tform(landmark)        # (3, 3)

            # crop image to DECA format #
            crop_image = warp(video_array[idx], tform.inverse, output_shape=(self.deca_size, self.deca_size))   # (224, 224, 3), val:[0, 1]

            cropped_lmk = np.dot(tform.params, np.hstack([landmark, np.ones([landmark.shape[0],1])]).T).T       # landmark in (224, 224)
            cropped_lmk[:,:2] = cropped_lmk[:,:2] / self.deca_size * 2  - 1       # normalized to [-1, 1]

            landmarks.append(landmark)
            tforms.append(tform.params)
            crop_images.append(crop_image)
            crop_lmks.append(cropped_lmk)

        ## to tensor ##
        video_array = torch.from_numpy(video_array).float()     # tensor, (T', H, W, 3), val: [0, 1]
        video_array = video_array.permute(0, 3, 1, 2)           # (T', H, W, 3) --> (T', 3, H, W)
        video_array = (video_array - 0.5) / 0.5                 # (T', 3, H, W), [-1, 1]

        crop_images = torch.tensor(np.asarray(crop_images)).float() # (T', 224, 224, 3), val:0-1
        crop_images = crop_images.permute(0, 3, 1, 2)             # (T', 3, 224, 224), val:0-1

        landmarks = torch.tensor(np.asarray(landmarks)).float()     # (T', 68, 2)   in (256, 256)
        tforms = torch.tensor(np.asarray(tforms)).float()           # (T', 3, 3)
        crop_lmks = torch.tensor(np.asarray(crop_lmks)).float()     # (T', 68, 2)   normalized to [-1, 1]

        ## >>>>>>>>>>>>>>>>>>>>>>>>> output >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        out = {}

        out['name'] = im_target1_path
        ## source image ##
        out['src_img'] = video_array[1]  # index: 0 # (3, 256, 256), val: [-1, 1]
        out['src_lmk'] = landmarks[1]   # index: 0 # (68, 2), in (256, 256)
        out["src_tform"] = tforms[1]   # index: 0 # (3, 3)
        out['src_img_crop'] = crop_images[1] # index: 0 # (3, 224, 224), val:[0, 1]
        out['src_lmk_crop'] = crop_lmks[1]  # index: 0 # (68, 2), in [-1, 1]

        if self.is_train:
            out['drv_img'] = video_array[1]         # (3, 256, 256), val: [-1, 1]
            out['drv_lmk'] = landmarks[1]           # (68, 2), in (256, 256)
            out["drv_tform"] = tforms[1]            # (3, 3)
            out['drv_img_crop'] = crop_images[1]    # (3, 224, 224), val: [0, 1]
            out['drv_lmk_crop'] = crop_lmks[1]      # (68, 2), in [-1, 1]

        else:
            out['drv_img'] = video_array # video_array[1:]         # (T-1, 3, 256, 256), val:0-1 -> (1, 3, 256, 256), val:0-1 
            out['drv_img_crop'] = crop_images # crop_images[1:]    # (T-1, 3, 224, 224), val:0-1 -> (1, 3, 256, 256), val:0-1
            out["drv_tform"] = tforms # tforms[1:]            # (3, 3)

        out['caption'] = caption 
        
        return out
