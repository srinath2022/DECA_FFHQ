import os, sys
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob

from . import detectors
import face_alignment


# FFHQ Dataset
class FFHQDataset(Dataset):
    def __init__(self, image_size, scale, trans_scale = 0, isEval=False):
        self.image_size  = image_size
        self.imagefolder = '/content/resized'
        self.images_list = os.listdir(self.imagefolder)
        self.scale = scale #[scale_min, scale_max]
        self.trans_scale = trans_scale # 0.5?
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        while(10):
            imgname = self.images_list[idx]
            image_path = os.path.join(self.imagefolder, imgname)
            image = imread(image_path)
            kpt = self.fan_landmarks(image)
            if len(kpt.shape) != 2:
                idx = np.random.randint(low=0, high=len(self.images_list))
                continue
            # print(kpt_path, kpt.shape)
            # kpt = kpt[:,:2]

            image = image/255.
            if len(image.shape) < 3:
                image = np.tile(image[:,:,None], 3)
            ### crop information
            tform = self.crop(image, kpt)
            ## crop 
            cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)

            # normalized kpt
            cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1

            ###
            images_array = torch.from_numpy(cropped_image.transpose(2,0,1)).type(dtype = torch.float32) #224,224,3
            kpt_array = torch.from_numpy(cropped_kpt).type(dtype = torch.float32) #224,224,3
                        
            data_dict = {
                'image': images_array,
                'landmark': kpt_array,
                # 'mask': mask_array
            }
            
            return data_dict
        
    def crop(self, image, kpt):
        left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
        top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])#+ old_size*0.1])
        trans_scale = (np.random.rand(2)*2 -1) * self.trans_scale
        center = center + trans_scale*old_size # 0.5
        
        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]

        size = int(old_size*scale)

        # crop image
        # src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        src_pts = np.array([[0,0], [0,h - 1], [w - 1, 0]])
        DST_PTS = np.array([[0,0], [0,self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform
    
    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            mask = np.zeros_like(vis_parsing_anno)
            # for i in range(1, 16):
            mask[vis_parsing_anno>0.5] = 1.
        else:
            mask = np.ones((h, w))
        return mask

    def fan_landmarks(self, image):
        preds = self.fa.get_landmarks(image)
        return np.array(preds)