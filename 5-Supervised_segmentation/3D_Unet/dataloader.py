
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip
import numpy as np
from torchvision import transforms
import tifffile
from patchify import patchify, unpatchify
import kornia
import skimage
import cv2
import random

from augmentation import MyAugmentationPipeline

rng = torch.manual_seed(0)
# Defining dataset
class CustomDataset(Dataset):

        
    def __init__(self, data_path, target_path, section, patch_size, transform, step):
        
        self.step = step
        
        self.data = tifffile.imread(data_path)

        self.data = self.contrast_stretching(self.data)
        
        self.target = tifffile.imread(target_path)
        self.transform = transform
        
        #padding to a proper shape that is power of 2
        #self.data = self.padd_to_proper(self.data)

        #self.target = self.padd_to_proper(self.target)
        self.original_shape = self.target.shape

        print ('data shape = ', self.data.shape)
        print ('data max = ', self.data.max())

        print ('target shape = ', self.target.shape)
        print ('target max = ', self.target.max())
        
        


        #Getting the proper section of the data (separate for train and test)
        start = int(section[0]*self.data.shape[0])
        end = int(section[1]*self.data.shape[0])
        
        self.data = self.data[start:end]
        self.target = self.target[start:end]
        #print ('section of data = ', self.data.shape)
        
        # Data and target into patches
        self.patch_size = patch_size
        self.data = self.patchyfy_img(self.data,self.patch_size)
        self.target = self.patchyfy_img(self.target,self.patch_size)
        
        

        
    def patchyfy_img(self,img, patch_size_):
        img = patchify(img,(patch_size_, patch_size_, patch_size_) ,  step=self.step )
        self.patchy_shape = img.shape
        img = img.reshape(img.shape[0]*img.shape[1]*img.shape[2],
                          patch_size_,patch_size_,patch_size_ )
        return img
        
    def padd_to_proper(self, data):
        padded_image = np.pad(data[:,53:-54,6:-10], ((0, 128-123), (0,0), (0, 0)), mode='constant') 
        #padded_image = np.pad(data[:,:,16:], ((0, 128-123), (11,10), (0, 0)), mode='constant') 
        return padded_image[:,:,:-21]
    
    def __len__(self):
        return len(self.data)
    
    def contrast_stretching(self,input_image):
        # Contrast stretching
        # Dropping extreems (artifacts)
        p2, p98 = np.percentile(input_image, (2, 98))
        stretched_image = skimage.exposure.rescale_intensity(input_image, in_range=(p2, p98))
        return stretched_image
    
    def __getitem__(self, index):

        data = torch.from_numpy(self.data[index]).unsqueeze(0).float()
        target = torch.from_numpy(self.target[index]).unsqueeze(0).float()
        
        
        
        def Random__gamma(img,gamma):
            gamma = random.randint(70, 130) * gamma
            print (gamma)
            image_float = img.astype(float)
            corrected_image = np.power(image_float, gamma)
            corrected_image = (corrected_image * 255).astype(np.uint8)
            return corrected_image
        
        
        def Random_Bluring_3D( img,sigma):

            sigma = random.randint(10, 20) * sigma
            blured = skimage.filters.gaussian(img,sigma=sigma,preserve_range=True)
            blured = blured.astype('uint8')
            blured = cv2.normalize(blured, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return blured
        
        
        def Random_speckle_noise_3D(image, sigma):
            sigma = random.randint(10, 20) * sigma
            # Generate random Gaussian noise
            depth, height, width = image.shape
            noise = np.random.randn(depth, height, width) * sigma

            # Add the noise to the image
            noisy_image = image + image * noise
        
            # Normalize the image to [0, 255]
            noisy_image = cv2.normalize(noisy_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            return noisy_image
        
        
        if self.transform is not None:
            
            #data = data.squeeze(0).numpy()

            # Adding random gamma
            #data = Random__gamma(data, 0.01)
            # Adding random noise
            #data = Random_speckle_noise_3D(data, 0.02)
            # Adding random bluriness
            #data = Random_Bluring_3D(data,0.1)
            
            # preparing for kornia operations
            #data = torch.from_numpy(data).float()
            #target = target.squeeze(0)
            # Concatinating the image and target to be able to pass through Kornia filters at once
            #concated = torch.concat([data/255,target], dim = 0)
            #data = data.unsqueeze(0)
            #target = target.unsqueeze(0)
            
            aug = MyAugmentationPipeline()
            data , target = aug.forward(data, target)
            
            target[target >= 0.5] = 1
            target[target < 0.5] = 0
            #print ('data ot of transformation max = ', data.max())
            # Expanding dimension for batching


        #target = torch.cat(((target<0.5),(target>0.5)), dim=0).float()
        #print ('final target shape', target.shape)
        
        return data, target


