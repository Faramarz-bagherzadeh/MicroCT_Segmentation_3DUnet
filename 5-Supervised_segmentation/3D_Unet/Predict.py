import torch
#import torch.nn as nn
#import tifffile
from patchify import patchify, unpatchify
import numpy as np
import skimage
import cv2



            
def contrast_stretching(input_image):
    # Contrast stretching + Normalizing 0-255
    p2, p98 = np.percentile(input_image, (2, 98))
    stretched_image = skimage.exposure.rescale_intensity(input_image, in_range=(p2, p98))

    
    return stretched_image

def padd_to_proper(data):
    padded_image = np.pad(data[:,:,16:], ((0, 128-123), (11,10), (0, 0)), mode='constant') 
    return padded_image[:,:,:-21]
    

def predict_for_one_patch(img, model):
    
    img = contrast_stretching(img)
    
    img = torch.from_numpy(img).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = img.to(device)
    output = model(img)
    output = output.cpu()
    output = (output>0.5).float().numpy()
    #output = output[:,0,:,:,:]
    return output

def prediction (img, patch_size_, model):
    
    reshaped_img = padd_to_proper(img)
    img = reshaped_img
    
    original_img_shape = img.shape
    
    #image into patches
    img = patchify(img,(patch_size_, patch_size_, patch_size_) ,  step=patch_size_ )
    
    # image shape after patching
    original_patches_shape = img.shape
    
    
    img = img.reshape(img.shape[0]*img.shape[1]*img.shape[2],1,
                         patch_size_,patch_size_,patch_size_ )
    
    output = np.zeros_like(img)
    for i in range (img.shape[0]):
        output[i]= predict_for_one_patch(img[i:i+1], model)


    
    output = output[:,0,:,:,:]
    output= output.reshape(original_patches_shape)
    
    output = unpatchify(output, original_img_shape)
    
    return reshaped_img, output
