import kornia.augmentation as K
import torch.nn as nn

# input data should have one extra dimenssion for batch 
class MyAugmentationPipeline(nn.Module):
   def __init__(self) -> None:
      super(MyAugmentationPipeline, self).__init__()
      self.DF = K.RandomDepthicalFlip3D(p=0.3,same_on_batch = True,keepdim = True)
      self.HF = K.RandomHorizontalFlip3D(p=0.3,same_on_batch = True,keepdim = True)
      self.VF = K.RandomVerticalFlip3D(p=0.3,same_on_batch = True,keepdim = True)
      self.RO = K.RandomRotation3D((10., 10., 10.), p=0.7, same_on_batch=True, keepdim=True)
      
   def forward(self, input, mask):

      DF_params = self.DF.forward_parameters(input.shape)
      input = self.DF(input, DF_params)
      mask = self.DF(mask, DF_params)
      
      HF_params = self.HF.forward_parameters(input.shape)
      input = self.HF(input, HF_params)
      mask = self.HF(mask, HF_params)
      
      VF_params = self.VF.forward_parameters(input.shape)
      input = self.VF(input, VF_params)
      mask = self.VF(mask, VF_params)
      
      #RO_params = self.RO.forward_parameters(input.shape)
      #input = self.RO(input, RO_params)
      #mask = self.RO(mask, RO_params)


      return input, mask
  

