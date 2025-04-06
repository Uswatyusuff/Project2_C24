import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image_utils import *
import torchvision.transforms.functional as F

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root *4
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
      assert index < len(self), "Index out of range"

      img_path = self.lines[index]
      img, target = load_data(img_path, self.train)  # target is numpy

      # Resize both image and target to fixed shape
      resize_shape = (768, 1024)  # H, W
      img = img.resize((resize_shape[1], resize_shape[0]), Image.BILINEAR)

      # Resize density map while preserving total count
      if isinstance(target, np.ndarray):
          original_count = np.sum(target)
          
          # 👇 Match model's output resolution
          downsample_shape = (96, 128)  # 1/8th of 768x1024
          target = cv2.resize(target, (downsample_shape[1], downsample_shape[0]), interpolation=cv2.INTER_CUBIC)
          
          # 👇 Preserve the original count
          if np.sum(target) > 0:
              target = target * (original_count / np.sum(target))

      # Transform image
      if self.transform:
          img = self.transform(img)

      # Convert target to torch tensor
      target = torch.from_numpy(target).unsqueeze(0).float()  # Shape: [1, H, W]

      if index == 0:
          print("Image shape:", img.shape)
          print("Target shape:", target.shape)

      return img, target


