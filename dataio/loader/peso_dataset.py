import torch.utils.data as data
import numpy as np
import datetime

from os import listdir
from os.path import join
from os.path import basename
from .utils import load_nifti_img, check_exceptions, is_image_file, open_image_np,open_target_np, open_target_np_peso;                   
import random

class peso_dataset(data.Dataset):
    def find_in_y(self,x):
        y_lis = self.target_filenames
        match = [y for y in y_lis if x in y]
        return match[0]

    def __init__(self, root_dir, split, transform=None, preload_data=False,train_pct=0.8,balance=True):
        super(peso_dataset, self).__init__()
        img_dir= root_dir
        # targets are a comob of two dirs 1- normal 1024 patches 2- Tum 1024
        img= join(img_dir,"train")
        mask= join(img_dir,"mask")
        self.image_filenames  = sorted([join(img, x) for x in listdir(img) if is_image_file(x)])
        self.target_filenames = sorted([join(mask, x) for x in listdir(mask) if is_image_file(x)])
        sp= self.target_filenames.__len__()
        sp= int(train_pct *sp)
        random.shuffle(self.image_filenames)
        if split == 'train':
            self.image_filenames = self.image_filenames[:sp]
        else:
            self.image_filenames = self.image_filenames[sp:]
            # find the mask for the image
        self.target_filenames = [ self.find_in_y(basename(x)) for x in self.image_filenames]
        assert len(self.image_filenames) == len(self.target_filenames)

        # report the number of images in the dataset
        print('Number of {0} images: {1} patches'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = [open_image_np(ii)[0] for ii in self.image_filenames]
            self.raw_labels = [open_target_np_peso(ii)[0] for ii in self.target_filenames]
            print('Loading is done\n')


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        if not self.preload_data:
            input  = open_image_np(self.image_filenames[index])
            target  = open_target_np_peso(self.target_filenames[index])
        else:
            input = np.copy(self.raw_images[index])
            target = np.copy(self.raw_labels[index])

        # handle exceptions
        # check_exceptions(input, target)
        if self.transform:
            input, target = self.transform(input, target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)