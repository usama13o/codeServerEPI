import torch.utils.data as data
import numpy as np
import os

from os import listdir
from os.path import join
from .utils import load_nifti_img, check_exceptions, is_image_file, open_image_np,open_target_np;                   


class TestDataset(data.Dataset):
    def __init__(self, root_dir, transform):
        super(TestDataset, self).__init__()
        image_dir = root_dir
        self.image_filenames = sorted([join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)])

        # Add the corresponding ground-truth images if they exist
        self.label_filenames = []
        label_dir = join(root_dir, 'labels')
        if os.path.isdir(label_dir):
            self.label_filenames = sorted([join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)])
            assert len(self.label_filenames) == len(self.image_filenames)

        # data pre-processing
        self.transform = transform

        # report the number of images in the dataset
        print('Number of test images: {0}'.format(self.__len__()))

    def __getitem__(self, index):

        # load the images
        input  = open_image_np(self.image_filenames[index])

        # load the label image if it exists
        if self.label_filenames:
            label, _ =  open_target_np(self.label_filenames[index])
            check_exceptions(input, label)
        else:
            label = []
            # check_exceptions(input)

        # Pre-process the input 3D Nifti image
        input = self.transform(input)

        return input, self.image_filenames[index], label

    def __len__(self):
        return len(self.image_filenames)