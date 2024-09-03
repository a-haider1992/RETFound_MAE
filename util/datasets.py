# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
import re

# Custom collate function to extract information from the image filenames
def custom_collate(batch):
    images, labels, filenames = zip(*batch)
    
    # Extract the desired information from the filenames using regex
    info = []
    for name in filenames:
        match = re.search(r'N1[^_]*', name)  # This regex matches "N" followed by any characters until "_"
        if match:
            info.append(match.group())  # Add the matched string to the info list
        else:
            info.append(None)  # In case no match is found (though ideally there should be one)
    
    return torch.stack(images), torch.tensor(labels), info


# Define a wrapper around the Dataset to also return the filename
class ImageFolderWithFilenames(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithFilenames, self).__getitem__(index)
        path, _ = self.samples[index]
        filename = path.split('/')[-1]  # Extract the filename
        return original_tuple + (filename,)


def build_dataset(is_train, args):
    
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    # dataset = datasets.ImageFolder(root, transform=transform)
    dataset = ImageFolderWithFilenames(root, transform=transform)

    return dataset, transform


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train=='train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
