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
import pdb

# Custom collate function to extract information from the image filenames
# def custom_collate(batch):
#     images, labels, filenames = zip(*batch)
    
#     # Extract the desired information from the filenames using regex
#     info = {"NicolaID": [], "Slice": []}
#     for name in filenames:
#         match = re.search(r'N1[^_]*', name)  # This regex matches "N" followed by any characters until "_"
#         match1 = re.search(r'oct_\d+', name)  # This regex matches "Slice" followed by any digits
#         if match:
#             info["NicolaID"].append(match.group())  # Add the matched string to the info list

#         if match1:
#             info["Slice"].append(match1.group())
    
#     return torch.stack(images), torch.tensor(labels), info


def custom_collate(batch):
    images, labels, filenames = zip(*batch)
    
    # Initialize info dictionary
    info = {"NicolaID": [], "Slice": [], "Timepoint": []}
    
    for name in filenames:
        # print(f"Filename: {name}")
        # Match NicolaID pattern (adjust the regex according to your filename format)
        match = re.search(r'N1[^_]*', name)  # This regex matches "N" followed by any characters until "_"
        
        # Match Slice pattern (adjust the regex according to your filename format)
        match1 = re.search(r'oct_(\d+)', name)  # This regex matches "oct_" followed by any chars until "."

        match2 = re.search(r'(\d+)_oct', name)  # This regex matches "oct_" followed by any digits
        
        # If matches are found, append to the lists; otherwise, append None or a placeholder
        info["NicolaID"].append(match.group() if match else "Unknown_NicolaID")
        info["Slice"].append(match1.group(1) if match1 else "Unknown_Slice")
        info["Timepoint"].append(match2.group(1) if match2 else "Unknown_Timepoint")

        if info["Slice"][-1] == "Unknown_Slice":
            print(f"Filename: {name}")
            print(info["Slice"])
            pdb.set_trace()
    
    # Stack images into a tensor and convert labels to tensor
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)

    assert len(info["NicolaID"]) == len(info["Slice"]) == len(info["Timepoint"]), "Mismatch between slices, NicolaID, and Timepoints count"
    
    return images_tensor, labels_tensor, info


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
