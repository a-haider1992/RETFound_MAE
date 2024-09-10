import os
import torch
import torchvision
from torchvision import datasets
from torch.utils.data import Dataset
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget, ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import pdb
import numpy as np
import cv2

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class ImagePathDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root=root_dir)  # Load dataset for paths and labels
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]  # Get the image file path and label
        return path, label  # Return the file path and label

def compute_and_save_heatmaps(model, save_dir, transform=None):
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Set the model to evaluation mode
    model.eval()
    target_layers = [model.module.blocks[-1].norm1]  # Adjust this line based on your model architecture

    test_dataset = ImagePathDataset(root_dir='Retfound_correct_predictions', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    scorecam = ScoreCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    targets = None
    cam.batch_size = 32

    # Iterate over the test images and their corresponding labels
    for batch_idx, (image_paths, targets) in enumerate(test_loader):
        for idx, img_path in enumerate(image_paths):
            # Load the image using cv2
            rgb_img = cv2.imread(img_path)[:, :, ::-1]  # Convert BGR (OpenCV) to RGB
            rgb_img = cv2.resize(rgb_img, (224, 224))
            rgb_img = np.float32(rgb_img) / 255.0  # Normalize to [0,1]

            # Preprocess the image for the model
            input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

            # Set the Grad-CAM target (specific class target for each image)
            target = ClassifierOutputTarget(targets[idx].item())

            # Compute the Grad-CAM heatmap
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=[target],
                                eigen_smooth=True,
                                aug_smooth=True)

            # Get the first grayscale CAM in the batch (only one image in this loop)
            grayscale_cam = grayscale_cam[0]

            # Overlay the Grad-CAM heatmap on the original image
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # Extract subfolder name from the image path
            subfolder = os.path.basename(os.path.dirname(img_path))
            image_name = os.path.basename(img_path)
            subfolder_dir = os.path.join(save_dir, subfolder)

            # Ensure subfolder exists
            os.makedirs(subfolder_dir, exist_ok=True)

            # Save the resulting heatmap
            save_path = os.path.join(subfolder_dir, f'heatmap_{batch_idx}_{idx}_{image_name}')
            cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
            # print(f'Heatmap saved at {save_path}')