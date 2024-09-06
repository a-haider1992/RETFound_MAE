import os
import torch
import torchvision
from torchvision import transforms
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2
import numpy as np
import timm

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def compute_and_save_heatmap_single_image(model, image_path, save_dir, transform=None):  
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Set the model to evaluation mode
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Choose the target layer for Grad-CAM (adapt for ViT architecture)
    target_layers = [model.blocks[-1].norm1]  # Adjust based on the ViT model structure

    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    # cam = ScoreCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    # If None, returns the map for the highest scoring category.
    # Set target for the dog class (class index for "Golden Retriever" is 207 in ImageNet)
    dog_class_index = 207  # Replace with the appropriate dog class index if necessary
    cat_class_index = 284  # Replace with the appropriate cat class index if necessary
    horse_class_index = 603  # Replace with the appropriate horse class index if necessary
    # cam_target = [ClassifierOutputTarget(dog_class_index)]
    # Otherwise, targets the requested category.
    targets = [ClassifierOutputTarget(dog_class_index)]

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=True,
                        aug_smooth=True)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    # Save the heatmap
    save_path = os.path.join(save_dir, 'heatmap_dog_class.png')
    cv2.imwrite(save_path, cam_image)

    print(f'Heatmap saved at {save_path}')

# Load a pre-trained Vision Transformer (ViT) model using timm
vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)

# Path to the single image (dog_cat.jpg)
image_path = 'dog_cat3.jpg'

# Directory where the heatmap will be saved
save_directory = 'single_image_heatmap'

# Compute and save the heatmap for the single image
compute_and_save_heatmap_single_image(vit_model, image_path, save_directory)
