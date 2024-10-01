import os
import shutil
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pdb

def save_correct_predictions(model, save_folder, data, transform=None):
    """
    Function to save correctly predicted images to class-specific folders.
    
    Args:
        model (torch.nn.Module): Trained model for predictions.
        dataloader (torch.utils.data.DataLoader): Dataloader for the dataset.
        dataset (torchvision.datasets.ImageFolder): Dataset containing image paths and labels.
        save_folder (str): Path to the folder where correctly predicted images will be saved.
    """

    # Ensure the save directory exists
    os.makedirs(save_folder, exist_ok=True)

    dataset = datasets.ImageFolder(root=data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Set model to evaluation mode
    model.eval()

    count = 0

    # Disable gradient calculation for faster inference
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Forward pass
            images = images.to('cuda')
            outputs = model(images)
            # pdb.set_trace()
            prediction_softmax = nn.Softmax(dim=1)(outputs)
            _, predicted = torch.max(prediction_softmax, 1)

            if count > 2000:
                break
            count += len(images)

            # Iterate over batch items and check predictions
            for i in range(len(images)):
                # Check if prediction is correct
                if predicted[i] == labels[i]:
                    # Get the correct class label (as a string)
                    class_label = dataset.classes[labels[i].item()]

                    # Create a folder for the class if it doesn't exist
                    class_folder = os.path.join(save_folder, class_label)
                    os.makedirs(class_folder, exist_ok=True)

                    # Get the original image path from the dataset
                    image_path = dataset.samples[batch_idx * dataloader.batch_size + i][0]
                    image_name = os.path.basename(image_path)

                    # Define the destination path to save the image
                    save_path = os.path.join(class_folder, image_name)

                    # Copy the image to the corresponding class folder
                    shutil.copy(image_path, save_path)
                    # print(f'Saved: {image_name} to {class_folder}')
