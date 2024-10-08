import os
import shutil
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from util.datasets import custom_collate
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

    # dataset = datasets.ImageFolder(root=data, transform=transform)
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=custom_collate)
    dataloader = data

    # Set model to evaluation mode
    model.eval()

    count = 0
    count_0 = 0
    count_1 = 0
    count_2 = 0
    num_images = 100

    # Disable gradient calculation for faster inference
    with torch.no_grad():
        for batch_idx, (images, labels, info) in enumerate(dataloader):
            # Forward pass
            images = images.to('cuda')
            outputs = model(images)

            # pdb.set_trace()

            prediction_softmax = nn.Softmax(dim=1)(outputs)
            _, predicted = torch.max(prediction_softmax, 1)

            # if count > 300:
            #     break

            # Iterate over batch items and check predictions
            for i in range(len(images)):
                # Global check to stop if all counts have reached num_images
                if count_0 == num_images and count_1 == num_images and count_2 == num_images:
                    break

                # Check if prediction is correct (true label matches predicted label)
                if predicted[i] == labels[i]:
                    # Get the correct class label (as a string) from the true label
                    # class_label = dataset.classes[labels[i].item()]
                    class_label = str(labels[i].item())

                    # Skip saving if the count for this class has reached num_images
                    if class_label == '0' and count_0 >= num_images:
                        continue
                    if class_label == '1' and count_1 >= num_images:
                        continue
                    if class_label == '2' and count_2 >= num_images:
                        continue

                    # # Increment the respective class count
                    if class_label == '0':
                        count_0 += 1
                    elif class_label == '1':
                        count_1 += 1
                    elif class_label == '2':
                        count_2 += 1

                    # Create a folder for the class if it doesn't exist
                    class_folder = os.path.join(save_folder, class_label)
                    os.makedirs(class_folder, exist_ok=True)

                    # pdb.set_trace()
                    # Get the original image path from the dataset
                    # image_path = dataset.samples[batch_idx * dataloader.batch_size + i][0]
                    # image_name = os.path.basename(image_path)

                    # pdb.set_trace()

                    image_name = info["Filename"][i]
                    image_path = os.path.join('/data/oct_CFH_retfound/test', class_label, image_name)

                    # Define the destination path to save the original image
                    save_path = os.path.join(class_folder, image_name)

                    # Copy the original image to the corresponding class folder
                    shutil.copy(image_path, save_path)
                    count += 1

            # Check after processing the batch
            if count_0 == num_images and count_1 == num_images and count_2 == num_images:
                break
