import os
import pandas as pd
import random
import torch
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import mlflow
import mlflow.pytorch
from sklearn.metrics import precision_score, recall_score, f1_score
import networks
from datasetmanager import *


# Device settings
if(torch.cuda.is_available()):
    device = 'cuda'
    print("\033[32mCUDA is used as the device.\033[0m")
else:
    device = 'cpu'
    print("\033[31mCPU is used as the device.\033[0m")


transform_ = transforms.Compose(
    [transforms.Resize((256, 256)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# Directories
dataset_directory = 'C:\\Users\\ardad\\Desktop\\Yazılım ve Elektronik\\Data'
train_directory = os.path.join(dataset_directory, "Train")
val_directory = os.path.join(dataset_directory, "Val")

# Data preparation
training_data = TripletImageDataset(train_directory)
validation_data = TripletImageDataset(val_directory)

train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
val_dataloader = DataLoader(validation_data, batch_size=1, shuffle=False)


# Network initialization and Hyperparameters
learning_rate = 0.000001
num_epochs = 10
model = networks.SiameseNetwork(in_channels=3, in_size=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.TripletMarginLoss(margin=1.0)


# Create an MLflow experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Face Recognition")

with mlflow.start_run():
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)

    # Train
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (data, labels) in enumerate(train_dataloader):
            anchor, positive, negative = data
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            
            anchor_output, positive_output, negative_output = model(anchor, positive, negative)

            loss = criterion(anchor_output, positive_output, negative_output)
            running_loss += loss.item()

            loss.backward()

            if batch_idx % 500 == 0:
                print(f'Epoch: [{epoch + 1}/{num_epochs}], Batch: [{batch_idx}], Loss: {running_loss / len(train_dataloader)}')

        
        optimizer.step()

        mlflow.log_metric("train_loss", running_loss / len(train_dataloader), step=epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / len(train_dataloader)}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(val_dataloader):
                anchor, positive, negative = data
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                anchor_output, positive_output, negative_output = model(anchor, positive, negative)
                loss = criterion(anchor_output, positive_output, negative_output)
                val_loss += loss.item()

                if batch_idx % 500 == 0:
                    print(f'Epoch: [{epoch + 1}/{num_epochs}], Batch: [{batch_idx}], Loss: {val_loss / len(val_dataloader)}')
                if batch_idx == 1168:
                    break
            


        mlflow.log_metric("val_loss", val_loss / len(val_dataloader), step=epoch)
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss}')

    print("Finished Training")

    # Save the model
    model_save_path = 'C:\\Users\\ardad\\Desktop\\Yazılım ve Elektronik\\Face_Recognition\\models\\siamese_model.pth'
    torch.save(model.state_dict(), model_save_path)
    mlflow.pytorch.log_model(model, "model")
    print(f'Model saved to {model_save_path}')
