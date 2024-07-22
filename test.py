import os
from matplotlib import pyplot as plt
import pandas as pd
import random
import torch
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import mlflow
import mlflow.pytorch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import networks
from datasetmanager import *

# Define device settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\033[32m{device.upper()} is used as the device.\033[0m")

# Transform
transform_ = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Directories
dataset_directory = 'C:\\Users\\ardad\\Desktop\\Yazılım ve Elektronik\\Data'
test_directory = os.path.join(dataset_directory, "Test")
db_directory = os.path.join(dataset_directory, "Database")


# Distance calculator
def calculated_distance(model, ref_img, compared_img):
    ref_vector = model.forward_one(ref_img)
    compared_vector = model.forward_one(compared_img)
    pdist = nn.PairwiseDistance()
    distance = pdist(ref_vector, compared_vector)
    return distance.item()



def main():
    # Data preparation
    test_data = TestImageDataset(test_directory, transform_)
    db_data = TestImageDataset(db_directory, transform_)
    print(type(test_data.__getitem__(0)[0]))
    print(len(test_data))
    print(type(db_data))
    print(len(db_data))

    # Model
    model_path = 'C:\\Users\\ardad\\Desktop\\Yazılım ve Elektronik\\Face_Recognition\\models\\siamese_model_f2_2.pth'
    model = networks.SiameseNetwork(in_channels=3, in_size=256).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create an MLflow experiment
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Face Recognition Test")

    
    # Prepares the test data and makes a prediction
    def predict(dblabel, threshold, samelabel, list_true, list_pred):
        test_index = 0
        while (test_data.__getitem__(test_index)[1].item() != dblabel) == samelabel:
            test_index = random.randint(0, len(test_data) - 1)
        test_image, test_label = test_data.__getitem__(test_index)
        test_image = test_image.to(device)

        distance = calculated_distance(model, db_image, test_image)
        print(f"{dblabel} - {test_label}: {distance}")
        prediction = db_label if distance < threshold else 0
        list_true.append(test_label if samelabel else 0)
        list_pred.append(prediction)

    # Start testing
    with torch.no_grad():
        threshold_distance_0 = 37.0
        threshold_distance_f1 = 36.0
        threshold_distance_f2 = 20.5        
        threshold_distance_f2_1 = 16.5
        threshold_distance_f2_2 = 14.0
        true_labels = []
        predicted_labels = []

        for db_image, db_label in db_data:
            db_image = db_image.to(device)

            predict(db_label, threshold_distance_f2_2, True, true_labels, predicted_labels)
            predict(db_label, threshold_distance_f2_2, False, true_labels, predicted_labels)

    print(true_labels)
    print(predicted_labels)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    print(f"Acc: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    # Logging metrics to MLflow
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

if __name__ == "__main__":
    main()
