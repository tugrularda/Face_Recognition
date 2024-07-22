import os
import pandas as pd
import random
import torch
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset


#Arrange the labels file
import pandas as pd

# Load the CSV file
file_path = 'C:\\Users\\ardad\\Desktop\\Yazılım ve Elektronik\\Data\\Train\\labels.txt'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
df.head()
# Count the occurrence of each label
label_counts = df['label'].value_counts()

# Find labels that are unique (appear only once)
unique_labels = label_counts[label_counts == 1].index

# Add '-' to the beginning of the unique labels in the dataframe
df['label'] = df['label'].apply(lambda x: f"-{x}" if x in unique_labels else x)

#import ace_tools as tools; tools.display_dataframe_to_user(name="Updated Labels Dataframe", dataframe=df)

# Save the updated dataframe to a new CSV file
updated_file_path = 'C:\\Users\\ardad\\Desktop\\Yazılım ve Elektronik\\Data\\Train\\updated_labels.txt'
df.to_csv(updated_file_path, index=False)

updated_file_path



#Device settings
if(torch.cuda.is_available()):
    device = 'cuda'
    print("\033[32mCUDA is used as the device.\033[0m")
else:
    device = 'cpu'
    print("\033[31mCPU is used as the device.\033[0m")

    
transform_ = transforms.Compose(
    [transforms.Resize((256,256)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


#Creates dataset from custom images
class TripletImageDataset(Dataset):
    def __init__(self, img_dir, transform=transform_, target_transform=None):
        labels_file = os.path.join(img_dir, 'labels.txt')
        self.img_labels = pd.read_csv(labels_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        while int(self.img_labels.iloc[idx, 1]) <= 0:
            if idx < len(self.img_labels) - 1:
                idx += 1
            else:
                idx = 1

        anchor_img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        anchor_image = read_image(anchor_img_path).float()
        anchor_label = self.img_labels.iloc[idx, 1]

        positive_idx = idx
        while positive_idx == idx or self.img_labels.iloc[positive_idx, 1] != anchor_label:
            positive_idx = random.randint(0, len(self.img_labels) - 1)

        positive_img_path = os.path.join(self.img_dir, self.img_labels.iloc[positive_idx, 0])
        positive_image = read_image(positive_img_path).float()

        negative_idx = idx
        while self.img_labels.iloc[negative_idx, 1] == anchor_label:
            negative_idx = random.randint(0, len(self.img_labels) - 1)

        negative_img_path = os.path.join(self.img_dir, self.img_labels.iloc[negative_idx, 0])
        negative_image = read_image(negative_img_path).float()

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return (anchor_image, positive_image, negative_image), anchor_label


class TestImageDataset(Dataset):
    def __init__(self, img_dir, transform=transform_):
        labels_file = os.path.join(img_dir, 'labels.txt')
        self.img_labels = pd.read_csv(labels_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).float()
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def raw_images(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path).float()
        label = self.img_labels.iloc[idx, 1]
        return image, label