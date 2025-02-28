from google.colab import drive
import os
import shutil
import zipfile


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
from pathlib import Path
import random
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

root_dir = Path('/kaggle/input/ct-kidney-dataset-normal-cyst-tumor-and-stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone')

cv = '/kaggle/input/ct-kidney-dataset-normal-cyst-tumor-and-stone/kidneyData.csv'

df = pd.read_csv('/kaggle/input/ct-kidney-dataset-normal-cyst-tumor-and-stone/kidneyData.csv')

df.head()

df = df.drop('Unnamed: 0', axis=1)

for dir_path,dir_names, file_names in os.walk(root_dir):
  print(f"There are {len(dir_names)} directories and {len(file_names)} images in {dir_path}")

sns.countplot(x=df['Class'])

img_path_list = list(root_dir.glob('*/*'))
len(img_path_list), len(df)

img_path_list[:5]

five = [name.stem for name in img_path_list[:5]]

five

df[df['image_id'].isin(five)]

df[df['image_id'] == 'Normal- (4786)']

labal = df[df['image_id'].isin(five)]['Class'].to_list()
labal

def plot_img(path, img_num,cols):
  img_path_list = list(path.glob('*/*'))
  img_num = min(img_num,len(img_path_list))
  ran_img_list = random.sample(img_path_list,img_num)

  img_name = [name.stem for name in ran_img_list]
  labal = df[df['image_id'].isin(img_name)]['Class'].to_list()

  cols = min(cols,img_num)
  rows = (img_num + cols+1) // cols

  fig,axes = plt.subplots(rows,cols,figsize = (10,10))
  axes = axes.flatten()

  for i in range(img_num):
    img = plt.imread(ran_img_list[i])
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f"label: {labal[i]}")

  for j in range(i +1, len(axes)):
    fig.delaxes(axes[j])

  plt.tight_layout()
  plt.show()

plot_img(root_dir,25,5)

class Kidney(Dataset):
  def __init__(self,root_dir, csv, transform=None):
    self.root_dir = root_dir
    self.csv = pd.read_csv(csv)
    self.transform = transform
    self.img_path_list = list(root_dir.glob('*/*'))
  def __len__(self):
     return len(self.img_path_list)

  def __getitem__(self,idx):
      img_path = self.root_dir /self.csv.iloc[idx,3] / (self.csv.iloc[idx,1] + '.jpg')
      img = Image.open(img_path).convert("RGB")
      labal = self.csv.iloc[idx,4]

      if self.transform is not None:
        img = self.transform(img)

      return img, labal

train_transsform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

dataset = Kidney(root_dir,cv,transform=train_transsform)

train, test = torch.utils.data.random_split(dataset,[int(0.8*len(dataset)),len(dataset) - int(0.8*len(dataset))])

train_dataloader = DataLoader(train,batch_size=8,shuffle=True)
test_dataloader = DataLoader(test,batch_size=8,shuffle=False)

dataset[4]

root_dir / (dataset.csv.iloc[2,1] + '.jpg')

root_dir / (df.iloc[2]['image_id'] + '.jpg')

root_dir / df.iloc[2,0]

df.iloc[2,0]

ran_img_list = random.sample(img_path_list,5)
for i in ran_img_list:
  fig,axes = plt.subplots(1,2,)
  img = Image.open(i).convert("RGB")

  axes[0].imshow(img)
  axes[0].axis('off')
  axes[0].set_title("Original")

  trans_img = train_transsform(img)
  trans_img = trans_img.permute(1, 2, 0).numpy()

  axes[1].imshow(trans_img)
  axes[1].axis('off')
  axes[1].set_title("Transformed")

class Model_0(nn.Module):
    def __init__(self,
                 class_num=4,
                 out_1=32,
                 out_2=64,
                 out_3=128,
                 out_4=256):
        super().__init__()

        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=out_1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_1)
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_2)

        self.cnn3 = nn.Conv2d(in_channels=out_2, out_channels=out_3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_3)
        self.cnn4 = nn.Conv2d(in_channels=out_3, out_channels=out_4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(out_4)

        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(out_4 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, class_num)

    def forward(self, x):
        x = F.relu(self.bn1(self.cnn1(x)))
        x = F.relu(self.bn2(self.cnn2(x)))
        x = self.max_pool(x)  # 128x128 -> 64x64

        x = F.relu(self.bn3(self.cnn3(x)))
        x = F.relu(self.bn4(self.cnn4(x)))
        x = self.max_pool(x)  # 64x64 -> 32x32

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_step(model, dataloader, loss_fn, acc_fn, optimizer, device):
  size = len(dataloader.dataset)
  batch_size = len(dataloader)
  model.train()
  train_loss, train_acc = 0, 0
  for X, y in dataloader:
    X, y = X.to(device), y.to(device)

    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    acc = acc_fn(y_pred.argmax(dim=1), y)
    train_acc += acc

  return train_acc / batch_size, train_loss / batch_size

def test_step(model, dataloader, loss_fn, acc_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, test_acc = 0.0, 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            acc = acc_fn(y_pred.argmax(dim=1), y)
            test_acc += acc

    return test_acc / num_batches, test_loss / num_batches

def trainx(model, train_dataloader, test_dataloader, loss_fn, acc_fn, optimizer, device, epochs):
    train_acc_list, test_acc_list = [], []
    train_loss_list, test_loss_list = [], []

    for epoch in range(epochs):
        train_acc, train_loss = train_step(model, train_dataloader, loss_fn, acc_fn, optimizer, device)
        test_acc, test_loss = test_step(model, test_dataloader, loss_fn, acc_fn, device)

        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    return train_acc_list, test_acc_list, train_loss_list, test_loss_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KidneyDiseaseModel(nn.Module):
    def __init__(self, num_classes=4):
        super(KidneyDiseaseModel, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = KidneyDiseaseModel(num_classes=4).to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def acc_fn(y_pred, y_true):
    _, y_pred_labels = torch.max(y_pred, 1)
    return (y_pred_labels == y_true).sum().item() / len(y_true)

epochs = 10
train_acc_list, test_acc_list, train_loss_list, test_loss_list= trainx(model, train_dataloader, test_dataloader, loss_fn, acc_fn, optimizer, device, epochs)

import matplotlib.pyplot as plt

def plot_metrics(train_acc, test_acc, train_loss, test_loss, epochs):
    fig, ax1 = plt.subplots(figsize=(8,5))

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.plot(range(1, epochs+1), train_acc, 'o-', label="Train Accuracy", color='tab:blue')
    ax1.plot(range(1, epochs+1), test_acc, 's-', label="Test Accuracy", color='tab:cyan')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:red')
    ax2.plot(range(1, epochs+1), train_loss, 'o--', label="Train Loss", color='tab:red')
    ax2.plot(range(1, epochs+1), test_loss, 's--', label="Test Loss", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2.grid(False)

    fig.suptitle('Training & Testing Accuracy and Loss')
    fig.legend(loc="upper center", bbox_to_anchor=(0.7, 0.55), ncol=2)

    plt.show()

print(len(train_acc_list), len(test_acc_list), len(train_loss_list), len(test_loss_list))

plot_metrics(train_acc_list, test_acc_list, train_loss_list, test_loss_list, epochs)

torch.save(model.state_dict(), "model.pth")



from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

# تحميل النموذج
class KidneyDiseaseModel(nn.Module):
    def __init__(self, num_classes=4):
        super(KidneyDiseaseModel, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KidneyDiseaseModel(num_classes=4)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# تحضير التحويلات نفسها المستخدمة في التدريب
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    return {"prediction": prediction}



