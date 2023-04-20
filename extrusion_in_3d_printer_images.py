import torch
import torch.nn as nn
import torchvision 
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch.optim as optim
from transformers import AdamW
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torchvision.transforms as transforms
from timm.models import vision_transformer 
from sklearn.model_selection import train_test_split
from custom_dataset import CustomDataset, CustomDatasetTest

# Loading train and test dataframes
train_df = pd.read_csv('/scratch/ag8766/yogya/train.csv')  
test_df = pd.read_csv('/scratch/ag8766/yogya/test.csv')    

train_df = train_df.drop(["printer_id", "print_id"], axis = 1)
test_df = test_df.drop(["printer_id", "print_id"], axis = 1)

image_dir = "/scratch/ag8766/yogya/images"

train_df['img_path'] = image_dir + '/' + train_df['img_path'] 
test_df['img_path'] = image_dir + '/' + test_df['img_path'] 

# Defining the data transforms
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_val_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])



train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)


# Creating custom datasets 
train_dataset = CustomDataset(train_df, transform_train)
val_dataset = CustomDataset(val_df, transform_val_test)
test_dataset = CustomDatasetTest(test_df, transform_val_test)


# Creating data loaders for train, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# Defining the Custom Model
class ViTCustom(nn.Module):
    def __init__(self, num_classes):
        super(ViTCustom, self).__init__()
        self.vit = vit.vit_small_patch16_224(pretrained=True)
        num_ftrs = self.vit.head.in_features
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.vit(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initializing the Custom ViT model with custom top layers
model = ViTCustom(num_classes=2)

# Putting the model on a GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.001)

# learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train
num_epochs = 40 
best_val_acc = 0.0 

for epoch in range(num_epochs):
    print(f'Epoch [{epoch+1}/{num_epochs}] Started')
    model.train() 
    train_loss = 0.0
    
    train_progress = tqdm(train_loader, desc=f"Training epoch {epoch}")

    for inputs, labels in train_progress:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        
    scheduler.step()

    # Validate 
    model.eval()  
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    val_progress = tqdm(val_loader, desc=f"Validation epoch {epoch}")

    with torch.no_grad():
        for inputs, labels in val_progress:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = val_correct / val_total
    val_f1_score = f1_score(labels.cpu().numpy(), predicted.cpu().numpy())

    print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss/len(train_dataset):.4f} ')
    print(f'Val Loss: {val_loss/len(val_dataset):.4f} Val Acc: {val_acc:.4f} ')
    print(f'Val F1 score: {val_f1_score:.4f}')

    # Saving the best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state_dict = model.state_dict()
        best_model = model
        torch.save(best_model, 'best_model_ViT.pth')

# Loading the best model state dict
model.load_state_dict(best_model_state_dict)

# Test
model.eval()  
test_preds = []
with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)

        outputs = model(inputs)
        _, predicted = outputs.max(1)

        test_preds.extend(predicted.cpu().numpy().tolist())

# Saveing the predicted labels to a CSV file
test_df['predicted_label'] = test_preds
test_df.to_csv('test_preds.csv', index=False)

print("Test predictions saved to test_preds.csv")


df = pd.read_csv("test_preds.csv")

df[['0', '1', '2', '3', '4', '5', '6', '7']] = df['img_path'].str.split('/', expand = True)

df = df.drop(['img_path'], axis = 1)

df['img_path'] = df[['5', '6', '7']].apply(lambda x: '/'.join(x), axis=1)

df = df.drop(['0', '1', '2', '3', '4', '5', '6', '7' ], axis = 1)

df.rename(columns={'predicted_label': 'has_under_extrusion'}, inplace=True)

df = df[['img_path', 'has_under_extrusion']]

df.to_csv('submission_7.csv', index=False)

