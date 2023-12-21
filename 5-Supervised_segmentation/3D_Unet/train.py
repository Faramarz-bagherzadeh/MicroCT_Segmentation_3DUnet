
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip
from sklearn.metrics import f1_score
import numpy as np
from torchvision import transforms
import UNET
import dataloader
import matplotlib.pyplot as plt

train_section = [0,0.8]
test_section = [0.8,1]
# while choosing patch size need to consider the test volume size (depth) 
# in such way to include at least one patch 
patch_size = 20
batch_size = 4




# Define the data augmentation transforms
transform = transforms.Compose([
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    ToTensor()
])


# Set the paths to your data and target files

data_path = 'data/Downsized_corespoinding_to_60micron_.tif'
target_path = "data/Downsized_to_60micron_56%.tif"

#test size is the same path but different part of the file will be selected

test_data_path = 'data/Downsized_corespoinding_to_60micron_.tif'
test_target_path= "data/Downsized_to_60micron_56%.tif"

# Create instances of your training and test datasets and data loaders
train_dataset = dataloader.CustomDataset(data_path, target_path,
                                         section=train_section,
                                         patch_size=patch_size ,
                                         transform=None)


test_dataset = dataloader.CustomDataset(test_data_path,
                                        test_target_path,
                                        section=test_section,
                                        patch_size=patch_size ,
                                        transform=None)

train_data_loader = DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=True)

test_data_loader = DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=False)

# Define your training function
def train(model, train_data_loader, test_data_loader, num_epochs, learning_rate):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        
        for data, target in train_data_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss / len(train_data_loader):.4f}")

        # Calculate F1-score on test set
        model.eval()
        with torch.no_grad():
            true_labels = []
            predicted_labels = []

            for data, target in test_data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                predicted = output.argmax(dim=1)

                true_labels.extend(target.cpu().numpy().flatten())
                predicted_labels.extend(predicted.cpu().numpy().flatten())

            f1 = f1_score(true_labels, predicted_labels, average='macro')
            print(f"F1-score: {f1:.4f}")

# Create an instance of the model
model = UNET.UNet(in_channels=1, out_channels=1)

# Set the hyperparameters
num_epochs = 1
learning_rate = 0.001

# Train the model
train(model, train_data_loader, test_data_loader, num_epochs, learning_rate)



#For checking the data loader:
#test = train_dataset.__getitem__(10)[0][0,:,:]
#plt.imshow(test[10],'gray')



















