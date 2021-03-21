#!/usr/bin/env python
# coding: utf-8

# In[11]:


from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import collections
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from lung_dataset import Lung_Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle


# In[2]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[3]:


# 8 convolutional layers following VGG18 format
class Net(nn.Module):
    def __init__(self, num_classes=2, pd=0.5):  # TODO: define other parameters here
        super(Net, self).__init__()
        # Conv2D: 1 input channel, 8 output channels, 3 by 3 kernel, stride of 2.
        self.conv1 = nn.Conv2d(1, 2, 3, 1)
        self.conv1_ = nn.Conv2d(2, 2, 3, 1)
        self.bn1 = nn.BatchNorm2d(2)
        self.conv2 = nn.Conv2d(2, 4, 3, 1)
        self.conv2_ = nn.Conv2d(4, 4, 3, 1)
        self.bn2 = nn.BatchNorm2d(4)
        self.conv3 = nn.Conv2d(4, 6, 3, 1)
        self.conv3_ = nn.Conv2d(6, 6, 3, 1)
        self.bn3 = nn.BatchNorm2d(6)
        self.conv4 = nn.Conv2d(6, 8, 3, 1)
        self.conv4_ = nn.Conv2d(8, 8, 3, 1)
        self.bn4 = nn.BatchNorm2d(8)
        self.mp = nn.MaxPool2d(2, stride=2)
        self.fc0 = nn.Linear(200, 200)
        self.fc1 = nn.Linear(200, 16)
        self.dropout = nn.Dropout(p=pd)
        self.fc2 = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv1_(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.mp(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv2_(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.mp(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv3_(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.mp(x)
        
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv4_(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = self.mp(x)
        
        x = torch.flatten(x, 1)
        x = self.fc0(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = x
        return output

    def count_parameters(self):
        # https://stackoverflow.com/a/62764464/5894029
        return sum(dict((p.data_ptr(), p.numel()) for p in self.parameters()).values())


# In[4]:


# 6 convolutional layers following VGG18 format
class Net2(nn.Module):
    def __init__(self, num_classes=2):  # TODO: define other parameters here
        super(Net2, self).__init__()
        # Conv2D: 1 input channel, 8 output channels, 3 by 3 kernel, stride of 2.
        self.conv1 = nn.Conv2d(1, 2, 3, 1)
        self.conv1_ = nn.Conv2d(2, 2, 3, 1)
        self.bn1 = nn.BatchNorm2d(2)
        self.conv2 = nn.Conv2d(2, 4, 3, 1)
        self.conv2_ = nn.Conv2d(4, 4, 3, 1)
        self.bn2 = nn.BatchNorm2d(4)
        self.conv3 = nn.Conv2d(4, 6, 3, 1)
        self.conv3_ = nn.Conv2d(6, 6, 3, 1)
        self.bn3 = nn.BatchNorm2d(6)
        self.conv4 = nn.Conv2d(6, 8, 3, 1)
        self.conv4_ = nn.Conv2d(8, 8, 3, 1)
        self.bn4 = nn.BatchNorm2d(8)
        self.mp = nn.MaxPool2d(2, stride=2)
        self.fc0 = nn.Linear(1350, 200)
        self.fc1 = nn.Linear(200, 16)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv1_(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.mp(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv2_(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.mp(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv3_(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.mp(x)

        x = torch.flatten(x, 1)
        x = self.fc0(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = x
        return output

    def count_parameters(self):
        # https://stackoverflow.com/a/62764464/5894029
        return sum(dict((p.data_ptr(), p.numel()) for p in self.parameters()).values())


# In[5]:


# 4 convolutional layers following VGG18 format
class Net4(nn.Module):
    def __init__(self, num_classes=2,pd=0.5):
        super(Net4, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 1)
        self.conv1_ = nn.Conv2d(2, 2, 3, 1)
        self.bn1 = nn.BatchNorm2d(2)
        self.conv2 = nn.Conv2d(2, 4, 3, 1)
        self.conv2_ = nn.Conv2d(4, 4, 3, 1)
        self.bn2 = nn.BatchNorm2d(4)
        self.conv3 = nn.Conv2d(4, 6, 3, 1)
        self.conv3_ = nn.Conv2d(6, 6, 3, 1)
        self.bn3 = nn.BatchNorm2d(6)
        self.conv4 = nn.Conv2d(6, 8, 3, 1)
        self.conv4_ = nn.Conv2d(8, 8, 3, 1)
        self.bn4 = nn.BatchNorm2d(8)
        self.mp = nn.MaxPool2d(2, stride=2)
        self.fc0=nn.Linear(4* 34* 34,200)
        self.fc1 = nn.Linear(200, 16)
        self.dropout = nn.Dropout(p=pd)
        self.fc2 = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv1_(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.mp(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv2_(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.mp(x)
        
        x = torch.flatten(x, 1)
        x = self.fc0(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = x
        return output

    def count_parameters(self):
        # https://stackoverflow.com/a/62764464/5894029
        return sum(dict((p.data_ptr(), p.numel()) for p in self.parameters()).values())


# In[6]:


def train(num_epochs, model, trainloader, testloader, criterion, optimizer, path, model_name, binary, verbose):
    num_param = model.count_parameters()
    print("Training {} parameters".format(num_param))
    # Initialize lists for plotting of graph
    loss_list = []
    acc_list = []
    test_loss_list = []
    test_acc_list = []
    path_list = []
    
    # Getting labels for binary or multi-class situation
    if binary:
        l = [0,1]
    else:
        l = [0,1,2]
        
    # Start training
    for epoch in range(num_epochs):
        
        # Set model to training mode
        model.train()
        
        # Initialize variables
        labels_full = torch.randn(0)
        pred_full = torch.randn(0)
        epoch_train_loss = 0
        epoch_test_loss = 0
        
        for i, data in tqdm(enumerate(trainloader, 0)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # get the prediction
            _, pred = torch.max(outputs, 1)
            # append data to generate confusion matrix and calculate loss/epoch
            pred_full = torch.cat((pred_full, pred))
            labels_full = torch.cat((labels_full, labels))
            epoch_train_loss += outputs.shape[0]*loss.item()

        # get training loss, accuracy and weighted accuracy       
        avg_train_loss = epoch_train_loss / len(trainloader.dataset)
        cm = confusion_matrix(labels_full.numpy(), pred_full.numpy())
        acc = get_acc(cm, l)
        _, sk_recall, _, _ = precision_recall_fscore_support(labels_full.numpy(), pred_full.numpy(),
                                                             average=None, labels=l, zero_division=0)
        weighted_acc, recall, recall_type = get_weighted_acc(acc, sk_recall, trainloader.dataset.classes)
        
        # Evaluate test set
        model.eval()
        with torch.no_grad():
            test_labels_full = torch.randn(0)
            test_pred_full = torch.randn(0)
            for test_idx, test_data in tqdm(enumerate(testloader, 0)):
                test_inputs, test_labels = test_data
                test_outputs = model(test_inputs)
                test_loss = criterion(test_outputs, test_labels)
                _, test_pred = torch.max(test_outputs, 1)
                test_pred_full = torch.cat((test_pred_full, test_pred))
                test_labels_full = torch.cat((test_labels_full, test_labels))
                epoch_test_loss += test_outputs.shape[0]*test_loss.item()
        test_cm = confusion_matrix(test_labels_full.numpy(), test_pred_full.numpy())
        test_acc = get_acc(test_cm, l)
        avg_test_loss = epoch_test_loss / len(testloader.dataset)
        _, test_sk_recall, _, _ = precision_recall_fscore_support(test_labels_full.numpy(), test_pred_full.numpy(), 
                                                                  average=None, labels=l, zero_division=0)
        test_weighted_acc, test_recall, test_recall_type = get_weighted_acc(test_acc, test_sk_recall, testloader.dataset.classes)
        
        # Ensure model is back to training mode
        model.train()
        
        # If verbose, print statistics
        if verbose:
            curr_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('--- Epoch: %d/%d @ ' % (epoch + 1, num_epochs) + curr_time, '\n Train Loss: %.5f \n Train Accuracy: %.5f' %
                  (avg_train_loss, acc), '\n Train', recall_type, recall, '\n Weighted Accuracy:', weighted_acc,
                 '\n Test Loss: %.5f \n Test Accuracy: %.5f' % (avg_test_loss, test_acc), '\n Test', test_recall_type, test_recall,
                 '\n Test weighted Accuracy:', test_weighted_acc)
        
        # Save to list for plotting after
        loss_list.append(avg_train_loss)
        acc_list.append(weighted_acc)
        test_loss_list.append(avg_test_loss)
        test_acc_list.append(test_weighted_acc)
        
#       Save model to path
        curr_time = datetime.now().strftime("%H:%M:%S")
        save_path = path + '/' + model_name + curr_time + '.pth'
        path_list.append(save_path)
        torch.save(model.state_dict(), save_path)

    print('Finished Training')
    return loss_list, acc_list, test_loss_list, test_acc_list, path_list, num_param


# In[7]:


def get_acc(cm, l):
    score = 0
    for j in range(len(l)):
        score += cm[j,j]
    acc = score/cm.sum()
    return acc

def get_weighted_acc(acc, recall, classes):
    if classes == [['covid'], ['non-covid']]:
        weighted_acc = 0.8*acc+0.2*recall[0]
        return weighted_acc, recall[0], "covid recall"
    elif classes == [['normal'], ['non-covid', 'covid']]:
        weighted_acc = 0.8*acc+0.2*recall[1]
        return weighted_acc, recall[1], "infected recall"
    elif classes == [['normal'], ['non-covid'], ['covid']]:
        weighted_acc = 0.8*acc+0.2*recall[2]
        return weighted_acc, recall[2], "covid recall"
    else:
        return 0, 0, "error"
    
def save_metrics(path, model_name, loss_list, acc_list, test_loss_list, test_acc_list, path_list, num_param):
    # TODO save best performing metrics
    best_idx = test_acc_list.index(max(test_acc_list))
    with open(path+'/'+model_name+"_results.txt", "w") as f:
        f.write("num of parameters: " + str(num_param) + "\n")
        f.write("best model weights saved at: " + path_list[best_idx] + "\n")
        f.write("train loss: " + str(loss_list[best_idx])+ "\n")
        f.write("train weighted accuracy: " + str(acc_list[best_idx])+ "\n")
        f.write("test loss: " + str(test_loss_list[best_idx])+ "\n")
        f.write("test weighted accuracy: " + str(test_acc_list[best_idx])+ "\n")


# In[8]:


def plot_learning_curve(train_loss_list, train_weighted_acc_list, test_loss_list, test_weighted_acc_list, save_path, model_name):
    plt.plot(range(1, len(train_loss_list)+1), train_loss_list, label="train_loss")
    plt.plot(range(1, len(test_loss_list)+1), test_loss_list, label="test_loss")
    plt.plot(range(1, len(train_weighted_acc_list)+1), train_weighted_acc_list, label="train_weighted_acc")
    plt.plot(range(1, len(test_weighted_acc_list)+1), test_weighted_acc_list, label="test_weighted_acc")
    plt.title(model_name + ': Weighted Accuracy and Loss against Epochs')
    plt.ylabel('metrics')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(save_path + "/" + model_name+'.jpg')
    plt.show()
    return None


# In[9]:


def two_stage_testing(testloader, first_model, second_model, device):
        first_model.eval()
        second_model.eval()
        first_model = first_model.to(device)
        second_model = second_model.to(device)
        
        first_stage_labels = []
        second_stage_labels = []
        actual_labels = []
        images=[]
        
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = first_model(inputs)
            _, predicted = torch.max(outputs, 1)
            first_stage_labels.extend(predicted)

            outputs = second_model(inputs)
            _, predicted = torch.max(outputs, 1)
            second_stage_labels.extend(predicted)
            
            actual_labels.extend(labels)
            
            images.append(inputs.cpu())
            
        
        predicted_labels = [0 if not first_label else 2 if not second_label else 1 for
                            first_label, second_label in zip(first_stage_labels, second_stage_labels)]
        actual_labels = [x.item() for x in actual_labels]
        
        img = images[0].squeeze(dim=0)
        
        cm = confusion_matrix(np.asarray(actual_labels), np.asarray(predicted_labels))
        acc = get_acc(cm, [0,1,2])
        _, sk_recall, _, _ = precision_recall_fscore_support(np.asarray(actual_labels), np.asarray(predicted_labels),
                                                             average=None, labels=[0,1,2], zero_division=0)
        weighted_acc, recall, recall_type = get_weighted_acc(acc, sk_recall, testloader.dataset.classes)
        print("Confusion matrix:")
        print(cm)
        print('Val Accuracy: %.5f' % (acc), '\nVal', recall_type, recall, '\nWeighted Accuracy:', weighted_acc)
        
        plotGrid(predicted_labels,actual_labels,img)

        return acc, recall, weighted_acc


# In[10]:


def plotGrid(predicted_labels,actual_labels,img):
    print(img.shape)
    fig, axs = plt.subplots(6, 4,figsize=(20,25))
    l=['normal','infected(non-covid)','infected(covid)']

    for i in range(6):
        for j in range(4):
            current_id =i*4+j
            colour = caption_colour(predicted_labels[current_id],actual_labels[current_id])
            axs[i,j].imshow(torchvision.utils.make_grid(img[current_id],nrow=1).permute(1,2,0))
            caption = "Ground Truth Label: {}\n Predicted Label: {}".format(l[actual_labels[current_id]],l[predicted_labels[current_id]])
            axs[i,j].set_title(caption, color=colour)
            axs[i,j].axis('off')
            
def caption_colour(predicted_class,actual_class):
    if predicted_class == actual_class:
        return "black"
    else :
        return "red"


# In[13]:


def loader_fn(loader_path, model1_path, model2_path):
    # Load validation dataloader
    with open(loader_path, 'rb') as file:
        valloader = pickle.load(file)

    # Load best models
    first_model = Net(pd=0.9)
    first_model.load_state_dict(torch.load(model1_path))
    second_model = Net4(pd=0.1)
    second_model.load_state_dict(torch.load(model2_path))
    return first_model, second_model, valloader


# In[15]:


if __name__ == "__main__":
    get_ipython().system('jupyter nbconvert --to script functions.ipynb')

