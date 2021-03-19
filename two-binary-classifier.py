#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


## what was updated:
## remove F.log_softmax(x, dim = 1) in Net
## changed optimised from SGD to Adam
### ok thanks - hk


# In[3]:


import torch
import torchvision
import torchvision.transforms as transforms


# In[4]:


from lung_dataset import Lung_Dataset
from torch.utils.data import DataLoader


# In[5]:


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# A simple mode
class Net(nn.Module):
    def __init__(self, num_classes=2):  # TODO: define other parameters here
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
        self.mp = nn.MaxPool2d(3, stride=2)
        self.fc1 = nn.Linear(128, 16)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_(x)
        x = self.bn1(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.conv2_(x)
        x = self.bn2(x)
        x = self.mp(x)
        x = self.conv3(x)
        x = self.conv3_(x)
        x = self.bn3(x)
        x = self.mp(x)
        x = self.conv4(x)
        x = self.conv4_(x)
        x = self.bn4(x)
        x = self.mp(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = x
        return output

    def count_parameters(self):
        # https://stackoverflow.com/a/62764464/5894029
        return sum(dict((p.data_ptr(), p.numel()) for p in self.parameters()).values())

criterion = nn.CrossEntropyLoss()


# In[6]:


# net.count_parameters()


# In[7]:


def train(trainloader, testloader, model, optimizer, num_epochs=2):
    print("Training {} parameters".format(net.count_parameters()))
    model = model.to(device)
    test_accuracies = []
    for epoch in range(20):  # loop over the dataset multiple times

        model.train()
        train_running_loss = 0.0
        train_total = 0
        train_correct = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum()
            train_running_loss += loss.item()
            train_total += labels.size(0)
            
        model.eval()
        
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum()
            test_running_loss += loss.item()
            test_total += labels.size(0)
            
        test_accuracy = test_correct / test_total
        test_accuracies.append(test_accuracy)
        print('[epoch {}] training loss/acc: {:.3f} {:.3f}, testing loss/acc: {:.3f} {:.3f}'.format(
            epoch + 1, 
            train_running_loss / train_total, train_correct / train_total, 
            test_running_loss / test_total, test_accuracy))

    print('Finished Training, average test accuracy of last 10 epochs: {:.3f}'.format(sum(test_accuracies[-10:])/10))


# In[8]:


bs_val = 16


# # Three classes

# In[9]:


net = Net(num_classes=3)

ld = Lung_Dataset("train", "normal/non-covid/covid")
trainloader = DataLoader(ld, batch_size = bs_val, shuffle = True)

ld = Lung_Dataset("test", "normal/non-covid/covid")
testloader = DataLoader(ld, batch_size = bs_val, shuffle = True)

# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)

print("Train batches {}, Test batches {}".format(len(trainloader), len(testloader)))
train(trainloader, testloader, net, optimizer)

# TODO: Visualisation here


# # Normal vs Infected

# In[10]:


first_net = Net(num_classes=2)

ld = Lung_Dataset("train", "normal/infected")
trainloader = DataLoader(ld, batch_size = bs_val, shuffle = True)

ld = Lung_Dataset("test", "normal/infected")
testloader = DataLoader(ld, batch_size = bs_val, shuffle = True)

optimizer = optim.Adam(first_net.parameters(), lr=0.001)

print("Train batches {}, Test batches {}".format(len(trainloader), len(testloader)))
train(trainloader, testloader, first_net, optimizer)


# # Infected vs Covid

# In[11]:


second_net = Net(num_classes=2)

ld = Lung_Dataset("train", "covid/non-covid")
trainloader = DataLoader(ld, batch_size = bs_val, shuffle = True)

ld = Lung_Dataset("test", "covid/non-covid")
testloader = DataLoader(ld, batch_size = bs_val, shuffle = True)

optimizer = optim.Adam(second_net.parameters(), lr=0.001)

print("Train batches {}, Test batches {}".format(len(trainloader), len(testloader)))
train(trainloader, testloader, second_net, optimizer)


# # Two stage classifcation

# In[12]:


import collections

def two_stage_testing(testloader, first_model, second_model):
        first_model.eval()
        second_model.eval()
        first_model = first_model.to(device)
        second_model = second_model.to(device)
        
        first_stage_labels = []
        second_stage_labels = []
        actual_labels = []
        
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
            
        
        predicted_labels = [0 if not first_label else 2 if not second_label else 1 for
                            first_label, second_label in zip(first_stage_labels, second_stage_labels)]

        first_stage_labels = [x.item() for x in first_stage_labels]
        second_stage_labels = [x.item() for x in second_stage_labels]
        actual_labels = [x.item() for x in actual_labels]
        print(first_stage_labels[:10])
        print(second_stage_labels[:10])
        print(predicted_labels[:10])
        print(actual_labels[:10])
        
        print(collections.Counter(predicted_labels))
        print(collections.Counter(actual_labels))
        
        accuracy = sum(actual == predicted for actual, predicted in zip(actual_labels, predicted_labels))/len(actual_labels)
        print("accuracy {:.3f}".format(accuracy))


# In[13]:


ld = Lung_Dataset("test", "normal/non-covid/covid")
testloader = DataLoader(ld, batch_size = bs_val, shuffle = True)

two_stage_testing(testloader, first_net, second_net)


# In[ ]:





# In[ ]:




