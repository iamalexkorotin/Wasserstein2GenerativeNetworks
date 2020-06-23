import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        top_hidden = F.relu(self.fc2(x))
        x = self.fc3(top_hidden)
        return x, top_hidden
    
def train_lenet(model, optimizer, epoch, train_loader, log_interval):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output,_ = model(data.cuda())
        loss = loss_fn(output, target.cuda())
        loss.backward()
        optimizer.step()
            
def test_lenet(model, epoch, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
    for data, target in test_loader:
        output,_ = model(data.cuda())
        test_loss += loss_fn(output, target.cuda()).data
        pred = np.argmax(output.cpu().data, axis=1)
        correct = correct + np.equal(pred, target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('Test set, Epoch {} , Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))