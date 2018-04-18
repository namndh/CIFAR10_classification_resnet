import torch 
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='/home/t3min4l/workspace/pytorch-tutorial/data', train=True, download=True, transform=transformer)
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=50 , sampler=None, shuffle=True, batch_sampler=None)

testset = torchvision.datasets.CIFAR10(root='/home/t3min4l/workspace/pytorch-tutorial/data', train=False, download=True, transform=transformer)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=50, shuffle=False, batch_sampler=None)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(trainloader.__len__())
print(testloader.__len__())
# images, labels = dataiter.next()
# print(images[0].shape)

class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
  
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()
net.cuda()  
print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(50):
    running_loss = 0.0
    for i, data in enumerate(trainloader,0):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print('Finished Training')

correct = 0
total = 0

for epoch in range(50):
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images).cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()

print('Accuracy of network over the test set:{}'.format(correct/total))

# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# for data in testloader:
#     images, labels = data

#     outputs = net(Variable(images).cuda())
#     _, predicted = torch.max(outputs.data, 1)
#     c = (predicted.cpu() == labels).squeeze()
#     for i in range(len(labels)):
#         label = labels[i]
#         class_correct[label] += c[i]
#         class_total[label] += 1

# for i in range(10):
#     print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))